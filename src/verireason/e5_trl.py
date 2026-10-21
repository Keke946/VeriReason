"""Narrow pinned-TRL replay seam; imports the real CPU-capable trainer and torch.

No loss formula is copied. GRPOTrainer._compute_loss owns ratios, clipping and DAPO.
Replay bypasses only generation/buffer shuffling, retaining original-ID row order.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import transformers
import trl
from transformers import TrainerCallback
from trl import GRPOTrainer

from verireason.e5_interface import advantage_matrix, terminal_advantages


def require_runtime():
    if trl.__version__ != "1.9.0" or transformers.__version__ != "5.14.1":
        raise RuntimeError("E5 requires TRL 1.9.0 and Transformers 5.14.1")


def attach_tensor_advantages(scored: dict, credits: tuple) -> dict:
    """Scalar -> (B,T), immediately before the saved batch enters _compute_loss.

    Retain every other tensor object, including old logps and original window D.
    Any unreliable mapping is already an exact E2 TokenCredit, not an ID rewrite.
    """
    a, mask = scored["advantages"], scored["completion_mask"]
    if a.ndim != 1 or a.shape[0] != mask.shape[0] or len(credits) != len(a):
        raise ValueError("scalar advantage/token credit row drift")
    for value, credit in zip(a.tolist(), credits, strict=True):
        if len(credit.mask) != mask.shape[1] or len(credit.advantages) != mask.shape[1]:
            raise ValueError("token credit shape drift")
        if credit.reason == "localized":
            expected = tuple(value * m for m in credit.mask)
            if value >= 0 or any(m not in (0, 1) for m in credit.mask):
                raise ValueError("only negative terminal credit may localize")
        else:
            expected = (value,) * mask.shape[1]
        if tuple(credit.advantages) != expected:
            raise ValueError("terminal anchor/fallback mismatch")
    values = advantage_matrix(tuple(a.tolist()), credits, tuple(map(tuple, mask.tolist())))
    matrix = a.new_tensor(values)
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite token advantages")
    return {**scored, "advantages": matrix}


def replay_batches(records: list[dict], credits: tuple, device) -> list[dict]:
    """One original four-rollout group/window, D independent of localized spans."""
    if len(records) != 32 or len(credits) != 32:
        raise ValueError("complete frozen replay required")
    advantages = terminal_advantages(tuple(r["reward"] for r in records))
    batches = []
    for start in range(0, 32, 4):
        group = records[start:start + 4]
        dtype_names = {r["old_logps_dtype"] for r in group}
        if len(dtype_names) != 1:
            raise ValueError("original log-prob dtype drift within group")
        old_dtype = getattr(torch, group[0]["old_logps_dtype"])
        p = max(len(r["prompt_ids"]) for r in group)
        t = max(len(r["completion_ids"]) for r in group)
        prompt_ids, prompt_mask, ids, masks, old, padded = [], [], [], [], [], []
        for r, c in zip(group, credits[start:start + 4], strict=True):
            np, nt = len(r["prompt_ids"]), len(r["completion_ids"])
            prompt_ids.append([r["pad_id"]] * (p - np) + r["prompt_ids"])
            prompt_mask.append([0] * (p - np) + [1] * np)
            ids.append(r["completion_ids"] + [r["pad_id"]] * (t - nt))
            masks.append(r["completion_mask"] + [0] * (t - nt))
            old.append(r["old_logps"] + [0.] * (t - nt))
            from verireason.e5_credit import TokenCredit

            a = advantages[start + len(padded)]
            padded.append(TokenCredit(
                c.advantages + (a,) * (t - nt), c.mask + (0,) * (t - nt), c.reason
            ))
            if c.reason == "localized":
                padded[-1] = TokenCredit(c.advantages + (0.,) * (t - nt),
                                         c.mask + (0,) * (t - nt), c.reason)
        batch = dict(
            prompt_ids=torch.tensor(prompt_ids, device=device),
            prompt_mask=torch.tensor(prompt_mask, device=device),
            completion_ids=torch.tensor(ids, device=device),
            completion_mask=torch.tensor(masks, device=device),
            old_per_token_logps=torch.tensor(old, device=device, dtype=old_dtype),
            advantages=torch.tensor(advantages[start:start + 4], device=device),
            num_items_in_batch=torch.tensor(sum(map(sum, masks)), device=device),
        )
        batches.append(attach_tensor_advantages(batch, tuple(padded)))
    return batches


class _UpdateGuard(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.owner
        if trainer._step != 4 * (state.global_step + 1):
            raise RuntimeError("replay accumulation/row order drift")
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("nonfinite/missing parameter gradients")
        trainer._e5_grad_norm = math.sqrt(sum(float(g.float().square().sum()) for g in gradients))
        trainer.e5_session.before_update(state.global_step)

    def on_optimizer_step(self, args, state, control, **kwargs):
        trainer = self.owner
        if any(not torch.isfinite(p).all() for p in trainer.model.parameters()
               if p.requires_grad):
            raise ValueError("nonfinite updated adapter")
        trainer.e5_session.after_update(state.global_step, {
            "loss": trainer._e5_group_loss, "gradient_norm": trainer._e5_grad_norm,
            "ratio_min_last_microbatch": trainer._e5_ratio_min,
            "ratio_max_last_microbatch": trainer._e5_ratio_max,
        })


class E5ReplayTrainer(GRPOTrainer):
    """Pinned standard single-process Trainer optimizer loop over already audited IDs.

    Scalar -> matrix happens in replay_batches, before _prepare_inputs slices the
    microbatch. No stock generation, shuffle, or regeneration is used during replay.
    """

    def __init__(self, *args, **kwargs):
        require_runtime()
        self.e5_session = None
        self.e5_batches = None
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise RuntimeError("E5 smoke supports exactly one process")
        if not (self.loss_type == "dapo" and self.beta == 0 and not self.use_vllm
                and self.num_iterations == 1 and self.importance_sampling_level == "token"
                and self.args.gradient_accumulation_steps == 4
                and self.args.per_device_train_batch_size == 1
                and self.args.max_steps == 8 and self.epsilon_low == .2
                and self.epsilon_high == .28 and not self.use_liger_kernel
                and not self._entropy_bonus_enabled and not self.aux_loss_enabled
                and self.top_entropy_quantile == 1 and self.off_policy_mask_threshold is None
                and self.args.delta is None):
            raise ValueError("unsupported E5 loss/training contract")
        self.add_callback(_UpdateGuard(self))
        # Pinned TRL intentionally sets this non-None sentinel to prevent HF's
        # extra GAS division; Accelerator also uses GAS=1 for Trainer's loop.
        if self.compute_loss_func is None or self.accelerator.gradient_accumulation_steps != 1:
            raise RuntimeError("pinned TRL/HF gradient accumulation contract drift")

    def prepare_replay(self, session, records, credits):
        if session.data["phase"] != "gated" or not session.data["gate"]["passed"]:
            raise RuntimeError("optimizer/replay before gate")
        self.e5_session = session
        self.e5_batches = replay_batches(records, credits, self.accelerator.device)

    def create_optimizer(self):
        if self.e5_session is None or self.e5_session.data["phase"] != "gated":
            raise RuntimeError("optimizer creation before gate")
        return super().create_optimizer()

    def _generate_and_score_completions(self, inputs):
        raise RuntimeError("E5 replay must never regenerate")

    def _prepare_inputs(self, generation_batch):
        if (self.e5_session is None or self.e5_batches is None or not self.model.training
                or self.e5_session.data["phase"] not in {"gated", "updating"}
                or not 0 <= self._step < 32):
            raise RuntimeError("replay before gate or excess microbatch")
        self.e5_session.check_deadline()
        batch = self.e5_batches[self._step // 4]
        row = self._step % 4
        if row == 0:
            self._e5_group_loss = 0.
        return {k: v if v.ndim == 0 else v[row:row + 1] for k, v in batch.items()}

    def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any):
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        if any(x is not None and not torch.isfinite(x).all() for x in result):
            raise ValueError("nonfinite model log probabilities/entropy")
        self._e5_current_logps = result[0]
        return result

    def _compute_loss(self, model, inputs):
        old = inputs.get("old_per_token_logps")
        if old is None or old.requires_grad or old.shape != inputs["completion_ids"].shape:
            raise ValueError("frozen E1 old log-probabilities required")
        for key in ("old_per_token_logps", "advantages", "num_items_in_batch"):
            if not torch.isfinite(inputs[key]).all():
                raise ValueError("nonfinite saved loss input")
        loss = super()._compute_loss(model, inputs)  # real pinned TRL, unchanged
        ratio = (self._e5_current_logps - old).exp()
        if not torch.isfinite(loss) or not torch.isfinite(ratio).all():
            raise ValueError("nonfinite loss or importance ratio")
        self._e5_loss = float(loss.detach())
        self._e5_group_loss = getattr(self, "_e5_group_loss", 0.) + self._e5_loss
        self._e5_ratio_min = float(ratio.detach().min())
        self._e5_ratio_max = float(ratio.detach().max())
        return loss
