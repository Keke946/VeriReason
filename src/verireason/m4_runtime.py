"""M4 primary-epoch QLoRA helpers, including exact partial-window semantics."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verireason.m3_contract import AnswerOnlyCollator, atomic_write_json
from verireason.m3_runtime import artifact_manifest


def build_accumulation_plan(
    examples: Sequence[dict[str, Any]], *, epochs: int = 1, accumulation_steps: int = 8
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Build the primary epoch; any later epoch belongs to a separately frozen experiment."""
    if not examples or epochs != 1 or accumulation_steps < 1:
        raise ValueError("M4 accumulation plan inputs are invalid")
    plan: list[dict[str, Any]] = []
    total_padding = 0
    for epoch in range(epochs):
        epoch_rows = [dict(example) for example in examples]
        remainder = len(epoch_rows) % accumulation_steps
        divisor = remainder or accumulation_steps
        start = len(epoch_rows) - remainder if remainder else len(epoch_rows)
        for index, example in enumerate(epoch_rows):
            scale = accumulation_steps / divisor if remainder and index >= start else 1.0
            plan.append(
                {
                    **example,
                    "_m4_loss_scale": scale,
                    "_m4_is_padding": False,
                    "_m4_epoch": epoch + 1,
                }
            )
        padding = (accumulation_steps - remainder) % accumulation_steps
        for index in range(padding):
            plan.append(
                {
                    **epoch_rows[index],
                    "_m4_loss_scale": 0.0,
                    "_m4_is_padding": True,
                    "_m4_epoch": epoch + 1,
                }
            )
        total_padding += padding
    updates_per_epoch = math.ceil(len(examples) / accumulation_steps)
    return plan, {
        "planned_microbatches": len(plan),
        "real_example_passes": len(examples) * epochs,
        "real_examples_per_epoch": len(examples),
        "epochs": epochs,
        "optimizer_steps": updates_per_epoch * epochs,
        "updates_per_epoch": updates_per_epoch,
        "final_real_microbatches": len(examples) % accumulation_steps or accumulation_steps,
        "zero_gradient_padding_microbatches_per_epoch": (
            accumulation_steps - len(examples) % accumulation_steps
        )
        % accumulation_steps,
        "zero_gradient_padding_microbatches": total_padding,
        "final_real_loss_multiplier": accumulation_steps
        / (len(examples) % accumulation_steps or accumulation_steps),
        "resume_microbatches_per_optimizer_step": accumulation_steps,
    }


def scaled_compute_loss(model: Any, inputs: dict[str, Any], return_outputs: bool = False):
    """The exact Trainer compute_loss boundary tested for real/padded M4 microbatches."""
    scale = inputs.pop("m4_loss_scale")
    outputs = model(**inputs)
    mean = scale.mean()
    if hasattr(mean, "to"):
        mean = mean.to(outputs.loss.device)
    loss = outputs.loss * mean
    return (loss, outputs) if return_outputs else loss


class RequiredUpdateEvidenceCallback:
    """Force save/log at pre-registered updates and retain their elapsed-time evidence."""

    def __init__(self, required_updates: set[int], training_contract: dict[str, Any]):
        self.required_updates = required_updates
        self.training_contract = training_contract
        self.started = time.monotonic()
        self.elapsed_seconds: dict[int, float] = {}

    def as_transformers_callback(self):
        from transformers import TrainerCallback

        parent = self

        class Callback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step in parent.required_updates:
                    parent.elapsed_seconds[state.global_step] = time.monotonic() - parent.started
                    control.should_save = True
                    control.should_log = True
                return control

            def on_save(self, args, state, control, **kwargs):
                if state.global_step in parent.required_updates:
                    atomic_write_json(
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}"
                        / "m4-contract.json",
                        parent.training_contract,
                    )
                return control

        return Callback()


def build_remainder_aware_trainer(
    *, model: Any, args: Any, train_dataset: Any, data_collator: Any, callbacks: list[Any]
):
    """Build the M4 Trainer and force the known 5.14.1 loss/GAS branch.

    Qwen3's ``forward(..., **kwargs)`` makes Transformers 5.14.1 infer that the model accepts
    loss kwargs. M4's custom loss deliberately does *not* consume ``num_items_in_batch``; leaving
    that inference enabled skips Trainer's gradient-accumulation division. Fail closed after
    construction so every microbatch uses the normal Trainer divide-by-current-GAS branch.
    """
    import torch
    from torch.utils.data import SequentialSampler
    from transformers import Trainer

    class RemainderAwareTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss = scaled_compute_loss(model, inputs, return_outputs=return_outputs)
            value = loss[0] if return_outputs else loss
            if not torch.isfinite(value):
                raise RuntimeError("M4 training loss is NaN or Inf")
            return loss

    trainer = RemainderAwareTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    original = trainer.model_accepts_loss_kwargs
    if original is not True:
        raise RuntimeError(
            "M4 expected Transformers to infer model_accepts_loss_kwargs=True for Qwen3 **kwargs"
        )
    trainer.model_accepts_loss_kwargs = False
    if trainer.model_accepts_loss_kwargs is not False:
        raise RuntimeError("M4 failed to force Trainer loss/GAS normalization branch")
    trainer._m4_original_model_accepts_loss_kwargs = original
    return trainer


def required_update_observations(
    *, trainer: Any, output_dir: Path, required_updates: set[int], elapsed_seconds: dict[int, float]
) -> list[dict[str, Any]]:
    """Read the authoritative Trainer log history; missing metric evidence is fatal."""
    observations = []
    for update in sorted(required_updates):
        rows = [row for row in trainer.state.log_history if row.get("step") == update]
        row = next(
            (
                item
                for item in reversed(rows)
                if all(item.get(key) is not None for key in ("loss", "grad_norm", "learning_rate"))
            ),
            None,
        )
        checkpoint = checkpoint_directory(output_dir) / f"checkpoint-{update}"
        values = {
            "training_loss_window": row["loss"] if row else None,
            "gradient_norm": row["grad_norm"] if row else None,
            "learning_rate": row["learning_rate"] if row else None,
            "elapsed_seconds": elapsed_seconds.get(update),
        }
        if (
            row is None
            or not checkpoint.exists()
            or any(value is None or not math.isfinite(float(value)) for value in values.values())
        ):
            raise RuntimeError(f"required M4 training evidence is incomplete at update {update}")
        observations.append(
            {
                "update": update,
                **values,
                "nan_or_inf_detected": False,
                "checkpoint_manifest_sha256": _checkpoint_digest(checkpoint),
            }
        )
    return observations


class RemainderAwareCollator:
    def __init__(self, tokenizer: Any):
        import torch

        self._torch = torch
        self._base = AnswerOnlyCollator(
            tokenizer.pad_token_id,
            tensor_factory=lambda values: torch.tensor(values, dtype=torch.long),
        )

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = self._base(features)
        batch["m4_loss_scale"] = self._torch.tensor(
            [float(item["_m4_loss_scale"]) for item in features], dtype=self._torch.float32
        )
        return batch


def _checkpoint_digest(path: Path) -> str | None:
    return (
        artifact_manifest(path)["manifest_sha256"] if (path / "m4-contract.json").exists() else None
    )


def checkpoint_directory(output_dir: str | Path) -> Path:
    """The outer M4 session owns an inner Trainer checkpoint directory."""
    return Path(output_dir) / "checkpoints"


def train_formal_adapter(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    resume_from_checkpoint: str | None = None,
    training_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exactly the approved primary epoch and atomically publish its adapter."""
    from transformers import TrainingArguments

    q, ckpt = config["qlora"], config["checkpoint"]
    required_updates = set(ckpt["required_loss_checkpoint_updates"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan, accumulation = build_accumulation_plan(
        examples, epochs=q["primary_epochs"], accumulation_steps=q["gradient_accumulation_steps"]
    )
    if accumulation["optimizer_steps"] != q["primary_updates"]:
        raise ValueError("prepared primary epoch does not match frozen optimizer-step contract")
    if training_contract is None:
        raise RuntimeError("M4 training requires a frozen training contract")
    evidence_callback = RequiredUpdateEvidenceCallback(required_updates, training_contract)
    args = TrainingArguments(
        output_dir=str(checkpoint_directory(output)),
        per_device_train_batch_size=q["per_device_train_batch_size"],
        gradient_accumulation_steps=q["gradient_accumulation_steps"],
        max_steps=q["primary_updates"],
        learning_rate=q["learning_rate"],
        optim=q["optimizer"],
        lr_scheduler_type=q["scheduler"],
        warmup_steps=q["warmup_optimizer_steps"],
        bf16=True,
        gradient_checkpointing=q["gradient_checkpointing"],
        max_grad_norm=q["maximum_gradient_norm"],
        save_strategy="steps",
        save_steps=ckpt["save_optimizer_steps"],
        save_total_limit=ckpt["save_total_limit"],
        logging_steps=8,
        seed=q["seed"],
        data_seed=q["seed"],
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = build_remainder_aware_trainer(
        model=model,
        args=args,
        train_dataset=plan,
        data_collator=RemainderAwareCollator(tokenizer),
        callbacks=[evidence_callback.as_transformers_callback()],
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if trainer.state.global_step != q["primary_updates"]:
        raise RuntimeError("M4 primary epoch did not stop at update 671")
    partial, final = output / f".adapter.partial-{os.getpid()}", output / "adapter"
    if partial.exists() or final.exists():
        raise FileExistsError("atomic adapter destination already exists")
    trainer.save_model(str(partial))
    tokenizer.save_pretrained(partial)
    if training_contract is not None:
        atomic_write_json(partial / "m4-contract.json", training_contract)
    os.replace(partial, final)
    observations = required_update_observations(
        trainer=trainer,
        output_dir=output,
        required_updates=required_updates,
        elapsed_seconds=evidence_callback.elapsed_seconds,
    )
    return {
        "optimizer_steps": trainer.state.global_step,
        "training_loss": result.training_loss,
        "resume_from_checkpoint": resume_from_checkpoint,
        "adapter_dir": str(final),
        "accumulation": accumulation,
        "required_update_observations": observations,
        "primary_epoch_complete": True,
        "automatic_extension": False,
        "trainer_loss_normalization": {
            "transformers_model_accepts_loss_kwargs_original": (
                trainer._m4_original_model_accepts_loss_kwargs
            ),
            "transformers_model_accepts_loss_kwargs_forced": trainer.model_accepts_loss_kwargs,
            "gradient_accumulation_steps": q["gradient_accumulation_steps"],
        },
    }
