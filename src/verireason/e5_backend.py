"""HF/TRL backend for a later-authorized E5 smoke; no model loaded on import."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from verireason.e5_credit import localize_execution
from verireason.e5_interface import terminal_advantages
from verireason.e5_trl import replay_batches, require_runtime
from verireason.m3_runtime import artifact_manifest
from verireason.m7_runtime import trainable_delta_l2, trainable_parameter_snapshot
from verireason.sql_evaluator import compare_strict_results, execute_read_only


class HFReplayBackend:
    def __init__(self, trainer, database_root: Path):
        self.trainer, self.tokenizer = trainer, trainer.processing_class
        self.database_root = Path(database_root).resolve()
        self.before = trainable_parameter_snapshot(trainer.model)
        self.numerics = None

    @property
    def optimizer_steps(self):
        return self.trainer.state.global_step

    def preflight(self, context):
        require_runtime()
        if self.optimizer_steps or self.trainer.optimizer is not None:
            raise RuntimeError("E1 capture must precede optimizer creation/update")
        if not self.tokenizer.is_fast:
            raise ValueError("HF fast tokenizer required")
        if context["parent"] != "e1_m4_formal_sft_001":
            raise ValueError("immutable E1 required")

    def database(self, row):
        if row["split"] != "train" or not row["operational_eligible"]:
            raise ValueError("training-only database required")
        p = (self.database_root / row["database_path"]).resolve()
        if not p.is_relative_to(self.database_root) or not p.is_file():
            raise ValueError("training database unavailable/outside root")
        return p

    def capture_group(self, row, group):
        if self.optimizer_steps or self.trainer.optimizer is not None:
            raise RuntimeError("capture after optimizer is forbidden")
        tokenizer, model = self.tokenizer, self.trainer.model
        inputs = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.trainer.accelerator.device) for k, v in inputs.items()}
        prompt_ids = inputs["input_ids"][0].tolist()
        model.eval()
        with torch.no_grad():
            # Actual generated IDs are captured before any decoding or re-encoding.
            output = model.generate(
                **inputs, do_sample=True, num_return_sequences=4, max_new_tokens=384,
                temperature=1.0, top_p=1.0, top_k=0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )[:, len(prompt_ids):]
        result = []
        database = self.database(row)
        gold = execute_read_only(database, row["gold_sql"])
        if not gold.successful:
            raise ValueError("gold execution failed")
        for j, generated in enumerate(output.tolist()):
            eos = next((i for i, x in enumerate(generated) if x == tokenizer.eos_token_id), None)
            truncated = eos is None
            text_ids = generated if truncated else generated[:eos]
            text = tokenizer.decode(text_ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            mask = [0] * len(generated) if truncated else (
                [1] * (eos + 1) + [0] * (len(generated) - eos - 1)
            )
            pids = inputs["input_ids"]
            cids = torch.tensor([generated], device=pids.device)
            cmask = torch.tensor([mask], device=pids.device)
            with torch.no_grad():
                old, _, _ = self.trainer._get_per_token_logps_and_entropies(
                    model, torch.cat([pids, cids], 1),
                    torch.cat([inputs["attention_mask"], cmask], 1), cids.shape[1],
                )
            prediction = execute_read_only(database, text)
            result.append(dict(
                example_id=row["example_id"], group=group, rollout=j,
                prompt_ids=prompt_ids, text=text, completion_ids=generated,
                completion_mask=mask, old_logps=old[0].tolist(),
                old_logps_dtype=str(old.dtype).removeprefix("torch."),
                truncated=truncated, eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                reward=int(compare_strict_results(prediction, gold).equal),
                terminal_status=prediction.status,
            ))
        return result

    def localize(self, record, row):
        return localize_execution(
            self.database(row), row["gold_sql"], record["text"], split="train"
        )

    def check_numerics(self, records, credits):
        """Actual pinned TRL loss + autograd, no optimizer. Retain diagnostic controls."""
        batches = replay_batches(records, credits, self.trainer.accelerator.device)
        controls = {"local_unscaled": batches, "terminal": [], "mass_matched": []}
        all_scalar = terminal_advantages(tuple(r["reward"] for r in records))
        for group, b in enumerate(batches):
            scalar = all_scalar[group * 4:group * 4 + 4]
            terminal = torch.tensor(scalar, device=b["advantages"].device)
            controls["terminal"].append({**b, "advantages": terminal})
            matrix = b["advantages"].clone()
            for i in range(4):
                if scalar[i] < 0:
                    n = int(b["completion_mask"][i].sum())
                    k = int(((matrix[i] != 0) * b["completion_mask"][i]).sum())
                    if k:
                        matrix[i] *= n / k
            controls["mass_matched"].append({**b, "advantages": matrix})
        self.trainer.model.eval()
        self.numerics = {}
        parameters = [p for p in self.trainer.model.parameters() if p.requires_grad]
        for mode, group_batches in controls.items():
            loss_total = 0.
            grad_max = 0.
            for batch in group_batches:
                # Four microbatches keep this pre-update check within the smoke memory budget.
                for i in range(4):
                    micro = {k: v if v.ndim == 0 else v[i:i + 1] for k, v in batch.items()}
                    loss = self.trainer._compute_loss(self.trainer.model, micro)
                    grad = torch.autograd.grad(loss, parameters, allow_unused=True)
                    if any(g is not None and not torch.isfinite(g).all() for g in grad):
                        raise ValueError("nonfinite pre-update autograd")
                    grad_max = max([grad_max] + [
                        float(g.abs().max()) for g in grad if g is not None
                    ])
                    loss_total += float(loss.detach())
            self.numerics[mode] = {"loss": loss_total, "parameter_gradient_max": grad_max}
        if trainable_delta_l2(self.before, self.trainer.model) != 0:
            raise RuntimeError("E1 changed before gate")

    def update(self, records, credits, session):
        if self.numerics is None:
            self.numerics = session.data["numerics"]
        self.trainer.prepare_replay(session, records, credits)
        self.trainer.train()

    def finish(self, directory):
        """Actual separate-process CPU reload; no second base-model load needed."""
        delta = trainable_delta_l2(self.before, self.trainer.model)
        path = Path(directory) / "adapter"
        self.trainer.save_model(str(path))
        self.tokenizer.save_pretrained(path)
        manifest = artifact_manifest(path)["manifest_sha256"]
        # Rehydrate the adapter into a fresh tiny architecture of the same configured shapes
        # for a tiny CPU test; production PEFT reload uses recorded base config, no base weights.
        config = copy.deepcopy(self.trainer.model.config)
        config.to_json_file(str(Path(directory) / "base-config.json"))
        command = [sys.executable, "-m", "verireason.e5_reload", str(path),
                   str(Path(directory) / "base-config.json"), manifest]
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        return dict(adapter_delta_l2=delta, independent_reload=result.returncode == 0,
                    adapter_manifest_sha256=manifest, numerics=self.numerics)


def load_training_selection(manifest_path, prepared_path, tokenizer):
    """Reuse frozen M7 selection; reject mixed prepared bundles before use."""
    from verireason.m3_contract import sha256_file
    from verireason.m7_contract import load_m7_smoke_config, select_smoke_rows

    config = load_m7_smoke_config("configs/m7_e2_smoke.toml")
    if sha256_file(manifest_path) != config["data"]["training_manifest_sha256"]:
        raise ValueError("frozen training manifest mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    prepared = json.loads(Path(prepared_path).read_text())
    if (prepared.get("heldout_examples_loaded", 0) or prepared.get("heldout_examples_evaluated", 0)
            or any(r.get("split") != "train" for r in prepared["examples"])):
        raise ValueError("prepared input must be entirely training-only")
    selected = select_smoke_rows(manifest["examples"], count=8,
                                seed=config["data"]["selector_seed"],
                                required_features=config["data"]["required_features"])
    by_id = {r["sample_id"]: r for r in prepared["examples"]}
    rows = []
    for row in selected:
        source = by_id[row["row_id"]]
        prompt = tokenizer.apply_chat_template(source["messages"], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        rows.append(dict(
            example_id=row["row_id"], split="train", operational_eligible=True,
            source="frozen_training_manifest", prompt=prompt, gold_sql=row["sql"],
            database_path=manifest["databases"][row["db_id"]]["path"],
        ))
    return rows
