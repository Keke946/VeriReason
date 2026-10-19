#!/usr/bin/env python3
"""Run a reviewed E2/E3/E4 GRPO contract and publish SQL-free adapter evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import artifact_manifest, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config
from verireason.m6_component_scorer import SQLiteRewardComponentScorer
from verireason.m7_runtime import (
    RewardEvidenceRecorder,
    grpo_argument_values,
    passes_signal_gate,
    summarize_trainer_diagnostics,
    trainable_delta_l2,
    trainable_parameter_snapshot,
    validate_smoke_bundle,
)
from verireason.posttraining_contract import load_posttraining_config
from verireason.trl_reward_adapter import TRLGroupRewardAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m7_e2_smoke.toml")
    parser.add_argument("--m4-config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--prepared-smoke", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--e1-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer

    started = time.monotonic()
    config = load_posttraining_config(args.config)
    m4 = load_m4_config(args.m4_config, "configs/m2_data.toml")
    bundle = json.loads(Path(args.prepared_smoke).read_text())
    validate_smoke_bundle(bundle, config)
    output = Path(args.output_dir).resolve()
    adapter_output = output / "adapter"
    if adapter_output.exists():
        raise FileExistsError("post-training adapter destination already exists")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)

    training_args = GRPOConfig(**grpo_argument_values(config, str(output / "checkpoints")))
    tokenizer = load_tokenizer(m4, False)
    rows = []
    for row in bundle["examples"]:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        rows.append(
            {"prompt": prompt, **{key: value for key, value in row.items() if key != "messages"}}
        )

    base = load_base(m4, trainable=False, allow_download=False)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.e1_adapter, is_trainable=True)
    model.config.use_cache = False
    before = trainable_parameter_snapshot(model)
    scorer = SQLiteRewardComponentScorer(args.database_root)
    reward = RewardEvidenceRecorder(
        TRLGroupRewardAdapter(
            arm=config["arm"],
            num_generations=config["generation"]["num_generations"],
            component_scorer=scorer,
            metadata_fields=("gold_sql", "database_path"),
        ),
        config["generation"]["num_generations"],
        metric_namespace="m7" if config["arm"] == "E2" else "m8",
        mean_reward_name=(
            "mean_terminal_reward" if config["arm"] == "E2" else "mean_group_relative_reward"
        ),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    result = trainer.train()
    if trainer.state.global_step != config["optimization"]["maximum_optimizer_steps"]:
        raise RuntimeError("GRPO run stopped before the planned optimizer step")
    delta = trainable_delta_l2(before, model)
    trainer.save_model(str(adapter_output))
    tokenizer.save_pretrained(adapter_output)
    diagnostics = reward.summary()
    component_diagnostics = scorer.summary()
    finite_logs = [
        row
        for row in trainer.state.log_history
        if all(
            value is None or not isinstance(value, int | float) or math.isfinite(value)
            for value in row.values()
        )
    ]
    if len(finite_logs) != len(trainer.state.log_history):
        raise RuntimeError("GRPO log history contains NaN or Inf")
    gradient_norms = [
        float(row["grad_norm"])
        for row in trainer.state.log_history
        if isinstance(row.get("grad_norm"), int | float)
    ]
    if not gradient_norms or any(not math.isfinite(value) for value in gradient_norms):
        raise RuntimeError("GRPO run lacks finite gradient-norm evidence")
    trainer_diagnostics = summarize_trainer_diagnostics(
        trainer.state.log_history,
        (
            "completions/mean_length",
            "completions/clipped_ratio",
            "entropy",
            "clip_ratio/region_mean",
            "step_time",
        ),
    )
    if diagnostics["groups"] != config["data"]["prompt_count"]:
        raise RuntimeError(
            "GRPO reward callback did not consume every selected prompt exactly once"
        )
    expected_completions = config["data"]["prompt_count"] * config["generation"]["num_generations"]
    if component_diagnostics["completion_count"] != expected_completions:
        raise RuntimeError("component telemetry does not cover every generated completion")
    signal_gate = config.get("signal_gate")
    signal_gate_passed = None
    if signal_gate is not None:
        signal_gate_passed = passes_signal_gate(diagnostics, signal_gate)
    status = (
        "trained" if delta > 0 and diagnostics["mixed_reward_groups"] > 0 else "stopped_no_signal"
    )
    evidence = {
        "schema_version": 1,
        "status": status,
        "experiment_id": config["experiment_id"],
        "arm": config["arm"],
        "optimizer_steps": trainer.state.global_step,
        "training_loss": float(result.training_loss),
        "gradient_norm_last": gradient_norms[-1],
        "gradient_norm_max": max(gradient_norms),
        "adapter_delta_l2": delta,
        "reward_diagnostics": diagnostics,
        "component_diagnostics": component_diagnostics,
        "trainer_diagnostics": trainer_diagnostics,
        "signal_gate_passed": signal_gate_passed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "wall_seconds": time.monotonic() - started,
        "adapter_manifest_sha256": artifact_manifest(adapter_output)["manifest_sha256"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(output / "training-result.json", evidence)
    print(json.dumps(evidence, indent=2))
    return 0 if status == "trained" else 2


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
