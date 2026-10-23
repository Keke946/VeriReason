#!/usr/bin/env python3
"""Train the fixed one-epoch E8 repair/KEEP adapter and publish SQL-free evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import DeviceMemorySampler, artifact_manifest, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config
from verireason.m4_runtime import (
    RemainderAwareCollator,
    build_accumulation_plan,
    build_remainder_aware_trainer,
)
from verireason.m7_runtime import trainable_delta_l2, trainable_parameter_snapshot
from verireason.repair_sft_runtime import (
    tokenize_repair_training,
    validate_private_split,
    validate_training_evidence,
)
from verireason.repair_sft_smoke import EXPERIMENT_ID, load_repair_sft_smoke_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e8_repair_sft_smoke.toml")
    parser.add_argument("--m4-config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--private-split", type=Path, required=True)
    parser.add_argument("--e1-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import TrainingArguments

    config = load_repair_sft_smoke_config(args.config)
    m4 = load_m4_config(args.m4_config, "configs/m2_data.toml")
    if (
        m4["model"]["model_id"] != config["model"]["model_id"]
        or m4["model"]["revision"] != config["model"]["revision"]
        or artifact_manifest(args.e1_adapter)["manifest_sha256"]
        != config["model"]["parent_adapter_manifest"]
    ):
        raise ValueError("E8 model or immutable E1 adapter identity drifted")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("E8 training requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)

    payload = json.loads(args.private_split.read_text())
    train, _audit = validate_private_split(payload, config)
    tokenizer = load_tokenizer(m4, False)
    encoded, token_receipt = tokenize_repair_training(train, tokenizer, config)
    training = config["training"]
    if len(encoded) != int(training["expected_train_examples"]):
        raise ValueError("E8 encoded training count drifted")
    plan, accumulation = build_accumulation_plan(
        encoded,
        epochs=int(training["epochs"]),
        accumulation_steps=int(training["gradient_accumulation_steps"]),
    )
    if accumulation["optimizer_steps"] != int(training["expected_optimizer_steps"]):
        raise ValueError("E8 optimizer-step plan drifted")

    base = load_base(m4, trainable=False, allow_download=False)
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=bool(training["gradient_checkpointing"])
    )
    model = PeftModel.from_pretrained(base, args.e1_adapter, is_trainable=True)
    model.config.use_cache = False
    before = trainable_parameter_snapshot(model)
    trainer_args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_steps=int(training["expected_optimizer_steps"]),
        learning_rate=float(training["learning_rate"]),
        optim=str(training["optimizer"]),
        lr_scheduler_type=str(training["scheduler"]),
        warmup_steps=int(training["warmup_optimizer_steps"]),
        bf16=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        max_grad_norm=float(training["maximum_gradient_norm"]),
        save_strategy="no",
        logging_steps=1,
        seed=int(training["seed"]),
        data_seed=int(training["seed"]),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = build_remainder_aware_trainer(
        model=model,
        args=trainer_args,
        train_dataset=plan,
        data_collator=RemainderAwareCollator(tokenizer),
        callbacks=[],
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with DeviceMemorySampler(interval_seconds=0.2) as memory:
        result = trainer.train()
    delta = trainable_delta_l2(before, model)
    gradients = [
        float(row["grad_norm"])
        for row in trainer.state.log_history
        if isinstance(row.get("grad_norm"), int | float)
    ]
    if not gradients or any(not math.isfinite(value) for value in gradients):
        raise RuntimeError("E8 training lacks finite gradient evidence")
    adapter = output / "adapter"
    partial = output / f".adapter.partial-{os.getpid()}"
    if adapter.exists() or partial.exists():
        raise FileExistsError("E8 adapter destination already exists")
    trainer.save_model(str(partial))
    tokenizer.save_pretrained(partial)
    os.replace(partial, adapter)
    evidence = {
        "schema_version": 1,
        "status": "trained",
        "experiment_id": EXPERIMENT_ID,
        "process_id": os.getpid(),
        "epochs": 1,
        "real_example_passes": accumulation["real_example_passes"],
        "optimizer_steps": trainer.state.global_step,
        "training_loss": float(result.training_loss),
        "gradient_norm": gradients[-1],
        "gradient_norm_max": max(gradients),
        "adapter_delta_l2": delta,
        "adapter_manifest_sha256": artifact_manifest(adapter)["manifest_sha256"],
        "token_receipt": token_receipt,
        "accumulation": accumulation,
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_memory_mib": memory.peak_mib,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
    evidence.update(
        validate_training_evidence(evidence, training_examples=len(encoded), config=config)
    )
    atomic_write_json(output / "training-result.json", evidence)
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
