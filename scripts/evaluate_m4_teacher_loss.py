#!/usr/bin/env python3
"""Measure the frozen dev256 answer-only teacher-forced token-weighted loss."""

import argparse
import json
import math
from pathlib import Path

from verireason.m3_contract import atomic_write_json, sha256_json
from verireason.m3_runtime import load_base
from verireason.m4_contract import load_m4_config


def teacher_forced_answer_only_loss(model, examples):
    import torch
    import torch.nn.functional as functional

    total, tokens = 0.0, 0
    model.eval()
    for example in examples:
        input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device="cuda:0")
        labels = torch.tensor([example["labels"]], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[:, :-1, :]
        shifted = labels[:, 1:]
        mask = shifted.ne(-100)
        token_count = int(mask.sum().item())
        if not token_count:
            raise RuntimeError("M4 dev example has no answer-only tokens")
        loss_sum = functional.cross_entropy(logits[mask], shifted[mask], reduction="sum")
        total += float(loss_sum)
        tokens += token_count
    value = total / tokens
    if not math.isfinite(value):
        raise RuntimeError("M4 teacher-forced loss is NaN or Inf")
    return value, tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--update", type=int, choices=(512, 671), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_m4_config(args.config, args.m2_config)
    bundle = json.loads(Path(args.prepared).read_text())
    if (
        bundle.get("split") != "dev"
        or len(bundle.get("examples", [])) != 256
        or bundle.get("heldout_examples_loaded") != 0
    ):
        raise SystemExit("teacher-forced diagnostic requires frozen dev256 with heldout=0")
    from peft import PeftModel

    base = load_base(config, trainable=False, allow_download=False)
    model = PeftModel.from_pretrained(base, args.adapter_checkpoint, is_trainable=False)
    loss, token_count = teacher_forced_answer_only_loss(model, bundle["examples"])
    payload = {
        "schema_version": 1,
        "scope": "dev256_teacher_forced_only",
        "update": args.update,
        "validation_selection_sha256": bundle["selection_sha256"],
        "sample_ids": bundle["sample_ids"],
        "prompt_hashes": bundle["prompt_hashes"],
        "loss": loss,
        "supervised_answer_tokens": token_count,
        "loss_definition": "teacher_forced_answer_only_token_weighted_mean",
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
        "checkpoint_contract_sha256": sha256_json(
            json.loads((Path(args.adapter_checkpoint) / "m4-contract.json").read_text())
        ),
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
