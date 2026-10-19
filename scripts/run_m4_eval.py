#!/usr/bin/env python3
"""Evaluate M4 Base or a freshly loaded adapter on one frozen prepared bundle."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import evaluate_model, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_m4_config(args.config, args.m2_config)
    bundle = json.loads(Path(args.prepared).read_text())
    tokenizer = load_tokenizer(config, False)
    base = load_base(config, trainable=False, allow_download=False)
    model = base
    condition = "base"
    if args.adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
        condition = "sft"
    result = evaluate_model(model, tokenizer, bundle["examples"], config)
    payload = {
        "schema_version": 1,
        "condition": condition,
        "validation_selection_sha256": bundle["selection_sha256"],
        "sample_ids": bundle["sample_ids"],
        "prompt_hashes": bundle["prompt_hashes"],
        "decoding": config["decoding"],
        "evaluator": config["evaluator"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        **result,
    }
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
