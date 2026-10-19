#!/usr/bin/env python3
"""Evaluate a registered M8 adapter on the frozen M4 full1209 validation exam."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import artifact_manifest, evaluate_model, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config
from verireason.m8_contract import load_m8_config, validate_prepared_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m8_common_eval.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_m8_config(args.config)
    m4 = load_m4_config(config["sources"]["m4_config"], config["sources"]["m2_config"])
    bundle = json.loads(Path(args.prepared).read_text())
    validate_prepared_bundle(bundle, config)
    if (
        artifact_manifest(args.adapter_dir)["manifest_sha256"]
        != config["model"]["adapter_manifest_sha256"]
    ):
        raise SystemExit(f"formal {config['model']['condition']} adapter manifest drifted")

    from peft import PeftModel

    tokenizer = load_tokenizer(m4, False)
    base = load_base(m4, trainable=False, allow_download=False)
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    result = evaluate_model(model, tokenizer, bundle["examples"], m4)
    payload = {
        "schema_version": 1,
        "condition": config["model"]["condition"],
        "adapter_experiment_id": config["model"]["adapter_experiment_id"],
        "adapter_manifest_sha256": config["model"]["adapter_manifest_sha256"],
        "validation_selection_sha256": bundle["selection_sha256"],
        "sample_ids": bundle["sample_ids"],
        "prompt_hashes": bundle["prompt_hashes"],
        "decoding": m4["decoding"],
        "evaluator": m4["evaluator"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        **result,
    }
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
