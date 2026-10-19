#!/usr/bin/env python3
"""Run or resume the frozen M4 formal QLoRA optimization."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json, sha256_json
from verireason.m3_runtime import load_base, load_tokenizer
from verireason.m4_contract import load_m4_config, validate_resume_contract
from verireason.m4_runtime import train_formal_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--prepared-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = load_m4_config(args.config, args.m2_config)
    bundle = json.loads(Path(args.prepared_train).read_text())
    if (
        bundle.get("heldout_examples_loaded") != 0
        or bundle.get("selection_sha256") is None
        or bundle.get("split") != "train"
        or len(bundle.get("examples", [])) != 5364
    ):
        raise SystemExit("prepared training bundle lacks frozen identity or reports heldout use")
    training_contract = {
        "schema_version": 1,
        "phase": "primary_epoch_1",
        "selection_sha256": bundle["selection_sha256"],
        "model_revision": config["model"]["revision"],
        "config_sha256": sha256_json(config),
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    if args.resume_from_checkpoint:
        observed_contract = json.loads(
            (Path(args.resume_from_checkpoint) / "m4-contract.json").read_text()
        )
        validate_resume_contract(training_contract, observed_contract)
    tokenizer = load_tokenizer(config, False)
    if args.resume_from_checkpoint:
        from peft import PeftModel

        base = load_base(config, trainable=False, allow_download=False)
        model = PeftModel.from_pretrained(base, args.resume_from_checkpoint, is_trainable=True)
        model.config.use_cache = False
    else:
        model = load_base(config, trainable=True, allow_download=False)
    evidence = train_formal_adapter(
        model,
        tokenizer,
        bundle["examples"],
        config,
        args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        training_contract=training_contract,
    )
    atomic_write_json(
        args.output or Path(args.output_dir) / "training-result.json",
        {"schema_version": 1, "phase": "primary_epoch_1", **evidence},
    )
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
