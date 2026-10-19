#!/usr/bin/env python3
"""Publish the advisory-only Day 5A epoch-extension decision packet."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m4_contract import extension_decision, load_m4_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--loss-512", required=True)
    parser.add_argument("--loss-671", required=True)
    parser.add_argument("--training-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_m4_config(args.config, args.m2_config)
    first, final, training = (
        json.loads(Path(path).read_text())
        for path in (args.loss_512, args.loss_671, args.training_result)
    )
    if (
        first["update"] != 512
        or final["update"] != 671
        or first["validation_selection_sha256"] != final["validation_selection_sha256"]
    ):
        raise SystemExit("epoch extension diagnostic identity/update drift")
    observations = {
        item["update"]: item for item in training.get("required_update_observations", [])
    }
    if set(observations) != {512, 671} or not training.get("primary_epoch_complete"):
        raise SystemExit("primary training evidence is incomplete")
    decision = extension_decision(
        first["loss"],
        final["loss"],
        advisory_threshold=config["epoch_extension_diagnostic"]["advisory_relative_dev_loss_drop"],
    )
    payload = {
        "schema_version": 1,
        "status": "primary_epoch_completed_control_review_required",
        "primary_updates": 671,
        "dev256_selection_sha256": first["validation_selection_sha256"],
        "raw_losses": {"512": first["loss"], "671": final["loss"]},
        "teacher_forced_evidence": {"512": first, "671": final},
        "training_update_observations": {str(k): observations[k] for k in sorted(observations)},
        **decision,
        "reason": (
            "Day 5A stops after the independent full5364x1 primary E1 baseline; the 1% value "
            "is advisory, not an automatic epoch-2 gate."
        ),
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
