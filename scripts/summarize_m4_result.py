#!/usr/bin/env python3
"""Publish SQL-free M4 evidence; final953 is derived from the unique full1209 generation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from verireason.m3_contract import atomic_write_json, sha256_json
from verireason.m4_contract import validate_comparison


def public_eval(payload: dict, ids: set[str] | None = None) -> dict:
    rows = [row for row in payload["examples"] if ids is None or row["sample_id"] in ids]
    if not rows:
        raise ValueError("empty M4 reporting view")
    count = len(rows)
    return {
        "example_count": count,
        "bird_ex_behavior_compatible": sum(row["bird_ex_behavior_compatible"] for row in rows)
        / count,
        "strict_execution_accuracy": sum(row["strict_execution_accuracy"] for row in rows) / count,
        "strict_sql_only_rate": sum(row["extraction"]["strict_sql_only"] for row in rows) / count,
        "valid_sql_rate": sum(row["prediction_status"] == "ok" for row in rows) / count,
        "failure_categories": dict(
            sorted(Counter(row["failure_category"] for row in rows).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.evidence_dir)
    base, sft = (
        json.loads((root / f"{condition}-full.json").read_text()) for condition in ("base", "sft")
    )
    validate_comparison(base, sft)
    selection = json.loads((root / "selection.json").read_text())
    training = json.loads((root / "training-result.json").read_text())
    decision = json.loads((root / "epoch-extension-decision.json").read_text())
    trainer_contract = json.loads((root / "trainer-contract.json").read_text())
    final_ids, dev_ids = (
        set(selection["final_validation"]["sample_ids"]),
        set(selection["development_validation"]["sample_ids"]),
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": "e1_m4_formal_sft_001",
        "selection": {
            "bundle_sha256": selection["selection_bundle_sha256"],
            "train_count": selection["train"]["count"],
            "development_validation_count": 256,
            "final_validation_count": 953,
            "full_validation_count": 1209,
            "dev_selection_sha256": selection["development_validation"]["selection_sha256"],
            "final_selection_sha256": selection["final_validation"]["selection_sha256"],
            "full_selection_sha256": selection["full_validation"]["selection_sha256"],
        },
        "training": {
            "checkpoint_selection": "final_primary_update_671",
            **training["accumulation"],
            "primary_epoch_complete": training["primary_epoch_complete"],
            "trainer_loss_normalization": training["trainer_loss_normalization"],
        },
        "trainer_contract": trainer_contract,
        "epoch_extension_decision": decision,
        "base_evaluation": {
            "full1209_primary": public_eval(base),
            "final953_derived_from_full": public_eval(base, final_ids),
            "dev256_derived_from_full": public_eval(base, dev_ids),
        },
        "sft_evaluation": {
            "full1209_primary": public_eval(sft),
            "final953_derived_from_full": public_eval(sft, final_ids),
            "dev256_derived_from_full": public_eval(sft, dev_ids),
        },
        "generation": {
            "unique_e0_full1209": True,
            "unique_e1_full1209": True,
            "no_separate_dev_generation": True,
        },
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "timing": {
            "base_generation_seconds": base["metrics"]["generation_seconds"],
            "sft_generation_seconds": sft["metrics"]["generation_seconds"],
        },
        "memory": {"source": "raw evaluation/training evidence"},
        "cost": {"actual_provider_cost_cny": None},
        "comparison_contract_sha256": sha256_json(
            {"full1209": base["validation_selection_sha256"]}
        ),
    }
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
