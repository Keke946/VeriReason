#!/usr/bin/env python3
"""Precheck fixed M2 labels without downloading any database archives."""

import argparse
import json
from pathlib import Path

from verireason.config import load_m2_config
from verireason.data_pipeline import (
    audit_leakage,
    build_split_precheck,
    load_examples,
    report_as_dict,
    sha256_file,
    validate_observed_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_labels", type=Path)
    parser.add_argument("--heldout-labels", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/m2_data.toml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_m2_config(args.config)
    split = config["split"]
    training = load_examples(args.training_labels)
    assignment, split_report = build_split_precheck(
        training,
        seed=split["seed"],
        validation_database_count=split["validation_database_count"],
        minimum_validation_ratio=split["minimum_validation_example_ratio"],
        maximum_validation_ratio=split["maximum_validation_example_ratio"],
        critical_features=split["critical_sql_features"],
    )
    expected_rows = config["training_source"]["expected_rows"]
    expected_databases = config["training_source"]["expected_database_count"]
    observed_databases = len({example.db_id for example in training})
    gate_errors = list(split_report.validation_gate_errors)
    training_sha256 = sha256_file(args.training_labels)
    try:
        validate_observed_identity(
            expected_sha256=config["training_source"]["labels_observed_sha256"],
            observed_sha256=training_sha256,
        )
    except ValueError as error:
        gate_errors.append(str(error))
    if len(training) != expected_rows:
        gate_errors.append(f"training row count {len(training)} != expected {expected_rows}")
    if observed_databases != expected_databases:
        gate_errors.append(
            f"training database count {observed_databases} != expected {expected_databases}"
        )

    split_examples = [(assignment[example.db_id], example) for example in training]
    heldout_identity = None
    if args.heldout_labels:
        heldout = load_examples(args.heldout_labels, source="heldout")
        heldout_sha256 = sha256_file(args.heldout_labels)
        heldout_identity = {
            "path": str(args.heldout_labels),
            "sha256": heldout_sha256,
            "row_count": len(heldout),
            "database_count": len({example.db_id for example in heldout}),
        }
        if len(heldout) != config["heldout_source"]["expected_rows"]:
            gate_errors.append("heldout row count differs from frozen expectation")
        if (
            heldout_identity["database_count"]
            != config["heldout_source"]["expected_database_count"]
        ):
            gate_errors.append("heldout database count differs from frozen expectation")
        try:
            validate_observed_identity(
                expected_sha256=config["heldout_source"]["labels_observed_sha256"],
                observed_sha256=heldout_sha256,
            )
        except ValueError as error:
            gate_errors.append(str(error))
        split_examples.extend(("heldout", example) for example in heldout)

    leakage = audit_leakage(split_examples)
    if not leakage.passed:
        gate_errors.append("hard leakage check failed")
    payload = {
        "schema_version": 1,
        "status": "passed" if not gate_errors else "control_review_required",
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "training_labels": {
            "path": str(args.training_labels),
            "sha256": training_sha256,
            "row_count": len(training),
            "database_count": observed_databases,
        },
        "heldout_labels": heldout_identity,
        "split_precheck": report_as_dict(split_report),
        "leakage_precheck": report_as_dict(leakage),
        "gate_errors": gate_errors,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not gate_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
