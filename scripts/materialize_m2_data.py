#!/usr/bin/env python3
"""Safely materialize and atomically publish frozen M2 data manifests."""

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

from verireason.config import load_m2_config
from verireason.data_pipeline import (
    Example,
    audit_leakage,
    build_split_precheck,
    inspect_database,
    load_examples,
    report_as_dict,
    sample_fingerprint,
    sha256_file,
)
from verireason.gold_preflight import require_gold_preflight_passed, run_gold_preflight
from verireason.materialization import (
    FilesystemCapacity,
    atomic_publish_directory,
    build_source_receipt,
    discover_database_root,
    evaluate_extraction_capacity,
    observe_zip_under_caps,
    require_free_space,
    require_outside_git_worktree,
    safe_extract_zip,
    safe_extract_zip_after_capacity_gate,
    write_json,
)
from verireason.sql_evaluator import EvaluatorLimits


def _example_payload(example: Example, split: str) -> dict[str, object]:
    payload = asdict(example)
    payload.pop("schema_tokens", None)
    payload["split"] = split
    payload["sample_fingerprint"] = sample_fingerprint(example)
    return payload


def _safe_relative_root(extraction_root: Path, value: str) -> Path:
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"database root must be a safe relative path: {value!r}")
    resolved = extraction_root.joinpath(*logical.parts)
    if not resolved.is_dir():
        raise ValueError(f"database root does not exist after extraction: {resolved}")
    return resolved


def _safe_relative_file(extraction_root: Path, value: str) -> Path:
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise ValueError(f"archive member must be a safe relative path: {value!r}")
    resolved = extraction_root.joinpath(*logical.parts)
    if not resolved.is_file():
        raise ValueError(f"nested archive does not exist after outer extraction: {resolved}")
    return resolved


def _evaluator_limits(config: dict[str, object]) -> EvaluatorLimits:
    evaluator = config["evaluator"]
    assert isinstance(evaluator, dict)
    return EvaluatorLimits(
        timeout_seconds=evaluator["timeout_seconds"],
        progress_steps=evaluator["progress_steps"],
        maximum_vm_bytes=evaluator["maximum_vm_bytes"],
        maximum_cpu_seconds=evaluator["maximum_cpu_seconds"],
        maximum_cpu_hard_seconds=evaluator["maximum_cpu_hard_seconds"],
        maximum_result_rows=evaluator["maximum_result_rows"],
        maximum_result_bytes=evaluator["maximum_result_bytes"],
        maximum_float_pair_checks=evaluator["maximum_float_pair_checks"],
        float_absolute_tolerance=evaluator["float_absolute_tolerance"],
        float_relative_tolerance=evaluator["float_relative_tolerance"],
    )


def _filesystem_evidence(capacity: FilesystemCapacity) -> dict[str, int]:
    return {
        "device": capacity.device,
        "total_bytes": capacity.total_bytes,
        "used_bytes": capacity.used_bytes,
        "free_bytes": capacity.free_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-label-headers", type=Path, required=True)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--training-archive-headers", type=Path, required=True)
    parser.add_argument("--training-database-root", default="auto")
    parser.add_argument("--heldout-labels", type=Path, required=True)
    parser.add_argument("--heldout-label-headers", type=Path, required=True)
    parser.add_argument("--heldout-archive", type=Path, required=True)
    parser.add_argument("--heldout-archive-headers", type=Path, required=True)
    parser.add_argument("--heldout-database-root", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/m2_data.toml"))
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_m2_config(args.config)
    materialization = config["materialization"]
    final_dir = args.target_dir.resolve()
    partial_dir = final_dir.with_name(final_dir.name + materialization["partial_suffix"])
    if final_dir.exists() or partial_dir.exists():
        raise SystemExit(f"final or partial target already exists: {final_dir}, {partial_dir}")
    require_outside_git_worktree(final_dir)
    capacity = require_free_space(final_dir, materialization["minimum_free_bytes"])
    partial_dir.mkdir(parents=True)

    receipt = build_source_receipt(
        args.config,
        config,
        training_labels=args.training_labels,
        heldout_labels=args.heldout_labels,
        training_archive=args.training_archive,
        heldout_archive=args.heldout_archive,
        training_label_headers=args.training_label_headers,
        heldout_label_headers=args.heldout_label_headers,
        training_archive_headers=args.training_archive_headers,
        heldout_archive_headers=args.heldout_archive_headers,
    )
    training_outer_extract = partial_dir / "sources" / "training-outer"
    training_extract = partial_dir / "databases" / "training"
    heldout_extract = partial_dir / "databases" / "heldout"
    safe_extract_zip(
        args.training_archive,
        training_outer_extract,
        expected_member_count=config["training_source"]["database_archive_member_count"],
        expected_uncompressed_bytes=config["training_source"][
            "database_archive_member_uncompressed_bytes"
        ],
    )
    nested_archive = _safe_relative_file(
        training_outer_extract,
        config["training_source"]["database_nested_archive_path"],
    )
    nested_validation = observe_zip_under_caps(
        nested_archive,
        maximum_member_count=config["training_source"][
            "database_nested_archive_maximum_member_count"
        ],
        maximum_uncompressed_bytes=config["training_source"][
            "database_nested_archive_maximum_uncompressed_bytes"
        ],
    )
    dynamic_capacity = evaluate_extraction_capacity(
        partial_dir,
        observed_inner_uncompressed_bytes=nested_validation.uncompressed_bytes,
        heldout_expected_uncompressed_bytes=config["heldout_source"][
            "archive_member_uncompressed_bytes"
        ],
        reserve_bytes=materialization["dynamic_capacity_reserve_bytes"],
    )
    capacity_evidence = {
        "schema_version": 1,
        "initial": {
            "filesystem": _filesystem_evidence(capacity),
            "minimum_free_bytes": materialization["minimum_free_bytes"],
            "passed": capacity.free_bytes >= materialization["minimum_free_bytes"],
        },
        "before_nested_extraction": {
            "filesystem": _filesystem_evidence(dynamic_capacity.filesystem),
            "observed_inner_uncompressed_bytes": (
                dynamic_capacity.observed_inner_uncompressed_bytes
            ),
            "heldout_expected_uncompressed_bytes": (
                dynamic_capacity.heldout_expected_uncompressed_bytes
            ),
            "reserve_bytes": dynamic_capacity.reserve_bytes,
            "required_bytes": dynamic_capacity.required_bytes,
            "passed": dynamic_capacity.passed,
        },
    }
    write_json(partial_dir / "capacity-gates.json", capacity_evidence)
    try:
        safe_extract_zip_after_capacity_gate(
            nested_archive,
            training_extract,
            observation=nested_validation,
            capacity_gate=dynamic_capacity,
        )
    except ValueError as error:
        raise SystemExit(f"{error}; inspect {partial_dir / 'capacity-gates.json'}") from error
    receipt["nested_archives"] = {
        "training_databases": {
            "path_within_outer_archive": config["training_source"]["database_nested_archive_path"],
            "size_bytes": nested_archive.stat().st_size,
            "sha256": sha256_file(nested_archive),
            "member_count": nested_validation.member_count,
            "uncompressed_bytes": nested_validation.uncompressed_bytes,
        }
    }
    receipt["capacity_gates"] = capacity_evidence
    shutil.rmtree(training_outer_extract)
    safe_extract_zip(
        args.heldout_archive,
        heldout_extract,
        expected_member_count=config["heldout_source"]["archive_member_count"],
        expected_uncompressed_bytes=config["heldout_source"]["archive_member_uncompressed_bytes"],
    )
    training = load_examples(args.training_labels)
    heldout = load_examples(args.heldout_labels, source="heldout")
    if (
        len(training) != config["training_source"]["expected_rows"]
        or len({example.db_id for example in training})
        != config["training_source"]["expected_database_count"]
    ):
        raise SystemExit("training label counts differ from the frozen source contract")
    if (
        len(heldout) != config["heldout_source"]["expected_rows"]
        or len({example.db_id for example in heldout})
        != config["heldout_source"]["expected_database_count"]
    ):
        raise SystemExit("heldout label counts differ from the frozen source contract")

    if args.training_database_root == "auto":
        training_database_root = discover_database_root(
            training_extract, {example.db_id for example in training}
        )
    else:
        training_database_root = _safe_relative_root(training_extract, args.training_database_root)
    heldout_database_root = _safe_relative_root(heldout_extract, args.heldout_database_root)
    receipt_path = partial_dir / "source-receipt.json"
    receipt_sha256 = write_json(receipt_path, receipt)

    split = config["split"]
    assignment, split_report = build_split_precheck(
        training,
        seed=split["seed"],
        validation_database_count=split["validation_database_count"],
        minimum_validation_ratio=split["minimum_validation_example_ratio"],
        maximum_validation_ratio=split["maximum_validation_example_ratio"],
        critical_features=split["critical_sql_features"],
    )
    if not split_report.passed:
        raise SystemExit(
            "split balance gate failed; do not change seed: "
            + "; ".join(split_report.validation_gate_errors)
        )

    training_db = {
        db_id: inspect_database(training_database_root, db_id, logical_root=partial_dir)
        for db_id in sorted({example.db_id for example in training})
    }
    heldout_db = {
        db_id: inspect_database(heldout_database_root, db_id, logical_root=partial_dir)
        for db_id in sorted({example.db_id for example in heldout})
    }
    training = [
        replace(
            example,
            database_sha256=training_db[example.db_id].sha256,
            schema_fingerprint=training_db[example.db_id].schema_fingerprint,
            schema_tokens=training_db[example.db_id].schema_tokens,
        )
        for example in training
    ]
    heldout = [
        replace(
            example,
            database_sha256=heldout_db[example.db_id].sha256,
            schema_fingerprint=heldout_db[example.db_id].schema_fingerprint,
            schema_tokens=heldout_db[example.db_id].schema_tokens,
        )
        for example in heldout
    ]
    leakage = audit_leakage(
        [(assignment[example.db_id], example) for example in training]
        + [("heldout", example) for example in heldout]
    )
    if not leakage.passed:
        raise SystemExit("hard database/sample leakage detected; manifest was not frozen")

    evaluator_limits = _evaluator_limits(config)
    training_preflight = run_gold_preflight(training, training_database_root, evaluator_limits)
    heldout_preflight = run_gold_preflight(heldout, heldout_database_root, evaluator_limits)
    preflight_summary = {
        "schema_version": 1,
        "training": training_preflight.as_dict(include_failure_ids=False),
        "heldout": heldout_preflight.as_dict(include_failure_ids=False),
    }
    write_json(partial_dir / "gold-preflight-summary.json", preflight_summary)
    write_json(
        partial_dir / "gold-preflight-private.json",
        {
            "schema_version": 1,
            "training": training_preflight.as_dict(include_failure_ids=True),
            "heldout": heldout_preflight.as_dict(include_failure_ids=True),
        },
        private=True,
    )
    try:
        require_gold_preflight_passed(training=training_preflight, heldout=heldout_preflight)
    except RuntimeError as error:
        raise SystemExit(f"{error}; inspect private report under {partial_dir}") from error

    common = {
        "schema_version": 1,
        "config_sha256": sha256_file(args.config),
        "source_receipt": {
            "path": "source-receipt.json",
            "sha256": receipt_sha256,
        },
        "split_algorithm": split["algorithm"],
        "split_seed": split["seed"],
    }
    training_manifest = {
        **common,
        "source": config["training_source"],
        "databases": {key: asdict(value) for key, value in training_db.items()},
        "examples": [_example_payload(example, assignment[example.db_id]) for example in training],
    }
    heldout_manifest = {
        **common,
        "source": config["heldout_source"],
        "databases": {key: asdict(value) for key, value in heldout_db.items()},
        "examples": [_example_payload(example, "heldout") for example in heldout],
    }
    training_manifest_sha = write_json(partial_dir / "training-manifest.json", training_manifest)
    heldout_manifest_sha = write_json(
        partial_dir / "heldout-manifest-private.json", heldout_manifest, private=True
    )
    audit_manifest_sha = write_json(
        partial_dir / "audit-manifest.json",
        {
            **common,
            "split_precheck": report_as_dict(split_report),
            "leakage": report_as_dict(leakage),
            "gold_preflight": preflight_summary,
        },
    )
    index_sha = write_json(
        partial_dir / "manifest-index.json",
        {
            "schema_version": 1,
            "source_receipt_sha256": receipt_sha256,
            "training_manifest_sha256": training_manifest_sha,
            "heldout_manifest_sha256": heldout_manifest_sha,
            "audit_manifest_sha256": audit_manifest_sha,
        },
    )
    atomic_publish_directory(partial_dir, final_dir)
    print(
        json.dumps(
            {
                "status": "published",
                "target": str(final_dir),
                "target_filesystem": asdict(capacity),
                "manifest_index_sha256": index_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
