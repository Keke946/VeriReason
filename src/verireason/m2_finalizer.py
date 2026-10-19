"""Evidence-only M2 operational-eligibility finalization and manifest loading."""

from __future__ import annotations

import json
import os
import stat
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from verireason.config import load_m2_config
from verireason.data_pipeline import (
    Example,
    audit_leakage,
    inspect_database,
    load_examples,
    report_as_dict,
    sample_fingerprint,
    sha256_file,
    split_database_ids,
    sql_complexity,
)
from verireason.m2_diagnostics import DiagnosticItem, select_failed_queries
from verireason.materialization import (
    atomic_publish_directory,
    discover_database_root,
    require_outside_git_worktree,
    write_json,
)


def load_finalizer_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("finalizer_only") is not True:
        raise ValueError("M2 finalizer config must be schema v1 and finalizer_only")
    expected = config.get("expected")
    if not isinstance(expected, dict) or (
        expected.get("training_eligible_unique_sql") != 6496
        or expected.get("training_ineligible_unique_sql") != 28
        or expected.get("heldout_eligible_examples") != 498
        or expected.get("heldout_ineligible_examples") != 2
    ):
        raise ValueError("M2 finalizer expected counts changed")
    return config


def _require_sha(path: str | Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return observed


def _require_private_mode(path: str | Path, label: str = "diagnostic checkpoint") -> None:
    mode = stat.S_IMODE(Path(path).stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"{label} must have mode 0600, observed {mode:04o}")


def verify_finalizer_evidence(
    config: Mapping[str, Any],
    *,
    private_preflight: str | Path,
    diagnostic_checkpoint: str | Path,
    source_receipt: str | Path,
    training_labels: str | Path,
    heldout_labels: str | Path,
    historical_formal_config: str | Path,
    amended_formal_config: str | Path,
) -> dict[str, str]:
    evidence = config["evidence"]
    paths = {
        "private_preflight_sha256": private_preflight,
        "diagnostic_checkpoint_sha256": diagnostic_checkpoint,
        "source_receipt_sha256": source_receipt,
        "training_labels_sha256": training_labels,
        "heldout_labels_sha256": heldout_labels,
        "historical_formal_config_sha256": historical_formal_config,
        "amended_formal_config_sha256": amended_formal_config,
    }
    observed = {
        name: _require_sha(path, evidence[name], name.removesuffix("_sha256"))
        for name, path in paths.items()
    }
    _require_private_mode(private_preflight, "private preflight")
    _require_private_mode(diagnostic_checkpoint)
    return observed


def _checkpoint_result(checkpoint: Mapping[str, Any], item: DiagnosticItem) -> Mapping[str, Any]:
    results = checkpoint.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("diagnostic checkpoint results are missing")
    entry = results.get(item.key)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("result"), Mapping):
        raise ValueError("diagnostic checkpoint is missing a selected result")
    return entry["result"]


def derive_operational_eligibility(
    private_preflight: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training: Sequence[Example],
    heldout: Sequence[Example],
    config: Mapping[str, Any],
) -> tuple[dict[str, tuple[bool, str | None]], dict[str, Any]]:
    expected = config["expected"]
    if checkpoint.get("complete") is not True:
        raise ValueError("diagnostic checkpoint is incomplete")
    if checkpoint.get("plan_sha256") != config["diagnostic_plan_sha256"]:
        raise ValueError("diagnostic checkpoint plan SHA-256 mismatch")
    results = checkpoint.get("results")
    if not isinstance(results, Mapping) or len(results) != expected["diagnostic_completed"]:
        raise ValueError("diagnostic checkpoint completed count differs from 244")

    items = select_failed_queries(private_preflight, {"training": training, "heldout": heldout})
    if len(items) != expected["diagnostic_completed"]:
        raise ValueError("selected diagnostic query count differs from 244")
    selected_counts = Counter((item.split, item.profile) for item in items)
    if selected_counts != Counter(
        {("training", "profile_a"): 220, ("training", "profile_b"): 22, ("heldout", "profile_a"): 2}
    ):
        raise ValueError("diagnostic selection split/profile counts changed")

    training_section = private_preflight.get("training")
    heldout_section = private_preflight.get("heldout")
    if not isinstance(training_section, Mapping) or not isinstance(heldout_section, Mapping):
        raise ValueError("private preflight split evidence is missing")
    if (
        training_section.get("unique_sql_count") != expected["training_unique_sql"]
        or training_section.get("status_counts", {}).get("ok")
        != expected["training_original_ok_unique_sql"]
        or heldout_section.get("unique_sql_count") != expected["heldout_total_examples"]
        or heldout_section.get("status_counts", {}).get("ok")
        != expected["heldout_eligible_examples"]
    ):
        raise ValueError("private preflight count identity changed")

    eligibility: dict[str, tuple[bool, str | None]] = {}
    recovered = 0
    exclusion_counts: Counter[str] = Counter()
    heldout_ineligible = 0
    for item in items:
        result = _checkpoint_result(checkpoint, item)
        status = result.get("status")
        limit_kind = result.get("limit_kind")
        if item.profile == "profile_b":
            eligible, reason = False, "formal_result_row_limit"
        elif status == "ok":
            eligible, reason = True, None
            if item.split == "training":
                recovered += 1
        elif status == "timeout" and limit_kind == "wall":
            eligible, reason = False, "formal_wall_timeout"
        elif status == "resource_limit" and limit_kind == "cpu":
            raise ValueError("unexpected Profile-A CPU limit in accepted diagnostic evidence")
        else:
            # Keep this classification identical to build_public_report(): after
            # ok, CPU limit, and wall timeout, every remaining Profile-A worker
            # outcome belongs to the accepted worker_error aggregate.  The raw
            # executor status is intentionally more specific and need not be the
            # literal string "worker_error".
            eligible, reason = False, "worker_error"
        eligibility[item.key] = (eligible, reason)
        if not eligible:
            exclusion_counts[f"{item.split}:{reason}"] += 1
            if item.split == "heldout":
                heldout_ineligible += 1

    training_excluded = sum(
        count for key, count in exclusion_counts.items() if key.startswith("training:")
    )
    eligible_unique = expected["training_original_ok_unique_sql"] + recovered
    if recovered != expected["training_recovered_unique_sql"]:
        raise ValueError("recovered training unique SQL count differs from 214")
    if training_excluded != expected["training_ineligible_unique_sql"]:
        raise ValueError("training unique SQL exclusion count differs from 28")
    if eligible_unique != expected["training_eligible_unique_sql"]:
        raise ValueError("eligible training unique SQL count differs from 6496")
    if expected["training_unique_sql"] - training_excluded != eligible_unique:
        raise ValueError("training operational eligibility identity failed")
    if heldout_ineligible != expected["heldout_ineligible_examples"]:
        raise ValueError("heldout ineligible count differs from 2")
    if (
        expected["heldout_total_examples"] - heldout_ineligible
        != expected["heldout_eligible_examples"]
    ):
        raise ValueError("heldout operational eligibility identity failed")

    return eligibility, {
        "training_original_ok_unique_sql": expected["training_original_ok_unique_sql"],
        "training_recovered_unique_sql": recovered,
        "training_eligible_unique_sql": eligible_unique,
        "training_ineligible_unique_sql": training_excluded,
        "heldout_eligible_examples": expected["heldout_eligible_examples"],
        "heldout_ineligible_examples": heldout_ineligible,
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
    }


def _query_key(split: str, example: Example) -> str:
    import hashlib

    return hashlib.sha256(f"{split}\0{example.db_id}\0{example.sql}".encode()).hexdigest()


def _hardlink_tree(source: Path, destination: Path) -> int:
    if source.is_symlink():
        raise ValueError("database tree cannot be a symbolic link")
    destination.mkdir(parents=True, exist_ok=False)
    ignored_non_database_sql_symlinks = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            # BIRD Mini-Dev contains two unreferenced .sql convenience links
            # whose absolute targets are outside the extracted dataset. Never
            # follow or reproduce them. Database discovery and integrity gates
            # operate exclusively on regular .sqlite/.db files, while every
            # other symlink type remains fatal.
            if path.suffix.casefold() == ".sql":
                ignored_non_database_sql_symlinks += 1
                continue
            raise ValueError("database tree contains a symbolic link")
        if path.is_dir():
            target.mkdir()
        elif path.is_file():
            os.link(path, target)
        else:
            raise ValueError("database tree contains an unsupported member")
    return ignored_non_database_sql_symlinks


def _example_payload(
    example: Example,
    split: str,
    operational_eligible: bool,
    exclusion_reason: str | None,
) -> dict[str, Any]:
    payload = asdict(example)
    payload.pop("schema_tokens", None)
    payload.update(
        {
            "split": split,
            "sample_fingerprint": sample_fingerprint(example),
            "operational_eligible": operational_eligible,
            "exclusion_reason": exclusion_reason,
        }
    )
    return payload


def _post_exclusion_report(
    examples: Sequence[tuple[str, Example, bool]], config: Mapping[str, Any]
) -> dict[str, Any]:
    eligible = [(split, example) for split, example, allowed in examples if allowed]
    counts = Counter(split for split, _example in eligible)
    databases: dict[str, set[str]] = defaultdict(set)
    complexity: dict[str, Counter[str]] = defaultdict(Counter)
    features: dict[str, Counter[str]] = defaultdict(Counter)
    for split, example in eligible:
        databases[split].add(example.db_id)
        parsed = sql_complexity(example.sql)
        complexity[split][parsed.label] += 1
        features[split].update(parsed.features)
    total = sum(counts.values())
    validation_ratio = counts["validation"] / total
    split_config = config["split"]
    errors: list[str] = []
    if (
        not split_config["minimum_validation_example_ratio"]
        <= validation_ratio
        <= split_config["maximum_validation_example_ratio"]
    ):
        errors.append("post-exclusion validation example ratio is outside 10%-25%")
    for feature in split_config["critical_sql_features"]:
        if features["validation"][feature] == 0:
            errors.append(f"post-exclusion validation is missing critical SQL feature: {feature}")
    expected_validation_databases = split_config["validation_database_count"]
    expected_train_databases = (
        config["training_source"]["expected_database_count"] - expected_validation_databases
    )
    if len(databases["train"]) != expected_train_databases:
        errors.append(
            "post-exclusion train database count changed: "
            f"{len(databases['train'])} != {expected_train_databases}"
        )
    if len(databases["validation"]) != expected_validation_databases:
        errors.append(
            "post-exclusion validation database count changed: "
            f"{len(databases['validation'])} != {expected_validation_databases}"
        )
    return {
        "passed": not errors,
        "gate_errors": errors,
        "database_counts": {split: len(databases[split]) for split in ("train", "validation")},
        "example_counts": {split: counts[split] for split in ("train", "validation")},
        "validation_example_ratio": validation_ratio,
        "complexity_distribution": {
            split: dict(sorted(complexity[split].items())) for split in ("train", "validation")
        },
        "feature_distribution": {
            split: {
                feature: features[split][feature]
                for feature in split_config["critical_sql_features"]
            }
            for split in ("train", "validation")
        },
    }


def load_operational_training_examples(
    manifest_path: str | Path, *, split: str
) -> tuple[dict[str, Any], ...]:
    if split not in {"train", "validation"}:
        raise ValueError("training loader split must be train or validation")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("examples")
    if not isinstance(rows, list):
        raise ValueError("training manifest examples are missing")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("source") == "heldout":
            raise ValueError("heldout data cannot enter a training/validation loader")
        if row.get("split") == split and row.get("operational_eligible") is True:
            selected.append(row)
    if any(row.get("operational_eligible") is not True for row in selected):
        raise ValueError("training loader emitted an operationally ineligible sample")
    return tuple(selected)


def score_heldout_scopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 500:
        raise ValueError("heldout metric denominator must remain 500")
    eligible = [row for row in rows if row.get("operational_eligible") is True]
    ineligible = [row for row in rows if row.get("operational_eligible") is False]
    if len(eligible) != 498 or len(ineligible) != 2:
        raise ValueError("heldout eligibility must remain 498/2")
    result: dict[str, Any] = {"full_denominator": 500, "eligible_denominator": 498}
    for metric in ("bird_ex_behavior_compatible", "strict_execution_accuracy"):
        eligible_correct = sum(bool(row.get(metric)) for row in eligible)
        result[metric] = {
            "full_heldout_conservative_execution_accuracy": eligible_correct / 500,
            "eligible_heldout_execution_accuracy": eligible_correct / 498,
            "full_correct": eligible_correct,
            "eligible_correct": eligible_correct,
        }
    return result


def finalize_m2(
    *,
    materialization_partial: str | Path,
    diagnostic_checkpoint: str | Path,
    training_labels: str | Path,
    heldout_labels: str | Path,
    historical_formal_config: str | Path,
    amended_formal_config: str | Path,
    finalizer_config: str | Path,
    final_dir: str | Path,
) -> dict[str, Any]:
    source = Path(materialization_partial).resolve()
    final = Path(final_dir).resolve()
    config = load_finalizer_config(finalizer_config)
    amended = load_m2_config(amended_formal_config)
    staging = final.with_name(final.name + config["publication"]["staging_suffix"])
    if final.exists() or staging.exists():
        raise ValueError("final or finalization partial directory already exists")
    require_outside_git_worktree(final)
    if not source.is_dir() or source == final or source == staging:
        raise ValueError("source materialization .partial is missing or aliases output")
    if source.stat().st_dev != final.parent.stat().st_dev:
        raise ValueError("source, staging, and final directories must share a filesystem")

    private_preflight_path = source / "gold-preflight-private.json"
    source_receipt_path = source / "source-receipt.json"
    identities = verify_finalizer_evidence(
        config,
        private_preflight=private_preflight_path,
        diagnostic_checkpoint=diagnostic_checkpoint,
        source_receipt=source_receipt_path,
        training_labels=training_labels,
        heldout_labels=heldout_labels,
        historical_formal_config=historical_formal_config,
        amended_formal_config=amended_formal_config,
    )
    private_preflight = json.loads(private_preflight_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(Path(diagnostic_checkpoint).read_text(encoding="utf-8"))
    training = load_examples(training_labels)
    heldout = load_examples(heldout_labels, source="heldout")
    if (
        len(training) != amended["training_source"]["expected_rows"]
        or len({example.db_id for example in training})
        != amended["training_source"]["expected_database_count"]
    ):
        raise ValueError("training label counts differ from the amended source contract")
    if (
        len(heldout) != amended["heldout_source"]["expected_rows"]
        or len({example.db_id for example in heldout})
        != amended["heldout_source"]["expected_database_count"]
    ):
        raise ValueError("heldout label counts differ from the amended source contract")
    eligibility, eligibility_counts = derive_operational_eligibility(
        private_preflight, checkpoint, training, heldout, config
    )

    staging.mkdir(mode=0o700)
    try:
        ignored_non_database_sql_symlinks = _hardlink_tree(
            source / "databases", staging / "databases"
        )
        os.link(source_receipt_path, staging / "source-receipt.json")
        training_root = discover_database_root(
            staging / "databases" / "training", {example.db_id for example in training}
        )
        heldout_root = discover_database_root(
            staging / "databases" / "heldout", {example.db_id for example in heldout}
        )
        training_db = {
            db_id: inspect_database(training_root, db_id, logical_root=staging)
            for db_id in sorted({example.db_id for example in training})
        }
        heldout_db = {
            db_id: inspect_database(heldout_root, db_id, logical_root=staging)
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
        split = amended["split"]
        assignment = split_database_ids(
            (example.db_id for example in training),
            split["seed"],
            split["validation_database_count"],
        )
        leakage = audit_leakage(
            [(assignment[example.db_id], example) for example in training]
            + [("heldout", example) for example in heldout]
        )
        if not leakage.passed or leakage.pending_checks:
            raise ValueError("database/schema/sample leakage audit did not pass completely")

        training_rows: list[dict[str, Any]] = []
        post_rows: list[tuple[str, Example, bool]] = []
        for example in training:
            key = _query_key("training", example)
            allowed, reason = eligibility.get(key, (True, None))
            split_name = assignment[example.db_id]
            training_rows.append(_example_payload(example, split_name, allowed, reason))
            post_rows.append((split_name, example, allowed))
        heldout_rows: list[dict[str, Any]] = []
        for example in heldout:
            key = _query_key("heldout", example)
            allowed, reason = eligibility.get(key, (True, None))
            heldout_rows.append(_example_payload(example, "heldout", allowed, reason))
        post_report = _post_exclusion_report(post_rows, amended)
        if not post_report["passed"]:
            raise ValueError("post-exclusion split balance/coverage gate failed")
        if sum(not row["operational_eligible"] for row in heldout_rows) != 2:
            raise ValueError("heldout manifest eligibility differs from 498/2")
        excluded_example_rows = Counter(
            row["split"] for row in training_rows if not row["operational_eligible"]
        )

        common = {
            "schema_version": 2,
            "amended_config_sha256": identities["amended_formal_config_sha256"],
            "historical_config_sha256": identities["historical_formal_config_sha256"],
            "source_receipt": {
                "path": "source-receipt.json",
                "sha256": identities["source_receipt_sha256"],
            },
            "diagnostic_plan_sha256": config["diagnostic_plan_sha256"],
        }
        training_sha = write_json(
            staging / "training-manifest.json",
            {
                **common,
                "source": amended["training_source"],
                "databases": {key: asdict(value) for key, value in training_db.items()},
                "examples": training_rows,
            },
        )
        heldout_sha = write_json(
            staging / "heldout-manifest-private.json",
            {
                **common,
                "source": amended["heldout_source"],
                "databases": {key: asdict(value) for key, value in heldout_db.items()},
                "examples": heldout_rows,
            },
            private=True,
        )
        _require_private_mode(staging / "heldout-manifest-private.json", "heldout manifest")
        eligibility_sha = write_json(
            staging / "operational-eligibility-audit.json",
            {
                **common,
                **eligibility_counts,
                "post_exclusion_split": post_report,
                "excluded_example_rows": {
                    "train": excluded_example_rows["train"],
                    "validation": excluded_example_rows["validation"],
                    "total": sum(excluded_example_rows.values()),
                },
                "heldout": {"total": 500, "eligible": 498, "ineligible": 2},
            },
        )
        leakage_sha = write_json(
            staging / "leakage-audit.json", {**common, "leakage": report_as_dict(leakage)}
        )
        evidence_sha = write_json(
            staging / "finalization-evidence.json",
            {
                **common,
                "input_evidence_sha256": dict(sorted(identities.items())),
                "diagnostic_completed": 244,
                "gold_sql_reexecuted": 0,
                "ignored_non_database_sql_symlinks": ignored_non_database_sql_symlinks,
            },
        )
        index_sha = write_json(
            staging / "manifest-index.json",
            {
                **common,
                "training_manifest_sha256": training_sha,
                "heldout_manifest_sha256": heldout_sha,
                "operational_eligibility_audit_sha256": eligibility_sha,
                "leakage_audit_sha256": leakage_sha,
                "finalization_evidence_sha256": evidence_sha,
            },
        )
        atomic_publish_directory(staging, final)
        return {
            "status": "published",
            "manifest_index_sha256": index_sha,
            "training_eligible_unique_sql": eligibility_counts["training_eligible_unique_sql"],
            "training_ineligible_unique_sql": eligibility_counts["training_ineligible_unique_sql"],
            "heldout_eligible_examples": 498,
            "heldout_ineligible_examples": 2,
            "ignored_non_database_sql_symlinks": ignored_non_database_sql_symlinks,
        }
    except BaseException as error:
        if staging.is_dir():
            failure = staging / "finalization-failure.json"
            if not failure.exists():
                write_json(
                    failure,
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    private=True,
                )
        raise
