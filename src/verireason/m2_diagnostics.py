"""Diagnostic-only reruns for previously failed M2 gold preflight queries."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verireason.data_pipeline import Example, resolve_database_path, sha256_file
from verireason.sql_evaluator import EvaluatorLimits, QueryResult, execute_read_only


@dataclass(frozen=True)
class DiagnosticItem:
    key: str
    split: str
    profile: str
    db_id: str
    example_ids: tuple[str, ...]
    sql: str


_EXPECTED_EVIDENCE = {
    "private_preflight_sha256": "2408b34df2e4e55544c77c49efb5a8c42364d86ac9e018168ea4b5dfeff98e02",
    "training_labels_sha256": "2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8",
    "heldout_labels_sha256": "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2",
    "formal_config_sha256": "d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b",
    "source_receipt_sha256": "da189fd0f69d7713c16acfe434738e1f867074665c2b95d7d78b3ace29741ffe",
}


def _query_key(split: str, db_id: str, sql: str) -> str:
    return hashlib.sha256(f"{split}\0{db_id}\0{sql}".encode()).hexdigest()


def _profile_for_failure(failure: Mapping[str, Any]) -> str | None:
    if failure.get("status") == "ok":
        return None
    error = str(failure.get("error") or "")
    if failure.get("limit_kind") == "rows" or "result row limit exceeded" in error:
        return "profile_b"
    if (
        failure.get("limit_kind") == "cpu"
        or failure.get("signal_name") == "SIGXCPU"
        or "exit code -24" in error
        or "exit code None" in error
    ):
        return "profile_a"
    raise ValueError(f"unrecognized prior non-ok failure category: {failure.get('status')!r}")


def select_failed_queries(
    private_preflight: Mapping[str, Any],
    examples_by_split: Mapping[str, Sequence[Example]],
) -> tuple[DiagnosticItem, ...]:
    selected: dict[str, DiagnosticItem] = {}
    for split in ("training", "heldout"):
        section = private_preflight.get(split)
        if not isinstance(section, Mapping):
            raise ValueError(f"private preflight is missing split {split!r}")
        failures = section.get("failures")
        if not isinstance(failures, list):
            raise ValueError(f"private preflight split {split!r} has no failure list")
        examples = examples_by_split.get(split)
        if examples is None:
            raise ValueError(f"examples are missing split {split!r}")
        by_identity = {(example.db_id, example.row_id): example for example in examples}
        for failure in failures:
            if not isinstance(failure, Mapping):
                raise ValueError("private preflight failure must be an object")
            profile = _profile_for_failure(failure)
            if profile is None:
                continue
            db_id = failure.get("db_id")
            example_ids = failure.get("example_ids")
            if not isinstance(db_id, str) or not isinstance(example_ids, list | tuple):
                raise ValueError("private preflight failure identity is invalid")
            matches = [by_identity.get((db_id, str(example_id))) for example_id in example_ids]
            if not matches or any(example is None for example in matches):
                raise ValueError("private preflight failure does not resolve to frozen labels")
            resolved = [example for example in matches if example is not None]
            sql_values = {example.sql for example in resolved}
            if len(sql_values) != 1:
                raise ValueError("one failure record resolves to multiple SQL strings")
            sql = next(iter(sql_values))
            key = _query_key(split, db_id, sql)
            item = DiagnosticItem(
                key=key,
                split=split,
                profile=profile,
                db_id=db_id,
                example_ids=tuple(sorted(str(value) for value in example_ids)),
                sql=sql,
            )
            previous = selected.get(key)
            if previous is not None and previous != item:
                raise ValueError("conflicting duplicate failure identity")
            selected[key] = item
    return tuple(selected[key] for key in sorted(selected))


def load_diagnostic_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("schema_version") != 1 or config.get("diagnostic_only") is not True:
        raise ValueError("M2 diagnostic config must be schema v1 and diagnostic_only")
    if config.get("maximum_session_wall_seconds") != 5400:
        raise ValueError("M2 diagnostic session wall limit must remain 5400 seconds")
    if config.get("evidence") != _EXPECTED_EVIDENCE:
        raise ValueError("M2 diagnostic evidence identities changed from accepted evidence")
    if config.get("selection") != {
        "training_profile_a": 220,
        "training_profile_b": 22,
        "heldout_profile_a": 2,
        "heldout_profile_b": 0,
    }:
        raise ValueError("M2 diagnostic selection counts changed from accepted evidence")
    expected = {
        "profile_a": (10.0, 12, 13, 100_000, 67_108_864),
        "profile_b": (30.0, 32, 33, 1_000_000, 536_870_912),
    }
    for name, values in expected.items():
        profile = config.get(name)
        if not isinstance(profile, dict):
            raise ValueError(f"missing diagnostic profile {name}")
        observed = (
            float(profile["timeout_seconds"]),
            profile["maximum_cpu_seconds"],
            profile["maximum_cpu_hard_seconds"],
            profile["maximum_result_rows"],
            profile["maximum_result_bytes"],
        )
        if (
            observed != values
            or profile["maximum_cpu_hard_seconds"] != profile["maximum_cpu_seconds"] + 1
        ):
            raise ValueError(f"diagnostic profile {name} changed from its preregistration")
        if profile["maximum_cpu_seconds"] <= profile["timeout_seconds"]:
            raise ValueError(f"diagnostic profile {name} CPU limit must exceed its wall timeout")
    return config


def verify_input_evidence(
    config: Mapping[str, Any], paths: Mapping[str, str | Path]
) -> dict[str, str]:
    expected = config["evidence"]
    required = {
        "private_preflight_sha256": "private_preflight",
        "training_labels_sha256": "training_labels",
        "heldout_labels_sha256": "heldout_labels",
        "formal_config_sha256": "formal_config",
        "source_receipt_sha256": "source_receipt",
    }
    observed: dict[str, str] = {}
    for identity_name, path_name in required.items():
        if path_name not in paths:
            raise ValueError(f"missing diagnostic evidence path: {path_name}")
        digest = sha256_file(paths[path_name])
        if digest != expected[identity_name]:
            raise ValueError(
                f"diagnostic evidence SHA-256 mismatch for {path_name}: "
                f"{digest} != {expected[identity_name]}"
            )
        observed[identity_name] = digest
    return observed


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def require_private_output_directory(
    output_dir: str | Path,
    *,
    git_worktree: str | Path,
    materialization_partial: str | Path,
) -> Path:
    resolved = Path(output_dir).resolve()
    git_root = Path(git_worktree).resolve()
    partial_root = Path(materialization_partial).resolve()
    if _is_within(resolved, git_root):
        raise ValueError("diagnostic output directory must be outside the Git worktree")
    if _is_within(resolved, partial_root):
        raise ValueError("diagnostic output directory must be outside materialization .partial")
    return resolved


def _require_checkpoint_mode(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"diagnostic checkpoint must have mode 0600, observed {mode:04o}")


def require_expected_selection(items: Sequence[DiagnosticItem], config: Mapping[str, Any]) -> None:
    observed = Counter(f"{item.split}_{item.profile}" for item in items)
    expected = config["selection"]
    for split in ("training", "heldout"):
        for profile in ("profile_a", "profile_b"):
            key = f"{split}_{profile}"
            if observed[key] != expected[key]:
                raise ValueError(
                    f"diagnostic selection count changed for {key}: "
                    f"{observed[key]} != {expected[key]}"
                )


def profile_limits(config: Mapping[str, Any], profile_name: str) -> EvaluatorLimits:
    profile = config[profile_name]
    return EvaluatorLimits(
        timeout_seconds=profile["timeout_seconds"],
        progress_steps=profile["progress_steps"],
        maximum_vm_bytes=profile["maximum_vm_bytes"],
        maximum_cpu_seconds=profile["maximum_cpu_seconds"],
        maximum_cpu_hard_seconds=profile["maximum_cpu_hard_seconds"],
        maximum_result_rows=profile["maximum_result_rows"],
        maximum_result_bytes=profile["maximum_result_bytes"],
        maximum_float_pair_checks=profile["maximum_float_pair_checks"],
        float_absolute_tolerance=profile["float_absolute_tolerance"],
        float_relative_tolerance=profile["float_relative_tolerance"],
    )


def _write_private_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _require_checkpoint_mode(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_diagnostics(
    items: Sequence[DiagnosticItem],
    database_roots: Mapping[str, str | Path],
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    source_identity: Mapping[str, str],
    executor: Callable[..., QueryResult] = execute_read_only,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    target = Path(checkpoint_path)
    plan_payload = {
        "items": [{"key": item.key, "profile": item.profile} for item in items],
        "config": config,
        "source_identity": dict(sorted(source_identity.items())),
    }
    plan_sha256 = hashlib.sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if target.exists():
        _require_checkpoint_mode(target)
        checkpoint = json.loads(target.read_text(encoding="utf-8"))
        if checkpoint.get("diagnostic_only") is not True:
            raise ValueError("refusing to resume a non-diagnostic checkpoint")
        if checkpoint.get("plan_sha256") != plan_sha256:
            raise ValueError("diagnostic checkpoint plan differs from the current selection/config")
    else:
        checkpoint = {
            "schema_version": 1,
            "diagnostic_only": True,
            "plan_sha256": plan_sha256,
            "source_identity": dict(sorted(source_identity.items())),
            "complete": False,
            "results": {},
        }
    results = checkpoint.get("results")
    if not isinstance(results, dict):
        raise ValueError("diagnostic checkpoint results must be an object")

    session_started = monotonic()
    for item in items:
        if item.key in results:
            continue
        if monotonic() - session_started >= config["maximum_session_wall_seconds"]:
            checkpoint["complete"] = False
            checkpoint["stop_reason"] = "maximum_session_wall_seconds"
            _write_private_checkpoint(target, checkpoint)
            return checkpoint
        database_path = resolve_database_path(database_roots[item.split], item.db_id)
        result = executor(
            database_path,
            item.sql,
            profile_limits(config, item.profile),
            collect_rows=False,
        )
        if result.exit_code is None:  # defensive contract for injected/custom executors
            raise ValueError("diagnostic result cannot persist exit_code=None")
        results[item.key] = {
            "split": item.split,
            "profile": item.profile,
            "db_id": item.db_id,
            "example_ids": list(item.example_ids),
            "result": asdict(result),
        }
        _write_private_checkpoint(target, checkpoint)
    checkpoint["complete"] = len(results) == len(items)
    checkpoint.pop("stop_reason", None)
    _write_private_checkpoint(target, checkpoint)
    return checkpoint


def _duration_bucket(seconds: float) -> str:
    if seconds <= 3:
        return "le_3_seconds"
    if seconds <= 5:
        return "gt_3_le_5_seconds"
    if seconds <= 10:
        return "gt_5_le_10_seconds"
    return "gt_10_seconds"


def _row_bucket(row_count: int) -> str:
    if row_count < 250_000:
        return "100k_to_250k"
    if row_count < 500_000:
        return "250k_to_500k"
    if row_count <= 1_000_000:
        return "500k_to_1m"
    return "gt_1m"


def build_public_report(
    checkpoint: Mapping[str, Any], items: Sequence[DiagnosticItem]
) -> dict[str, Any]:
    results = checkpoint.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("diagnostic checkpoint results must be an object")
    expected_count = len(items)
    completed_count = sum(item.key in results for item in items)
    complete = checkpoint.get("complete") is True and completed_count == expected_count
    source_identity = checkpoint.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("diagnostic checkpoint source identity is missing")
    report: dict[str, Any] = {
        "schema_version": 1,
        "diagnostic_only": True,
        "plan_sha256": checkpoint.get("plan_sha256"),
        "complete": complete,
        "completed_count": completed_count,
        "expected_count": expected_count,
        "input_evidence_sha256": dict(sorted(source_identity.items())),
        "splits": {},
    }
    for split in ("training", "heldout"):
        split_items = [item for item in items if item.split == split]
        cpu_items = [item for item in split_items if item.profile == "profile_a"]
        large_items = [item for item in split_items if item.profile == "profile_b"]
        cpu_counts: Counter[str] = Counter()
        large_counts: Counter[str] = Counter()
        unstable = 0
        for item in cpu_items:
            entry = results.get(item.key)
            if not isinstance(entry, Mapping):
                continue
            result = entry["result"]
            status = result["status"]
            if status == "ok":
                cpu_counts["succeeded"] += 1
                cpu_counts[_duration_bucket(float(result["duration_seconds"]))] += 1
            elif status == "resource_limit" and result.get("limit_kind") == "cpu":
                cpu_counts["cpu_limit"] += 1
            elif status == "timeout" and result.get("limit_kind") == "wall":
                cpu_counts["wall_timeout"] += 1
            else:
                cpu_counts["worker_error"] += 1
            if (
                result.get("exit_code") is None
                or result.get("worker_state") == "exitcode_unavailable"
            ):
                unstable += 1
        for item in large_items:
            entry = results.get(item.key)
            if not isinstance(entry, Mapping):
                continue
            result = entry["result"]
            if result["status"] == "ok" or (
                result["status"] == "resource_limit" and result.get("limit_kind") == "rows"
            ):
                large_counts[_row_bucket(int(result["row_count"]))] += 1
            elif result["status"] == "resource_limit" and result.get("limit_kind") == "bytes":
                large_counts["byte_limit"] += 1
            elif result["status"] == "timeout":
                large_counts["timeout"] += 1
            else:
                large_counts["worker_error"] += 1
        report["splits"][split] = {
            "cpu_ipc": {
                "original_failure_count": len(cpu_items),
                "completed_count": sum(item.key in results for item in cpu_items),
                "succeeded": cpu_counts["succeeded"],
                "duration_buckets": {
                    key: cpu_counts[key]
                    for key in (
                        "le_3_seconds",
                        "gt_3_le_5_seconds",
                        "gt_5_le_10_seconds",
                        "gt_10_seconds",
                    )
                },
                "cpu_limit": cpu_counts["cpu_limit"],
                "wall_timeout": cpu_counts["wall_timeout"],
                "worker_error": cpu_counts["worker_error"],
                "unstable_exitcode": unstable,
            },
            "large_result": {
                "original_failure_count": len(large_items),
                "completed_count": sum(item.key in results for item in large_items),
                **{
                    key: large_counts[key]
                    for key in (
                        "100k_to_250k",
                        "250k_to_500k",
                        "500k_to_1m",
                        "gt_1m",
                        "byte_limit",
                        "timeout",
                        "worker_error",
                    )
                },
            },
        }
    return report
