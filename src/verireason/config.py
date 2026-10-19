"""Typed loading helpers for frozen VeriReason TOML configuration files."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str
    role: str
    license: str
    published_repository_size_gb: float


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    question: str
    expected_rows: list[list[Any]]


@dataclass(frozen=True)
class SmokeConfig:
    benchmark: dict[str, Any]
    models: tuple[ModelSpec, ...]
    cases: tuple[SmokeCase, ...]

    def model(self, key: str) -> ModelSpec:
        matches = [model for model in self.models if model.key == key]
        if len(matches) != 1:
            raise ValueError(f"model key must resolve exactly once: {key!r}")
        return matches[0]


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_smoke_config(path: str | Path) -> SmokeConfig:
    raw = load_toml(path)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported M1 smoke config schema")

    benchmark = raw.get("benchmark")
    model_rows = raw.get("models")
    case_rows = raw.get("cases")
    if not isinstance(benchmark, dict) or not isinstance(model_rows, list) or not case_rows:
        raise ValueError("M1 smoke config must define benchmark, models, and cases")

    models = tuple(ModelSpec(**row) for row in model_rows)
    if len({model.key for model in models}) != len(models):
        raise ValueError("M1 smoke model keys must be unique")

    cases = tuple(
        SmokeCase(
            case_id=row["id"],
            question=row["question"],
            expected_rows=json.loads(row["expected_rows_json"]),
        )
        for row in case_rows
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("M1 smoke case IDs must be unique")

    return SmokeConfig(benchmark=benchmark, models=models, cases=cases)


def load_m2_config(path: str | Path) -> dict[str, Any]:
    raw = load_toml(path)
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported M2 data config schema")
    required = {
        "training_source",
        "heldout_source",
        "split",
        "identity",
        "materialization",
        "evaluator",
        "reward",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"M2 config is missing sections: {', '.join(missing)}")
    split = raw["split"]
    if (
        not 0
        < split["minimum_validation_example_ratio"]
        < split["maximum_validation_example_ratio"]
        < 1
    ):
        raise ValueError("M2 validation ratio bounds are invalid")
    feature_weights = raw["reward"]["structural_feature_weights"]
    if not math.isclose(sum(feature_weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("M2 structural feature weights must sum to 1")
    if raw["reward"]["structural_normalization"] != "active_dimensions_only":
        raise ValueError("M2 structural normalization contract changed")
    if raw["reward"]["all_structural_dimensions_empty_score"] != 0.0:
        raise ValueError("M2 all-empty structural score must remain zero")
    if raw["materialization"]["minimum_free_bytes"] < 70 * 1024**3:
        raise ValueError("M2 materialization free-space floor must be at least 70 GiB")
    if raw["materialization"]["dynamic_capacity_reserve_bytes"] < 10 * 1024**3:
        raise ValueError("M2 dynamic extraction reserve must be at least 10 GiB")
    training_source = raw["training_source"]
    nested_path = PurePosixPath(training_source["database_nested_archive_path"])
    if (
        nested_path.is_absolute()
        or ".." in nested_path.parts
        or not nested_path.parts
        or nested_path.suffix.casefold() != ".zip"
    ):
        raise ValueError("M2 nested training archive path must identify a ZIP")
    if training_source["database_nested_archive_maximum_member_count"] <= 0:
        raise ValueError("M2 nested training member cap must be positive")
    if training_source["database_nested_archive_maximum_uncompressed_bytes"] <= 0:
        raise ValueError("M2 nested training byte cap must be positive")
    if training_source["database_nested_archive_maximum_uncompressed_bytes"] != 50 * 1024**3:
        raise ValueError("M2 nested training byte cap must remain exactly 50 GiB")
    if raw["evaluator"]["maximum_float_pair_checks"] <= 0:
        raise ValueError("M2 float-comparison pair budget must be positive")
    evaluator = raw["evaluator"]
    expected_limits = {
        "timeout_seconds": 10.0,
        "maximum_cpu_seconds": 12,
        "maximum_cpu_hard_seconds": 13,
        "maximum_result_rows": 100_000,
        "maximum_result_bytes": 67_108_864,
        "maximum_vm_bytes": 2_147_483_648,
    }
    for key, expected in expected_limits.items():
        if evaluator.get(key) != expected:
            raise ValueError(f"M2 formal evaluator {key} must remain {expected}")
    if evaluator["maximum_cpu_hard_seconds"] != evaluator["maximum_cpu_seconds"] + 1:
        raise ValueError("M2 formal CPU hard limit must equal soft limit plus one")
    if evaluator.get("metric_scopes") != [
        "full_heldout_conservative_execution_accuracy",
        "eligible_heldout_execution_accuracy",
    ]:
        raise ValueError("M2 heldout metric scopes changed")
    if evaluator.get("formal_consumers") != [
        "rl_terminal_reward",
        "validation_evaluator",
        "benchmark_candidate_execution",
    ]:
        raise ValueError("M2 formal evaluator consumers changed")
    eligibility = raw.get("operational_eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("M2 operational eligibility contract is missing")
    if (
        eligibility.get("expected_training_eligible_unique_sql") != 6496
        or eligibility.get("expected_training_ineligible_unique_sql") != 28
        or eligibility.get("expected_heldout_eligible_examples") != 498
        or eligibility.get("expected_heldout_ineligible_examples") != 2
    ):
        raise ValueError("M2 operational eligibility counts changed")
    return raw
