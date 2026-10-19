"""Full gold-query execution preflight required before M2 manifests can freeze."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from verireason.data_pipeline import Example, resolve_database_path
from verireason.sql_evaluator import DEFAULT_LIMITS, EvaluatorLimits, QueryResult, execute_read_only

_REPORT_STATUSES = (
    "ok",
    "unsafe",
    "syntax_error",
    "execution_error",
    "timeout",
    "resource_limit",
    "worker_error",
    "database_error",
)


@dataclass(frozen=True)
class FailureRecord:
    db_id: str
    example_ids: tuple[str, ...]
    status: str
    error: str | None
    exit_code: int
    signal_name: str | None
    limit_kind: str | None
    worker_state: str


@dataclass(frozen=True)
class GoldPreflightReport:
    total_sql_count: int
    unique_sql_count: int
    status_counts: dict[str, int]
    duration_seconds: dict[str, float | int]
    failures_by_database: dict[str, dict[str, int]]
    failures: tuple[FailureRecord, ...]

    @property
    def passed(self) -> bool:
        return self.status_counts["ok"] == self.unique_sql_count

    def as_dict(self, *, include_failure_ids: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "total_sql_count": self.total_sql_count,
            "unique_sql_count": self.unique_sql_count,
            "status_counts": self.status_counts,
            "duration_seconds": self.duration_seconds,
            "passed": self.passed,
        }
        if include_failure_ids:
            payload["failures_by_database"] = self.failures_by_database
            payload["failures"] = [asdict(failure) for failure in self.failures]
        return payload


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def _duration_summary(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else 0.0,
        "minimum": ordered[0] if ordered else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "maximum": ordered[-1] if ordered else 0.0,
    }


def _report_status(status: str) -> str:
    return status if status in _REPORT_STATUSES else "worker_error"


def run_gold_preflight(
    examples: Sequence[Example],
    database_root: str | Path,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
    executor: Callable[..., QueryResult] = execute_read_only,
) -> GoldPreflightReport:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for example in examples:
        grouped[(example.db_id, example.sql)].append(example.row_id)

    status_counts: Counter[str] = Counter({status: 0 for status in _REPORT_STATUSES})
    durations: list[float] = []
    failures: list[FailureRecord] = []
    failures_by_database: dict[str, Counter[str]] = defaultdict(Counter)
    database_paths = {
        db_id: resolve_database_path(database_root, db_id)
        for db_id in sorted({db_id for db_id, _sql in grouped})
    }

    for (db_id, sql), example_ids in sorted(grouped.items()):
        started = time.perf_counter()
        result = executor(database_paths[db_id], sql, limits, collect_rows=False)
        parent_duration = time.perf_counter() - started
        durations.append(result.duration_seconds or parent_duration)
        status = _report_status(result.status)
        status_counts[status] += 1
        if status != "ok":
            failures.append(
                FailureRecord(
                    db_id=db_id,
                    example_ids=tuple(sorted(example_ids)),
                    status=status,
                    error=result.error,
                    exit_code=result.exit_code,
                    signal_name=result.signal_name,
                    limit_kind=result.limit_kind,
                    worker_state=result.worker_state,
                )
            )
            failures_by_database[db_id]["failed_unique_sql"] += 1
            failures_by_database[db_id]["failed_examples"] += len(example_ids)

    return GoldPreflightReport(
        total_sql_count=len(examples),
        unique_sql_count=len(grouped),
        status_counts=dict(status_counts),
        duration_seconds=_duration_summary(durations),
        failures_by_database={
            db_id: dict(counts) for db_id, counts in sorted(failures_by_database.items())
        },
        failures=tuple(failures),
    )


def require_gold_preflight_passed(**reports: GoldPreflightReport) -> None:
    failed = [name for name, report in reports.items() if not report.passed]
    if failed:
        raise RuntimeError(
            "gold execution preflight failed; final manifests are forbidden: "
            + ", ".join(sorted(failed))
        )
