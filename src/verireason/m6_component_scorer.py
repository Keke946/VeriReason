"""Production training-only conversion from SQL completions to frozen reward components."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from verireason.reward_contract import RewardComponents
from verireason.sql_evaluator import (
    DEFAULT_LIMITS,
    EvaluatorLimits,
    QueryResult,
    compare_strict_results,
    execute_read_only,
)
from verireason.sql_features import UnsupportedGoldStructureError, score_sql_structure
from verireason.sqlite_schema import load_sqlite_schema
from verireason.trl_reward_adapter import RewardInput


class SQLiteRewardComponentScorer:
    """Score contiguous TRL completions using the frozen evaluator and dense gate."""

    def __init__(self, database_root: str | Path, limits: EvaluatorLimits = DEFAULT_LIMITS) -> None:
        self.database_root = Path(database_root).resolve()
        if not self.database_root.is_dir():
            raise ValueError("database_root must be an existing directory")
        self.limits = limits
        self._cached_key: tuple[Path, str] | None = None
        self._cached_gold: QueryResult | None = None
        self._cached_schema: dict[str, set[str]] | None = None
        self._prediction_statuses: Counter[str] = Counter()
        self._components: list[RewardComponents] = []

    def _record(self, status: str, component: RewardComponents) -> RewardComponents:
        self._prediction_statuses[status] += 1
        self._components.append(component)
        return component

    def summary(self) -> dict[str, Any]:
        """Return SQL-free aggregate telemetry for reward-hacking review."""
        count = len(self._components)
        if not count:
            raise RuntimeError("component scorer has not evaluated any completions")
        return {
            "completion_count": count,
            "strict_correct_count": sum(item.r_exec for item in self._components),
            "valid_count": sum(item.r_valid for item in self._components),
            "confidence_gate_count": sum(item.g_conf for item in self._components),
            "mean_structure_score": sum(item.r_struct for item in self._components) / count,
            "prediction_status_counts": dict(sorted(self._prediction_statuses.items())),
        }

    def _database_path(self, logical_path: object) -> Path:
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError("database_path metadata must be a non-empty relative path")
        candidate = (self.database_root / logical_path).resolve()
        try:
            candidate.relative_to(self.database_root)
        except ValueError as error:
            raise ValueError("database_path escapes database_root") from error
        if not candidate.is_file():
            raise ValueError("database_path does not identify a file")
        return candidate

    def __call__(self, item: RewardInput) -> RewardComponents:
        gold_sql = item.metadata.get("gold_sql")
        if not isinstance(gold_sql, str) or not gold_sql.strip():
            raise ValueError("gold_sql metadata must be a non-empty string")
        database_path = self._database_path(item.metadata.get("database_path"))
        key = (database_path, gold_sql)
        if key != self._cached_key:
            gold = execute_read_only(database_path, gold_sql, self.limits)
            if not gold.successful:
                raise ValueError(f"gold SQL preflight failed: {gold.status}: {gold.error}")
            self._cached_key = key
            self._cached_gold = gold
            self._cached_schema = load_sqlite_schema(database_path)
        assert self._cached_gold is not None and self._cached_schema is not None

        prediction = execute_read_only(database_path, item.completion, self.limits)
        r_exec = int(compare_strict_results(prediction, self._cached_gold, self.limits).equal)
        r_valid = int(prediction.successful)
        try:
            structure = score_sql_structure(
                gold_sql,
                item.completion,
                self._cached_schema,
                prediction_execution_status=prediction.status,
            )
        except UnsupportedGoldStructureError:
            return self._record(
                prediction.status,
                RewardComponents(r_exec=r_exec, r_valid=r_valid, r_struct=0.0, g_conf=0),
            )
        if structure.r_valid != r_valid:
            raise RuntimeError("evaluator and structural validity signals disagree")
        return self._record(
            prediction.status,
            RewardComponents(
                r_exec=r_exec,
                r_valid=r_valid,
                r_struct=structure.r_struct,
                g_conf=structure.g_conf,
            ),
        )
