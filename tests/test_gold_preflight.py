import sqlite3
from pathlib import Path

import pytest

from verireason.data_pipeline import Example
from verireason.gold_preflight import require_gold_preflight_passed, run_gold_preflight
from verireason.sql_evaluator import EvaluatorLimits, QueryResult


def _database(root: Path, db_id: str) -> None:
    directory = root / db_id
    directory.mkdir()
    connection = sqlite3.connect(directory / f"{db_id}.sqlite")
    connection.execute("CREATE TABLE values_table(value INTEGER)")
    connection.close()


def test_gold_preflight_deduplicates_and_records_failures_without_sql(tmp_path: Path) -> None:
    _database(tmp_path, "a")
    _database(tmp_path, "b")
    examples = [
        Example("1", "a", "q1", "SELECT 1"),
        Example("2", "a", "q2", "SELECT 1"),
        Example("3", "b", "q3", "SELECT bad"),
    ]

    collect_rows_values: list[bool] = []

    def fake_executor(
        _path: str | Path,
        sql: str,
        _limits: EvaluatorLimits,
        *,
        collect_rows: bool,
    ) -> QueryResult:
        collect_rows_values.append(collect_rows)
        return QueryResult("ok") if sql == "SELECT 1" else QueryResult("syntax_error", error="bad")

    report = run_gold_preflight(examples, tmp_path, executor=fake_executor)
    public = report.as_dict(include_failure_ids=False)
    private = report.as_dict(include_failure_ids=True)

    assert report.total_sql_count == 3
    assert report.unique_sql_count == 2
    assert report.status_counts["ok"] == 1
    assert report.status_counts["syntax_error"] == 1
    assert report.failures_by_database == {"b": {"failed_unique_sql": 1, "failed_examples": 1}}
    assert "failures" not in public
    assert "failures_by_database" not in public
    assert private["failures"][0]["example_ids"] == ("3",)
    assert private["failures"][0]["exit_code"] == 0
    assert "SELECT bad" not in repr(private)
    assert collect_rows_values == [False, False]
    assert not report.passed
    with pytest.raises(RuntimeError, match="final manifests are forbidden"):
        require_gold_preflight_passed(training=report)


def test_gold_preflight_status_only_traverses_large_result(tmp_path: Path) -> None:
    _database(tmp_path, "large")
    sql = """
        WITH RECURSIVE count(x) AS (
          SELECT 1 UNION ALL SELECT x + 1 FROM count WHERE x < 20000
        ) SELECT x FROM count
    """
    report = run_gold_preflight(
        [Example("large-1", "large", "large result", sql)],
        tmp_path,
        EvaluatorLimits(
            timeout_seconds=10,
            maximum_result_rows=30_000,
            maximum_result_bytes=10_000_000,
        ),
    )

    assert report.passed
    assert report.status_counts["ok"] == 1


def test_gold_preflight_preserves_worker_error_classification(tmp_path: Path) -> None:
    _database(tmp_path, "worker")

    def fake_executor(*_args: object, **_kwargs: object) -> QueryResult:
        return QueryResult(
            "worker_error",
            error="RuntimeError: child failed",
            exit_code=0,
            worker_state="exception",
        )

    report = run_gold_preflight(
        [Example("worker-1", "worker", "worker failure", "SELECT 1")],
        tmp_path,
        executor=fake_executor,
    )

    assert report.status_counts["worker_error"] == 1
    assert report.status_counts["execution_error"] == 0
    assert report.failures[0].worker_state == "exception"
