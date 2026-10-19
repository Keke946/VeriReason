import hashlib
import signal
import sqlite3
import time
from pathlib import Path

import pytest

from verireason.sql_evaluator import (
    EvaluatorLimits,
    QueryResult,
    analyze_sql,
    bird_ex_behavior_compatible,
    compare_strict_results,
    evaluate_prediction,
    execute_read_only,
    strict_results_equal,
)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE items(id INTEGER PRIMARY KEY, kind TEXT, value REAL, optional TEXT);
        INSERT INTO items VALUES (1, 'a', 1.0, NULL);
        INSERT INTO items VALUES (2, 'b', 2.0, 'x');
        INSERT INTO items VALUES (3, 'a', 2.0000004, NULL);
        """
    )
    connection.close()
    return path


def test_sql_analysis_allows_quotes_and_one_trailing_semicolon() -> None:
    result = analyze_sql("SELECT ';' AS value /* ; */;")

    assert result.accepted
    assert result.sql == "SELECT ';' AS value /* ; */"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CASE WHEN 1 THEN 'yes' ELSE 'no' END",
        "SELECT CASE WHEN 1 THEN CASE WHEN 2 THEN 3 ELSE 4 END ELSE 5 END",
        "SELECT REPLACE('abc', 'a', 'z')",
        "WITH value AS (SELECT 1 AS n) SELECT n FROM value",
        "SELECT 1; -- trailing comment",
        "SELECT 1; /* trailing block comment */",
    ],
)
def test_sql_analysis_accepts_valid_query_constructs(sql: str) -> None:
    assert analyze_sql(sql).accepted


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "DROP TABLE items",
        "WITH x AS (SELECT 1) DELETE FROM items",
        "WITH x AS (SELECT 1) UPDATE items SET kind = 'x'",
        "WITH x AS (SELECT 1) INSERT INTO items(kind) VALUES ('x')",
        "WITH x AS (SELECT 1) REPLACE INTO items(id) VALUES (1)",
        "PRAGMA user_version",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "DETACH DATABASE other",
        "SELECT load_extension('bad')",
        "SELECT readfile('/etc/passwd')",
        "SELECT writefile('/tmp/bad', 'x')",
        "BEGIN TRANSACTION",
        "SELECT 1; garbage comment",
    ],
)
def test_unsafe_and_multi_statement_sql_is_rejected(sql: str) -> None:
    assert not analyze_sql(sql).accepted


def test_behavior_compatible_golden_semantics() -> None:
    gold = QueryResult("ok", rows=((1, "a"), (2, "b"), (2, "b")), column_count=2)
    reordered_deduplicated = QueryResult("ok", rows=((2, "b"), (1, "a")), column_count=2)
    swapped_columns = QueryResult("ok", rows=(("a", 1), ("b", 2)), column_count=2)

    assert bird_ex_behavior_compatible(reordered_deduplicated, gold)
    assert not bird_ex_behavior_compatible(swapped_columns, gold)
    assert not bird_ex_behavior_compatible(QueryResult("execution_error"), gold)


def test_formal_default_resource_limits_use_amended_protocol() -> None:
    limits = EvaluatorLimits()

    assert limits.timeout_seconds == 10.0
    assert limits.maximum_cpu_seconds == 12
    assert limits.maximum_cpu_hard_seconds == 13
    assert limits.maximum_result_rows == 100_000
    assert limits.maximum_result_bytes == 64 * 1024**2
    assert limits.maximum_vm_bytes == 2 * 1024**3


def test_strict_golden_semantics_for_bags_order_null_float_and_empty() -> None:
    limits = EvaluatorLimits()
    bag_gold = QueryResult("ok", rows=((1, None), (1, None), (2, 2.0)), column_count=2)
    bag_pred = QueryResult("ok", rows=((2, 2.0000004), (1, None), (1, None)), column_count=2)
    missing_duplicate = QueryResult("ok", rows=((2, 2.0), (1, None)), column_count=2)
    ordered_gold = QueryResult("ok", rows=((1,), (2,)), column_count=1, top_level_order_by=True)
    ordered_wrong = QueryResult("ok", rows=((2,), (1,)), column_count=1)

    assert strict_results_equal(bag_pred, bag_gold, limits)
    assert not strict_results_equal(missing_duplicate, bag_gold, limits)
    assert not strict_results_equal(ordered_wrong, ordered_gold, limits)
    assert strict_results_equal(
        QueryResult("ok", column_count=2), QueryResult("ok", column_count=2)
    )
    assert not strict_results_equal(
        QueryResult("ok", column_count=1), QueryResult("ok", column_count=2)
    )


def test_float_bag_matching_is_complete_and_permutation_invariant() -> None:
    limits = EvaluatorLimits(float_absolute_tolerance=1e-6, float_relative_tolerance=0)
    prediction_rows = ((0.0,), (1.5e-6,))
    gold_rows = ((0.75e-6,), (0.0,))

    for prediction in (prediction_rows, tuple(reversed(prediction_rows))):
        for gold in (gold_rows, tuple(reversed(gold_rows))):
            assert strict_results_equal(
                QueryResult("ok", rows=prediction, column_count=1),
                QueryResult("ok", rows=gold, column_count=1),
                limits,
            )


def test_float_comparison_budget_fails_before_pair_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    limits = EvaluatorLimits(maximum_float_pair_checks=1_000)
    rows = tuple((float(index),) for index in range(40))

    def unexpected_pair_check(*_args: object) -> bool:
        raise AssertionError("pair comparison ran after the frozen budget was exceeded")

    monkeypatch.setattr("verireason.sql_evaluator._row_equal", unexpected_pair_check)
    result = compare_strict_results(
        QueryResult("ok", rows=rows, column_count=1),
        QueryResult("ok", rows=rows, column_count=1),
        limits,
    )

    assert result.status == "comparator_resource_limit"
    assert not result.equal


def test_read_only_execution_and_equivalent_sql(database: Path) -> None:
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    score = evaluate_prediction(
        database,
        "SELECT kind FROM items WHERE id IN (3, 1) ORDER BY id DESC",
        "SELECT kind FROM items WHERE id = 3 OR id = 1 ORDER BY id DESC",
    )
    after = hashlib.sha256(database.read_bytes()).hexdigest()

    assert score.bird_ex_behavior_compatible == 1
    assert score.strict_execution_accuracy == 1
    assert before == after


def test_illegal_and_exceptional_predictions_score_zero(database: Path) -> None:
    unsafe = evaluate_prediction(database, "DELETE FROM items", "SELECT id FROM items")
    invalid = evaluate_prediction(database, "SELECT missing FROM items", "SELECT id FROM items")

    assert unsafe.bird_ex_behavior_compatible == unsafe.strict_execution_accuracy == 0
    assert unsafe.prediction_status == "unsafe"
    assert unsafe.strict_comparison_status == "sql_execution_error"
    assert invalid.bird_ex_behavior_compatible == invalid.strict_execution_accuracy == 0
    assert invalid.prediction_status == "execution_error"
    assert invalid.strict_comparison_status == "sql_execution_error"


def test_evaluation_reports_comparator_resource_limit(database: Path) -> None:
    sql = """
        WITH RECURSIVE count(x) AS (
          SELECT 1 UNION ALL SELECT x + 1 FROM count WHERE x < 40
        ) SELECT x * 1.0 FROM count
    """
    score = evaluate_prediction(
        database,
        sql,
        sql,
        EvaluatorLimits(maximum_float_pair_checks=1_000),
    )

    assert score.bird_ex_behavior_compatible == 1
    assert score.strict_execution_accuracy == 0
    assert score.strict_comparison_status == "comparator_resource_limit"


def test_large_result_is_received_before_worker_join(database: Path) -> None:
    limits = EvaluatorLimits(
        timeout_seconds=10,
        maximum_result_rows=30_000,
        maximum_result_bytes=10_000_000,
    )
    result = execute_read_only(
        database,
        """
        WITH RECURSIVE count(x) AS (
          SELECT 1 UNION ALL SELECT x + 1 FROM count WHERE x < 20000
        ) SELECT x FROM count
        """,
        limits,
    )

    assert result.status == "ok"
    assert result.row_count == 20_000
    assert len(result.rows) == 20_000
    assert result.rows[-1] == (20_000,)


@pytest.mark.parametrize(
    ("limits", "expected_error"),
    [
        (EvaluatorLimits(maximum_result_rows=10), "result row limit exceeded"),
        (EvaluatorLimits(maximum_result_bytes=10), "result byte limit exceeded"),
    ],
)
def test_status_only_execution_preserves_result_caps(
    database: Path, limits: EvaluatorLimits, expected_error: str
) -> None:
    result = execute_read_only(
        database,
        """
        WITH RECURSIVE count(x) AS (
          SELECT 1 UNION ALL SELECT x + 1 FROM count WHERE x < 20
        ) SELECT x FROM count
        """,
        limits,
        collect_rows=False,
    )

    assert result.status == "resource_limit"
    assert result.error == expected_error
    assert result.limit_kind == ("rows" if "row" in expected_error else "bytes")
    assert result.exit_code == 0
    assert result.worker_state == "exited"
    assert result.rows == ()
    assert result.row_count > 0
    assert result.result_bytes > 0


def test_worker_exception_has_stable_failure_code(database: Path) -> None:
    limits = EvaluatorLimits(maximum_result_rows=None)  # type: ignore[arg-type]

    result = execute_read_only(database, "SELECT 1", limits)

    assert result.status == "worker_error"
    assert result.error is not None and result.error.startswith("TypeError:")
    assert result.exit_code == 0
    assert result.worker_state == "exited"


def test_timeout_is_bounded_and_scores_zero(database: Path) -> None:
    limits = EvaluatorLimits(
        timeout_seconds=0.05,
        progress_steps=1_000_000_000,
        maximum_result_rows=100_000_000,
        maximum_result_bytes=1_000_000_000,
    )
    started = time.monotonic()
    result = execute_read_only(
        database,
        """
        WITH RECURSIVE count(x) AS (
          SELECT 1 UNION ALL SELECT x + 1 FROM count
        ) SELECT x FROM count
        """,
        limits,
        collect_rows=False,
    )
    elapsed = time.monotonic() - started

    assert result.status == "timeout"
    assert result.limit_kind == "wall"
    assert result.exit_code is not None
    assert elapsed < 2


def test_ipc_eof_joins_before_classifying_sigxcpu(
    database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FakeConnection:
        def poll(self, _timeout: float) -> bool:
            return True

        def recv(self) -> QueryResult:
            events.append("recv")
            raise EOFError

        def close(self) -> None:
            events.append("pipe_close")

    class FakeProcess:
        exitcode: int | None = None

        def __init__(self, **_kwargs: object) -> None:
            self.alive = True

        def start(self) -> None:
            events.append("start")

        def join(self, _timeout: float) -> None:
            events.append("join")
            self.alive = False
            self.exitcode = -signal.SIGXCPU

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

        def close(self) -> None:
            events.append("process_close")

    class FakeContext:
        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert not duplex
            return FakeConnection(), FakeConnection()

        def Process(self, **kwargs: object) -> FakeProcess:
            return FakeProcess(**kwargs)

    monkeypatch.setattr(
        "verireason.sql_evaluator.multiprocessing.get_context", lambda _kind: FakeContext()
    )
    result = execute_read_only(database, "SELECT 1")

    assert result.status == "resource_limit"
    assert result.limit_kind == "cpu"
    assert result.exit_code == -signal.SIGXCPU
    assert result.signal_name == "SIGXCPU"
    assert result.worker_state == "signaled"
    assert events.index("recv") < events.index("join")
    assert "terminate" not in events and "kill" not in events
    assert events.count("pipe_close") >= 2
    assert events.count("process_close") == 1


def test_gold_query_failure_is_a_preflight_error(database: Path) -> None:
    with pytest.raises(ValueError, match="gold SQL preflight failed"):
        evaluate_prediction(database, "SELECT 1", "SELECT missing FROM items")
