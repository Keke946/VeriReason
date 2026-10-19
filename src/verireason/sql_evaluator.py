"""Independent M2 SQLite execution evaluator with strict safety boundaries."""

from __future__ import annotations

import math
import multiprocessing
import signal
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any
from urllib.parse import quote

_UNSAFE_TOKENS = {
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "BEGIN",
    "COMMIT",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "EXPLAIN",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "RELEASE",
    "ROLLBACK",
    "SAVEPOINT",
    "UPDATE",
    "VACUUM",
}
_WITH_MAIN_STATEMENTS = {"SELECT", "INSERT", "UPDATE", "DELETE", "REPLACE"}
_UNSAFE_FUNCTIONS = {"LOAD_EXTENSION", "READFILE", "WRITEFILE"}
_ALLOWED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_SELECT,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_AUTHORIZER_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


@dataclass(frozen=True)
class EvaluatorLimits:
    timeout_seconds: float = 10.0
    progress_steps: int = 1000
    maximum_vm_bytes: int = 2 * 1024**3
    maximum_cpu_seconds: int = 12
    maximum_cpu_hard_seconds: int | None = None
    maximum_result_rows: int = 100_000
    maximum_result_bytes: int = 64 * 1024**2
    maximum_float_pair_checks: int = 1_000_000
    float_absolute_tolerance: float = 1e-6
    float_relative_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.maximum_cpu_hard_seconds is None:
            object.__setattr__(self, "maximum_cpu_hard_seconds", self.maximum_cpu_seconds + 1)


DEFAULT_LIMITS = EvaluatorLimits()


@dataclass(frozen=True)
class SqlAnalysis:
    accepted: bool
    sql: str | None
    error: str | None
    top_level_order_by: bool


@dataclass(frozen=True)
class QueryResult:
    status: str
    rows: tuple[tuple[Any, ...], ...] = ()
    column_count: int = 0
    top_level_order_by: bool = False
    error: str | None = None
    row_count: int = 0
    result_bytes: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    signal_name: str | None = None
    limit_kind: str | None = None
    worker_state: str = "not_started"

    @property
    def successful(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class EvaluationScore:
    bird_ex_behavior_compatible: int
    strict_execution_accuracy: int
    prediction_status: str
    gold_status: str
    strict_comparison_status: str


@dataclass(frozen=True)
class ComparisonResult:
    status: str

    @property
    def equal(self) -> bool:
        return self.status == "equal"


def _scan_sql(sql: str) -> tuple[list[tuple[str, int]], list[int]]:
    tokens: list[tuple[str, int]] = []
    semicolons: list[int] = []
    current: list[str] = []
    current_depth = 0
    depth = 0
    state = "plain"
    index = 0

    def flush() -> None:
        nonlocal current
        if current:
            tokens.append(("".join(current).upper(), current_depth))
            current = []

    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "line_comment":
            if char == "\n":
                state = "plain"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                state = "plain"
                index += 2
            else:
                index += 1
            continue
        if state in {"single", "double", "backtick"}:
            delimiter = {"single": "'", "double": '"', "backtick": "`"}[state]
            if char == delimiter:
                if following == delimiter:
                    index += 2
                    continue
                state = "plain"
            index += 1
            continue
        if state == "bracket":
            if char == "]":
                state = "plain"
            index += 1
            continue

        if char == "-" and following == "-":
            flush()
            state = "line_comment"
            index += 2
        elif char == "/" and following == "*":
            flush()
            state = "block_comment"
            index += 2
        elif char in "'\"`[":
            flush()
            state = {"'": "single", '"': "double", "`": "backtick", "[": "bracket"}[char]
            index += 1
        elif char == "(":
            flush()
            depth += 1
            index += 1
        elif char == ")":
            flush()
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced closing parenthesis")
            index += 1
        elif char == ";":
            flush()
            semicolons.append(index)
            index += 1
        elif char.isalnum() or char == "_":
            if not current:
                current_depth = depth
            current.append(char)
            index += 1
        else:
            flush()
            index += 1
    flush()
    if state in {"single", "double", "backtick", "bracket", "block_comment"}:
        raise ValueError("unterminated quote or comment")
    if depth:
        raise ValueError("unbalanced opening parenthesis")
    return tokens, semicolons


def _comments_and_whitespace_only(sql: str) -> bool:
    state = "plain"
    index = 0
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "line_comment":
            if char == "\n":
                state = "plain"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                state = "plain"
                index += 2
            else:
                index += 1
            continue
        if char.isspace():
            index += 1
        elif char == "-" and following == "-":
            state = "line_comment"
            index += 2
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 2
        else:
            return False
    return state != "block_comment"


def _main_statement(token_names: list[str], top_level: list[str]) -> str | None:
    if not token_names:
        return None
    if token_names[0] != "WITH":
        return token_names[0]
    return next((token for token in top_level[1:] if token in _WITH_MAIN_STATEMENTS), None)


def analyze_sql(sql: str) -> SqlAnalysis:
    candidate = sql.strip()
    if not candidate:
        return SqlAnalysis(False, None, "empty SQL", False)
    try:
        tokens, semicolons = _scan_sql(candidate)
    except ValueError as error:
        return SqlAnalysis(False, None, str(error), False)
    if semicolons:
        if len(semicolons) != 1 or not _comments_and_whitespace_only(
            candidate[semicolons[0] + 1 :]
        ):
            return SqlAnalysis(False, None, "multiple SQL statements are forbidden", False)
        candidate = candidate[: semicolons[0]].rstrip()
    token_names = [token for token, _ in tokens]
    top_level = [token for token, depth in tokens if depth == 0]
    main_statement = _main_statement(token_names, top_level)
    if main_statement != "SELECT":
        if token_names and token_names[0] == "WITH" and main_statement:
            return SqlAnalysis(False, None, f"unsafe WITH statement: {main_statement}", False)
        return SqlAnalysis(False, None, "only SELECT or WITH queries are allowed", False)
    denied = sorted(set(token_names) & _UNSAFE_TOKENS)
    if denied:
        return SqlAnalysis(False, None, f"unsafe SQL token: {denied[0]}", False)
    denied_functions = sorted(set(token_names) & _UNSAFE_FUNCTIONS)
    if denied_functions:
        return SqlAnalysis(False, None, f"unsafe SQL function: {denied_functions[0]}", False)
    ordered = any(
        top_level[index : index + 2] == ["ORDER", "BY"] for index in range(len(top_level) - 1)
    )
    return SqlAnalysis(True, candidate, None, ordered)


def _apply_resource_limits(limits: EvaluatorLimits) -> None:
    if (
        limits.maximum_cpu_seconds > 0
        and limits.maximum_cpu_hard_seconds != limits.maximum_cpu_seconds + 1
    ):
        raise ValueError("CPU hard limit must equal soft limit plus one")
    try:
        import resource

        if limits.maximum_vm_bytes > 0:
            resource.setrlimit(
                resource.RLIMIT_AS, (limits.maximum_vm_bytes, limits.maximum_vm_bytes)
            )
        if limits.maximum_cpu_seconds > 0:
            assert limits.maximum_cpu_hard_seconds is not None
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (limits.maximum_cpu_seconds, limits.maximum_cpu_hard_seconds),
            )
    except (ImportError, OSError, ValueError):
        pass


def _sqlite_error_status(message: str) -> str:
    normalized = message.casefold()
    syntax_markers = (
        "syntax error",
        "incomplete input",
        "unrecognized token",
        "unterminated",
    )
    if any(marker in normalized for marker in syntax_markers):
        return "syntax_error"
    return "execution_error"


def _execute_worker(
    database_path: str,
    analysis: SqlAnalysis,
    limits: EvaluatorLimits,
    output: Connection,
    collect_rows: bool,
) -> None:
    started = time.monotonic()
    _apply_resource_limits(limits)
    connection: sqlite3.Connection | None = None
    result = QueryResult("worker_error", error="worker did not produce a result")
    try:
        encoded_path = quote(str(Path(database_path).resolve()), safe="/")
        database_uri = f"file:{encoded_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(database_uri, uri=True, timeout=0)
        # Some CPython builds omit extension-loading support entirely. In that
        # case extensions are already unavailable; when the API exists, keep
        # the frozen M2 behavior of disabling it explicitly.
        if hasattr(connection, "enable_load_extension"):
            connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")

        def authorize(
            action: int,
            _arg1: str | None,
            arg2: str | None,
            _database: str | None,
            _source: str | None,
        ) -> int:
            if action not in _ALLOWED_AUTHORIZER_ACTIONS:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").upper() in _UNSAFE_FUNCTIONS:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        deadline = time.monotonic() + limits.timeout_seconds

        def progress() -> int:
            return 1 if time.monotonic() >= deadline else 0

        connection.set_authorizer(authorize)
        connection.set_progress_handler(progress, limits.progress_steps)
        cursor = connection.execute(analysis.sql or "")
        column_count = len(cursor.description or ())
        rows: list[tuple[Any, ...]] = []
        row_count = 0
        result_bytes = 0
        while batch := cursor.fetchmany(1024):
            batch_rows = tuple(tuple(row) for row in batch)
            row_count += len(batch_rows)
            result_bytes += sum(len(repr(row).encode("utf-8")) for row in batch_rows)
            if row_count > limits.maximum_result_rows:
                result = QueryResult(
                    "resource_limit",
                    column_count=column_count,
                    top_level_order_by=analysis.top_level_order_by,
                    error="result row limit exceeded",
                    row_count=row_count,
                    result_bytes=result_bytes,
                    limit_kind="rows",
                    worker_state="running",
                )
                break
            if result_bytes > limits.maximum_result_bytes:
                result = QueryResult(
                    "resource_limit",
                    column_count=column_count,
                    top_level_order_by=analysis.top_level_order_by,
                    error="result byte limit exceeded",
                    row_count=row_count,
                    result_bytes=result_bytes,
                    limit_kind="bytes",
                    worker_state="running",
                )
                break
            if collect_rows:
                rows.extend(batch_rows)
        else:
            result = QueryResult(
                status="ok",
                rows=tuple(rows),
                column_count=column_count,
                top_level_order_by=analysis.top_level_order_by,
                row_count=row_count,
                result_bytes=result_bytes,
                worker_state="running",
            )
    except sqlite3.Error as error:
        message = str(error)
        status = "timeout" if "interrupted" in message.lower() else _sqlite_error_status(message)
        result = QueryResult(
            status,
            error=message,
            limit_kind="wall" if status == "timeout" else None,
            worker_state="running",
        )
    except BaseException as error:  # child failures become data, not evaluator crashes
        result = QueryResult(
            "worker_error", error=f"{type(error).__name__}: {error}", worker_state="exception"
        )
    finally:
        if connection is not None:
            connection.close()
        result = replace(result, duration_seconds=time.monotonic() - started)
        try:
            output.send(result)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            output.close()


def _terminate_and_reap(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(1)
    if process.is_alive():
        process.kill()
        process.join(1)


def _signal_name(exit_code: int) -> str | None:
    if exit_code >= 0:
        return None
    try:
        return signal.Signals(-exit_code).name
    except ValueError:
        return f"UNKNOWN_SIGNAL_{-exit_code}"


def _stable_exit_code(process: multiprocessing.Process, *, join_seconds: float = 1.0) -> int:
    process.join(join_seconds)
    if process.is_alive():
        _terminate_and_reap(process)
    exit_code = process.exitcode
    if exit_code is None:
        return 255
    return exit_code


def _worker_exit_result(process: multiprocessing.Process, error: BaseException) -> QueryResult:
    exit_code = _stable_exit_code(process)
    signal_name = _signal_name(exit_code)
    prefix = f"worker IPC failed ({type(error).__name__}); exit code {exit_code}"
    if signal_name == "SIGXCPU":
        return QueryResult(
            "resource_limit",
            error=f"{prefix}; signal {signal_name}",
            exit_code=exit_code,
            signal_name=signal_name,
            limit_kind="cpu",
            worker_state="signaled",
        )
    state = (
        "exitcode_unavailable" if exit_code == 255 else "signaled" if exit_code < 0 else "exited"
    )
    return QueryResult(
        "worker_error",
        error=f"{prefix}" + (f"; signal {signal_name}" if signal_name else ""),
        exit_code=exit_code,
        signal_name=signal_name,
        worker_state=state,
    )


def execute_read_only(
    database_path: str | Path,
    sql: str,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
    *,
    collect_rows: bool = True,
) -> QueryResult:
    analysis = analyze_sql(sql)
    if not analysis.accepted:
        return QueryResult("unsafe", error=analysis.error)
    path = Path(database_path)
    if not path.is_file():
        return QueryResult("database_error", error="database file does not exist")

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_execute_worker,
        args=(str(path), analysis, limits, send, collect_rows),
        daemon=True,
    )
    started = False
    try:
        process.start()
        started = True
        send.close()
        if not receive.poll(limits.timeout_seconds + 0.5):
            _terminate_and_reap(process)
            exit_code = process.exitcode if process.exitcode is not None else 255
            return QueryResult(
                "timeout",
                error="wall-clock timeout exceeded",
                exit_code=exit_code,
                signal_name=_signal_name(exit_code),
                limit_kind="wall",
                worker_state="terminated_for_wall_timeout",
            )
        try:
            result = receive.recv()
        except (EOFError, OSError) as error:
            return _worker_exit_result(process, error)
        process.join(1)
        if process.is_alive():
            _terminate_and_reap(process)
            exit_code = process.exitcode if process.exitcode is not None else 255
            return QueryResult(
                "worker_error",
                error="worker did not exit after sending result",
                exit_code=exit_code,
                signal_name=_signal_name(exit_code),
                worker_state="reaped_after_payload",
            )
        exit_code = process.exitcode if process.exitcode is not None else 255
        return replace(
            result,
            exit_code=exit_code,
            signal_name=_signal_name(exit_code),
            worker_state="exited" if exit_code == 0 else "signaled",
        )
    finally:
        receive.close()
        send.close()
        if started:
            _terminate_and_reap(process)
            close_process = getattr(process, "close", None)
            if close_process is not None and not process.is_alive():
                close_process()


def bird_ex_behavior_compatible(prediction: QueryResult, gold: QueryResult) -> bool:
    if not prediction.successful or not gold.successful:
        return False
    return set(prediction.rows) == set(gold.rows)


def _scalar_equal(left: Any, right: Any, limits: EvaluatorLimits) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return False
        return math.isclose(
            float(left),
            float(right),
            rel_tol=limits.float_relative_tolerance,
            abs_tol=limits.float_absolute_tolerance,
        )
    return type(left) is type(right) and left == right


def _row_equal(left: Sequence[Any], right: Sequence[Any], limits: EvaluatorLimits) -> bool:
    return len(left) == len(right) and all(
        _scalar_equal(left_value, right_value, limits)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _contains_float(rows: Sequence[Sequence[Any]]) -> bool:
    return any(isinstance(value, float) for row in rows for value in row)


def _bag_comparison(
    left: Sequence[tuple[Any, ...]], right: Sequence[tuple[Any, ...]], limits: EvaluatorLimits
) -> ComparisonResult:
    if len(left) != len(right):
        return ComparisonResult("not_equal")
    if not _contains_float(left) and not _contains_float(right):
        return ComparisonResult("equal" if Counter(left) == Counter(right) else "not_equal")
    if len(left) * len(right) > limits.maximum_float_pair_checks:
        return ComparisonResult("comparator_resource_limit")
    adjacency = [
        [
            right_index
            for right_index, candidate in enumerate(right)
            if _row_equal(row, candidate, limits)
        ]
        for row in left
    ]
    if any(not candidates for candidates in adjacency):
        return ComparisonResult("not_equal")
    matched_left_by_right = [-1] * len(right)
    matched_right_by_left = [-1] * len(left)

    def augment(start_left: int) -> bool:
        seen_left = {start_left}
        seen_right: set[int] = set()
        parent_left_by_right: dict[int, int] = {}
        stack: list[tuple[int, int]] = [(start_left, 0)]
        while stack:
            left_index, candidate_offset = stack[-1]
            if candidate_offset >= len(adjacency[left_index]):
                stack.pop()
                continue
            right_index = adjacency[left_index][candidate_offset]
            stack[-1] = (left_index, candidate_offset + 1)
            if right_index in seen_right:
                continue
            seen_right.add(right_index)
            parent_left_by_right[right_index] = left_index
            previous_left = matched_left_by_right[right_index]
            if previous_left == -1:
                while right_index != -1:
                    path_left = parent_left_by_right[right_index]
                    previous_right = matched_right_by_left[path_left]
                    matched_left_by_right[right_index] = path_left
                    matched_right_by_left[path_left] = right_index
                    right_index = previous_right
                return True
            if previous_left not in seen_left:
                seen_left.add(previous_left)
                stack.append((previous_left, 0))
        return False

    matched = all(augment(left_index) for left_index in range(len(left)))
    return ComparisonResult("equal" if matched else "not_equal")


def compare_strict_results(
    prediction: QueryResult,
    gold: QueryResult,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
) -> ComparisonResult:
    if not prediction.successful or not gold.successful:
        return ComparisonResult("sql_execution_error")
    if prediction.column_count != gold.column_count:
        return ComparisonResult("not_equal")
    if gold.top_level_order_by:
        equal = len(prediction.rows) == len(gold.rows) and all(
            _row_equal(left, right, limits)
            for left, right in zip(prediction.rows, gold.rows, strict=True)
        )
        return ComparisonResult("equal" if equal else "not_equal")
    return _bag_comparison(prediction.rows, gold.rows, limits)


def strict_results_equal(
    prediction: QueryResult,
    gold: QueryResult,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
) -> bool:
    return compare_strict_results(prediction, gold, limits).equal


def evaluate_prediction(
    database_path: str | Path,
    prediction_sql: str,
    gold_sql: str,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
) -> EvaluationScore:
    gold = execute_read_only(database_path, gold_sql, limits)
    if not gold.successful:
        raise ValueError(f"gold SQL preflight failed: {gold.status}: {gold.error}")
    prediction = execute_read_only(database_path, prediction_sql, limits)
    strict_comparison = compare_strict_results(prediction, gold, limits)
    return EvaluationScore(
        bird_ex_behavior_compatible=int(bird_ex_behavior_compatible(prediction, gold)),
        strict_execution_accuracy=int(strict_comparison.equal),
        prediction_status=prediction.status,
        gold_status=gold.status,
        strict_comparison_status=strict_comparison.status,
    )
