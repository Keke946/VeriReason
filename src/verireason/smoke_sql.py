"""Tiny, in-memory SQL checks used only by the M1 model gate."""

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

_FENCED_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_QUERY_START = re.compile(r"\b(?:SELECT|WITH)\b", flags=re.IGNORECASE)
_DENIED_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}


@dataclass(frozen=True)
class SmokeScore:
    extracted_sql: str | None
    valid_read_only_sql: bool
    execution_correct: bool
    rows: list[list[Any]] | None
    error: str | None


def extract_sql(completion: str) -> str | None:
    fenced = _FENCED_SQL.findall(completion)
    candidate = fenced[-1].strip() if fenced else completion.strip()
    match = _QUERY_START.search(candidate)
    if not match:
        return None
    candidate = candidate[match.start() :].strip()
    if candidate.endswith(";"):
        candidate = candidate[:-1].rstrip()
    return candidate or None


def is_strict_sql_only(completion: str) -> bool:
    """Return true only when the complete response is one bare SELECT/WITH query.

    Leading/trailing whitespace and one trailing semicolon are allowed. Markdown fences,
    explanatory prose, and any text before or after the query are not allowed.
    """
    raw = completion.strip()
    if not raw or "```" in raw:
        return False
    normalized = raw[:-1].rstrip() if raw.endswith(";") else raw
    if ";" in normalized:
        return False
    extracted = extract_sql(raw)
    return extracted is not None and normalized == extracted


def _authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str, _src: str) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK


def score_completion(setup_sql: str, completion: str, expected_rows: list[list[Any]]) -> SmokeScore:
    query = extract_sql(completion)
    if not query:
        return SmokeScore(None, False, False, None, "no SELECT or WITH query found")
    if ";" in query:
        return SmokeScore(query, False, False, None, "multiple statements are not allowed")

    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(setup_sql)
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_authorizer)
        steps = 0

        def limit_steps() -> int:
            nonlocal steps
            steps += 1
            return 1 if steps > 10_000 else 0

        connection.set_progress_handler(limit_steps, 1_000)
        rows = [list(row) for row in connection.execute(query).fetchall()]
        return SmokeScore(query, True, rows == expected_rows, rows, None)
    except sqlite3.Error as error:
        return SmokeScore(query, False, False, None, str(error))
    finally:
        connection.close()
