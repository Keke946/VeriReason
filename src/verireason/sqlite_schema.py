"""Read deterministic table/column metadata from an immutable SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from urllib.parse import quote


def load_sqlite_schema(path: str | Path) -> dict[str, set[str]]:
    database = Path(path)
    uri = f"file:{quote(str(database.resolve()), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        table_names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE BINARY"
            )
        ]
        schema = {
            table.casefold(): {
                str(row[1]).casefold()
                for row in connection.execute(f"PRAGMA table_info({json.dumps(table)})")
            }
            for table in table_names
        }
    finally:
        connection.close()
    if not schema or any(not columns for columns in schema.values()):
        raise ValueError(f"database schema is empty or incomplete: {database}")
    return schema
