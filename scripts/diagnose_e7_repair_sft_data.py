#!/usr/bin/env python3
"""Run the authored CPU-only E7 repair-data feasibility gate."""

from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.repair_sft_data import build_repair_sft_examples, load_repair_sft_config


def _make_database(path: Path, *, prefix: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            f"""
            CREATE TABLE {prefix}_people (id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER);
            CREATE TABLE {prefix}_teams (team_id INTEGER PRIMARY KEY, code TEXT, score REAL);
            CREATE TABLE {prefix}_events (event_id INTEGER PRIMARY KEY, person_id INTEGER);
            INSERT INTO {prefix}_people VALUES (1, 'Ada', 10), (2, 'Lin', 20);
            INSERT INTO {prefix}_teams VALUES (10, 'A', 2.5), (20, 'B', 3.5);
            INSERT INTO {prefix}_events VALUES (100, 1), (101, 2);
            """
        )
    finally:
        connection.close()


def _row(sample_id: str, db_id: str, database: Path, table: str, column: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": "train",
        "db_id": db_id,
        "database_path": str(database),
        "target_sql": f"SELECT p.{column} FROM {table} AS p ORDER BY p.id",
        "messages": [
            {"role": "system", "content": "Return SQLite SQL only."},
            {"role": "user", "content": f"List the requested {db_id} field for {sample_id}."},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e7_repair_sft_data.toml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_repair_sft_config(args.config)
    with tempfile.TemporaryDirectory(prefix="verireason-e7-") as temporary:
        root = Path(temporary)
        first, second = root / "first.sqlite", root / "second.sqlite"
        _make_database(first, prefix="a")
        _make_database(second, prefix="b")
        rows = [
            _row("fixture-1", "fixture-a", first, "a_people", "name"),
            _row("fixture-2", "fixture-a", first, "a_people", "name"),
            _row("fixture-3", "fixture-b", second, "b_people", "name"),
            _row("fixture-4", "fixture-b", second, "b_people", "name"),
        ]
        _private, receipt = build_repair_sft_examples(
            rows, root, config, evidence_scope="authored_cpu_fixture"
        )
    atomic_write_json(args.output, receipt)
    if receipt["status"] != "pass":
        raise SystemExit("E7 authored CPU gate did not pass")


if __name__ == "__main__":
    main()
