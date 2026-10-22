import json
import sqlite3
from pathlib import Path

import pytest

import verireason.repair_sft_data as repair_data
from verireason.repair_sft_data import (
    build_repair_sft_examples,
    load_repair_sft_config,
    normalize_m7_formal_bundle,
)


def _database(path: Path, prefix: str = "x") -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            f"""
            CREATE TABLE {prefix}_people (id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER);
            CREATE TABLE {prefix}_teams (team_id INTEGER PRIMARY KEY, code TEXT, score REAL);
            CREATE TABLE {prefix}_events (event_id INTEGER PRIMARY KEY, person_id INTEGER);
            INSERT INTO {prefix}_people VALUES (1, 'SECRET_CELL', 10), (2, 'Lin', 20);
            INSERT INTO {prefix}_teams VALUES (10, 'A', 2.5), (20, 'B', 3.5);
            INSERT INTO {prefix}_events VALUES (100, 1), (101, 2);
            """
        )
    finally:
        connection.close()


def _row(sample_id: str, database: Path, *, prefix: str = "x", split: str = "train") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "db_id": f"db-{prefix}",
        "database_path": str(database),
        "target_sql": f"SELECT p.name FROM {prefix}_people AS p ORDER BY p.id",
        "messages": [
            {"role": "system", "content": "SQL only"},
            {"role": "user", "content": f"List names for {sample_id}"},
        ],
    }


def _passing_rows(tmp_path: Path) -> list[dict]:
    first, second = tmp_path / "first.sqlite", tmp_path / "second.sqlite"
    _database(first, "x")
    _database(second, "y")
    return [
        _row("private-sample-alpha", first, prefix="x"),
        _row("private-sample-beta", first, prefix="x"),
        _row("private-sample-gamma", second, prefix="y"),
        _row("private-sample-delta", second, prefix="y"),
    ]


def test_authored_gate_builds_verified_repairs_and_keep_examples(tmp_path: Path) -> None:
    private, public = build_repair_sft_examples(
        _passing_rows(tmp_path), tmp_path, load_repair_sft_config("configs/e7_repair_sft_data.toml")
    )

    repairs = [row for row in private if row["kind"] == "repair"]
    keeps = [row for row in private if row["kind"] == "keep"]
    assert public["status"] == "pass" and public["real_training_yield_known"] is False
    assert set(public["mutation_counts"]) == {"wrong_alias", "wrong_column", "wrong_table"}
    assert repairs and keeps and all(row["target_status"] == "ok" for row in private)
    assert all(row["initial_status"] in {"execution_error", "syntax_error"} for row in repairs)
    assert all(row["target"] == "KEEP" for row in keeps)
    assert "SECRET_CELL" not in json.dumps(private)


def test_public_receipt_is_sql_free_and_identity_free(tmp_path: Path) -> None:
    rows = _passing_rows(tmp_path)
    _private, public = build_repair_sft_examples(
        rows, tmp_path, load_repair_sft_config("configs/e7_repair_sft_data.toml")
    )
    serialized = json.dumps(public)
    assert "SELECT" not in serialized and "List names" not in serialized
    assert all(row["sample_id"] not in serialized for row in rows)
    assert all(row["db_id"] not in serialized for row in rows)


def test_non_train_duplicate_and_escaping_database_are_rejected(tmp_path: Path) -> None:
    database = tmp_path / "db.sqlite"
    _database(database)
    config = load_repair_sft_config("configs/e7_repair_sft_data.toml")
    with pytest.raises(ValueError, match="training rows only"):
        build_repair_sft_examples([_row("a", database, split="validation")], tmp_path, config)
    with pytest.raises(ValueError, match="duplicate"):
        build_repair_sft_examples([_row("a", database), _row("a", database)], tmp_path, config)
    outside = tmp_path.parent / "outside-e7.sqlite"
    _database(outside)
    try:
        with pytest.raises(ValueError, match="below database_root"):
            build_repair_sft_examples([_row("a", outside)], tmp_path, config)
    finally:
        outside.unlink()


def test_executable_or_equivalent_mutation_is_not_accepted(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "db.sqlite"
    _database(database)
    row = _row("a", database)
    monkeypatch.setattr(
        repair_data,
        "_mutation_candidates",
        lambda sql, _schema: [("wrong_column", sql)],
    )
    private, public = build_repair_sft_examples(
        [row], tmp_path, load_repair_sft_config("configs/e7_repair_sft_data.toml")
    )
    assert private == [] and public["status"] == "revise"


def test_bad_message_shape_and_unsafe_gold_do_not_create_supervision(tmp_path: Path) -> None:
    database = tmp_path / "db.sqlite"
    _database(database)
    config = load_repair_sft_config("configs/e7_repair_sft_data.toml")
    bad = _row("a", database)
    bad["messages"] = [*bad["messages"], {"role": "assistant", "content": "gold"}]
    with pytest.raises(ValueError, match="system/user"):
        build_repair_sft_examples([bad], tmp_path, config)
    unsafe = _row("b", database)
    unsafe["target_sql"] = "DROP TABLE x_people"
    private, public = build_repair_sft_examples([unsafe], tmp_path, config)
    assert private == [] and public["gate_checks"]["all_verified"] is False


def test_m7_formal_bundle_adapter_requires_frozen_training_identity(tmp_path: Path) -> None:
    database = tmp_path / "db.sqlite"
    _database(database)
    config = load_repair_sft_config("configs/e7_repair_sft_data.toml")
    examples = [
        {
            "reward_group_id": index,
            "database_path": "databases/training/db.sqlite",
            "gold_sql": "SELECT 1",
            "messages": [{"role": "user", "content": f"q{index}"}],
        }
        for index in range(256)
    ]
    payload = {
        "schema_version": 1,
        "selection_policy": "m7_e2_formal_hash_v1",
        "bundle_sha256": config["source"]["bundle_sha256"],
        "sample_ids": list(range(256)),
        "examples": examples,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }

    normalized = normalize_m7_formal_bundle(payload, config)

    assert len(normalized) == 256 and all(row["split"] == "train" for row in normalized)
    with pytest.raises(ValueError, match="training-only contract"):
        normalize_m7_formal_bundle({**payload, "heldout_examples_loaded": 1}, config)
    drifted = {**payload, "sample_ids": [999, *list(range(1, 256))]}
    with pytest.raises(ValueError, match="alignment"):
        normalize_m7_formal_bundle(drifted, config)
