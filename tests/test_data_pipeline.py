import sqlite3
from pathlib import Path

import pytest

from verireason.data_pipeline import (
    Example,
    audit_leakage,
    build_split_precheck,
    inspect_database,
    normalize_sql,
    split_database_ids,
    sql_complexity,
    validate_observed_identity,
)


def _examples() -> list[Example]:
    sql_by_db = {
        "a": "SELECT id FROM t ORDER BY id",
        "b": "SELECT COUNT(*) FROM t",
        "c": "SELECT t.id FROM t JOIN u ON t.id = u.id",
        "d": "SELECT kind, COUNT(*) FROM t GROUP BY kind",
        "e": "SELECT id FROM t WHERE id IN (SELECT id FROM u)",
        "f": "SELECT id FROM t UNION SELECT id FROM u",
    }
    return [
        Example(str(index), db_id, f"Question {db_id}", sql)
        for index, (db_id, sql) in enumerate(sql_by_db.items())
    ]


def test_database_split_is_deterministic_and_disjoint() -> None:
    first = split_database_ids(["c", "a", "b"], "seed", 1)
    second = split_database_ids(["b", "c", "a", "a"], "seed", 1)

    assert first == second
    assert list(first.values()).count("validation") == 1


def test_split_precheck_reports_counts_features_and_gate() -> None:
    examples = _examples()
    assignment, report = build_split_precheck(
        examples,
        seed="seed",
        validation_database_count=2,
        minimum_validation_ratio=0.10,
        maximum_validation_ratio=0.50,
        critical_features=("aggregate",),
    )

    assert report.database_counts == {"train": 4, "validation": 2}
    assert sum(report.example_counts.values()) == len(examples)
    assert set(assignment) == {example.db_id for example in examples}
    assert "simple" in report.complexity_distribution["train"]
    assert report.feature_distribution["validation"]["aggregate"] in {0, 1}


def test_complexity_proxy_and_sql_normalization_are_deterministic() -> None:
    parsed = sql_complexity("SELECT a, COUNT(*) FROM t JOIN u ON t.id=u.id GROUP BY a ORDER BY a")

    assert parsed.label == "complex"
    assert {"join", "aggregate", "group_by", "order_by"} <= set(parsed.features)
    assert normalize_sql("SELECT  1; -- comment\n") == "select 1"


def test_review_duplicates_do_not_hard_fail() -> None:
    rows = [
        ("train", Example("1", "a", "How many?", "SELECT COUNT(*) FROM t")),
        ("train", Example("2", "a", "How many?", "SELECT COUNT(*) FROM t")),
    ]
    report = audit_leakage(rows)

    assert report.passed
    assert report.review_findings["normalized_question"].within_split_pairs == 1
    assert "database_sha256_cross_split" in report.pending_checks


def test_cross_split_database_and_joint_identity_hard_fail() -> None:
    common = dict(question="Same question", sql="SELECT id FROM t", database_sha256="a" * 64)
    report = audit_leakage(
        [
            ("train", Example("1", "db", **common)),
            ("validation", Example("2", "renamed", **common)),
        ]
    )

    assert not report.passed
    assert report.hard_failures["database_sha256"].cross_split_pairs == 1
    assert report.hard_failures["joint_sample_fingerprint"].cross_split_pairs == 1
    assert report.hard_failures["renamed_identical_database"].cross_split_pairs == 1


def test_cross_split_similar_schema_is_review_only() -> None:
    shared = tuple(f"token_{index}" for index in range(10))
    report = audit_leakage(
        [
            (
                "train",
                Example(
                    "1",
                    "a",
                    "first",
                    "SELECT 1",
                    database_sha256="a" * 64,
                    schema_fingerprint="1" * 64,
                    schema_tokens=shared,
                ),
            ),
            (
                "validation",
                Example(
                    "2",
                    "b",
                    "second",
                    "SELECT 2",
                    database_sha256="b" * 64,
                    schema_fingerprint="2" * 64,
                    schema_tokens=shared + ("extra",),
                ),
            ),
        ]
    )

    assert report.passed
    assert report.review_findings["schema_jaccard_gte_0_90"].cross_split_pairs == 1


def test_heldout_gold_cannot_enter_training_interface() -> None:
    report = audit_leakage([("train", Example("mini-1", "db", "q", "SELECT 1", source="heldout"))])

    assert not report.passed
    assert report.hard_failures["heldout_gold_interface_exposure"].within_split_pairs == 1


def test_artifact_identity_warns_on_headers_but_fails_on_sha() -> None:
    digest = "a" * 64
    warnings = validate_observed_identity(
        expected_sha256=digest,
        observed_sha256=digest,
        previous_headers={"etag": "old", "content_length": 10, "last_modified": None},
        observed_headers={"etag": "new", "content_length": 10, "last_modified": None},
    )

    assert warnings == ("provenance header changed with stable SHA-256: etag: 'old' -> 'new'",)
    with pytest.raises(ValueError, match="SHA-256 changed"):
        validate_observed_identity(expected_sha256=digest, observed_sha256="b" * 64)


def test_database_inspection_records_integrity_hash_and_schema(tmp_path: Path) -> None:
    db_dir = tmp_path / "sample"
    db_dir.mkdir()
    path = db_dir / "sample.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE values_table(id INTEGER PRIMARY KEY, value TEXT)")
    connection.commit()
    connection.close()

    identity = inspect_database(tmp_path, "sample")

    assert identity.integrity_check == "ok"
    assert identity.path == "sample/sample.sqlite"
    assert identity.size_bytes == path.stat().st_size
    assert len(identity.sha256) == len(identity.schema_fingerprint) == 64
    assert {"values_table", "value", "text"} <= set(identity.schema_tokens)
