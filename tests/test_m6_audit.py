import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from verireason.m6_audit import audit_training_manifest


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    database = tmp_path / "databases" / "toy.sqlite"
    database.parent.mkdir()
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE users(id INTEGER, age INTEGER)")
    connection.commit()
    connection.close()
    payload = {
        "schema_version": 2,
        "databases": {"toy": {"path": "databases/toy.sqlite"}},
        "examples": [
            {
                "source": "training",
                "db_id": "toy",
                "sql": "SELECT id FROM users WHERE age > 1",
                "operational_eligible": True,
            },
            {
                "source": "training",
                "db_id": "toy",
                "sql": "SELECT id FROM users WHERE age > 1",
                "operational_eligible": True,
            },
            {
                "source": "training",
                "db_id": "toy",
                "sql": "SELECT missing FROM users",
                "operational_eligible": True,
            },
            {
                "source": "training",
                "db_id": "toy",
                "sql": "SELECT random() FROM users",
                "operational_eligible": False,
            },
        ],
    }
    manifest = tmp_path / "training-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return manifest, digest


def test_training_audit_is_deduplicated_aggregate_and_fail_closed(tmp_path: Path) -> None:
    manifest, digest = _fixture(tmp_path)
    result = audit_training_manifest(
        manifest,
        expected_manifest_sha256=digest,
        expected_eligible_examples=3,
        expected_unique_sql=2,
        expected_database_count=1,
    )

    assert result["supported_unique_sql_count"] == 1
    assert result["unsupported_unique_sql_count"] == 1
    assert result["reason_counts"] == {
        "schema_resolution_error:OptimizeError": 1,
        "supported": 1,
    }
    assert result["heldout_examples_loaded"] == 0
    serialized = json.dumps(result).casefold()
    assert "select id from users" not in serialized
    assert "select missing from users" not in serialized
    assert '"db_id"' not in serialized


def test_training_audit_rejects_identity_drift_and_heldout(tmp_path: Path) -> None:
    manifest, digest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        audit_training_manifest(
            manifest,
            expected_manifest_sha256="0" * 64,
            expected_eligible_examples=3,
            expected_unique_sql=2,
            expected_database_count=1,
        )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["examples"][0]["source"] = "heldout"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    changed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="heldout"):
        audit_training_manifest(
            manifest,
            expected_manifest_sha256=changed,
            expected_eligible_examples=3,
            expected_unique_sql=2,
            expected_database_count=1,
        )


def test_training_audit_uses_frozen_exact_sql_identity(tmp_path: Path) -> None:
    manifest, _ = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["examples"][1]["sql"] = "SELECT  id  FROM users WHERE age > 1"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    result = audit_training_manifest(
        manifest,
        expected_manifest_sha256=digest,
        expected_eligible_examples=3,
        expected_unique_sql=3,
        expected_database_count=1,
    )

    assert result["eligible_unique_sql_count"] == 3
