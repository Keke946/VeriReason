"""Training-only real-schema coverage audit for the M6 dense extractor."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sha256_file
from verireason.sql_features import extract_sql_features
from verireason.sqlite_schema import load_sqlite_schema


def _resolve_database(manifest_dir: Path, logical_path: str) -> Path:
    candidate = (manifest_dir / logical_path).resolve()
    try:
        candidate.relative_to(manifest_dir.resolve())
    except ValueError as error:
        raise ValueError("database path escapes the finalized M2 directory") from error
    if not candidate.is_file():
        raise ValueError(f"database file is missing: {logical_path}")
    return candidate


def audit_training_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_eligible_examples: int = 6573,
    expected_unique_sql: int = 6496,
    expected_database_count: int = 69,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    if sha256_file(path) != expected_manifest_sha256:
        raise ValueError("training manifest SHA-256 differs from frozen M2 identity")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("training manifest must use frozen schema version 2")
    rows = payload.get("examples")
    databases = payload.get("databases")
    if not isinstance(rows, list) or not isinstance(databases, dict):
        raise ValueError("training manifest is missing examples or databases")
    if any(row.get("source") == "heldout" for row in rows):
        raise ValueError("heldout content appeared in the training manifest")
    eligible = [row for row in rows if row.get("operational_eligible") is True]
    if len(eligible) != expected_eligible_examples:
        raise ValueError("eligible training example count changed")
    if len(databases) != expected_database_count:
        raise ValueError("training database count changed")

    unique: dict[tuple[str, str], str] = {}
    for row in eligible:
        db_id = str(row.get("db_id", ""))
        sql = str(row.get("sql", ""))
        if not db_id or not sql or db_id not in databases:
            raise ValueError("eligible row has incomplete database or SQL identity")
        # M2 freezes unique SQL identity as the exact (db_id, SQL) pair.
        unique.setdefault((db_id, sql), sql)
    if len(unique) != expected_unique_sql:
        raise ValueError("eligible unique SQL count changed")

    schemas: dict[str, dict[str, set[str]]] = {}
    for db_id, identity in databases.items():
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise ValueError("database identity is malformed")
        schemas[str(db_id)] = load_sqlite_schema(
            _resolve_database(path.parent, str(identity["path"]))
        )

    reasons: Counter[str] = Counter()
    dimension_nonempty: Counter[str] = Counter()
    digest = hashlib.sha256()
    for (db_id, sql_identity), sql in sorted(unique.items()):
        result = extract_sql_features(sql, schemas[db_id])
        reason = "supported" if result.supported else str(result.reason)
        reasons[reason] += 1
        if result.supported:
            for dimension, values in result.features.items():
                if values:
                    dimension_nonempty[dimension] += 1
        digest.update(f"{db_id}\0{sql_identity}\0{reason}\n".encode())

    supported = reasons["supported"]
    return {
        "schema_version": 1,
        "status": "completed",
        "scope": "training_only_operationally_eligible_unique_sql",
        "training_manifest_sha256": expected_manifest_sha256,
        "eligible_example_count": len(eligible),
        "eligible_unique_sql_count": len(unique),
        "database_count": len(databases),
        "supported_unique_sql_count": supported,
        "unsupported_unique_sql_count": len(unique) - supported,
        "support_rate": supported / len(unique),
        "reason_counts": dict(sorted(reasons.items())),
        "nonempty_dimension_counts": dict(sorted(dimension_nonempty.items())),
        "private_row_outcome_digest_sha256": digest.hexdigest(),
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
