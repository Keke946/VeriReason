"""Build and verify the deterministic public-evidence manifest."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

FORBIDDEN_DATA_KEYS = {
    "database_path",
    "db_path",
    "example_id",
    "generated_sql",
    "gold_sql",
    "predicted_sql",
    "question",
    "sql_text",
}
SQL_PREFIX = re.compile(
    r"^\s*(?:select|with|insert|update|delete|create|drop|alter|pragma|attach)\b",
    re.IGNORECASE,
)


class ReleaseVerificationError(ValueError):
    """Raised when a public release artifact violates the M10 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_paths(config_path: Path) -> tuple[str, list[str]]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ReleaseVerificationError("unsupported M10 artifact config schema")
    python_version = config.get("python_version")
    artifacts = config.get("artifacts")
    if not isinstance(python_version, str) or not python_version:
        raise ReleaseVerificationError("python_version must be a non-empty string")
    if not isinstance(artifacts, list) or not artifacts:
        raise ReleaseVerificationError("artifacts must be a non-empty list")
    if artifacts != sorted(set(artifacts)):
        raise ReleaseVerificationError("artifact paths must be unique and sorted")
    for relative in artifacts:
        if not isinstance(relative, str):
            raise ReleaseVerificationError("artifact paths must be strings")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ReleaseVerificationError(f"unsafe public artifact path: {relative}")
        if path.parts[:1] != ("results",) or "raw" in path.parts:
            raise ReleaseVerificationError(f"artifact is outside curated results: {relative}")
    return python_version, artifacts


def _privacy_errors(value: Any, location: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if str(key).lower() in FORBIDDEN_DATA_KEYS:
                errors.append(f"forbidden data-bearing key at {child_location}")
            errors.extend(_privacy_errors(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_privacy_errors(child, f"{location}[{index}]"))
    elif isinstance(value, str) and SQL_PREFIX.search(value):
        errors.append(f"SQL-like string at {location}")
    return errors


def build_release_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    python_version, artifacts = load_artifact_paths(config_path)
    entries: list[dict[str, Any]] = []
    for relative in artifacts:
        path = root / relative
        if not path.is_file():
            raise ReleaseVerificationError(f"missing public artifact: {relative}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReleaseVerificationError(f"invalid JSON artifact: {relative}: {error}") from error
        privacy_errors = _privacy_errors(payload)
        if privacy_errors:
            raise ReleaseVerificationError(f"{relative}: {privacy_errors[0]}")
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "sql_free_privacy_check": "passed",
            }
        )
    return {
        "schema_version": 1,
        "project": "VeriReason",
        "canonical_python": python_version,
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def render_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def verify_release_manifest(root: Path, config_path: Path, manifest_path: Path) -> None:
    expected = render_manifest(build_release_manifest(root, config_path))
    if not manifest_path.is_file():
        raise ReleaseVerificationError(f"missing release manifest: {manifest_path}")
    actual = manifest_path.read_text(encoding="utf-8")
    if actual != expected:
        raise ReleaseVerificationError(
            "release manifest is stale; regenerate it with scripts/verify_m10_release.py --write"
        )
