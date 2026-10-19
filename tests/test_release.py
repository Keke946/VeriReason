import json
from pathlib import Path

import pytest

from verireason.release import (
    ReleaseVerificationError,
    build_release_manifest,
    render_manifest,
    verify_release_manifest,
)


def _fixture(tmp_path: Path, payload: object) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "results" / "m1" / "summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        'schema_version = 1\npython_version = "3.11.15"\n'
        'artifacts = ["results/m1/summary.json"]\n',
        encoding="utf-8",
    )
    return artifact, config, tmp_path / "manifest.json"


def test_release_manifest_round_trip_and_drift_detection(tmp_path: Path) -> None:
    artifact, config, manifest = _fixture(tmp_path, {"accuracy": 0.5})
    built = build_release_manifest(tmp_path, config)
    manifest.write_text(render_manifest(built), encoding="utf-8")

    verify_release_manifest(tmp_path, config, manifest)
    artifact.write_text('{"accuracy": 0.6}', encoding="utf-8")

    with pytest.raises(ReleaseVerificationError, match="stale"):
        verify_release_manifest(tmp_path, config, manifest)


@pytest.mark.parametrize(
    "payload",
    [
        {"gold_sql": "redacted"},
        {"note": "SELECT secret FROM private_table"},
    ],
)
def test_release_manifest_rejects_private_sql_payloads(tmp_path: Path, payload: object) -> None:
    _, config, _ = _fixture(tmp_path, payload)

    with pytest.raises(ReleaseVerificationError):
        build_release_manifest(tmp_path, config)


def test_dockerfile_uses_canonical_cpu_environment_and_non_root_user() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11.15-slim-bookworm\n")
    assert "requirements/m10-cpu-lock.txt" in dockerfile
    assert "USER verireason" in dockerfile
    assert dockerfile.index("USER verireason") < dockerfile.index("RUN make m10-verify")
    assert ".[gpu]" not in dockerfile
