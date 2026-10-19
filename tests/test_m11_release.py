from pathlib import Path

import pytest

from scripts.verify_m11_release import ReleaseGateError, verify_release

ROOT = Path(__file__).resolve().parents[1]


def test_current_m11_release_surface_passes():
    result = verify_release(ROOT, ROOT / "configs/m11_release.toml")

    assert result == {
        "status": "passed",
        "required_file_count": 13,
        "validation_examples": 1209,
        "serving_requests": 192,
        "public_artifacts": 19,
    }


def test_release_gate_rejects_metric_drift(tmp_path):
    config = (ROOT / "configs/m11_release.toml").read_text(encoding="utf-8")
    config = config.replace(
        "e2_strict_execution_accuracy = 0.42349048800661704",
        "e2_strict_execution_accuracy = 0.99",
    )
    config_path = tmp_path / "m11_release.toml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(ReleaseGateError, match="E2 strict accuracy"):
        verify_release(ROOT, config_path)


def test_release_gate_rejects_missing_required_file(tmp_path):
    config_path = tmp_path / "m11_release.toml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'status = "release_candidate"',
                'required_files = ["not-present"]',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseGateError, match="missing or empty"):
        verify_release(ROOT, config_path)
