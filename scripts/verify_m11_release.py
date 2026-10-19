#!/usr/bin/env python3
"""Verify the public M11 release surface against measured evidence."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class ReleaseGateError(ValueError):
    """Raised when the public release surface drifts from accepted evidence."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if isinstance(expected, float):
        if not isinstance(observed, int | float) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ReleaseGateError(f"{label}: expected {expected!r}, observed {observed!r}")
    elif observed != expected:
        raise ReleaseGateError(f"{label}: expected {expected!r}, observed {observed!r}")


def verify_release(root: Path, config_path: Path) -> dict[str, Any]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("status") != "release_candidate":
        raise ReleaseGateError("M11 config identity is invalid")

    for relative in config["required_files"]:
        path = root / relative
        if not path.is_file() or not path.read_bytes():
            raise ReleaseGateError(f"required release file is missing or empty: {relative}")

    expected = config["expected"]
    evaluation = _read_json(root / "results/m8/m8-common-eval-summary.json")
    serving = _read_json(root / "results/m9/m9-serving-summary.json")
    manifest = _read_json(root / "results/m10/release-manifest.json")

    _require_equal("validation count", evaluation["example_count"], expected["validation_examples"])
    _require_equal(
        "validation selection",
        evaluation["validation_selection_sha256"],
        expected["validation_selection_sha256"],
    )
    _require_equal(
        "E1 strict accuracy",
        evaluation["E1"]["strict_execution_accuracy"],
        expected["e1_strict_execution_accuracy"],
    )
    _require_equal(
        "E2 strict accuracy",
        evaluation["E2"]["strict_execution_accuracy"],
        expected["e2_strict_execution_accuracy"],
    )
    _require_equal(
        "E2-E1 strict delta",
        evaluation["E2_minus_E1"]["strict_execution_accuracy"],
        expected["e2_minus_e1_strict_execution_accuracy"],
    )
    _require_equal(
        "E2-E1 McNemar p",
        evaluation["paired_analysis"]["strict_execution_accuracy"]["exact_two_sided_mcnemar_p"],
        expected["e2_vs_e1_exact_mcnemar_p"],
    )
    request_count = sum(item["request_count"] for item in serving["scenarios"])
    _require_equal("serving requests", request_count, expected["serving_requests"])
    _require_equal(
        "public evidence count", manifest["artifact_count"], expected["public_artifacts"]
    )

    for label, payload in (("evaluation", evaluation), ("serving", serving)):
        _require_equal(
            f"{label} heldout loaded",
            payload["heldout_examples_loaded"],
            expected["heldout_examples_loaded"],
        )
        _require_equal(
            f"{label} heldout evaluated",
            payload["heldout_examples_evaluated"],
            expected["heldout_examples_evaluated"],
        )

    readme = (root / "README.md").read_text(encoding="utf-8")
    for required_text in (
        config["repository_url"],
        config["base_model_revision"],
        config["selected_adapter_manifest_sha256"],
        config["selected_adapter_archive_sha256"],
        "verireason-demo",
    ):
        if required_text not in readme:
            raise ReleaseGateError(f"README is missing release identity: {required_text}")

    return {
        "status": "passed",
        "required_file_count": len(config["required_files"]),
        "validation_examples": evaluation["example_count"],
        "serving_requests": request_count,
        "public_artifacts": manifest["artifact_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m11_release.toml")
    args = parser.parse_args()
    result = verify_release(ROOT, args.config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
