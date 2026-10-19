#!/usr/bin/env python3
"""Run the approved M3 session once, continuously, under automatic hard stops."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from verireason.m3_contract import atomic_write_json, load_m3_config
from verireason.m3_orchestrator import (
    M3GateError,
    SessionRunner,
    prepare_output_directory,
    require_approved_commit,
)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise M3GateError(f"required environment variable is missing: {name}")
    return value


def main() -> int:
    session_started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    args = parser.parse_args()

    worktree = Path(__file__).resolve().parents[1]
    approved = _required_environment("M3_APPROVED_COMMIT")
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=worktree, text=True
        ).strip()
    )
    require_approved_commit(approved, observed, dirty)

    dataset_root = _required_environment("M3_DATASET_ROOT")
    training_manifest = _required_environment("M3_TRAINING_MANIFEST")
    manifest_index = _required_environment("M3_MANIFEST_INDEX")
    requested_output = _required_environment("M3_OUTPUT_DIR")
    config = load_m3_config(worktree / args.config, worktree / args.m2_config)
    output, free_bytes = prepare_output_directory(
        requested_output,
        worktree,
        config["evidence"]["minimum_output_free_bytes"],
    )
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "approved_commit": approved,
        "observed_commit": observed,
        "git_dirty": dirty,
        "output_dir": str(output),
        "output_filesystem_free_bytes": free_bytes,
        "minimum_output_free_bytes": config["evidence"]["minimum_output_free_bytes"],
        "maximum_session_wall_seconds": config["evidence"]["maximum_session_wall_seconds"],
        "fixed_train_count": config["data"]["train_count"],
        "fixed_validation_count": config["data"]["validation_count"],
        "started_utc": started_utc,
    }
    atomic_write_json(output / "preflight.json", preflight)

    runner = SessionRunner(
        output,
        config["evidence"]["maximum_session_wall_seconds"],
        config["evidence"]["terminate_grace_seconds"],
        started_monotonic=session_started_monotonic,
    )
    environment = {**os.environ, "PYTHONPATH": str(worktree / "src")}
    python = sys.executable

    stages: list[tuple[str, list[str], Path | None]] = [
        ("m3-verify", ["make", "m3-verify"], None),
        ("gpu-environment", ["make", "env-check-gpu"], None),
        (
            "environment-evidence",
            [
                python,
                "-m",
                "verireason.envcheck",
                "--profile",
                "gpu",
                "--output",
                str(output / "environment.json"),
            ],
            None,
        ),
        ("pip-freeze", [python, "-m", "pip", "freeze"], output / "gpu-pip-freeze.txt"),
        ("nvidia-smi", ["nvidia-smi"], output / "nvidia-smi.txt"),
        (
            "token-lengths",
            [
                python,
                "scripts/inspect_m3_candidates.py",
                "--dataset-root",
                dataset_root,
                "--manifest-index",
                manifest_index,
                "--training-manifest",
                training_manifest,
                "--output",
                str(output / "token-length-report.json"),
            ],
            None,
        ),
        (
            "freeze-slice",
            [
                python,
                "scripts/prepare_m3_slice.py",
                "--dataset-root",
                dataset_root,
                "--manifest-index",
                manifest_index,
                "--training-manifest",
                training_manifest,
                "--length-report",
                str(output / "token-length-report.json"),
                "--output",
                str(output / "slice.json"),
            ],
            None,
        ),
        (
            "base-and-train",
            [
                python,
                "scripts/run_m3_base_train.py",
                "--slice",
                str(output / "slice.json"),
                "--adapter-dir",
                str(output / "adapter"),
                "--output",
                str(output / "base-train.json"),
            ],
            None,
        ),
        (
            "fresh-reload-eval",
            [
                python,
                "scripts/run_m3_reload_eval.py",
                "--slice",
                str(output / "slice.json"),
                "--adapter-dir",
                str(output / "adapter"),
                "--prior-result",
                str(output / "base-train.json"),
                "--output",
                str(output / "m3-result.json"),
            ],
            None,
        ),
        (
            "validate-result",
            [python, "scripts/validate_m3_result.py", str(output / "m3-result.json")],
            None,
        ),
    ]
    for stage, command, stdout_file in stages:
        runner.run_stage(stage, command, cwd=worktree, env=environment, stdout_file=stdout_file)

    completion = {
        "schema_version": 1,
        "status": "completed",
        "approved_commit": approved,
        "elapsed_seconds": runner.elapsed_seconds,
        "maximum_session_wall_seconds": runner.maximum_wall_seconds,
        "result": str(output / "m3-result.json"),
    }
    atomic_write_json(output / "session-completion.json", completion)
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
