#!/usr/bin/env python3
"""Run an approved E2/E3/E4 post-training contract as one bounded cloud session."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_orchestrator import (
    SessionRunner,
    prepare_output_directory,
    require_approved_commit,
)
from verireason.posttraining_contract import load_posttraining_config
from verireason.posttraining_result import close_posttraining_result


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get("M7_CONFIG", "configs/m7_e2_smoke.toml"))
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_posttraining_config(config_path)
    approved = required("M7_APPROVED_COMMIT")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True).strip()
    )
    require_approved_commit(approved, observed, dirty)
    output, free = prepare_output_directory(
        required("M7_OUTPUT_DIR"), root, config["evidence"]["minimum_output_free_bytes"]
    )
    dataset_root = Path(required("M7_DATASET_ROOT")).resolve()
    manifest = Path(required("M7_TRAINING_MANIFEST")).resolve()
    m4_output = Path(required("M7_M4_OUTPUT_DIR")).resolve()
    prepared_train, e1_adapter = m4_output / "prepared-train.json", m4_output / "adapter"
    for path in (dataset_root, manifest, prepared_train, e1_adapter):
        if not path.exists():
            raise FileNotFoundError(f"required retained input is unavailable: {path}")
    atomic_write_json(
        output / "preflight.json",
        {
            "schema_version": 1,
            "status": "passed",
            "approved_commit": approved,
            "free_bytes": free,
            "maximum_session_wall_seconds": config["evidence"]["maximum_session_wall_seconds"],
            "maximum_session_cost_cny": config["evidence"]["maximum_session_cost_cny"],
            "heldout_examples_loaded": 0,
            "heldout_examples_evaluated": 0,
        },
    )
    runner = SessionRunner(
        output,
        config["evidence"]["maximum_session_wall_seconds"],
        30,
        started_monotonic=time.monotonic(),
    )
    python = sys.executable
    interpreter_bin = str(Path(python).parent)
    env = {
        **os.environ,
        "PATH": f"{interpreter_bin}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(root / "src"),
    }
    stages = [
        (
            "verify",
            ["make", "m7-preflight-verify" if config["arm"] == "E2" else "m8-preflight-verify"],
        ),
        ("gpu-env", ["make", "env-check-gpu"]),
        (
            "prepare",
            [
                python,
                "scripts/prepare_m7_smoke.py",
                "--config",
                str(config_path),
                "--training-manifest",
                str(manifest),
                "--prepared-train",
                str(prepared_train),
                "--database-root",
                str(dataset_root),
                "--output",
                str(output / "prepared-bundle.json"),
            ],
        ),
        (
            "train",
            [
                python,
                "scripts/run_m7_smoke_train.py",
                "--config",
                str(config_path),
                "--prepared-smoke",
                str(output / "prepared-bundle.json"),
                "--database-root",
                str(dataset_root),
                "--e1-adapter",
                str(e1_adapter),
                "--output-dir",
                str(output),
            ],
        ),
        (
            "reload",
            [
                python,
                "scripts/reload_m7_smoke_adapter.py",
                "--adapter",
                str(output / "adapter"),
                "--output",
                str(output / "reload-result.json"),
            ],
        ),
    ]
    for name, command in stages:
        runner.run_stage(name, command, cwd=root, env=env)
    summary = close_posttraining_result(
        config,
        json.loads((output / "training-result.json").read_text()),
        json.loads((output / "reload-result.json").read_text()),
        session_wall_seconds=runner.elapsed_seconds,
    )
    atomic_write_json(output / "posttraining-summary.json", summary)
    atomic_write_json(
        output / "session-completion.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": config["experiment_id"],
            "arm": config["arm"],
            "elapsed_seconds": runner.elapsed_seconds,
            "posttraining_summary_status": summary["status"],
            "heldout_examples_loaded": 0,
            "heldout_examples_evaluated": 0,
        },
    )
    print(json.dumps({"status": "completed", "elapsed_seconds": runner.elapsed_seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
