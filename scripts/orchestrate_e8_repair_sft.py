#!/usr/bin/env python3
"""Run the approved E8 train and fresh-process four-cell audit as one bounded session."""

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
from verireason.repair_sft_smoke import load_repair_sft_smoke_config


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get("E8_CONFIG", "configs/e8_repair_sft_smoke.toml"))
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_repair_sft_smoke_config(config_path)
    approved = required("E8_APPROVED_COMMIT")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True).strip()
    )
    require_approved_commit(approved, observed, dirty)
    budget = config["budget"]
    output, free = prepare_output_directory(
        required("E8_OUTPUT_DIR"), root, int(budget["minimum_output_free_bytes"])
    )
    private_split = Path(required("E8_PRIVATE_SPLIT")).resolve()
    database_root = Path(required("E8_DATABASE_ROOT")).resolve()
    e1_adapter = Path(required("E8_E1_ADAPTER")).resolve()
    for path in (private_split, database_root, e1_adapter):
        if not path.exists():
            raise FileNotFoundError(f"required E8 input is unavailable: {path}")
    atomic_write_json(
        output / "preflight.json",
        {
            "schema_version": 1,
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "approved_commit": approved,
            "free_bytes": free,
            "maximum_session_wall_seconds": int(budget["maximum_minutes"]) * 60,
            "maximum_session_cost_cny": budget["maximum_cost_cny"],
            "validation_payloads_read": 0,
            "heldout_payloads_read": 0,
        },
    )
    runner = SessionRunner(
        output,
        int(budget["maximum_minutes"]) * 60,
        int(budget["terminate_grace_seconds"]),
        started_monotonic=time.monotonic(),
    )
    python = sys.executable
    env = {
        **os.environ,
        "PATH": f"{Path(python).parent}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(root / "src"),
    }
    training_output = output / "training"
    evaluation_output = output / "evaluation"
    stages = [
        ("verify", ["make", "e8-repair-sft-verify"]),
        ("gpu-env", ["make", "env-check-gpu"]),
        (
            "train",
            [
                python,
                "scripts/run_e8_repair_sft_train.py",
                "--config",
                str(config_path),
                "--private-split",
                str(private_split),
                "--e1-adapter",
                str(e1_adapter),
                "--output-dir",
                str(training_output),
            ],
        ),
        (
            "evaluate",
            [
                python,
                "scripts/run_e8_repair_sft_eval.py",
                "--config",
                str(config_path),
                "--private-split",
                str(private_split),
                "--database-root",
                str(database_root),
                "--e1-adapter",
                str(e1_adapter),
                "--repair-adapter",
                str(training_output / "adapter"),
                "--training-result",
                str(training_output / "training-result.json"),
                "--output-dir",
                str(evaluation_output),
            ],
        ),
    ]
    for name, command in stages:
        runner.run_stage(name, command, cwd=root, env=env)
    summary = json.loads((evaluation_output / "e8-repair-sft-summary.json").read_text())
    atomic_write_json(
        output / "session-completion.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": config["experiment_id"],
            "research_gate_status": summary["status"],
            "elapsed_seconds": runner.elapsed_seconds,
            "validation_payloads_read": 0,
            "heldout_payloads_read": 0,
        },
    )
    print(json.dumps({"status": "completed", "research_gate_status": summary["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
