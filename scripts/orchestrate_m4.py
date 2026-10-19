#!/usr/bin/env python3
"""Run or resume one M4 session under a cumulative fail-closed wall deadline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_orchestrator import SessionRunner, require_approved_commit
from verireason.m4_contract import load_m4_config


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = load_m4_config(root / "configs/m4_formal_sft.toml", root / "configs/m2_data.toml")
    approved = required("M4_APPROVED_COMMIT")
    observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=root, text=True).strip()
    )
    require_approved_commit(approved, observed, dirty)
    output = Path(required("M4_OUTPUT_DIR")).resolve()
    if output == root or output.is_relative_to(root):
        raise RuntimeError("M4 output must be outside Git")
    if (output / "session-completion.json").exists():
        raise RuntimeError("M4 session is already complete")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    free = shutil.disk_usage(output).free
    if free < config["evidence"]["minimum_output_free_bytes"]:
        raise RuntimeError("M4 persistent output filesystem lacks the frozen 30 GiB reserve")
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "approved_commit": approved,
        "primary_protocol": "full5364x1_update671",
        "output": str(output),
        "free_bytes": free,
        "maximum_session_wall_seconds": config["evidence"]["maximum_session_wall_seconds"],
        "maximum_session_cost_cny": config["evidence"]["maximum_session_cost_cny"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    existing_preflight = output / "preflight.json"
    if existing_preflight.exists():
        prior = json.loads(existing_preflight.read_text())
        for key in (
            "approved_commit",
            "output",
            "maximum_session_wall_seconds",
            "maximum_session_cost_cny",
            "primary_protocol",
        ):
            if prior.get(key) != preflight[key]:
                raise RuntimeError(f"M4 resume preflight drift: {key}")
    atomic_write_json(output / "preflight.json", preflight)
    runner = SessionRunner(
        output,
        config["evidence"]["maximum_session_wall_seconds"],
        config["evidence"]["terminate_grace_seconds"],
        started_monotonic=time.monotonic(),
    )
    python = sys.executable
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    dataset = required("M4_DATASET_ROOT")
    index = required("M4_MANIFEST_INDEX")
    manifest = required("M4_TRAINING_MANIFEST")
    stages = [
        ("verify", ["make", "m4-verify"], output / ".stage-verify"),
        ("gpu-env", ["make", "env-check-gpu"], output / ".stage-gpu-env"),
        (
            "trainer-contract",
            [
                python,
                "scripts/verify_m4_trainer_contract.py",
                "--output",
                str(output / "trainer-contract.json"),
            ],
            output / "trainer-contract.json",
        ),
        (
            "trainer-regression",
            [python, "-m", "pytest", "-rs", "tests/test_m4_trainer_regression.py"],
            output / ".stage-trainer-regression",
        ),
        (
            "token-report",
            [
                python,
                "scripts/inspect_m3_candidates.py",
                "--config",
                "configs/m3_vertical_slice.toml",
                "--dataset-root",
                dataset,
                "--manifest-index",
                index,
                "--training-manifest",
                manifest,
                "--output",
                str(output / "token-length-report.json"),
            ],
            output / "token-length-report.json",
        ),
        (
            "selection",
            [
                python,
                "scripts/prepare_m4_selection.py",
                "--dataset-root",
                dataset,
                "--manifest-index",
                index,
                "--training-manifest",
                manifest,
                "--token-report",
                str(output / "token-length-report.json"),
                "--output",
                str(output / "selection.json"),
                "--prepared-dir",
                str(output),
            ],
            output / "prepared-full-validation.json",
        ),
        (
            "train",
            [
                python,
                "scripts/run_m4_train.py",
                "--prepared-train",
                str(output / "prepared-train.json"),
                "--output-dir",
                str(output),
            ],
            output / "adapter",
        ),
        (
            "teacher-loss-512",
            [
                python,
                "scripts/evaluate_m4_teacher_loss.py",
                "--prepared",
                str(output / "prepared-dev.json"),
                "--adapter-checkpoint",
                str(output / "checkpoints" / "checkpoint-512"),
                "--update",
                "512",
                "--output",
                str(output / "teacher-loss-512.json"),
            ],
            output / "teacher-loss-512.json",
        ),
        (
            "teacher-loss-671",
            [
                python,
                "scripts/evaluate_m4_teacher_loss.py",
                "--prepared",
                str(output / "prepared-dev.json"),
                "--adapter-checkpoint",
                str(output / "checkpoints" / "checkpoint-671"),
                "--update",
                "671",
                "--output",
                str(output / "teacher-loss-671.json"),
            ],
            output / "teacher-loss-671.json",
        ),
        (
            "epoch-extension-decision",
            [
                python,
                "scripts/decide_m4_epoch_extension.py",
                "--loss-512",
                str(output / "teacher-loss-512.json"),
                "--loss-671",
                str(output / "teacher-loss-671.json"),
                "--training-result",
                str(output / "training-result.json"),
                "--output",
                str(output / "epoch-extension-decision.json"),
            ],
            output / "epoch-extension-decision.json",
        ),
        (
            "base-full",
            [
                python,
                "scripts/run_m4_eval.py",
                "--prepared",
                str(output / "prepared-full-validation.json"),
                "--output",
                str(output / "base-full.json"),
            ],
            output / "base-full.json",
        ),
        (
            "sft-full",
            [
                python,
                "scripts/run_m4_eval.py",
                "--prepared",
                str(output / "prepared-full-validation.json"),
                "--adapter-dir",
                str(output / "adapter"),
                "--output",
                str(output / "sft-full.json"),
            ],
            output / "sft-full.json",
        ),
        (
            "compare-full",
            [
                python,
                "scripts/validate_m4_comparison.py",
                str(output / "base-full.json"),
                str(output / "sft-full.json"),
            ],
            output / ".stage-compare-full",
        ),
    ]
    for name, command, marker in stages:
        if marker.exists():
            continue
        if name == "train":
            required_checkpoint_files = {
                "adapter_config.json",
                "adapter_model.safetensors",
                "m4-contract.json",
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
            }
            checkpoints = (
                sorted(
                    (
                        path
                        for path in (output / "checkpoints").glob("checkpoint-*")
                        if required_checkpoint_files <= {item.name for item in path.iterdir()}
                    ),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if (output / "checkpoints").exists()
                else []
            )
            if checkpoints:
                command += ["--resume-from-checkpoint", str(checkpoints[-1])]
        runner.run_stage(name, command, cwd=root, env=env)
        if marker.name.startswith(".stage-"):
            marker.touch(mode=0o600)
    completion = {
        "schema_version": 1,
        "status": "evaluation_completed_summary_pending",
        "elapsed_seconds": runner.elapsed_seconds,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(output / "session-evaluation-completion.json", completion)
    runner.run_stage(
        "public-summary",
        [
            python,
            "scripts/summarize_m4_result.py",
            "--evidence-dir",
            str(output),
            "--output",
            str(output / "m4-public-summary.json"),
        ],
        cwd=root,
        env=env,
    )
    completion["status"] = "completed"
    atomic_write_json(output / "session-completion.json", completion)
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
