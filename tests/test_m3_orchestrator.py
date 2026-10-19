import json
import os
import sys
from pathlib import Path

import pytest

from verireason.m3_contract import load_m3_config
from verireason.m3_orchestrator import (
    DiskUsage,
    M3GateError,
    SessionRunner,
    prepare_output_directory,
    require_approved_commit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_counts_make_two_complete_accumulation_updates() -> None:
    loaded = load_m3_config(ROOT / "configs/m3_vertical_slice.toml")
    assert loaded["data"]["train_count"] == 8
    assert loaded["data"]["validation_count"] == 8
    assert loaded["evidence"]["maximum_session_wall_seconds"] == 14_400
    assert loaded["evidence"]["maximum_session_cost_cny"] == 10.36
    assert loaded["data"]["train_count"] == (
        loaded["qlora"]["per_device_train_batch_size"]
        * loaded["qlora"]["gradient_accumulation_steps"]
        * loaded["qlora"]["maximum_optimizer_steps"]
    )


def test_approved_commit_is_required_and_must_match() -> None:
    commit = "a" * 40
    require_approved_commit(commit, commit, False)
    with pytest.raises(M3GateError, match="externally supplied"):
        require_approved_commit(None, commit, False)
    with pytest.raises(M3GateError, match="does not match"):
        require_approved_commit(commit, "b" * 40, False)
    with pytest.raises(M3GateError, match="dirty"):
        require_approved_commit(commit, commit, True)


def test_output_must_be_fresh_external_and_have_space(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    external = tmp_path / "artifacts" / "new-run"
    output, free = prepare_output_directory(
        external,
        worktree,
        20 * 1024**3,
        disk_usage=lambda _path: DiskUsage(30 * 1024**3, 1, 25 * 1024**3),
    )
    assert output.is_dir()
    assert free == 25 * 1024**3
    with pytest.raises(M3GateError, match="must not already exist"):
        prepare_output_directory(output, worktree, 1)
    with pytest.raises(M3GateError, match="outside"):
        prepare_output_directory(worktree / "inside", worktree, 1)
    with pytest.raises(M3GateError, match="at least"):
        prepare_output_directory(
            tmp_path / "low-space",
            worktree,
            20 * 1024**3,
            disk_usage=lambda _path: DiskUsage(10, 9, 1),
        )


def test_output_symlink_cannot_hide_worktree_destination(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(worktree, target_is_directory=True)
    with pytest.raises(M3GateError, match="outside"):
        prepare_output_directory(alias / "run", worktree, 1)


def test_runner_enforces_cumulative_timeout_and_preserves_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    runner = SessionRunner(output, maximum_wall_seconds=0.05, terminate_grace_seconds=0.05)
    with pytest.raises(TimeoutError, match="cumulative"):
        runner.run_stage(
            "slow",
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            env=dict(os.environ),
        )
    evidence = json.loads((output / "session-timeout.json").read_text())
    assert evidence["status"] == "session_wall_timeout"
    assert evidence["stage"] == "slow"
    assert evidence["maximum_session_wall_seconds"] == 0.05


def test_runner_does_not_reset_deadline_between_stages(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    ticks = iter([10.0, 11.1, 11.1])
    runner = SessionRunner(
        output,
        maximum_wall_seconds=1.0,
        terminate_grace_seconds=0.01,
        monotonic=lambda: next(ticks),
    )
    with pytest.raises(TimeoutError, match="before stage"):
        runner.run_stage(
            "next",
            [sys.executable, "-c", "raise SystemExit(0)"],
            cwd=tmp_path,
            env=dict(os.environ),
        )
