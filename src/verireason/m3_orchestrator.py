"""Fail-closed orchestration for the single paid M3 vertical-slice session."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TextIO

from verireason.m3_contract import atomic_write_json


class M3GateError(RuntimeError):
    """Raised before paid work when a frozen precondition is not satisfied."""


class DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def require_approved_commit(approved: str | None, observed: str, dirty: bool) -> None:
    """Require an externally supplied, exact Git identity before GPU work."""
    invalid_sha = (
        approved is None
        or len(approved) != 40
        or any(c not in "0123456789abcdef" for c in approved)
    )
    if invalid_sha:
        raise M3GateError("M3_APPROVED_COMMIT must be an externally supplied 40-character SHA")
    if dirty:
        raise M3GateError("Git worktree is dirty; refusing the paid M3 session")
    if observed != approved:
        raise M3GateError(f"HEAD {observed} does not match approved commit {approved}")


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise M3GateError(f"no existing ancestor for output path: {path}")
        candidate = candidate.parent
    return candidate.resolve(strict=True)


def prepare_output_directory(
    output: str | Path,
    worktree: str | Path,
    minimum_free_bytes: int,
    *,
    disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
) -> tuple[Path, int]:
    """Validate a fresh external output directory and create it with mode 0700."""
    raw = Path(output).expanduser()
    if not raw.is_absolute():
        raise M3GateError("M3_OUTPUT_DIR must be an absolute path")
    if raw.exists() or raw.is_symlink():
        raise M3GateError("M3_OUTPUT_DIR must not already exist; use a new experiment directory")

    ancestor = _nearest_existing_ancestor(raw.parent)
    resolved_output = raw.resolve(strict=False)
    resolved_worktree = Path(worktree).resolve(strict=True)
    if resolved_output == resolved_worktree or resolved_output.is_relative_to(resolved_worktree):
        raise M3GateError("M3_OUTPUT_DIR must resolve outside the Git worktree")

    free_bytes = disk_usage(ancestor).free
    if free_bytes < minimum_free_bytes:
        raise M3GateError(
            f"output filesystem has {free_bytes} free bytes; at least {minimum_free_bytes} required"
        )
    resolved_output.mkdir(parents=True, mode=0o700)
    return resolved_output, free_bytes


@dataclass
class SessionRunner:
    """Run every stage against one cumulative monotonic wall-clock deadline."""

    output_dir: Path
    maximum_wall_seconds: float
    terminate_grace_seconds: float
    started_monotonic: float | None = None
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.started_monotonic is None:
            self.started_monotonic = self.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        assert self.started_monotonic is not None
        return self.monotonic() - self.started_monotonic

    @property
    def remaining_seconds(self) -> float:
        return self.maximum_wall_seconds - self.elapsed_seconds

    def _record_timeout(self, stage: str, command: Sequence[str]) -> None:
        payload = {
            "schema_version": 1,
            "status": "session_wall_timeout",
            "stage": stage,
            "command": list(command),
            "elapsed_seconds": self.elapsed_seconds,
            "maximum_session_wall_seconds": self.maximum_wall_seconds,
        }
        atomic_write_json(self.output_dir / "session-timeout.json", payload)

    def run_stage(
        self,
        stage: str,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout_file: Path | None = None,
    ) -> None:
        remaining = self.remaining_seconds
        if remaining <= 0:
            self._record_timeout(stage, command)
            raise TimeoutError("M3 cumulative session wall limit reached before stage start")

        atomic_write_json(
            self.output_dir / "session-progress.json",
            {
                "schema_version": 1,
                "status": "running",
                "stage": stage,
                "elapsed_seconds": self.elapsed_seconds,
                "remaining_seconds": self.remaining_seconds,
            },
        )
        remaining = self.remaining_seconds
        if remaining <= 0:
            self._record_timeout(stage, command)
            raise TimeoutError("M3 cumulative session wall limit reached before process start")
        log_path = self.output_dir / f"stage-{stage}.log"
        with log_path.open("ab") as log_handle:
            output_handle: TextIO | None = None
            try:
                if stdout_file is not None:
                    output_handle = stdout_file.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    list(command),
                    cwd=cwd,
                    env=env,
                    stdout=output_handle if output_handle is not None else log_handle,
                    stderr=log_handle,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as error:
                    self._record_timeout(stage, command)
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=self.terminate_grace_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError(
                        f"M3 cumulative session wall limit reached during {stage}"
                    ) from error
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, list(command))
                atomic_write_json(
                    self.output_dir / "session-progress.json",
                    {
                        "schema_version": 1,
                        "status": "stage_completed",
                        "stage": stage,
                        "elapsed_seconds": self.elapsed_seconds,
                        "remaining_seconds": self.remaining_seconds,
                    },
                )
            finally:
                if output_handle is not None:
                    output_handle.close()
