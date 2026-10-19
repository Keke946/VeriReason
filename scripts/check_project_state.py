#!/usr/bin/env python3
"""Fail fast when VeriReason's durable project state is inconsistent."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "README.md",
    "docs/PROJECT.md",
    "docs/PLAN.md",
    "docs/STATE.md",
    "docs/DECISIONS.md",
    "docs/INTERVIEW.md",
    "docs/COMPUTE.md",
    "docs/M1_RUNBOOK.md",
    "docs/M2_PROTOCOL.md",
    "configs/environment.toml",
    "configs/m1_smoke.toml",
    "configs/m2_data.toml",
    "configs/m2_data_v1_3s.toml",
    "configs/m2_finalizer.toml",
    "configs/m3_vertical_slice.toml",
    "configs/m10_public_artifacts.toml",
    "configs/m11_release.toml",
    "experiments/registry.yaml",
    "experiments/spend.csv",
    "pyproject.toml",
    "results/schemas/m2-manifest-index.schema.json",
    "results/schemas/m3-result.schema.json",
    "results/schemas/m3-closure-summary.schema.json",
    "results/m3/m3-vertical-slice-summary.json",
    "docs/M3_RUNBOOK.md",
    "docs/M3_LEARNING.md",
    "docs/M10_RUNBOOK.md",
    "docs/TECHNICAL_REPORT.md",
    "docs/MODEL_CARD.md",
    "docs/DATA_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESEARCH_FOLLOWUP.md",
    "docs/RELEASE_PACKET.md",
    "LICENSE",
    "CITATION.cff",
    "requirements/m10-cpu-lock.txt",
    "results/m10/release-manifest.json",
    "Dockerfile",
)
FORBIDDEN_SUFFIXES = {".bin", ".gguf", ".pem", ".pt", ".pth", ".safetensors"}


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def repository_files() -> list[Path]:
    """Return tracked and untracked, non-ignored files that could enter a commit."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not {".git", ".venv"}.intersection(path.parts)
        ]
    return [ROOT / relative for relative in completed.stdout.split("\0") if relative]


def main() -> int:
    errors: list[str] = []

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"missing required file: {relative_path}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"required file is empty: {relative_path}")

    if errors:
        return report(errors)

    agents = read("AGENTS.md")
    for required_phrase in (
        "Required context at the start of every task",
        "Source of truth",
        "Completion protocol",
        "Learning and interview readiness",
    ):
        if required_phrase not in agents:
            errors.append(f"AGENTS.md lost required section: {required_phrase}")

    project = read("docs/PROJECT.md")
    if "## Definition of done" not in project:
        errors.append("PROJECT.md must contain the definition of done")

    plan = read("docs/PLAN.md")
    milestone_rows = re.findall(
        r"^\|\s*(M\d+)\s+[^|]*\|[^|]*\|\s*"
        r"(pending|in_progress|blocked|completed)\s*\|",
        plan,
        flags=re.MULTILINE,
    )
    if not milestone_rows:
        errors.append("PLAN.md contains no parseable milestone rows")
    else:
        active = [milestone for milestone, status in milestone_rows if status == "in_progress"]
        if len(active) > 1:
            errors.append(
                "PLAN.md may contain at most one in_progress milestone; "
                f"found {len(active)}: {active}"
            )

    state = read("docs/STATE.md")
    state_milestone_match = re.search(r"^- Milestone:\s*(M\d+)\b", state, re.MULTILINE)
    if not state_milestone_match:
        errors.append("STATE.md must name the current milestone")
    elif milestone_rows:
        state_milestone = state_milestone_match.group(1)
        active = [milestone for milestone, status in milestone_rows if status == "in_progress"]
        if len(active) == 1 and state_milestone != active[0]:
            errors.append(
                f"STATE.md says {state_milestone}, but PLAN.md marks {active[0]} in progress"
            )
        elif not active:
            milestone_status = dict(milestone_rows).get(state_milestone)
            state_status = re.search(r"^- Status:\s*(.+)$", state, re.MULTILINE)
            if (
                milestone_status != "completed"
                or not state_status
                or not state_status.group(1).startswith("completed")
            ):
                errors.append(
                    "PLAN.md has no in_progress milestone, but STATE.md does not describe a "
                    "completed handoff state"
                )

    next_action_match = re.search(
        r"^## Exactly one next action\s*\n+(.*?)(?=\n## |\Z)",
        state,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not next_action_match or not next_action_match.group(1).strip():
        errors.append("STATE.md must contain one non-empty next action")

    decisions = read("docs/DECISIONS.md")
    if "Decisions are append-only" not in decisions:
        errors.append("DECISIONS.md must preserve append-only history")

    registry = read("experiments/registry.yaml")
    if not re.search(r"^schema_version:\s*1\s*$", registry, re.MULTILINE):
        errors.append("experiment registry schema_version must be 1")
    if not re.search(r"^experiments:\s*\{\}\s*$", registry, re.MULTILINE) and not re.search(
        r"^experiments:\s*$", registry, re.MULTILINE
    ):
        errors.append("experiment registry must define experiments")

    spend_header = read("experiments/spend.csv").splitlines()[0]
    for required_column in ("session_id", "experiment_ids", "planned_cost_cny", "actual_cost_cny"):
        if required_column not in spend_header.split(","):
            errors.append(f"spend.csv missing required column: {required_column}")

    for path in repository_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden model/secret artifact in repository: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"file larger than 5 MiB requires explicit handling: {relative}")

    return report(errors)


def report(errors: list[str]) -> int:
    if errors:
        print("Project guard failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project guard passed: durable state and workflow invariants are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
