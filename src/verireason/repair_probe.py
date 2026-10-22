"""Execution-feedback repair probe contracts; no model or CUDA import."""

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verireason.m3_contract import extract_generated_sql
from verireason.sql_evaluator import QueryResult

EXPERIMENT = "e6_execution_repair_probe_001"
ARMS = ("feedback", "control")


def load_repair_probe_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"source", "model", "generation", "feedback", "gate", "evidence"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("repair probe config is incomplete")
    if config.get("experiment_id") != EXPERIMENT:
        raise ValueError("repair probe experiment identity drifted")
    source, generation, evidence = (
        config["source"],
        config["generation"],
        config["evidence"],
    )
    if (
        source.get("initial_states") != 32
        or source.get("training_examples") != 8
        or source.get("validation_payloads_read") != 0
        or source.get("heldout_payloads_read") != 0
    ):
        raise ValueError("repair probe source scope drifted")
    if generation.get("arms") != list(ARMS) or generation.get("completions_per_arm") != 2:
        raise ValueError("repair probe requires two matched two-completion arms")
    if evidence.get("optimizer_updates") != 0:
        raise ValueError("repair probe must remain inference-only")
    return config


def _safe_cell(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    text = str(value)
    return text if len(text) <= 96 else text[:93] + "..."


def _bounded_error(result: QueryResult) -> str:
    message = result.error or ""
    patterns = (
        (r"no such column:\s*([^\n;]+)", "unknown column: {}"),
        (r"no such table:\s*([^\n;]+)", "unknown table: {}"),
        (r"ambiguous column name:\s*([^\n;]+)", "ambiguous column: {}"),
        (r"near\s+\"([^\"]+)\":\s*syntax error", "syntax error near: {}"),
    )
    for pattern, template in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            identifier = re.sub(r"[^A-Za-z0-9_.$ -]", "", match.group(1)).strip()[:96]
            return template.format(identifier or "unknown token")
    if result.status == "unsafe":
        return "query rejected by the read-only safety policy"
    if result.status == "resource_limit":
        return f"query exceeded the {result.limit_kind or 'resource'} limit"
    if result.status == "timeout":
        return "query exceeded the wall-clock limit"
    if result.status == "syntax_error":
        return "SQLite reported a syntax error"
    if result.status == "execution_error":
        return "SQLite reported an execution error"
    return f"executor status: {result.status}"


def execution_feedback(result: QueryResult, *, max_rows: int, max_characters: int) -> str:
    """Return bounded tool feedback without gold SQL/results or correctness labels."""
    if max_rows < 0 or max_characters < 64:
        raise ValueError("invalid feedback bounds")
    if result.successful:
        preview = [tuple(_safe_cell(value) for value in row) for row in result.rows[:max_rows]]
        payload = json.dumps(preview, ensure_ascii=False, allow_nan=False)
        body = (
            f"Execution succeeded. Columns: {result.column_count}. "
            f"Rows returned: {result.row_count}. Preview: {payload}"
        )
    else:
        body = f"Execution did not succeed. {_bounded_error(result)}."
    return body[:max_characters]


def repair_messages(
    source_messages: Sequence[Mapping[str, str]],
    initial_sql: str,
    *,
    arm: str,
    feedback: str,
) -> list[dict[str, str]]:
    if arm not in ARMS or not source_messages or not initial_sql.strip():
        raise ValueError("invalid repair state")
    observation = (
        f"Database feedback:\n{feedback}"
        if arm == "feedback"
        else "Database feedback is withheld for this controlled comparison."
    )
    instruction = (
        "Review your previous SQL against the original question and schema. "
        "If it should remain unchanged, return KEEP. Otherwise return exactly one corrected "
        "read-only SQLite SELECT or WITH ... SELECT query. Return only KEEP or SQL, with no "
        f"explanation or Markdown.\n\n{observation}"
    )
    return [
        *[dict(message) for message in source_messages],
        {"role": "assistant", "content": initial_sql},
        {"role": "user", "content": instruction},
    ]


def resolve_repair_action(completion: str, initial_sql: str) -> tuple[str | None, str, bool]:
    raw = completion.strip()
    if raw.upper() == "KEEP":
        return initial_sql, "keep", True
    parsed = extract_generated_sql(completion)
    if parsed["sql"] is None or not parsed["strict_sql_only"]:
        return None, "invalid_format", False
    return parsed["sql"], "rewrite", True


def summarize_repair_probe(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict:
    expected = (
        config["source"]["initial_states"] * len(ARMS) * config["generation"]["completions_per_arm"]
    )
    if len(records) != expected:
        raise ValueError("repair probe completion count is incomplete")
    state_keys = {(int(row["state_index"]), str(row["arm"])) for row in records}
    if len(state_keys) != config["source"]["initial_states"] * len(ARMS):
        raise ValueError("repair probe state/arm coverage is incomplete")

    per_arm: dict[str, dict[str, Any]] = {}
    state_rewards: dict[tuple[int, str], list[int]] = {}
    state_examples: dict[int, str] = {}
    initial_rewards: dict[int, int] = {}
    for row in records:
        state = int(row["state_index"])
        arm = str(row["arm"])
        state_examples[state] = str(row["example_id"])
        initial_rewards[state] = int(row["initial_reward"])
        state_rewards.setdefault((state, arm), []).append(int(row["final_reward"]))

    for arm in ARMS:
        arm_rows = [row for row in records if row["arm"] == arm]
        wrong = [row for row in arm_rows if not row["initial_reward"]]
        correct = [row for row in arm_rows if row["initial_reward"]]
        repaired_states = {
            state
            for state, reward in initial_rewards.items()
            if reward == 0 and any(state_rewards[(state, arm)])
        }
        damaged_states = {
            state
            for state, reward in initial_rewards.items()
            if reward == 1 and not all(state_rewards[(state, arm)])
        }
        mixed = sum(
            len(set(values)) > 1
            for (state, state_arm), values in state_rewards.items()
            if state_arm == arm
        )
        per_arm[arm] = {
            "completions": len(arm_rows),
            "repaired_completions": sum(int(row["final_reward"]) for row in wrong),
            "wrong_state_completions": len(wrong),
            "damaged_completions": sum(not int(row["final_reward"]) for row in correct),
            "correct_state_completions": len(correct),
            "repaired_states_any": len(repaired_states),
            "damaged_states_any": len(damaged_states),
            "mixed_states": mixed,
            "strict_outputs": sum(bool(row["strict_output"]) for row in arm_rows),
            "unsafe_actions": sum(row["final_status"] == "unsafe" for row in arm_rows),
            "resource_limited_actions": sum(
                row["final_status"] in {"resource_limit", "timeout"} for row in arm_rows
            ),
            "final_statuses": dict(Counter(str(row["final_status"]) for row in arm_rows)),
        }

    feedback_repaired = {
        state
        for state, reward in initial_rewards.items()
        if reward == 0 and any(state_rewards[(state, "feedback")])
    }
    control_repaired = {
        state
        for state, reward in initial_rewards.items()
        if reward == 0 and any(state_rewards[(state, "control")])
    }
    feedback_only = feedback_repaired - control_repaired
    control_only = control_repaired - feedback_repaired
    gates = config["gate"]
    checks = {
        "feedback_only_repairs": len(feedback_only)
        >= gates["minimum_feedback_only_repaired_states"],
        "distinct_examples": len({state_examples[state] for state in feedback_only})
        >= gates["minimum_distinct_feedback_only_examples"],
        "mixed_feedback_states": per_arm["feedback"]["mixed_states"]
        >= gates["minimum_feedback_mixed_states"],
        "repair_advantage": per_arm["feedback"]["repaired_completions"]
        > per_arm["control"]["repaired_completions"],
        "damage_control": per_arm["feedback"]["damaged_completions"]
        <= per_arm["control"]["damaged_completions"],
        "safe": per_arm["feedback"]["unsafe_actions"] == 0
        and per_arm["control"]["unsafe_actions"] == 0,
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT,
        "status": "promote_to_rl_design" if all(checks.values()) else "stop_or_revise",
        "initial_states": config["source"]["initial_states"],
        "initial_correct": sum(initial_rewards.values()),
        "initial_wrong": len(initial_rewards) - sum(initial_rewards.values()),
        "per_arm": per_arm,
        "feedback_only_repaired_states": len(feedback_only),
        "control_only_repaired_states": len(control_only),
        "distinct_feedback_only_examples": len({state_examples[state] for state in feedback_only}),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "optimizer_updates": 0,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
