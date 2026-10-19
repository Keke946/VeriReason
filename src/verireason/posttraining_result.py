"""Close an E2/E3/E4 training session from aggregate, SQL-free evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or (positive and observed <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return observed


def _require_zero_heldout(evidence: Mapping[str, Any], name: str) -> None:
    if (
        evidence.get("heldout_examples_loaded") != 0
        or evidence.get("heldout_examples_evaluated") != 0
    ):
        raise ValueError(f"{name} reports heldout access")


def close_posttraining_result(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    reload: Mapping[str, Any],
    *,
    session_wall_seconds: float,
) -> dict[str, Any]:
    """Validate the minimum evidence needed to call a post-training smoke complete."""
    experiment_id, arm = config["experiment_id"], config["arm"]
    if (
        training.get("schema_version") != 1
        or training.get("status") != "trained"
        or training.get("experiment_id") != experiment_id
        or training.get("arm") != arm
    ):
        raise ValueError("training result identity or status mismatch")
    if reload.get("schema_version") != 1 or reload.get("status") != "passed":
        raise ValueError("adapter reload did not pass")
    _require_zero_heldout(training, "training result")
    _require_zero_heldout(reload, "reload result")

    steps = config["optimization"]["maximum_optimizer_steps"]
    prompts = config["data"]["prompt_count"]
    generations = config["generation"]["num_generations"]
    if training.get("optimizer_steps") != steps:
        raise ValueError("optimizer step count drifted")
    training_loss = _finite_number(training.get("training_loss"), "training loss")
    gradient_norm_last = _finite_number(training.get("gradient_norm_last"), "last gradient norm")
    gradient_norm_max = _finite_number(
        training.get("gradient_norm_max"), "maximum gradient norm", positive=True
    )
    adapter_delta = _finite_number(training.get("adapter_delta_l2"), "adapter delta", positive=True)
    peak_memory = _finite_number(
        training.get("peak_gpu_memory_mib"), "peak GPU memory", positive=True
    )
    training_wall = _finite_number(
        training.get("wall_seconds"), "training wall time", positive=True
    )
    session_wall = _finite_number(session_wall_seconds, "session wall time", positive=True)
    if session_wall > config["evidence"]["maximum_session_wall_seconds"]:
        raise ValueError("session exceeded the frozen wall limit")

    reward = training.get("reward_diagnostics")
    components = training.get("component_diagnostics")
    if not isinstance(reward, Mapping) or not isinstance(components, Mapping):
        raise ValueError("reward or component diagnostics are missing")
    mixed = reward.get("mixed_reward_groups")
    zero_variance = reward.get("zero_variance_groups")
    if (
        reward.get("groups") != prompts
        or not isinstance(mixed, int)
        or isinstance(mixed, bool)
        or mixed <= 0
        or not isinstance(zero_variance, int)
        or mixed + zero_variance != prompts
    ):
        raise ValueError("reward-group diagnostics are incomplete or show no signal")
    expected_fraction = mixed / prompts
    observed_fraction = _finite_number(
        reward.get("mixed_reward_group_fraction"), "mixed reward fraction"
    )
    if not math.isclose(observed_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mixed reward fraction is inconsistent")

    mean_name = "mean_terminal_reward" if arm == "E2" else "mean_group_relative_reward"
    mean_reward = _finite_number(reward.get(mean_name), mean_name)
    if arm in {"E3", "E4"}:
        if "mean_terminal_reward" in reward:
            raise ValueError(f"{arm} dense reward was mislabeled as terminal reward")
        if not math.isclose(mean_reward, 0.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{arm} group-normalized reward mean is not approximately zero")

    completion_count = prompts * generations
    if components.get("completion_count") != completion_count:
        raise ValueError("component diagnostics do not cover every completion")
    for name in ("strict_correct_count", "valid_count", "confidence_gate_count"):
        value = components.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= completion_count
        ):
            raise ValueError(f"invalid component count: {name}")
    mean_structure = _finite_number(components.get("mean_structure_score"), "mean structure score")
    if not 0.0 <= mean_structure <= 1.0:
        raise ValueError("mean structure score must be in [0, 1]")
    statuses = components.get("prediction_status_counts")
    if (
        not isinstance(statuses, Mapping)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in statuses.values()
        )
        or sum(statuses.values()) != completion_count
    ):
        raise ValueError("prediction status counts are incomplete")
    if arm == "E2" and not math.isclose(
        mean_reward,
        components["strict_correct_count"] / completion_count,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("E2 terminal reward mean disagrees with strict-correct count")

    adapter_sha = training.get("adapter_manifest_sha256")
    if (
        not isinstance(adapter_sha, str)
        or len(adapter_sha) != 64
        or reload.get("adapter_manifest_sha256") != adapter_sha
    ):
        raise ValueError("training and reload adapter identities disagree")
    adapter_names = reload.get("adapter_names")
    if not isinstance(adapter_names, list) or not adapter_names:
        raise ValueError("reload result contains no adapter names")

    signal_gate_passed = training.get("signal_gate_passed")
    if "signal_gate" in config and not isinstance(signal_gate_passed, bool):
        raise ValueError("formal run lacks a boolean signal-gate result")
    summary_status = "completed_below_signal_gate" if signal_gate_passed is False else "passed"
    return {
        "schema_version": 1,
        "status": summary_status,
        "experiment_id": experiment_id,
        "arm": arm,
        "multi_objective_aggregation": config["optimization"]["multi_objective_aggregation"],
        "training_prompts": prompts,
        "completions_per_prompt": generations,
        "rollouts": completion_count,
        "optimizer_steps": steps,
        "training_loss": training_loss,
        "gradient_norm_last": gradient_norm_last,
        "gradient_norm_max": gradient_norm_max,
        "adapter_delta_l2": adapter_delta,
        "mixed_reward_groups": mixed,
        "zero_variance_groups": zero_variance,
        "mixed_reward_group_fraction": observed_fraction,
        mean_name: mean_reward,
        "strict_correct_count": components["strict_correct_count"],
        "valid_count": components["valid_count"],
        "confidence_gate_count": components["confidence_gate_count"],
        "mean_structure_score": mean_structure,
        "prediction_status_counts": dict(components["prediction_status_counts"]),
        "peak_gpu_memory_mib": peak_memory,
        "training_wall_seconds": training_wall,
        "session_wall_seconds": session_wall,
        "adapter_manifest_sha256": adapter_sha,
        "separate_process_reload_passed": True,
        "signal_gate_passed": signal_gate_passed,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
