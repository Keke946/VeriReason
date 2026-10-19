"""Minimal pre-cloud contract for the M7 outcome-only GRPO engineering smoke."""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sql_complexity

MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
TRAINING_MANIFEST_SHA256 = "d90ffab32e296248376c542b736aef5ce7f547acc702a9eab41073789cf5b72a"


def load_m7_smoke_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"parent", "data", "generation", "optimization", "reward", "evidence", "stops"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M7 smoke config must be schema v1 with every contract section")
    if config.get("status") != "precloud_preregistered_pending_teach_back_and_control_approval":
        raise ValueError("M7 smoke must remain unapproved during local preparation")
    if config.get("arm") != "E2":
        raise ValueError("the first M7 smoke is outcome-only E2")

    parent, data = config["parent"], config["data"]
    generation, opt = config["generation"], config["optimization"]
    if parent.get("revision") != MODEL_REVISION or parent.get("immutable") is not True:
        raise ValueError("M7 must start from the immutable frozen E1 parent")
    if data.get("training_manifest_sha256") != TRAINING_MANIFEST_SHA256:
        raise ValueError("M7 training manifest identity drifted")
    if data.get("split") != "train" or not data.get("outcome_blind"):
        raise ValueError("M7 smoke selection must be outcome-blind and training-only")
    if data.get("heldout_examples_loaded") or data.get("heldout_examples_evaluated"):
        raise ValueError("M7 smoke cannot access heldout")
    if data.get("prompt_count") != 8 or generation.get("num_generations") != 4:
        raise ValueError("M7 smoke freezes eight prompts with four completions each")
    if generation.get("use_vllm") is not False:
        raise ValueError("M7 smoke uses direct generation to minimize infrastructure")
    if opt.get("generation_batch_size") != generation["num_generations"]:
        raise ValueError("each generation batch must contain one complete prompt group")
    if opt.get("gradient_accumulation_steps") != generation["num_generations"]:
        raise ValueError("M7 smoke freezes one complete group per accumulation window")
    if opt.get("maximum_optimizer_steps") != data["prompt_count"]:
        raise ValueError("M7 smoke requires one optimizer update per selected prompt group")
    if opt.get("scale_rewards") != "none":
        raise ValueError("project-relative rewards must not be normalized by TRL a second time")
    if opt.get("loss_type") != "dapo" or not opt.get("mask_truncated_completions"):
        raise ValueError("M7 smoke requires DAPO token normalization and truncation masking")
    if opt.get("beta") != 0.0 or opt.get("num_iterations") != 1:
        raise ValueError("the minimal smoke excludes a reference model and rollout reuse")
    if config["reward"].get("terminal_metric") != "strict_execution_accuracy":
        raise ValueError("E2 terminal reward must use frozen strict execution")
    if config["evidence"].get("maximum_session_cost_cny") != 7.77:
        raise ValueError("M7 smoke cost ceiling changed")
    return config


def load_m7_formal_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "parent",
        "data",
        "generation",
        "optimization",
        "reward",
        "signal_gate",
        "reward_hacking_review",
        "evidence",
        "stops",
    }
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M7 formal config must be schema v1 with every contract section")
    if config.get("status") != "precloud_preregistered_pending_control_approval":
        raise ValueError("M7 formal run must remain unapproved during local preparation")
    if config.get("experiment_id") != "e2_m7_outcome_grpo_formal_001":
        raise ValueError("unexpected M7 formal experiment identity")
    if config.get("arm") != "E2":
        raise ValueError("M7 formal baseline must remain outcome-only E2")

    parent, data = config["parent"], config["data"]
    generation, opt = config["generation"], config["optimization"]
    if parent.get("revision") != MODEL_REVISION or parent.get("immutable") is not True:
        raise ValueError("M7 must start from the immutable frozen E1 parent")
    if data.get("training_manifest_sha256") != TRAINING_MANIFEST_SHA256:
        raise ValueError("M7 training manifest identity drifted")
    if data.get("split") != "train" or not data.get("outcome_blind"):
        raise ValueError("M7 formal selection must be outcome-blind and training-only")
    if data.get("heldout_examples_loaded") or data.get("heldout_examples_evaluated"):
        raise ValueError("M7 formal run cannot access heldout")
    if data.get("prompt_count") != 256 or generation.get("num_generations") != 4:
        raise ValueError("M7 formal E2 freezes 256 prompts with four completions each")
    if generation.get("use_vllm") is not False:
        raise ValueError("M7 formal E2 retains measured direct generation")
    if opt.get("generation_batch_size") != generation["num_generations"]:
        raise ValueError("each generation batch must contain one complete prompt group")
    if opt.get("gradient_accumulation_steps") != generation["num_generations"]:
        raise ValueError("each accumulation window must contain one complete prompt group")
    if opt.get("maximum_optimizer_steps") != data["prompt_count"]:
        raise ValueError("M7 formal E2 requires one optimizer update per selected prompt group")
    checkpoint_steps = opt.get("checkpoint_steps")
    if checkpoint_steps != 64 or opt["maximum_optimizer_steps"] % checkpoint_steps:
        raise ValueError("M7 formal checkpoints must divide the complete run into four blocks")
    if opt.get("scale_rewards") != "none":
        raise ValueError("project-relative rewards must not be normalized by TRL a second time")
    if opt.get("loss_type") != "dapo" or not opt.get("mask_truncated_completions"):
        raise ValueError("M7 formal E2 requires DAPO token normalization and truncation masking")
    if opt.get("beta") != 0.0 or opt.get("num_iterations") != 1:
        raise ValueError("formal E2 retains the measured no-reference single-iteration baseline")
    if config["reward"].get("terminal_metric") != "strict_execution_accuracy":
        raise ValueError("E2 terminal reward must use frozen strict execution")
    gate = config["signal_gate"]
    if gate.get("minimum_mixed_reward_groups") != 32:
        raise ValueError("M7 formal promotion requires at least 32 informative groups")
    if gate.get("minimum_mixed_reward_group_fraction") != 0.125:
        raise ValueError("M7 formal signal fraction gate changed")
    if config["evidence"].get("maximum_session_cost_cny") != 7.77:
        raise ValueError("M7 formal cost ceiling changed")
    return config


def load_m7_config(path: str | Path) -> dict[str, Any]:
    """Dispatch to the immutable smoke or pending formal contract."""
    with Path(path).open("rb") as handle:
        experiment_id = tomllib.load(handle).get("experiment_id")
    if experiment_id == "e2_m7_outcome_grpo_smoke_001":
        return load_m7_smoke_config(path)
    if experiment_id == "e2_m7_outcome_grpo_formal_001":
        return load_m7_formal_config(path)
    raise ValueError("unknown M7 experiment identity")


def select_smoke_rows(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: str, required_features: Sequence[str]
) -> list[dict[str, Any]]:
    """Select training rows without consulting any model outcome or reward."""
    eligible = [
        dict(row)
        for row in rows
        if row.get("split") == "train"
        and row.get("operational_eligible") is True
        and row.get("source") != "heldout"
    ]
    identities = [row.get("row_id") for row in eligible]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("every M7 candidate needs a stable row_id")
    if len(set(identities)) != len(identities):
        raise ValueError("M7 candidate row_ids must be unique")

    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['db_id']}\0{row['row_id']}".encode()
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for feature in required_features:
        match = next(
            (
                row
                for row in ranked
                if row["row_id"] not in used
                and feature in sql_complexity(str(row.get("sql", ""))).features
            ),
            None,
        )
        if match is None:
            raise ValueError(f"no eligible smoke row covers required feature: {feature}")
        selected.append(match)
        used.add(match["row_id"])
    selected.extend(row for row in ranked if row["row_id"] not in used)
    selected = selected[:count]
    if len(selected) != count:
        raise ValueError("insufficient eligible rows for M7 smoke")
    return selected


def summarize_group_rewards(rewards: Sequence[float], *, num_generations: int) -> dict[str, Any]:
    if not rewards or len(rewards) % num_generations:
        raise ValueError("reward diagnostics require complete groups")
    groups = [rewards[i : i + num_generations] for i in range(0, len(rewards), num_generations)]
    mixed = sum(len(set(group)) > 1 for group in groups)
    return {
        "groups": len(groups),
        "mixed_reward_groups": mixed,
        "zero_variance_groups": len(groups) - mixed,
        "mixed_reward_group_fraction": mixed / len(groups),
        "mean_reward": sum(rewards) / len(rewards),
    }
