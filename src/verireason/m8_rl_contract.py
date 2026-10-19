"""Frozen E3/E4 post-training contracts for the M8 reward comparison."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from verireason.m7_contract import MODEL_REVISION, TRAINING_MANIFEST_SHA256

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _require_e2_scientific_parity(config: dict[str, Any], *, formal: bool) -> None:
    reference_name = "m7_e2_formal.toml" if formal else "m7_e2_smoke.toml"
    with (CONFIG_ROOT / reference_name).open("rb") as handle:
        reference = tomllib.load(handle)
    sections = ["parent", "data", "generation", "optimization", "reward", "stops"]
    if formal:
        sections.extend(("signal_gate", "reward_hacking_review"))
    drifted = [section for section in sections if config.get(section) != reference.get(section)]
    if drifted:
        raise ValueError(f"E3 scientific control drifted from E2: {', '.join(drifted)}")


def _require_e4_parity(config: dict[str, Any], *, formal: bool) -> None:
    """Require E4 to differ from the matching E3 run only at the intervention."""
    reference_name = "m8_e3_formal.toml" if formal else "m8_e3_smoke.toml"
    stage = "formal" if formal else "smoke"
    with (CONFIG_ROOT / reference_name).open("rb") as handle:
        reference = tomllib.load(handle)
    if set(config) != set(reference):
        raise ValueError(f"E4/E3 {stage} top-level contract drifted")
    drifted: list[str] = []
    for section in config:
        if section in {"experiment_id", "arm"}:
            continue
        if section == "optimization":
            candidate = dict(config[section])
            baseline = dict(reference[section])
            candidate.pop("multi_objective_aggregation", None)
            baseline.pop("multi_objective_aggregation", None)
            if candidate != baseline:
                drifted.append(section)
        elif config[section] != reference[section]:
            drifted.append(section)
    if drifted:
        raise ValueError(f"E4 scientific control drifted from E3: {', '.join(drifted)}")


def load_m8_rl_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    experiment_id = config.get("experiment_id")
    expected = {
        "e3_m8_dense_grpo_smoke_001": {
            "arm": "E3",
            "aggregation": "sum_then_normalize",
            "prompt_count": 8,
            "steps": 8,
            "checkpoint_steps": None,
            "selector_version": "m7_e2_smoke_hash_v1",
            "selector_seed": "verireason-m7-e2-smoke-v1",
            "required_features": ["join", "aggregate", "subquery"],
            "wall": 1800,
            "cost": 1.30,
        },
        "e3_m8_dense_grpo_formal_001": {
            "arm": "E3",
            "aggregation": "sum_then_normalize",
            "prompt_count": 256,
            "steps": 256,
            "checkpoint_steps": 64,
            "selector_version": "m7_e2_formal_hash_v1",
            "selector_seed": "verireason-m7-e2-formal-v1",
            "required_features": [
                "join",
                "aggregate",
                "subquery",
                "group_by",
                "set_operation",
                "order_by",
            ],
            "wall": 10800,
            "cost": 7.77,
        },
        "e4_m8_gdpo_smoke_001": {
            "arm": "E4",
            "aggregation": "normalize_then_sum",
            "prompt_count": 8,
            "steps": 8,
            "checkpoint_steps": None,
            "selector_version": "m7_e2_smoke_hash_v1",
            "selector_seed": "verireason-m7-e2-smoke-v1",
            "required_features": ["join", "aggregate", "subquery"],
            "wall": 1800,
            "cost": 1.30,
        },
        "e4_m8_gdpo_formal_001": {
            "arm": "E4",
            "aggregation": "normalize_then_sum",
            "prompt_count": 256,
            "steps": 256,
            "checkpoint_steps": 64,
            "selector_version": "m7_e2_formal_hash_v1",
            "selector_seed": "verireason-m7-e2-formal-v1",
            "required_features": [
                "join",
                "aggregate",
                "subquery",
                "group_by",
                "set_operation",
                "order_by",
            ],
            "wall": 10800,
            "cost": 7.77,
        },
    }.get(experiment_id)
    required = {"parent", "data", "generation", "optimization", "reward", "evidence", "stops"}
    if expected is not None and expected["prompt_count"] == 256:
        required.update(("signal_gate", "reward_hacking_review"))
    if config.get("schema_version") != 1 or required - config.keys() or expected is None:
        raise ValueError("unknown or incomplete M8 RL contract")
    if config.get("status") != "precloud_preregistered_pending_control_approval":
        raise ValueError("M8 RL must remain unapproved during local preparation")
    if config.get("arm") != expected["arm"]:
        raise ValueError("M8 reward arm does not match the registered experiment")

    parent, data = config["parent"], config["data"]
    generation, optimization = config["generation"], config["optimization"]
    if (
        parent.get("model_id") != "Qwen/Qwen3-4B-Base"
        or parent.get("revision") != MODEL_REVISION
        or parent.get("adapter_experiment_id") != "e1_m4_formal_sft_001"
        or parent.get("immutable") is not True
    ):
        raise ValueError("E3 must start independently from immutable E1")
    if (
        data.get("training_manifest_sha256") != TRAINING_MANIFEST_SHA256
        or data.get("split") != "train"
        or data.get("outcome_blind") is not True
        or data.get("heldout_examples_loaded") != 0
        or data.get("heldout_examples_evaluated") != 0
    ):
        raise ValueError("E3 must use the frozen training-only manifest")
    for key in ("prompt_count", "selector_version", "selector_seed", "required_features"):
        if data.get(key) != expected[key]:
            raise ValueError(f"E3/E2 comparison drifted: {key}")
    if generation != {
        "num_generations": 4,
        "max_completion_length": 384,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "use_vllm": False,
    }:
        raise ValueError("E3 generation settings drifted from E2")
    if (
        optimization.get("maximum_optimizer_steps") != expected["steps"]
        or optimization.get("gradient_accumulation_steps") != 4
        or optimization.get("generation_batch_size") != 4
        or optimization.get("learning_rate") != 0.000005
        or optimization.get("beta") != 0.0
        or optimization.get("loss_type") != "dapo"
        or optimization.get("scale_rewards") != "none"
        or optimization.get("multi_objective_aggregation") != expected["aggregation"]
    ):
        raise ValueError("E3 optimization settings drifted from the controlled comparison")
    if optimization.get("checkpoint_steps") != expected["checkpoint_steps"]:
        if expected["checkpoint_steps"] is not None or "checkpoint_steps" in optimization:
            raise ValueError("E3 checkpoint schedule drifted")
    if config["reward"].get("terminal_metric") != "strict_execution_accuracy":
        raise ValueError("E3 terminal component must remain strict execution correctness")
    if (
        config["evidence"].get("maximum_session_wall_seconds") != expected["wall"]
        or config["evidence"].get("maximum_session_cost_cny") != expected["cost"]
    ):
        raise ValueError("E3 cost or wall gate drifted")
    if "component_telemetry" not in config["evidence"].get("required", []):
        raise ValueError("E3 requires component telemetry")
    if expected["prompt_count"] == 256:
        gate = config.get("signal_gate", {})
        if (
            gate.get("minimum_mixed_reward_groups") != 32
            or gate.get("minimum_mixed_reward_group_fraction") != 0.125
        ):
            raise ValueError("formal E3 signal gate drifted")
    if expected["arm"] == "E3":
        _require_e2_scientific_parity(config, formal=expected["prompt_count"] == 256)
    else:
        _require_e4_parity(config, formal=expected["prompt_count"] == 256)
    return config
