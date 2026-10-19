"""Runtime helpers shared by the controlled E2/E3/E4 post-training runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from verireason.m7_contract import summarize_group_rewards


def grpo_argument_values(config: Mapping[str, Any], output_dir: str) -> dict[str, Any]:
    """Translate the reviewed TOML contract to pinned TRL 1.9 argument names."""
    generation, opt = config["generation"], config["optimization"]
    return {
        "output_dir": output_dir,
        "max_steps": opt["maximum_optimizer_steps"],
        "per_device_train_batch_size": opt["per_device_train_batch_size"],
        "gradient_accumulation_steps": opt["gradient_accumulation_steps"],
        "generation_batch_size": opt["generation_batch_size"],
        "num_generations": generation["num_generations"],
        "max_completion_length": generation["max_completion_length"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": generation["top_k"],
        "use_vllm": generation["use_vllm"],
        "learning_rate": opt["learning_rate"],
        "optim": opt["optimizer"],
        "lr_scheduler_type": opt["scheduler"],
        "beta": opt["beta"],
        "epsilon": opt["epsilon"],
        "epsilon_high": opt["epsilon_high"],
        "loss_type": opt["loss_type"],
        "scale_rewards": opt["scale_rewards"],
        "multi_objective_aggregation": opt["multi_objective_aggregation"],
        "mask_truncated_completions": opt["mask_truncated_completions"],
        "num_iterations": opt["num_iterations"],
        "gradient_checkpointing": opt["gradient_checkpointing"],
        "bf16": True,
        "seed": opt["seed"],
        "data_seed": opt["seed"],
        "shuffle_dataset": False,
        "remove_unused_columns": False,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": opt.get("checkpoint_steps", opt["maximum_optimizer_steps"]),
        "save_total_limit": 1,
        "report_to": "none",
        "auto_find_batch_size": False,
    }


class RewardEvidenceRecorder:
    """Wrap the M6 adapter while retaining aggregate, SQL-free group diagnostics."""

    def __init__(
        self,
        reward_function: Any,
        num_generations: int,
        *,
        metric_namespace: str = "m7",
        mean_reward_name: str = "mean_terminal_reward",
    ) -> None:
        if not metric_namespace or not mean_reward_name:
            raise ValueError("reward evidence metric names must be non-empty")
        self.reward_function = reward_function
        self.num_generations = num_generations
        self.metric_namespace = metric_namespace
        self.mean_reward_name = mean_reward_name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[float]:
        rewards = self.reward_function(**kwargs)
        if any(not math.isfinite(value) for value in rewards):
            raise RuntimeError("reward callback returned NaN or Inf")
        self.calls.append(summarize_group_rewards(rewards, num_generations=self.num_generations))
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            summary = self.calls[-1]
            log_metric(
                f"{self.metric_namespace}/mixed_reward_group_fraction",
                summary["mixed_reward_group_fraction"],
            )
            log_metric(f"{self.metric_namespace}/{self.mean_reward_name}", summary["mean_reward"])
        return rewards

    def summary(self) -> dict[str, Any]:
        if not self.calls:
            raise RuntimeError("reward callback was never invoked")
        groups = sum(item["groups"] for item in self.calls)
        mixed = sum(item["mixed_reward_groups"] for item in self.calls)
        weighted_reward = sum(item["mean_reward"] * item["groups"] for item in self.calls)
        summary = {
            "callback_calls": len(self.calls),
            "groups": groups,
            "mixed_reward_groups": mixed,
            "zero_variance_groups": groups - mixed,
            "mixed_reward_group_fraction": mixed / groups,
        }
        summary[self.mean_reward_name] = weighted_reward / groups
        return summary


def trainable_parameter_snapshot(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def trainable_delta_l2(before: Mapping[str, Any], model: Any) -> float:
    current = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(current) != set(before) or not current:
        raise RuntimeError("trainable parameter identity changed or is empty")
    squared = 0.0
    for name, parameter in current.items():
        delta = parameter.detach().float().cpu() - before[name]
        squared += float((delta * delta).sum().item())
    return math.sqrt(squared)


def summarize_trainer_diagnostics(
    log_history: Sequence[Mapping[str, Any]], metric_names: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    """Compress trainer logs into SQL-free finite metric ranges."""
    output: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = [
            float(row[name])
            for row in log_history
            if isinstance(row.get(name), int | float) and math.isfinite(float(row[name]))
        ]
        if values:
            output[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
            }
    return output


def passes_signal_gate(diagnostics: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    """Apply the preregistered post-run informative-group promotion boundary."""
    return (
        diagnostics["mixed_reward_groups"] >= gate["minimum_mixed_reward_groups"]
        and diagnostics["mixed_reward_group_fraction"]
        >= gate["minimum_mixed_reward_group_fraction"]
    )


def validate_smoke_bundle(bundle: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    examples = bundle.get("examples")
    count = config["data"]["prompt_count"]
    if bundle.get("schema_version") != 1 or not isinstance(examples, Sequence):
        raise ValueError("invalid post-training prepared bundle")
    if len(examples) != count or len(bundle.get("sample_ids", [])) != count:
        raise ValueError("post-training prepared bundle count drifted")
    if bundle.get("heldout_examples_loaded") != 0 or bundle.get("heldout_examples_evaluated") != 0:
        raise ValueError("post-training bundle contains heldout evidence")
    required = {"reward_group_id", "messages", "gold_sql", "database_path"}
    if any(required - row.keys() for row in examples):
        raise ValueError("post-training prepared row lacks reward metadata")
