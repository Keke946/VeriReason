from types import SimpleNamespace

import pytest

from verireason.m7_contract import load_m7_formal_config, load_m7_smoke_config
from verireason.m7_runtime import (
    RewardEvidenceRecorder,
    grpo_argument_values,
    passes_signal_gate,
    summarize_trainer_diagnostics,
    trainable_delta_l2,
    validate_smoke_bundle,
)


def test_grpo_arguments_preserve_project_reward_and_group_math() -> None:
    config = load_m7_smoke_config("configs/m7_e2_smoke.toml")
    values = grpo_argument_values(config, "/tmp/output")
    assert values["scale_rewards"] == "none"
    assert values["generation_batch_size"] == values["num_generations"] == 4
    assert values["gradient_accumulation_steps"] == 4
    assert values["max_steps"] == 8
    assert values["use_vllm"] is False


def test_formal_grpo_arguments_checkpoint_every_64_updates() -> None:
    config = load_m7_formal_config("configs/m7_e2_formal.toml")
    values = grpo_argument_values(config, "/tmp/output")
    assert values["max_steps"] == 256
    assert values["save_steps"] == 64
    assert values["num_generations"] == 4
    assert values["scale_rewards"] == "none"


def test_reward_recorder_aggregates_without_private_content() -> None:
    calls = iter(([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]))
    logged = []
    recorder = RewardEvidenceRecorder(lambda **kwargs: list(next(calls)), 4)
    recorder(log_metric=lambda name, value: logged.append((name, value)))
    recorder(log_metric=lambda name, value: logged.append((name, value)))
    assert recorder.summary() == {
        "callback_calls": 2,
        "groups": 2,
        "mixed_reward_groups": 1,
        "zero_variance_groups": 1,
        "mixed_reward_group_fraction": 0.5,
        "mean_terminal_reward": 0.75,
    }
    assert len(logged) == 4


def test_reward_recorder_rejects_nonfinite() -> None:
    recorder = RewardEvidenceRecorder(lambda **kwargs: [0.0, float("nan"), 0.0, 1.0], 4)
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        recorder()


def test_reward_recorder_names_dense_reward_evidence_without_calling_it_terminal() -> None:
    logged = []
    recorder = RewardEvidenceRecorder(
        lambda **kwargs: [-1.0, -0.5, 0.5, 1.0],
        4,
        metric_namespace="m8",
        mean_reward_name="mean_group_relative_reward",
    )
    recorder(log_metric=lambda name, value: logged.append((name, value)))
    assert recorder.summary()["mean_group_relative_reward"] == 0.0
    assert "mean_terminal_reward" not in recorder.summary()
    assert {name for name, _ in logged} == {
        "m8/mixed_reward_group_fraction",
        "m8/mean_group_relative_reward",
    }


def test_trainer_diagnostics_are_finite_aggregate_only() -> None:
    summary = summarize_trainer_diagnostics(
        [
            {"entropy": 0.4, "completions/clipped_ratio": 0.0},
            {"entropy": 0.2, "completions/clipped_ratio": 0.25, "private": "sql"},
            {"entropy": float("nan")},
        ],
        ("entropy", "completions/clipped_ratio"),
    )
    assert summary == {
        "entropy": {
            "count": 2,
            "min": 0.2,
            "max": 0.4,
            "mean": 0.30000000000000004,
            "last": 0.2,
        },
        "completions/clipped_ratio": {
            "count": 2,
            "min": 0.0,
            "max": 0.25,
            "mean": 0.125,
            "last": 0.25,
        },
    }


@pytest.mark.parametrize(
    ("mixed", "fraction", "expected"),
    ((32, 0.125, True), (31, 0.5, False), (64, 0.124, False)),
)
def test_formal_signal_gate_uses_both_boundaries(mixed, fraction, expected) -> None:
    assert (
        passes_signal_gate(
            {"mixed_reward_groups": mixed, "mixed_reward_group_fraction": fraction},
            {
                "minimum_mixed_reward_groups": 32,
                "minimum_mixed_reward_group_fraction": 0.125,
            },
        )
        is expected
    )


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def __sub__(self, other):
        return FakeTensor(self.value - other.value)

    def __mul__(self, other):
        return FakeTensor(self.value * other.value)

    def sum(self):
        return self

    def item(self):
        return self.value


def test_trainable_delta_requires_change() -> None:
    parameter = SimpleNamespace(requires_grad=True, detach=lambda: FakeTensor(4.0))
    model = SimpleNamespace(named_parameters=lambda: [("adapter", parameter)])
    assert trainable_delta_l2({"adapter": FakeTensor(1.0)}, model) == 3.0


def test_validate_smoke_bundle_rejects_heldout() -> None:
    config = load_m7_smoke_config("configs/m7_e2_smoke.toml")
    row = {
        "reward_group_id": "x",
        "messages": [],
        "gold_sql": "SELECT 1",
        "database_path": "x.sqlite",
    }
    bundle = {
        "schema_version": 1,
        "sample_ids": [str(i) for i in range(8)],
        "examples": [row] * 8,
        "heldout_examples_loaded": 1,
        "heldout_examples_evaluated": 0,
    }
    with pytest.raises(ValueError, match="heldout"):
        validate_smoke_bundle(bundle, config)
