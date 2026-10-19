from copy import deepcopy

import pytest

from verireason.m7_contract import load_m7_smoke_config
from verireason.m8_rl_contract import load_m8_rl_config
from verireason.posttraining_result import close_posttraining_result


@pytest.fixture
def evidence():
    config = load_m8_rl_config("configs/m8_e3_smoke.toml")
    adapter_sha = "a" * 64
    training = {
        "schema_version": 1,
        "status": "trained",
        "experiment_id": config["experiment_id"],
        "arm": "E3",
        "optimizer_steps": 8,
        "training_loss": -0.01,
        "gradient_norm_last": 0.0,
        "gradient_norm_max": 0.7,
        "adapter_delta_l2": 0.1,
        "reward_diagnostics": {
            "groups": 8,
            "mixed_reward_groups": 5,
            "zero_variance_groups": 3,
            "mixed_reward_group_fraction": 0.625,
            "mean_group_relative_reward": 0.0,
        },
        "component_diagnostics": {
            "completion_count": 32,
            "strict_correct_count": 8,
            "valid_count": 24,
            "confidence_gate_count": 23,
            "mean_structure_score": 0.4,
            "prediction_status_counts": {"ok": 24, "execution_error": 8},
        },
        "peak_gpu_memory_mib": 12000.0,
        "wall_seconds": 200.0,
        "adapter_manifest_sha256": adapter_sha,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    reload = {
        "schema_version": 1,
        "status": "passed",
        "adapter_manifest_sha256": adapter_sha,
        "adapter_names": ["default"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    return config, training, reload


def test_close_e3_smoke_returns_sql_free_aggregate(evidence) -> None:
    config, training, reload = evidence
    summary = close_posttraining_result(config, training, reload, session_wall_seconds=300.0)
    assert summary["status"] == "passed"
    assert summary["arm"] == "E3"
    assert summary["rollouts"] == 32
    assert summary["mean_group_relative_reward"] == 0.0
    assert "mean_terminal_reward" not in summary
    assert summary["separate_process_reload_passed"] is True


def test_close_e4_smoke_returns_centered_sql_free_aggregate(evidence) -> None:
    _, training, reload = deepcopy(evidence)
    config = load_m8_rl_config("configs/m8_e4_smoke.toml")
    training["experiment_id"] = config["experiment_id"]
    training["arm"] = "E4"
    summary = close_posttraining_result(config, training, reload, session_wall_seconds=300.0)
    assert summary["status"] == "passed"
    assert summary["arm"] == "E4"
    assert summary["multi_objective_aggregation"] == "normalize_then_sum"
    assert summary["mean_group_relative_reward"] == 0.0
    assert "mean_terminal_reward" not in summary


def test_close_e4_rejects_terminal_label_or_noncentered_reward(evidence) -> None:
    _, training, reload = deepcopy(evidence)
    config = load_m8_rl_config("configs/m8_e4_smoke.toml")
    training["experiment_id"] = config["experiment_id"]
    training["arm"] = "E4"
    training["reward_diagnostics"]["mean_terminal_reward"] = 0.25
    with pytest.raises(ValueError, match="E4 dense reward was mislabeled"):
        close_posttraining_result(config, training, reload, session_wall_seconds=300.0)
    training["reward_diagnostics"].pop("mean_terminal_reward")
    training["reward_diagnostics"]["mean_group_relative_reward"] = 0.1
    with pytest.raises(ValueError, match="E4 group-normalized"):
        close_posttraining_result(config, training, reload, session_wall_seconds=300.0)


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    (
        ("training", "arm", "E2", "identity"),
        ("training", "optimizer_steps", 7, "step count"),
        ("training", "adapter_delta_l2", 0.0, "adapter delta"),
        ("reload", "adapter_manifest_sha256", "b" * 64, "identities"),
        ("reload", "heldout_examples_loaded", 1, "heldout"),
    ),
)
def test_close_rejects_inconsistent_direct_evidence(evidence, target, key, value, message) -> None:
    config, training, reload = deepcopy(evidence)
    (training if target == "training" else reload)[key] = value
    with pytest.raises(ValueError, match=message):
        close_posttraining_result(config, training, reload, session_wall_seconds=300.0)


def test_close_rejects_mislabeled_or_noncentered_e3_reward(evidence) -> None:
    config, training, reload = evidence
    mislabeled = deepcopy(training)
    mislabeled["reward_diagnostics"]["mean_terminal_reward"] = 0.5
    with pytest.raises(ValueError, match="mislabeled"):
        close_posttraining_result(config, mislabeled, reload, session_wall_seconds=300.0)
    noncentered = deepcopy(training)
    noncentered["reward_diagnostics"]["mean_group_relative_reward"] = 0.1
    with pytest.raises(ValueError, match="approximately zero"):
        close_posttraining_result(config, noncentered, reload, session_wall_seconds=300.0)


def test_close_rejects_incomplete_component_telemetry(evidence) -> None:
    config, training, reload = evidence
    training["component_diagnostics"]["prediction_status_counts"] = {"ok": 31}
    with pytest.raises(ValueError, match="status counts"):
        close_posttraining_result(config, training, reload, session_wall_seconds=300.0)


def test_close_preserves_e2_terminal_reward_semantics(evidence) -> None:
    _, training, reload = deepcopy(evidence)
    config = load_m7_smoke_config("configs/m7_e2_smoke.toml")
    training["experiment_id"] = config["experiment_id"]
    training["arm"] = "E2"
    training["reward_diagnostics"].pop("mean_group_relative_reward")
    training["reward_diagnostics"]["mean_terminal_reward"] = 0.25
    summary = close_posttraining_result(config, training, reload, session_wall_seconds=300.0)
    assert summary["mean_terminal_reward"] == 0.25
    assert summary["signal_gate_passed"] is None

    training["reward_diagnostics"]["mean_terminal_reward"] = 0.5
    with pytest.raises(ValueError, match="strict-correct"):
        close_posttraining_result(config, training, reload, session_wall_seconds=300.0)
