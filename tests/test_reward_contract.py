from itertools import combinations_with_replacement

import pytest

from verireason.reward_contract import (
    RewardComponents,
    active_dimension_structural_reward,
    group_rewards,
    population_zscores,
    raw_utility,
)

WEIGHTS = {"select": 0.2, "tables": 0.1, "joins": 0.15}


def test_empty_dimensions_are_inactive_and_all_empty_is_zero() -> None:
    assert active_dimension_structural_reward({}, {}, WEIGHTS) == 0.0
    assert (
        active_dimension_structural_reward(
            {"tables": {"users"}},
            {"tables": {"users"}},
            WEIGHTS,
        )
        == 1.0
    )


def test_extra_structure_reduces_precision_and_repetition_cannot_help() -> None:
    extra = active_dimension_structural_reward(
        {"tables": {"users"}},
        {"tables": ["users", "orders"]},
        WEIGHTS,
    )
    repeated = active_dimension_structural_reward(
        {"tables": {"users"}},
        {"tables": ["users", "users", "users"]},
        WEIGHTS,
    )

    assert extra == pytest.approx(2 / 3)
    assert repeated == 1.0


def test_only_active_dimension_weights_are_normalized() -> None:
    score = active_dimension_structural_reward(
        {"select": {"column"}, "tables": {"users"}},
        {"select": {"column"}, "tables": {"orders"}},
        WEIGHTS,
    )

    assert score == pytest.approx(0.2 / 0.3)


def test_unknown_nonempty_dimension_fails() -> None:
    with pytest.raises(ValueError, match="unknown non-empty"):
        active_dimension_structural_reward({"invented": {"x"}}, {}, WEIGHTS)


def test_raw_utility_preserves_correct_incorrect_margin() -> None:
    maximum_incorrect = raw_utility(r_exec=0, r_valid=1, r_struct=1.0, g_conf=1)
    minimum_correct = raw_utility(r_exec=1, r_valid=0, r_struct=0.0, g_conf=0)

    assert maximum_incorrect == 0.25
    assert minimum_correct == 1.0
    assert minimum_correct - maximum_incorrect == 0.75


def test_population_zscores_use_population_variance_and_zero_constant_groups() -> None:
    assert population_zscores([3.0, 3.0, 3.0]) == (0.0, 0.0, 0.0)
    observed = population_zscores([0.0, 1.0])
    assert observed == pytest.approx((-1.0, 1.0))
    assert sum(observed) == pytest.approx(0.0)


def test_outcome_only_has_no_signal_when_every_completion_is_wrong() -> None:
    group = [
        RewardComponents(r_exec=0, r_valid=1, r_struct=0.0, g_conf=1),
        RewardComponents(r_exec=0, r_valid=1, r_struct=1.0, g_conf=1),
    ]

    assert group_rewards(group, arm="E2") == (0.0, 0.0)
    assert group_rewards(group, arm="E3") == pytest.approx((-1.0, 1.0))


def test_e3_and_e4_isolate_different_normalization_rules() -> None:
    group = [
        RewardComponents(r_exec=0, r_valid=1, r_struct=0.0, g_conf=1),
        RewardComponents(r_exec=0, r_valid=1, r_struct=1.0, g_conf=1),
        RewardComponents(r_exec=1, r_valid=1, r_struct=0.5, g_conf=1),
    ]

    e3 = group_rewards(group, arm="E3")
    e4 = group_rewards(group, arm="E4")

    assert e3 != pytest.approx(e4)
    assert sum(e3) == pytest.approx(0.0)
    assert sum(e4) == pytest.approx(0.0)


def test_confidence_gate_blocks_dense_signal_and_inputs_fail_closed() -> None:
    gated = [
        RewardComponents(r_exec=0, r_valid=1, r_struct=1.0, g_conf=0),
        RewardComponents(r_exec=0, r_valid=0, r_struct=0.0, g_conf=0),
    ]
    assert group_rewards(gated, arm="E3") == (0.0, 0.0)
    assert group_rewards(gated, arm="E4") == (0.0, 0.0)

    with pytest.raises(ValueError, match="must not be empty"):
        population_zscores([])
    with pytest.raises(ValueError, match="finite"):
        population_zscores([float("nan")])
    with pytest.raises(ValueError, match="unknown reward arm"):
        group_rewards(gated, arm="invented")


def test_e4_terminal_correctness_dominates_for_registered_group_sizes() -> None:
    structural_values = (0.0, 0.25, 0.5, 0.75, 1.0)
    states = (
        RewardComponents(0, 0, 0.0, 0),
        RewardComponents(0, 1, 0.0, 0),
        *(RewardComponents(0, 1, value, 1) for value in structural_values),
        RewardComponents(1, 1, 0.0, 0),
        *(RewardComponents(1, 1, value, 1) for value in structural_values),
    )

    for group_size in (2, 4, 8):
        for group in combinations_with_replacement(states, group_size):
            if {item.r_exec for item in group} != {0, 1}:
                continue
            rewards = group_rewards(group, arm="E4")
            correct = [value for item, value in zip(group, rewards, strict=True) if item.r_exec]
            wrong = [value for item, value in zip(group, rewards, strict=True) if not item.r_exec]
            assert min(correct) > max(wrong)
