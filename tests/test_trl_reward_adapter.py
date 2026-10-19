from __future__ import annotations

import pytest

from verireason.reward_contract import RewardComponents, group_rewards
from verireason.trl_reward_adapter import RewardInput, TRLGroupRewardAdapter

COMPONENTS = {
    "wrong": RewardComponents(0, 1, 0.0, 1),
    "close": RewardComponents(0, 1, 0.8, 1),
    "correct": RewardComponents(1, 1, 1.0, 1),
    "gated": RewardComponents(0, 0, 0.0, 0),
}


def _scorer(item: RewardInput) -> RewardComponents:
    assert item.metadata["gold_sql"] == "SELECT 1"
    return COMPONENTS[item.completion]


def _call(adapter: TRLGroupRewardAdapter, completions: list[str], **overrides):
    size = len(completions)
    kwargs = {
        "prompts": ["prompt-a"] * 4 + ["prompt-b"] * 4,
        "completions": completions,
        "completion_ids": [[index] for index in range(size)],
        "reward_group_id": ["a"] * 4 + ["b"] * 4,
        "gold_sql": ["SELECT 1"] * size,
    }
    kwargs.update(overrides)
    return adapter(**kwargs)


@pytest.mark.parametrize("arm", ["E2", "E3", "E4"])
def test_adapter_matches_framework_independent_group_math(arm: str) -> None:
    adapter = TRLGroupRewardAdapter(
        arm=arm,
        num_generations=4,
        component_scorer=_scorer,
        metadata_fields=("gold_sql",),
    )
    names = ["wrong", "close", "correct", "gated", "correct", "gated", "close", "wrong"]
    expected = [
        *group_rewards([COMPONENTS[name] for name in names[:4]], arm=arm),
        *group_rewards([COMPONENTS[name] for name in names[4:]], arm=arm),
    ]
    assert _call(adapter, names) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reward_group_id": ["a"] * 3 + ["b"] * 5}, "contiguous and constant"),
        ({"reward_group_id": ["a"] * 4 + ["a"] * 4}, "multiple blocks"),
        ({"gold_sql": ["SELECT 1"] * 7}, "missing or misaligned"),
        ({"prompts": ["prompt"] * 7}, "misaligned"),
    ],
)
def test_adapter_rejects_group_or_metadata_misalignment(overrides, message: str) -> None:
    adapter = TRLGroupRewardAdapter(
        arm="E4",
        num_generations=4,
        component_scorer=_scorer,
        metadata_fields=("gold_sql",),
    )
    with pytest.raises(ValueError, match=message):
        _call(adapter, ["wrong"] * 8, **overrides)


def test_adapter_propagates_component_scorer_failure() -> None:
    def fail_on_close(item: RewardInput) -> RewardComponents:
        if item.completion == "close":
            raise RuntimeError("evaluator failed")
        return COMPONENTS[item.completion]

    adapter = TRLGroupRewardAdapter(
        arm="E3",
        num_generations=4,
        component_scorer=fail_on_close,
        metadata_fields=("gold_sql",),
    )
    with pytest.raises(RuntimeError, match="evaluator failed"):
        _call(adapter, ["wrong", "close", "correct", "gated"] * 2)


def test_adapter_rejects_incomplete_group_and_conversational_completion() -> None:
    adapter = TRLGroupRewardAdapter(
        arm="E2",
        num_generations=4,
        component_scorer=_scorer,
        metadata_fields=("gold_sql",),
    )
    with pytest.raises(ValueError, match="complete generation groups"):
        adapter(
            prompts=["p"] * 3,
            completions=["wrong"] * 3,
            completion_ids=[[1]] * 3,
            reward_group_id=["a"] * 3,
            gold_sql=["SELECT 1"] * 3,
        )
    with pytest.raises(TypeError, match="standard string"):
        _call(adapter, [[{"role": "assistant", "content": "wrong"}]] * 8)
    with pytest.raises(TypeError, match="integer token IDs"):
        _call(adapter, ["wrong"] * 8, completion_ids=[[1.5]] * 8)


def test_adapter_rejects_invalid_configuration_and_scorer_output() -> None:
    with pytest.raises(TypeError, match="sequence of field names"):
        TRLGroupRewardAdapter(
            arm="E2", num_generations=4, component_scorer=_scorer, metadata_fields="gold_sql"
        )
    adapter = TRLGroupRewardAdapter(
        arm="E2",
        num_generations=4,
        component_scorer=lambda _: 1.0,
        metadata_fields=("gold_sql",),
    )
    with pytest.raises(TypeError, match="RewardComponents"):
        _call(adapter, ["wrong"] * 8)
