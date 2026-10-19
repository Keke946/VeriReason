from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import verireason.m6_component_scorer as scorer_module
from verireason.m6_component_scorer import SQLiteRewardComponentScorer
from verireason.reward_contract import group_rewards
from verireason.trl_reward_adapter import RewardInput, TRLGroupRewardAdapter


@pytest.fixture
def database_root(tmp_path: Path) -> Path:
    root = tmp_path / "databases"
    root.mkdir()
    connection = sqlite3.connect(root / "toy.sqlite")
    connection.executescript(
        """
        CREATE TABLE items(id INTEGER PRIMARY KEY, value INTEGER);
        INSERT INTO items VALUES (1, 1), (2, 2);
        """
    )
    connection.close()
    return root


def _adapter(root: Path, scorer=None, *, arm: str = "E2") -> TRLGroupRewardAdapter:
    return TRLGroupRewardAdapter(
        arm=arm,
        num_generations=4,
        component_scorer=scorer or SQLiteRewardComponentScorer(root),
        metadata_fields=("gold_sql", "database_path"),
    )


def _call(adapter: TRLGroupRewardAdapter, completions: list[str], gold_sql: str):
    return adapter(
        prompts=["return SQL"] * 4,
        completions=completions,
        completion_ids=[[index] for index in range(4)],
        reward_group_id=["example-1"] * 4,
        gold_sql=[gold_sql] * 4,
        database_path=["toy.sqlite"] * 4,
    )


def test_real_sqlite_components_flow_through_adapter(database_root: Path) -> None:
    scorer = SQLiteRewardComponentScorer(database_root)
    rewards = _call(
        _adapter(database_root, scorer),
        [
            "SELECT id FROM items ORDER BY id",
            "SELECT id FROM items WHERE value > 0 ORDER BY id",
            "SELECT id FROM items WHERE value > 1 ORDER BY id",
            "DROP TABLE items",
        ],
        "SELECT id FROM items ORDER BY id",
    )

    assert rewards == [1.0, 1.0, 0.0, 0.0]
    assert scorer.summary() == {
        "completion_count": 4,
        "strict_correct_count": 2,
        "valid_count": 3,
        "confidence_gate_count": 3,
        "mean_structure_score": pytest.approx(0.6021505376344085),
        "prediction_status_counts": {"ok": 3, "unsafe": 1},
    }
    connection = sqlite3.connect(database_root / "toy.sqlite")
    assert connection.execute("SELECT COUNT(*) FROM items").fetchone() == (2,)
    connection.close()


def test_real_sqlite_components_flow_through_e4_reward(database_root: Path) -> None:
    completions = [
        "SELECT id FROM items ORDER BY id",
        "SELECT id FROM items WHERE value > 0 ORDER BY id",
        "SELECT id FROM items WHERE value > 1 ORDER BY id",
        "DROP TABLE items",
    ]
    gold = "SELECT id FROM items ORDER BY id"
    actual_scorer = SQLiteRewardComponentScorer(database_root)
    rewards = _call(_adapter(database_root, actual_scorer, arm="E4"), completions, gold)

    expected_scorer = SQLiteRewardComponentScorer(database_root)
    components = [
        expected_scorer(
            RewardInput(
                prompt="return SQL",
                completion=completion,
                completion_ids=(index,),
                metadata={"gold_sql": gold, "database_path": "toy.sqlite"},
            )
        )
        for index, completion in enumerate(completions)
    ]
    assert rewards == pytest.approx(group_rewards(components, arm="E4"))
    assert sum(rewards) == pytest.approx(0.0, abs=1e-12)
    assert max(rewards) > min(rewards)


def test_gold_execution_is_reused_within_contiguous_group(database_root: Path, monkeypatch) -> None:
    calls: list[str] = []
    original = scorer_module.execute_read_only

    def counted(path, sql, limits):
        calls.append(sql)
        return original(path, sql, limits)

    monkeypatch.setattr(scorer_module, "execute_read_only", counted)
    gold = "SELECT id FROM items ORDER BY id"
    _call(_adapter(database_root), [gold] * 4, gold)

    assert calls.count(gold) == 5  # one Gold execution plus four completion executions


def test_unsupported_gold_structure_keeps_terminal_reward(database_root: Path) -> None:
    scorer = SQLiteRewardComponentScorer(database_root)
    adapter = _adapter(database_root, scorer)
    gold = "SELECT date('now')"

    rewards = _call(adapter, [gold] * 4, gold)
    components = scorer(
        RewardInput(
            prompt="p",
            completion=gold,
            completion_ids=(1,),
            metadata={"gold_sql": gold, "database_path": "toy.sqlite"},
        )
    )

    assert rewards == [1.0] * 4
    assert (components.r_exec, components.r_valid, components.r_struct, components.g_conf) == (
        1,
        1,
        0.0,
        0,
    )


def test_component_scorer_rejects_database_path_escape(database_root: Path) -> None:
    adapter = _adapter(database_root)
    with pytest.raises(ValueError, match="escapes database_root"):
        adapter(
            prompts=["p"] * 4,
            completions=["SELECT 1"] * 4,
            completion_ids=[[1]] * 4,
            reward_group_id=["a"] * 4,
            gold_sql=["SELECT 1"] * 4,
            database_path=["../outside.sqlite"] * 4,
        )


def test_component_summary_requires_real_evaluations(database_root: Path) -> None:
    with pytest.raises(RuntimeError, match="has not evaluated"):
        SQLiteRewardComponentScorer(database_root).summary()
