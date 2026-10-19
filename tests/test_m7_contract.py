from pathlib import Path

import pytest

from verireason.m7_contract import (
    load_m7_config,
    load_m7_formal_config,
    load_m7_smoke_config,
    select_smoke_rows,
    summarize_group_rewards,
)

ROOT = Path(__file__).parents[1]


def test_m7_smoke_config_is_minimal_and_unapproved() -> None:
    config = load_m7_smoke_config(ROOT / "configs/m7_e2_smoke.toml")
    assert config["data"]["prompt_count"] * config["generation"]["num_generations"] == 32
    assert config["optimization"]["maximum_optimizer_steps"] == 8
    assert config["data"]["heldout_examples_loaded"] == 0


def test_m7_config_rejects_double_reward_normalization(tmp_path: Path) -> None:
    source = (ROOT / "configs/m7_e2_smoke.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(source.replace('scale_rewards = "none"', 'scale_rewards = "group"'))
    with pytest.raises(ValueError, match="second time"):
        load_m7_smoke_config(path)


def test_m7_formal_config_freezes_bounded_signal_aware_run() -> None:
    path = ROOT / "configs/m7_e2_formal.toml"
    config = load_m7_formal_config(path)
    assert load_m7_config(path) == config
    assert config["data"]["prompt_count"] == 256
    assert config["generation"]["num_generations"] == 4
    assert config["optimization"]["maximum_optimizer_steps"] == 256
    assert config["optimization"]["checkpoint_steps"] == 64
    assert config["signal_gate"]["minimum_mixed_reward_groups"] == 32
    assert config["data"]["heldout_examples_loaded"] == 0


def test_m7_formal_config_rejects_smaller_signal_gate(tmp_path: Path) -> None:
    source = (ROOT / "configs/m7_e2_formal.toml").read_text()
    path = tmp_path / "bad-formal.toml"
    path.write_text(
        source.replace("minimum_mixed_reward_groups = 32", "minimum_mixed_reward_groups = 8")
    )
    with pytest.raises(ValueError, match="at least 32"):
        load_m7_formal_config(path)


def test_selection_is_deterministic_training_only_and_covers_features() -> None:
    sql_by_feature = {
        "join": "SELECT * FROM a JOIN b ON a.id = b.id",
        "aggregate": "SELECT COUNT(*) FROM a",
        "subquery": "SELECT * FROM a WHERE id IN (SELECT id FROM b)",
        "": "SELECT id FROM a",
    }
    rows = [
        {
            "row_id": str(i),
            "db_id": f"db-{i}",
            "split": "train",
            "operational_eligible": True,
            "source": "training",
            "sql": sql_by_feature[feature[0] if feature else ""],
        }
        for i, feature in enumerate((["join"], ["aggregate"], ["subquery"], [], [], [], [], []))
    ]
    rows += [
        {"row_id": "validation", "db_id": "x", "split": "validation", "operational_eligible": True},
        {
            "row_id": "heldout",
            "db_id": "y",
            "split": "train",
            "operational_eligible": True,
            "source": "heldout",
        },
    ]
    one = select_smoke_rows(
        rows, count=8, seed="fixed", required_features=["join", "aggregate", "subquery"]
    )
    two = select_smoke_rows(
        list(reversed(rows)),
        count=8,
        seed="fixed",
        required_features=["join", "aggregate", "subquery"],
    )
    assert [row["row_id"] for row in one] == [row["row_id"] for row in two]
    selected_sql = " ".join(row["sql"] for row in one)
    assert "JOIN" in selected_sql and "COUNT" in selected_sql and "SELECT id FROM b" in selected_sql


def test_reward_group_diagnostics_distinguish_informative_groups() -> None:
    result = summarize_group_rewards([0, 0, 1, 1, 1, 1, 1, 1], num_generations=4)
    assert result == {
        "groups": 2,
        "mixed_reward_groups": 1,
        "zero_variance_groups": 1,
        "mixed_reward_group_fraction": 0.5,
        "mean_reward": 0.75,
    }


def test_reward_group_diagnostics_reject_partial_group() -> None:
    with pytest.raises(ValueError, match="complete groups"):
        summarize_group_rewards([0, 1, 0], num_generations=4)
