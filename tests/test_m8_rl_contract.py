import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from verireason.m7_runtime import grpo_argument_values
from verireason.m8_rl_contract import load_m8_rl_config
from verireason.posttraining_contract import load_posttraining_config

ROOT = Path(__file__).parents[1]


def _read_toml(name: str) -> dict:
    with (ROOT / "configs" / name).open("rb") as handle:
        return tomllib.load(handle)


@pytest.mark.parametrize(
    ("name", "count", "steps"),
    (("m8_e3_smoke.toml", 8, 8), ("m8_e3_formal.toml", 256, 256)),
)
def test_e3_configs_start_from_e1_and_preserve_e2_budget(name, count, steps) -> None:
    config = load_m8_rl_config(ROOT / "configs" / name)
    assert load_posttraining_config(ROOT / "configs" / name) == config
    assert config["arm"] == "E3"
    assert config["parent"]["adapter_experiment_id"] == "e1_m4_formal_sft_001"
    assert config["data"]["prompt_count"] == count
    assert config["optimization"]["maximum_optimizer_steps"] == steps
    assert config["optimization"]["multi_objective_aggregation"] == "sum_then_normalize"
    assert config["data"]["heldout_examples_loaded"] == 0
    assert "component_telemetry" in config["evidence"]["required"]


@pytest.mark.parametrize(
    ("e2_name", "e3_name"),
    (
        ("m7_e2_smoke.toml", "m8_e3_smoke.toml"),
        ("m7_e2_formal.toml", "m8_e3_formal.toml"),
    ),
)
def test_e3_changes_reward_arm_without_changing_scientific_budget(e2_name, e3_name) -> None:
    e2 = _read_toml(e2_name)
    e3 = _read_toml(e3_name)
    for section in ("parent", "data", "generation", "optimization", "reward", "stops"):
        assert e3[section] == e2[section]
    if "signal_gate" in e2:
        assert e3["signal_gate"] == e2["signal_gate"]
        assert e3["reward_hacking_review"] == e2["reward_hacking_review"]


def test_e3_rejects_e2_parent_or_per_reward_aggregation(tmp_path: Path) -> None:
    source = (ROOT / "configs/m8_e3_smoke.toml").read_text()
    bad_parent = tmp_path / "bad-parent.toml"
    bad_parent.write_text(
        source.replace(
            'adapter_experiment_id = "e1_m4_formal_sft_001"',
            'adapter_experiment_id = "e2_m7_outcome_grpo_formal_001"',
        )
    )
    with pytest.raises(ValueError, match="immutable E1"):
        load_m8_rl_config(bad_parent)
    bad_aggregation = tmp_path / "bad-aggregation.toml"
    bad_aggregation.write_text(
        source.replace(
            'multi_objective_aggregation = "sum_then_normalize"',
            'multi_objective_aggregation = "normalize_then_sum"',
        )
    )
    with pytest.raises(ValueError, match="optimization settings"):
        load_m8_rl_config(bad_aggregation)


def test_e3_rejects_prompt_or_budget_drift(tmp_path: Path) -> None:
    source = (ROOT / "configs/m8_e3_formal.toml").read_text()
    for old, new, message in (
        ("prompt_count = 256", "prompt_count = 255", "prompt_count"),
        ("maximum_optimizer_steps = 256", "maximum_optimizer_steps = 255", "optimization"),
        ('selector_seed = "verireason-m7-e2-formal-v1"', 'selector_seed = "new"', "selector_seed"),
    ):
        path = tmp_path / f"bad-{message}.toml"
        path.write_text(source.replace(old, new))
        with pytest.raises(ValueError, match=message):
            load_m8_rl_config(path)


@pytest.mark.parametrize(
    ("e3_name", "e4_name", "count", "steps"),
    (
        ("m8_e3_smoke.toml", "m8_e4_smoke.toml", 8, 8),
        ("m8_e3_formal.toml", "m8_e4_formal.toml", 256, 256),
    ),
)
def test_e4_changes_only_registered_aggregation_intervention(
    e3_name: str, e4_name: str, count: int, steps: int
) -> None:
    e3 = _read_toml(e3_name)
    e4 = _read_toml(e4_name)
    normalized = deepcopy(e4)
    normalized["experiment_id"] = e3["experiment_id"]
    normalized["arm"] = e3["arm"]
    normalized["optimization"]["multi_objective_aggregation"] = e3["optimization"][
        "multi_objective_aggregation"
    ]
    assert normalized == e3

    config = load_m8_rl_config(ROOT / "configs" / e4_name)
    assert load_posttraining_config(ROOT / "configs" / e4_name) == config
    assert config["arm"] == "E4"
    assert config["data"]["prompt_count"] == count
    assert config["optimization"]["maximum_optimizer_steps"] == steps
    assert config["optimization"]["multi_objective_aggregation"] == "normalize_then_sum"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'adapter_experiment_id = "e1_m4_formal_sft_001"',
            'adapter_experiment_id = "e2_m7_outcome_grpo_formal_001"',
            "immutable E1",
        ),
        ("prompt_count = 8", "prompt_count = 7", "prompt_count"),
        ("learning_rate = 0.000005", "learning_rate = 0.000006", "optimization"),
        (
            'component_scorer = "SQLiteRewardComponentScorer"',
            'component_scorer = "OtherScorer"',
            "reward",
        ),
        ('terminal_metric = "strict_execution_accuracy"', 'terminal_metric = "other"', "terminal"),
    ),
)
def test_e4_rejects_scientific_drift(tmp_path: Path, old: str, new: str, message: str) -> None:
    source = (ROOT / "configs/m8_e4_smoke.toml").read_text()
    path = tmp_path / "bad-e4.toml"
    path.write_text(source.replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_m8_rl_config(path)


def test_e4_rejects_extra_optimizer_drift(tmp_path: Path) -> None:
    source = (ROOT / "configs/m8_e4_smoke.toml").read_text()
    path = tmp_path / "bad-e4-extra.toml"
    path.write_text(source.replace("seed = 23", "seed = 23\nweight_decay = 0.1"))
    with pytest.raises(ValueError, match="optimization"):
        load_m8_rl_config(path)


@pytest.mark.parametrize("name", ("m8_e4_smoke.toml", "m8_e4_formal.toml"))
def test_pinned_trl_accepts_e4_aggregation(tmp_path: Path, name: str) -> None:
    trl = pytest.importorskip("trl")
    config = load_m8_rl_config(ROOT / "configs" / name)
    arguments = trl.GRPOConfig(**grpo_argument_values(config, str(tmp_path / "output")))
    assert arguments.multi_objective_aggregation == "normalize_then_sum"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("checkpoint_steps = 64", "checkpoint_steps = 32", "checkpoint schedule"),
        ("minimum_mixed_reward_groups = 32", "minimum_mixed_reward_groups = 31", "signal gate"),
        ("prompt_count = 256", "prompt_count = 255", "prompt_count"),
    ),
)
def test_e4_formal_rejects_budget_or_gate_drift(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    source = (ROOT / "configs/m8_e4_formal.toml").read_text()
    path = tmp_path / "bad-e4-formal.toml"
    path.write_text(source.replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_m8_rl_config(path)
