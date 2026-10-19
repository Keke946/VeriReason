from pathlib import Path

import pytest

from verireason.config import load_m2_config, load_smoke_config

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_has_preferred_and_fallback() -> None:
    config = load_smoke_config(ROOT / "configs/m1_smoke.toml")

    assert config.model("qwen3_4b_base").role == "preferred"
    assert config.model("qwen3_1_7b_base").role == "fallback"
    assert len(config.cases) == 4


def test_unknown_model_key_fails() -> None:
    config = load_smoke_config(ROOT / "configs/m1_smoke.toml")

    with pytest.raises(ValueError, match="resolve exactly once"):
        config.model("missing")


def test_m2_config_freezes_sources_metrics_and_reward_weights() -> None:
    config = load_m2_config(ROOT / "configs/m2_data.toml")

    assert config["training_source"]["revision"] == "4068469807b255fcfc0816bdd520946fe460d256"
    assert config["heldout_source"]["expected_rows"] == 500
    assert config["evaluator"]["metrics"] == [
        "bird_ex_behavior_compatible",
        "strict_execution_accuracy",
    ]
    assert config["reward"]["w_struct"] == 0.20
    assert config["reward"]["structural_normalization"] == "active_dimensions_only"
    assert config["reward"]["all_structural_dimensions_empty_score"] == 0.0
    assert config["evaluator"]["maximum_float_pair_checks"] == 1_000_000
    assert config["evaluator"]["timeout_seconds"] == 10.0
    assert config["evaluator"]["maximum_cpu_seconds"] == 12
    assert config["evaluator"]["maximum_cpu_hard_seconds"] == 13
    assert config["evaluator"]["maximum_result_rows"] == 100_000
    assert config["evaluator"]["maximum_result_bytes"] == 64 * 1024**2
    assert config["evaluator"]["maximum_vm_bytes"] == 2 * 1024**3
    assert config["materialization"]["minimum_free_bytes"] == 70 * 1024**3
    assert config["materialization"]["dynamic_capacity_reserve_bytes"] == 10 * 1024**3
    assert config["training_source"]["database_nested_archive_path"] == (
        "train/train_databases.zip"
    )
    assert config["training_source"]["database_archive_observed_sha256"] == (
        "66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc"
    )
    assert (
        config["training_source"]["database_nested_archive_maximum_uncompressed_bytes"]
        == 50 * 1024**3
    )


def test_diagnostic_profile_b_limits_do_not_enter_formal_evaluator() -> None:
    formal = load_m2_config(ROOT / "configs/m2_data.toml")["evaluator"]
    import tomllib

    with (ROOT / "configs/m2_diagnostic.toml").open("rb") as handle:
        diagnostic = tomllib.load(handle)
    assert formal["maximum_result_rows"] == 100_000
    assert formal["maximum_result_bytes"] == 64 * 1024**2
    assert diagnostic["profile_b"]["maximum_result_rows"] == 1_000_000
    assert diagnostic["profile_b"]["maximum_result_bytes"] == 512 * 1024**2


def test_m2_config_rejects_unsafe_nested_archive_path(tmp_path: Path) -> None:
    source = ROOT / "configs/m2_data.toml"
    raw = source.read_text(encoding="utf-8")
    target = tmp_path / "m2.toml"
    target.write_text(
        raw.replace('"train/train_databases.zip"', '"../train_databases.zip"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nested training archive path"):
        load_m2_config(target)
