import json
from pathlib import Path

import pytest

from verireason.m8_contract import (
    build_e3_public_comparison,
    build_e4_public_comparison,
    build_public_comparison,
    load_m8_config,
    paired_transitions,
    validate_candidate_evaluation,
    validate_e2_evaluation,
    validate_prepared_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(path="configs/m8_common_eval.toml"):
    return load_m8_config(ROOT / path, root=ROOT)


def _bundle(count=1209, config=None):
    config = config or _config()
    ids = [str(index) for index in range(count)]
    hashes = [f"{index:064x}" for index in range(count)]
    return {
        "split": "full-validation",
        "selection_sha256": config["data"]["selection_sha256"],
        "sample_ids": ids,
        "prompt_hashes": hashes,
        "examples": [
            {"sample_id": sample_id, "prompt_sha256": prompt_hash}
            for sample_id, prompt_hash in zip(ids, hashes, strict=True)
        ],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }


def _evaluation(bundle, config=None):
    config = config or _config()
    examples = []
    for index, sample_id in enumerate(bundle["sample_ids"]):
        correct = index % 2 == 0
        examples.append(
            {
                "sample_id": sample_id,
                "prompt_sha256": bundle["prompt_hashes"][index],
                "bird_ex_behavior_compatible": int(correct),
                "strict_execution_accuracy": int(correct),
                "prediction_status": "ok",
                "failure_category": "strict_correct" if correct else "wrong_result",
                "extraction": {"strict_sql_only": True},
            }
        )
    return {
        "condition": config["model"]["condition"],
        "adapter_experiment_id": config["model"]["adapter_experiment_id"],
        "adapter_manifest_sha256": config["model"]["adapter_manifest_sha256"],
        "validation_selection_sha256": bundle["selection_sha256"],
        "sample_ids": bundle["sample_ids"],
        "prompt_hashes": bundle["prompt_hashes"],
        "decoding": config["_frozen_m4"]["decoding"],
        "evaluator": config["_frozen_m4"]["evaluator"],
        "examples": examples,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }


def test_config_and_full_bundle_pass() -> None:
    config = _config()
    validate_prepared_bundle(_bundle(), config)


def test_bundle_rejects_drift_and_heldout() -> None:
    config = _config()
    bundle = _bundle()
    bundle["selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selection drifted"):
        validate_prepared_bundle(bundle, config)
    bundle = _bundle()
    bundle["heldout_examples_loaded"] = 1
    with pytest.raises(ValueError, match="heldout"):
        validate_prepared_bundle(bundle, config)


def test_e2_evaluation_requires_same_exam() -> None:
    config, bundle = _config(), _bundle()
    evaluation = _evaluation(bundle)
    validate_e2_evaluation(evaluation, bundle, config)
    evaluation["prompt_hashes"] = list(reversed(evaluation["prompt_hashes"]))
    with pytest.raises(ValueError, match="prompt_hashes"):
        validate_e2_evaluation(evaluation, bundle, config)


def test_e2_evaluation_rejects_decoding_and_row_drift() -> None:
    config, bundle = _config(), _bundle()
    evaluation = _evaluation(bundle)
    evaluation["decoding"] = {**evaluation["decoding"], "do_sample": True}
    with pytest.raises(ValueError, match="decoding protocol"):
        validate_e2_evaluation(evaluation, bundle, config)
    evaluation = _evaluation(bundle)
    evaluation["examples"] = list(reversed(evaluation["examples"]))
    with pytest.raises(ValueError, match="sample order"):
        validate_e2_evaluation(evaluation, bundle, config)


def test_paired_transition_counts_and_exact_test() -> None:
    e1 = {
        "examples": [
            {"sample_id": "a", "strict_execution_accuracy": 1},
            {"sample_id": "b", "strict_execution_accuracy": 0},
            {"sample_id": "c", "strict_execution_accuracy": 0},
        ]
    }
    e2 = {
        "examples": [
            {"sample_id": "a", "strict_execution_accuracy": 0},
            {"sample_id": "b", "strict_execution_accuracy": 1},
            {"sample_id": "c", "strict_execution_accuracy": 1},
        ]
    }
    result = paired_transitions(e1, e2, "strict_execution_accuracy")
    assert result["e1_correct_e2_wrong"] == 1
    assert result["e1_wrong_e2_correct"] == 2
    assert result["exact_two_sided_mcnemar_p"] == 1.0


def test_public_comparison_has_no_raw_rows() -> None:
    config, bundle = _config(), _bundle()
    evaluation = _evaluation(bundle)
    m4 = json.loads((ROOT / "results/m4/m4-public-summary.json").read_text())
    result = build_public_comparison(config=config, m4_summary=m4, e2_evaluation=evaluation)
    assert result["example_count"] == 1209
    assert result["paired_analysis"]["status"] == "not_available_without_existing_raw_e1"
    assert "examples" not in result["E2"]


def test_e3_config_and_evaluation_use_the_same_exam() -> None:
    config = _config("configs/m8_e3_common_eval.toml")
    bundle = _bundle(config=config)
    evaluation = _evaluation(bundle, config)
    validate_candidate_evaluation(evaluation, bundle, config)
    evaluation["condition"] = "E2"
    with pytest.raises(ValueError, match="not labeled E3"):
        validate_candidate_evaluation(evaluation, bundle, config)


def test_candidate_evaluation_rejects_adapter_drift() -> None:
    config = _config("configs/m8_e4_common_eval.toml")
    bundle = _bundle(config=config)
    evaluation = _evaluation(bundle, config)
    evaluation["adapter_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter identity drifted"):
        validate_candidate_evaluation(evaluation, bundle, config)


def test_e3_public_comparison_reuses_e1_e2_without_raw_rows() -> None:
    config = _config("configs/m8_e3_common_eval.toml")
    bundle = _bundle(config=config)
    evaluation = _evaluation(bundle, config)
    m4 = json.loads((ROOT / "results/m4/m4-public-summary.json").read_text())
    common = json.loads((ROOT / "results/m8/m8-common-eval-summary.json").read_text())
    result = build_e3_public_comparison(
        config=config,
        m4_summary=m4,
        existing_common=common,
        e3_evaluation=evaluation,
        e2_raw=evaluation,
    )
    assert result["experiment_id"] == "e3_m8_common_validation_001"
    assert result["E1"] == common["E1"]
    assert result["E2"] == common["E2"]
    assert result["E3"]["example_count"] == 1209
    assert "examples" not in result["E3"]
    e2_vs_e3 = result["paired_analysis"]["E2_vs_E3"]
    assert e2_vs_e3["status"] == "completed_from_existing_raw_evidence"
    assert e2_vs_e3["strict_execution_accuracy"]["e2_correct_e3_wrong"] == 0
    assert e2_vs_e3["strict_execution_accuracy"]["e2_wrong_e3_correct"] == 0


def test_e4_config_and_public_comparison_reuse_all_prior_arms() -> None:
    config = _config("configs/m8_e4_common_eval.toml")
    bundle = _bundle(config=config)
    evaluation = _evaluation(bundle, config)
    validate_candidate_evaluation(evaluation, bundle, config)
    common = json.loads((ROOT / "results/m8/m8-e3-common-eval-summary.json").read_text())
    result = build_e4_public_comparison(
        config=config,
        existing_e3_common=common,
        e4_evaluation=evaluation,
        e3_raw=evaluation,
    )
    assert result["experiment_id"] == "e4_m8_common_validation_001"
    assert result["E1"] == common["E1"]
    assert result["E2"] == common["E2"]
    assert result["E3"] == common["E3"]
    assert result["E4"]["example_count"] == 1209
    assert "examples" not in result["E4"]
    e3_vs_e4 = result["paired_analysis"]["E3_vs_E4"]
    assert e3_vs_e4["status"] == "completed_from_existing_raw_evidence"
    assert e3_vs_e4["candidate_condition"] == "E4"
    assert e3_vs_e4["strict_execution_accuracy"]["e3_correct_e4_wrong"] == 0
    assert e3_vs_e4["strict_execution_accuracy"]["e3_wrong_e4_correct"] == 0


def test_e4_config_rejects_e3_regeneration(tmp_path: Path) -> None:
    config_path = ROOT / "configs/m8_e4_common_eval.toml"
    text = config_path.read_text().replace(
        "e3_regeneration_allowed = false", "e3_regeneration_allowed = true"
    )
    temporary = tmp_path / "m8_e4_common_eval.toml"
    temporary.write_text(text)
    with pytest.raises(ValueError, match="reuse, not regenerate, the accepted E3"):
        load_m8_config(temporary, root=ROOT)
