import copy
import json
from pathlib import Path

import pytest

from verireason.m4_contract import (
    extension_decision,
    load_m4_config,
    select_formal_rows,
    validate_comparison,
    validate_resume_contract,
    validate_validation_partition,
)
from verireason.m4_runtime import build_accumulation_plan, scaled_compute_loss

ROOT = Path(__file__).resolve().parents[1]


def row(index, split, db, sql="SELECT 1"):
    return {
        "row_id": str(index),
        "split": split,
        "db_id": db,
        "sql": sql,
        "operational_eligible": True,
        "token_eligible": True,
    }


def test_config_freezes_one_primary_epoch_advisory_diagnostic_and_zero_heldout():
    config = load_m4_config(ROOT / "configs/m4_formal_sft.toml", ROOT / "configs/m2_data.toml")
    assert config["data"]["train_count"] == 5364
    assert (
        config["data"]["development_validation_count"],
        config["data"]["final_validation_count"],
        config["data"]["full_validation_count"],
    ) == (256, 953, 1209)
    assert (
        config["qlora"]["primary_epochs"],
        config["qlora"]["primary_updates"],
        config["qlora"]["scheduler_total_optimizer_steps"],
    ) == (1, 671, 671)
    assert config["qlora"]["transformers_version"] == "5.14.1"
    assert config["qlora"]["automatic_extension"] is False
    assert config["epoch_extension_diagnostic"]["advisory_relative_dev_loss_drop"] == 0.01
    assert config["checkpoint"]["required_loss_checkpoint_updates"] == [512, 671]


def test_full_primary_epoch_keeps_all_rows_and_rejects_a_continuous_second_epoch():
    examples = [{"input_ids": [index], "labels": [index]} for index in range(5364)]
    plan, evidence = build_accumulation_plan(examples, epochs=1, accumulation_steps=8)
    assert evidence["real_example_passes"] == 5364 and evidence["optimizer_steps"] == 671
    assert evidence["zero_gradient_padding_microbatches"] == 4
    assert [item["_m4_loss_scale"] for item in plan[-8:]] == [2.0] * 4 + [0.0] * 4
    with pytest.raises(ValueError, match="inputs"):
        build_accumulation_plan(examples, epochs=2, accumulation_steps=8)


def test_actual_compute_loss_boundary_makes_four_real_losses_a_four_sample_mean():
    class Scale:
        def __init__(self, value):
            self.value = value

        def mean(self):
            return self.value

    class Output:
        def __init__(self, loss):
            self.loss = loss

    class Model:
        def __init__(self, loss):
            self.loss = loss

        def __call__(self, **kwargs):
            return Output(self.loss)

    real_losses = [2.0, 3.0, 5.0, 7.0]
    micro_losses = [
        scaled_compute_loss(Model(loss), {"m4_loss_scale": Scale(2.0)}) for loss in real_losses
    ]
    padding_losses = [
        scaled_compute_loss(Model(99.0), {"m4_loss_scale": Scale(0.0)}) for _ in range(4)
    ]
    trainer_window_gradient = sum(micro_losses + padding_losses) / 8
    assert trainer_window_gradient == pytest.approx(sum(real_losses) / 4)


def test_selector_partition_is_disjoint_and_full_is_exact_union():
    rows = [row(i, "validation", f"db{i % 3}") for i in range(1209)]
    dev = select_formal_rows(
        rows,
        split="validation",
        count=256,
        seed="x",
        required_features=[],
        source_manifest_sha256="a" * 64,
    )
    final_rows = [item for item in rows if item["row_id"] not in set(dev["sample_ids"])]
    final = select_formal_rows(
        final_rows,
        split="validation",
        count=953,
        seed="x",
        required_features=[],
        source_manifest_sha256="a" * 64,
    )
    full = {"sample_ids": [*dev["sample_ids"], *final["sample_ids"]]}
    validate_validation_partition(dev, final, full)


def test_selector_is_deterministic_preserves_database_and_features():
    rows = [
        row(
            i,
            "train",
            f"db{i % 3}",
            "SELECT COUNT(*) FROM a JOIN b ON a.id=b.id GROUP BY a.id"
            if i == 4
            else "SELECT * FROM a WHERE id IN (SELECT id FROM b) ORDER BY id",
        )
        for i in range(20)
    ]
    first = select_formal_rows(
        rows,
        split="train",
        count=10,
        seed="fixed",
        required_features=["join", "aggregate", "subquery", "group_by", "order_by"],
        source_manifest_sha256="a" * 64,
    )
    second = select_formal_rows(
        list(reversed(rows)),
        split="train",
        count=10,
        seed="fixed",
        required_features=["join", "aggregate", "subquery", "group_by", "order_by"],
        source_manifest_sha256="a" * 64,
    )
    assert first["sample_ids"] == second["sample_ids"] and first["database_ids"] == [
        "db0",
        "db1",
        "db2",
    ]


def test_extension_reference_is_advisory_not_an_automatic_gate():
    decision = extension_decision(1.0, 0.995)
    assert decision["advisory_threshold_met"] is False and decision["automatic_extension"] is False
    assert decision["decision"] == "primary_epoch_stopped_pending_control_review"


def test_result_schemas_preregister_primary_and_advisory_evidence():
    schema = json.loads((ROOT / "results/schemas/m4-result.schema.json").read_text())
    assert schema["properties"]["training"]["properties"]["optimizer_steps"]["const"] == 671
    assert (
        schema["properties"]["training"]["properties"]["trainer_loss_normalization"]["properties"][
            "transformers_model_accepts_loss_kwargs_forced"
        ]["const"]
        is False
    )
    assert (
        schema["properties"]["trainer_contract"]["properties"]["transformers_version"]["const"]
        == "5.14.1"
    )
    decision = json.loads(
        (ROOT / "results/schemas/m4-epoch-extension-decision.schema.json").read_text()
    )
    assert decision["properties"]["automatic_extension"]["const"] is False


def test_resume_contract_and_same_exam_guard():
    contract = {"phase": "primary_epoch_1", "selection_sha256": "a", "config_sha256": "b"}
    validate_resume_contract(contract, copy.deepcopy(contract))
    with pytest.raises(ValueError, match="resume checkpoint"):
        validate_resume_contract(contract, {**contract, "phase": "epoch2"})
    common = {
        "validation_selection_sha256": "a",
        "sample_ids": ["1"],
        "prompt_hashes": ["p"],
        "decoding": {"do_sample": False},
        "evaluator": {"timeout": 10},
        "heldout_examples_loaded": 0,
    }
    validate_comparison(common, copy.deepcopy(common))
    with pytest.raises(ValueError, match="sample_ids"):
        validate_comparison(common, {**common, "sample_ids": ["2"]})
