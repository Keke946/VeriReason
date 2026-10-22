from copy import deepcopy

import pytest

from verireason.repair_sft_smoke import (
    load_repair_sft_smoke_config,
    prepare_repair_sft_split,
    summarize_repair_sft_evaluation,
)


def _config():
    config = deepcopy(load_repair_sft_smoke_config("configs/e8_repair_sft_smoke.toml"))
    config["source"].update(source_examples=12, repair_examples=24, keep_examples=12, databases=6)
    config["split"]["audit_databases"] = 2
    config["gate"].update(
        minimum_adapter_feedback_repairs=2,
        minimum_repaired_databases=2,
        minimum_repair_rate_gain_over_e1=0.25,
        minimum_feedback_gain_over_control=0.25,
    )
    return config


def _records():
    rows = []
    for source in range(12):
        database = f"db-{source // 2}"
        for mutation in ("wrong_column", "wrong_table"):
            rows.append(
                {
                    "record_id": f"r-{source}-{mutation}",
                    "source_sample_id": f"s-{source}",
                    "db_id": database,
                    "kind": "repair",
                    "mutation_type": mutation,
                    "messages": [{"role": "user", "content": "q"}],
                    "target": "SELECT 1",
                }
            )
        rows.append(
            {
                "record_id": f"k-{source}",
                "source_sample_id": f"s-{source}",
                "db_id": database,
                "kind": "keep",
                "mutation_type": None,
                "messages": [{"role": "user", "content": "q"}],
                "target": "KEEP",
            }
        )
    return rows


def test_split_is_deterministic_database_disjoint_and_balanced():
    first_private, first_public = prepare_repair_sft_split(_records(), _config())
    second_private, second_public = prepare_repair_sft_split(_records()[::-1], _config())

    assert first_private == second_private and first_public == second_public
    assert first_public["audit_databases"] == 2
    assert first_public["database_overlap"] == first_public["source_overlap"] == 0
    assert first_public["train_repair_examples"] == first_public["train_keep_examples"]
    train_db = {row["db_id"] for row in first_private["train"]}
    audit_db = {row["db_id"] for row in first_private["audit"]}
    assert train_db.isdisjoint(audit_db)


def test_split_rejects_duplicate_or_incomplete_sources():
    records = _records()
    duplicated = [dict(row) for row in records]
    duplicated[-1]["record_id"] = duplicated[0]["record_id"]
    with pytest.raises(ValueError, match="unique"):
        prepare_repair_sft_split(duplicated, _config())
    incomplete = [row for row in records if row["record_id"] != "k-0"]
    config = _config()
    config["source"]["keep_examples"] -= 1
    with pytest.raises(ValueError, match="one KEEP"):
        prepare_repair_sft_split(incomplete, config)


def _evaluation_rows(*, adapter_feedback_correct=4, adapter_control_correct=1):
    rows = []
    for condition in ("e1_baseline", "repair_sft"):
        for arm in ("feedback", "control"):
            for index in range(4):
                if condition == "repair_sft" and arm == "feedback":
                    correct = index < adapter_feedback_correct
                elif condition == "repair_sft" and arm == "control":
                    correct = index < adapter_control_correct
                else:
                    correct = False
                rows.append(
                    {
                        "condition": condition,
                        "arm": arm,
                        "record_id": f"repair-{index}",
                        "source_sample_id": f"repair-{index}",
                        "db_id": f"db-{index % 2}",
                        "kind": "repair",
                        "final_correct": correct,
                        "final_status": "ok",
                    }
                )
            for index in range(4):
                rows.append(
                    {
                        "condition": condition,
                        "arm": arm,
                        "record_id": f"keep-{index}",
                        "source_sample_id": f"keep-{index}",
                        "db_id": f"db-{index % 2}",
                        "kind": "keep",
                        "final_correct": True,
                        "final_status": "ok",
                    }
                )
    return rows


def test_evaluation_gate_requires_gain_feedback_specificity_keep_and_safety():
    passed = summarize_repair_sft_evaluation(_evaluation_rows(), _config())
    assert passed["status"] == "pass" and all(passed["gate_checks"].values())

    no_feedback_gain = summarize_repair_sft_evaluation(
        _evaluation_rows(adapter_feedback_correct=4, adapter_control_correct=4), _config()
    )
    assert no_feedback_gain["status"] == "stop"
    assert no_feedback_gain["gate_checks"]["feedback_specific"] is False

    unsafe = _evaluation_rows()
    unsafe[0]["final_status"] = "unsafe"
    result = summarize_repair_sft_evaluation(unsafe, _config())
    assert result["status"] == "stop" and result["gate_checks"]["safe"] is False


def test_evaluation_rejects_nonidentical_condition_coverage():
    rows = _evaluation_rows()
    rows.pop()
    with pytest.raises(ValueError, match="identical audit identities"):
        summarize_repair_sft_evaluation(rows, _config())


def test_evaluation_rejects_duplicate_id_within_cell():
    rows = _evaluation_rows()
    rows[1]["record_id"] = rows[0]["record_id"]
    with pytest.raises(ValueError, match="duplicate record identities"):
        summarize_repair_sft_evaluation(rows, _config())
