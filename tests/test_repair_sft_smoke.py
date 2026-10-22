from copy import deepcopy

import pytest

from verireason.repair_sft_runtime import (
    audit_messages,
    tokenize_repair_training,
    validate_private_split,
    validate_reload_evidence,
    validate_training_evidence,
)
from verireason.repair_sft_smoke import (
    load_repair_sft_smoke_config,
    prepare_repair_sft_split,
    summarize_repair_sft_evaluation,
)


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return [10 + index for index, _message in enumerate(messages)]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [100 + ord(character) % 23 for character in text]


def _config():
    config = deepcopy(load_repair_sft_smoke_config("configs/e8_repair_sft_smoke.toml"))
    config["source"].update(source_examples=12, repair_examples=24, keep_examples=12, databases=6)
    config["split"]["audit_databases"] = 2
    config["evaluation"]["expected_audit_records"] = 12
    config["gate"].update(
        minimum_adapter_feedback_repairs=2,
        minimum_repaired_databases=2,
        minimum_repair_rate_gain_over_e1=0.25,
        minimum_feedback_gain_over_control=0.25,
    )
    return config


def _messages(initial="SELECT broken"):
    return [
        {"role": "system", "content": "Return SQL."},
        {"role": "user", "content": "Question and schema"},
        {"role": "assistant", "content": initial},
        {
            "role": "user",
            "content": (
                "Return only KEEP or corrected SQL.\n\nDatabase feedback:\nunknown column: broken"
            ),
        },
    ]


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
                    "messages": _messages(),
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
                "messages": _messages("SELECT 1"),
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


def test_runtime_validates_split_tokenizes_answer_only_and_builds_control():
    private, _public = prepare_repair_sft_split(_records(), _config())
    train, audit = validate_private_split(private, _config())
    encoded, receipt = tokenize_repair_training(train, TinyTokenizer(), _config())
    assert len(encoded) == receipt["examples"] == len(train)
    assert all(row["labels"][:4] == [-100, -100, -100, -100] for row in encoded)

    feedback = audit_messages(audit[0], "feedback")
    control = audit_messages(audit[0], "control")
    assert "Database feedback:" in feedback[-1]["content"]
    assert "withheld" in control[-1]["content"]


def test_runtime_training_and_independent_reload_contracts():
    config = _config()
    evidence = {
        "experiment_id": "e8_repair_sft_smoke_001",
        "epochs": 1,
        "real_example_passes": 24,
        "optimizer_steps": 3,
        "training_loss": 1.2,
        "gradient_norm": 0.4,
        "adapter_delta_l2": 0.03,
        "adapter_manifest_sha256": "a" * 64,
        "process_id": 12,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
    assert (
        validate_training_evidence(evidence, training_examples=24, config=config)[
            "expected_optimizer_steps"
        ]
        == 3
    )
    reload = {
        "adapter_manifest_sha256": "a" * 64,
        "training_process_id": 12,
        "reload_process_id": 13,
        "adapter_loaded": True,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
    assert validate_reload_evidence(evidence, reload)["independent_reload_passed"] is True

    bad = dict(evidence, optimizer_steps=2)
    with pytest.raises(ValueError, match="training evidence"):
        validate_training_evidence(bad, training_examples=24, config=config)
    with pytest.raises(ValueError, match="reload contract"):
        validate_reload_evidence(evidence, dict(reload, reload_process_id=12))
