"""CPU-testable contracts shared by the E8 training and evaluation processes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from verireason.m3_contract import tokenize_answer_only, validate_answer_only_token_contract
from verireason.repair_probe import repair_messages
from verireason.repair_sft_smoke import EXPERIMENT_ID


def validate_private_split(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the private E8 payload without depending on its public receipt."""
    if payload.get("schema_version") != 1 or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E8 private split identity drifted")
    train = [dict(row) for row in payload.get("train", ())]
    audit = [dict(row) for row in payload.get("audit", ())]
    if not train or not audit:
        raise ValueError("E8 private split is incomplete")

    def identities(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
        record_ids = {str(row.get("record_id", "")) for row in rows}
        source_ids = {str(row.get("source_sample_id", "")) for row in rows}
        database_ids = {str(row.get("db_id", "")) for row in rows}
        if (
            "" in record_ids | source_ids | database_ids
            or len(record_ids) != len(rows)
            or any(row.get("kind") not in {"repair", "keep"} for row in rows)
        ):
            raise ValueError("E8 private rows have invalid identities or kinds")
        return record_ids, source_ids, database_ids

    _train_records, train_sources, train_databases = identities(train)
    _audit_records, audit_sources, audit_databases = identities(audit)
    if train_sources & audit_sources or train_databases & audit_databases:
        raise ValueError("E8 private split leaks sources or databases")
    if len(audit_databases) != int(config["split"]["audit_databases"]):
        raise ValueError("E8 audit database count drifted")
    train_repairs = sum(row["kind"] == "repair" for row in train)
    train_keeps = sum(row["kind"] == "keep" for row in train)
    if train_repairs != train_keeps or train_repairs != len(train_sources):
        raise ValueError("E8 train rows must contain one repair and one KEEP per source")
    evaluation = config["evaluation"]
    if len(audit) != int(evaluation["expected_audit_records"]):
        raise ValueError("E8 selected audit count drifted")
    for database in audit_databases:
        selected = [row for row in audit if str(row["db_id"]) == database]
        if sum(row["kind"] == "repair" for row in selected) != int(
            evaluation["repairs_per_database"]
        ) or sum(row["kind"] == "keep" for row in selected) != int(
            evaluation["keeps_per_database"]
        ):
            raise ValueError("E8 per-database audit composition drifted")
    return train, audit


def tokenize_repair_training(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tokenize E8 supervision while masking every prompt token."""
    maximum_sequence = int(config["training"]["maximum_sequence_tokens"])
    maximum_answer = int(config["training"]["maximum_answer_tokens"])
    encoded: list[dict[str, Any]] = []
    total_supervised = 0
    maximum_observed = 0
    for row in rows:
        messages, target = row.get("messages"), str(row.get("target", "")).strip()
        if not isinstance(messages, list) or not messages or not target:
            raise ValueError("E8 training row lacks messages or target")
        input_ids, labels, prompt_tokens = tokenize_answer_only(tokenizer, messages, target)
        answer_tokens = len(input_ids) - prompt_tokens
        validate_answer_only_token_contract(
            input_ids,
            labels,
            prompt_tokens,
            answer_tokens=answer_tokens,
            supervised_tokens=answer_tokens,
        )
        if len(input_ids) > maximum_sequence or answer_tokens > maximum_answer:
            raise ValueError("E8 training row exceeds the frozen token budget")
        encoded.append(
            {
                "record_id": str(row["record_id"]),
                "kind": str(row["kind"]),
                "input_ids": input_ids,
                "labels": labels,
            }
        )
        total_supervised += answer_tokens
        maximum_observed = max(maximum_observed, len(input_ids))
    return encoded, {
        "examples": len(encoded),
        "supervised_tokens": total_supervised,
        "maximum_sequence_tokens_observed": maximum_observed,
    }


def audit_messages(row: Mapping[str, Any], arm: str) -> list[dict[str, str]]:
    """Rebuild the fixed feedback/control prompt from one E7/E8 audit record."""
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) < 4
        or messages[-2].get("role") != "assistant"
        or messages[-1].get("role") != "user"
    ):
        raise ValueError("E8 audit row does not contain a repair turn")
    initial_sql = str(messages[-2].get("content", "")).strip()
    feedback_prompt = str(messages[-1].get("content", ""))
    marker = "Database feedback:\n"
    if marker not in feedback_prompt:
        raise ValueError("E8 audit row lacks bounded database feedback")
    feedback = feedback_prompt.split(marker, 1)[1]
    return repair_messages(messages[:-2], initial_sql, arm=arm, feedback=feedback)


def validate_training_evidence(
    evidence: Mapping[str, Any], *, training_examples: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on an incomplete or numerically invalid one-epoch result."""
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    batch_size = int(config["training"]["per_device_train_batch_size"])
    expected_updates = math.ceil(training_examples / (accumulation * batch_size))
    finite = all(
        isinstance(evidence.get(key), int | float) and math.isfinite(float(evidence[key]))
        for key in ("training_loss", "gradient_norm", "adapter_delta_l2")
    )
    passed = (
        evidence.get("experiment_id") == EXPERIMENT_ID
        and evidence.get("epochs") == 1
        and evidence.get("real_example_passes") == training_examples
        and evidence.get("optimizer_steps") == expected_updates
        and finite
        and float(evidence.get("adapter_delta_l2", 0)) > 0
        and evidence.get("validation_payloads_read") == 0
        and evidence.get("heldout_payloads_read") == 0
    )
    if not passed:
        raise ValueError("E8 training evidence failed its fixed contract")
    return {"expected_optimizer_steps": expected_updates, "training_contract_passed": True}


def validate_reload_evidence(
    training: Mapping[str, Any], reload: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the saved adapter to load in a distinct process before audit."""
    if (
        training.get("adapter_manifest_sha256") is None
        or reload.get("adapter_manifest_sha256") != training["adapter_manifest_sha256"]
        or reload.get("training_process_id") != training.get("process_id")
        or reload.get("reload_process_id") == training.get("process_id")
        or reload.get("adapter_loaded") is not True
        or reload.get("validation_payloads_read") != 0
        or reload.get("heldout_payloads_read") != 0
    ):
        raise ValueError("E8 independent adapter reload contract failed")
    return {"independent_reload_passed": True}
