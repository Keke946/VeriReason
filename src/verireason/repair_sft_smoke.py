"""Deterministic E8 repair-SFT split, balanced training set, and evaluation gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "e8_repair_sft_smoke_001"


def load_repair_sft_smoke_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"source", "split", "training", "evaluation", "gate", "budget"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("E8 config is incomplete")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("E8 experiment identity drifted")
    source = config["source"]
    if source.get("validation_payloads_read") != 0 or source.get("heldout_payloads_read") != 0:
        raise ValueError("E8 must remain training-only")
    if config["training"].get("epochs") != 1:
        raise ValueError("E8 smoke requires exactly one epoch")
    return config


def _rank(seed: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _write_private(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def prepare_repair_sft_split(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create DB-disjoint audit data and a 1:1 repair/KEEP training subset."""
    source_config, split = config["source"], config["split"]
    if len(records) != source_config["repair_examples"] + source_config["keep_examples"]:
        raise ValueError("E7 private record count drifted")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_records: set[str] = set()
    for raw in records:
        row = dict(raw)
        record_id = str(row.get("record_id", ""))
        source_id = str(row.get("source_sample_id", ""))
        if not record_id or record_id in seen_records or not source_id:
            raise ValueError("E7 records require unique record/source identities")
        seen_records.add(record_id)
        if row.get("kind") not in {"repair", "keep"}:
            raise ValueError("unsupported E7 record kind")
        by_source[source_id].append(row)
    if len(by_source) != source_config["source_examples"]:
        raise ValueError("E7 source count drifted")

    source_database: dict[str, str] = {}
    for source_id, rows in by_source.items():
        databases = {str(row.get("db_id", "")) for row in rows}
        repairs = [row for row in rows if row["kind"] == "repair"]
        keeps = [row for row in rows if row["kind"] == "keep"]
        if len(databases) != 1 or "" in databases or len(keeps) != 1 or not repairs:
            raise ValueError("each source requires one database, one KEEP, and repairs")
        source_database[source_id] = next(iter(databases))
    databases = sorted(set(source_database.values()))
    if len(databases) != source_config["databases"]:
        raise ValueError("E7 database count drifted")
    ranked = sorted(databases, key=lambda value: (_rank(split["seed"], value), value))
    audit_databases = set(ranked[: int(split["audit_databases"])])
    if not audit_databases or len(audit_databases) == len(databases):
        raise ValueError("database split is degenerate")

    train: list[dict[str, Any]] = []
    audit_pool: list[dict[str, Any]] = []
    train_sources: set[str] = set()
    audit_sources: set[str] = set()
    for source_id in sorted(by_source):
        rows = by_source[source_id]
        if source_database[source_id] in audit_databases:
            audit_pool.extend(sorted(rows, key=lambda row: str(row["record_id"])))
            audit_sources.add(source_id)
            continue
        repairs = sorted(
            (row for row in rows if row["kind"] == "repair"),
            key=lambda row: (_rank(split["seed"], str(row["record_id"])), str(row["record_id"])),
        )
        keep = next(row for row in rows if row["kind"] == "keep")
        train.extend((repairs[0], keep))
        train_sources.add(source_id)

    train_databases = set(databases) - audit_databases
    if train_sources & audit_sources or train_databases & audit_databases:
        raise RuntimeError("E8 split leakage detected")
    if sum(row["kind"] == "repair" for row in train) != sum(row["kind"] == "keep" for row in train):
        raise RuntimeError("E8 training set is not repair/KEEP balanced")
    evaluation = config["evaluation"]
    audit: list[dict[str, Any]] = []
    for database in sorted(audit_databases):
        database_rows = [row for row in audit_pool if row["db_id"] == database]
        for kind, count_key in (("repair", "repairs_per_database"), ("keep", "keeps_per_database")):
            candidates = sorted(
                (row for row in database_rows if row["kind"] == kind),
                key=lambda row: (
                    _rank(split["seed"] + ":audit", str(row["record_id"])),
                    str(row["record_id"]),
                ),
            )
            required = int(evaluation[count_key])
            if len(candidates) < required:
                raise ValueError(f"audit database lacks required {kind} examples")
            audit.extend(candidates[:required])
    if len(audit) != int(evaluation["expected_audit_records"]):
        raise RuntimeError("E8 selected audit count drifted")
    selected_audit_sources = {str(row["source_sample_id"]) for row in audit}
    private = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "train": train,
        "audit": audit,
    }
    public = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "prepared",
        "train_sources": len(train_sources),
        "audit_sources": len(selected_audit_sources),
        "audit_pool_sources": len(audit_sources),
        "train_databases": len(train_databases),
        "audit_databases": len(audit_databases),
        "train_repair_examples": sum(row["kind"] == "repair" for row in train),
        "train_keep_examples": sum(row["kind"] == "keep" for row in train),
        "audit_repair_examples": sum(row["kind"] == "repair" for row in audit),
        "audit_keep_examples": sum(row["kind"] == "keep" for row in audit),
        "audit_pool_repair_examples": sum(row["kind"] == "repair" for row in audit_pool),
        "audit_pool_keep_examples": sum(row["kind"] == "keep" for row in audit_pool),
        "database_overlap": 0,
        "source_overlap": 0,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
        "privacy": {
            "sql_included": False,
            "questions_included": False,
            "sample_ids_included": False,
            "database_ids_included": False,
        },
    }
    return private, public


def summarize_repair_sft_evaluation(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize the fixed 2x2 evaluation and apply the preregistered smoke gate."""
    conditions, arms = config["evaluation"]["conditions"], config["evaluation"]["arms"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for condition in conditions:
        for arm in arms:
            selected = [
                row for row in rows if row.get("condition") == condition and row.get("arm") == arm
            ]
            if not selected:
                raise ValueError("E8 evaluation matrix is incomplete")
            grouped[(condition, arm)] = selected
    reference = grouped[(conditions[0], arms[0])]
    expected = {str(row["record_id"]) for row in reference}
    if len(expected) != len(reference):
        raise ValueError("E8 evaluation cell contains duplicate record identities")
    for selected in grouped.values():
        observed = {str(row["record_id"]) for row in selected}
        if len(observed) != len(selected):
            raise ValueError("E8 evaluation cell contains duplicate record identities")
        if observed != expected:
            raise ValueError("E8 conditions do not cover identical audit identities")

    metrics: dict[str, dict[str, Any]] = {}
    for (condition, arm), selected in grouped.items():
        repairs = [row for row in selected if row["kind"] == "repair"]
        keeps = [row for row in selected if row["kind"] == "keep"]
        repaired = [row for row in repairs if row.get("final_correct") is True]
        retained = [row for row in keeps if row.get("final_correct") is True]
        metrics[f"{condition}:{arm}"] = {
            "repair_total": len(repairs),
            "repair_correct": len(repaired),
            "repair_rate": len(repaired) / len(repairs) if repairs else 0.0,
            "keep_total": len(keeps),
            "keep_correct": len(retained),
            "keep_rate": len(retained) / len(keeps) if keeps else 0.0,
            "repaired_databases": len({str(row["db_id"]) for row in repaired}),
            "unsafe_actions": sum(row.get("final_status") == "unsafe" for row in selected),
        }
    baseline = metrics["e1_baseline:feedback"]
    adapter_feedback = metrics["repair_sft:feedback"]
    adapter_control = metrics["repair_sft:control"]
    gate = config["gate"]
    checks = {
        "minimum_repairs": adapter_feedback["repair_correct"]
        >= gate["minimum_adapter_feedback_repairs"],
        "cross_database": adapter_feedback["repaired_databases"]
        >= gate["minimum_repaired_databases"],
        "gain_over_e1": adapter_feedback["repair_rate"] - baseline["repair_rate"]
        >= gate["minimum_repair_rate_gain_over_e1"],
        "feedback_specific": adapter_feedback["repair_rate"] - adapter_control["repair_rate"]
        >= gate["minimum_feedback_gain_over_control"],
        "keep_rate": adapter_feedback["keep_rate"] >= gate["minimum_keep_rate"],
        "keep_damage": baseline["keep_rate"] - adapter_feedback["keep_rate"]
        <= gate["maximum_keep_rate_drop_vs_e1"],
        "safe": all(metric["unsafe_actions"] == 0 for metric in metrics.values()),
    }
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "stop",
        "metrics": metrics,
        "gate_checks": checks,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }


def write_private_split(path: str | Path, payload: Mapping[str, Any]) -> None:
    _write_private(path, payload)
