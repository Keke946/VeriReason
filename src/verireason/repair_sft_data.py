"""Training-only repair-SFT example construction with execution-verified mutations."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from verireason.repair_probe import execution_feedback, repair_messages
from verireason.sql_evaluator import DEFAULT_LIMITS, EvaluatorLimits, execute_read_only
from verireason.sqlite_schema import load_sqlite_schema

EXPERIMENT_ID = "e7_repair_sft_data_feasibility_001"
_ACTIONABLE_PREFIXES = ("unknown column:", "unknown table:", "ambiguous column:")


def load_repair_sft_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"source", "feedback", "construction", "synthetic_gate"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("repair-SFT config is incomplete")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("repair-SFT experiment identity drifted")
    source = config["source"]
    if (
        source.get("allowed_split") != "train"
        or source.get("validation_payloads_read") != 0
        or source.get("heldout_payloads_read") != 0
    ):
        raise ValueError("repair-SFT construction must remain training-only")
    allowed = {"wrong_column", "wrong_table", "wrong_alias"}
    mutation_types = config["construction"].get("mutation_types")
    if not mutation_types or set(mutation_types) - allowed:
        raise ValueError("unsupported repair-SFT mutation type")
    return config


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _database_path(row: Mapping[str, Any], database_root: str | Path) -> Path:
    root = Path(database_root).resolve()
    supplied = Path(str(row.get("database_path", "")))
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("database_path must identify a file below database_root")
    return path


def _source_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes) or not messages:
        raise ValueError("prepared row requires source messages")
    clean: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("source messages have an invalid shape")
        role, content = message["role"], message["content"]
        if role not in {"system", "user"} or not isinstance(content, str) or not content:
            raise ValueError("source messages may contain only non-empty system/user context")
        clean.append({"role": role, "content": content})
    return clean


def _plausible_typo(name: str, occupied: set[str]) -> str | None:
    candidates = []
    if len(name) > 3:
        candidates.append(name[:-2] + name[-1] + name[-2])
    candidates.extend((name + "s", name[:-1] if name.endswith("s") else name + "_old"))
    for candidate in candidates:
        folded = candidate.casefold()
        if candidate and candidate != name and folded not in occupied:
            return candidate
    return None


def _mutation_candidates(sql: str, schema: dict[str, set[str]]) -> list[tuple[str, str]]:
    try:
        tree = parse_one(sql, read="sqlite")
    except SqlglotError:
        return []
    tables = [table.name for table in tree.find_all(exp.Table) if table.name]
    used_tables = {name.casefold() for name in tables if name.casefold() in schema}
    used_columns = set().union(*(schema[name] for name in used_tables)) if used_tables else set()
    decoys = sorted(set().union(*schema.values()) - used_columns)
    mutations: list[tuple[str, str]] = []

    if decoys:
        column = next(tree.find_all(exp.Column), None)
        if column is not None:
            changed = tree.copy()
            changed_column = next(changed.find_all(exp.Column))
            changed_column.set("this", exp.to_identifier(decoys[0]))
            mutations.append(("wrong_column", changed.sql(dialect="sqlite")))

    table = next(tree.find_all(exp.Table), None)
    if table is not None:
        typo = _plausible_typo(table.name, set(schema))
        if typo:
            changed = tree.copy()
            next(changed.find_all(exp.Table)).set("this", exp.to_identifier(typo))
            mutations.append(("wrong_table", changed.sql(dialect="sqlite")))

    qualified = next((column for column in tree.find_all(exp.Column) if column.table), None)
    other_tables = sorted(set(schema) - used_tables)
    if qualified is not None and other_tables:
        changed = tree.copy()
        changed_column = next(column for column in changed.find_all(exp.Column) if column.table)
        changed_column.set("table", exp.to_identifier(other_tables[0]))
        mutations.append(("wrong_alias", changed.sql(dialect="sqlite")))
    return mutations


def build_repair_sft_examples(
    prepared_examples: Sequence[Mapping[str, Any]],
    database_root: str | Path,
    config: Mapping[str, Any],
    *,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
    evidence_scope: str = "cpu_interface",
    real_training_yield_known: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build private supervised examples and a SQL-free public receipt."""
    if not prepared_examples:
        raise ValueError("prepared training examples must not be empty")
    maximum = int(config["construction"]["maximum_mutations_per_source"])
    allowed_types = tuple(config["construction"]["mutation_types"])
    feedback_config = config["feedback"]
    records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_inputs: set[str] = set()

    for row in sorted(prepared_examples, key=lambda item: str(item["sample_id"])):
        sample_id = str(row["sample_id"])
        if sample_id in seen_sources:
            raise ValueError("prepared examples contain duplicate sample_id values")
        seen_sources.add(sample_id)
        if row.get("split") != "train":
            raise ValueError("repair-SFT construction accepts training rows only")
        database = _database_path(row, database_root)
        source_messages = _source_messages(row)
        gold_sql = str(row["target_sql"]).strip()
        gold = execute_read_only(database, gold_sql, limits)
        if not gold.successful:
            continue
        db_id = str(row["db_id"])
        schema = load_sqlite_schema(database)
        accepted = 0
        for mutation_type, corrupted_sql in _mutation_candidates(gold_sql, schema):
            if mutation_type not in allowed_types or accepted >= maximum:
                continue
            corrupted = execute_read_only(database, corrupted_sql, limits)
            if corrupted.status not in {"execution_error", "syntax_error"}:
                continue
            feedback = execution_feedback(
                corrupted,
                max_rows=int(feedback_config["maximum_preview_rows"]),
                max_characters=int(feedback_config["maximum_preview_characters"]),
            )
            if not feedback.casefold().startswith(_ACTIONABLE_PREFIXES) and not any(
                prefix in feedback.casefold() for prefix in _ACTIONABLE_PREFIXES
            ):
                continue
            messages = repair_messages(
                source_messages, corrupted_sql, arm="feedback", feedback=feedback
            )
            input_identity = _fingerprint(messages)
            if input_identity in seen_inputs or gold_sql in feedback:
                continue
            seen_inputs.add(input_identity)
            records.append(
                {
                    "record_id": _fingerprint([sample_id, mutation_type, input_identity]),
                    "source_sample_id": sample_id,
                    "db_id": db_id,
                    "database_path": str(database.relative_to(Path(database_root).resolve())),
                    "kind": "repair",
                    "mutation_type": mutation_type,
                    "messages": messages,
                    "target": gold_sql,
                    "initial_status": corrupted.status,
                    "target_status": gold.status,
                }
            )
            accepted += 1

        if accepted and config["construction"].get("include_keep_example") is True:
            feedback = execution_feedback(
                gold,
                max_rows=int(feedback_config["maximum_preview_rows"]),
                max_characters=int(feedback_config["maximum_preview_characters"]),
            )
            messages = repair_messages(source_messages, gold_sql, arm="feedback", feedback=feedback)
            input_identity = _fingerprint(messages)
            if input_identity not in seen_inputs:
                seen_inputs.add(input_identity)
                records.append(
                    {
                        "record_id": _fingerprint([sample_id, "keep", input_identity]),
                        "source_sample_id": sample_id,
                        "db_id": db_id,
                        "database_path": str(database.relative_to(Path(database_root).resolve())),
                        "kind": "keep",
                        "mutation_type": None,
                        "messages": messages,
                        "target": "KEEP",
                        "initial_status": gold.status,
                        "target_status": gold.status,
                    }
                )

    repairs = [record for record in records if record["kind"] == "repair"]
    keeps = [record for record in records if record["kind"] == "keep"]
    represented_databases = {record["db_id"] for record in records}
    mutation_counts = Counter(record["mutation_type"] for record in repairs)
    gate = config["synthetic_gate"]
    checks = {
        "source_examples": len(prepared_examples) >= int(gate["minimum_source_examples"]),
        "repair_examples": len(repairs) >= int(gate["minimum_repair_examples"]),
        "keep_examples": len(keeps) >= int(gate["minimum_keep_examples"]),
        "mutation_types": len(mutation_counts) >= int(gate["minimum_mutation_types"]),
        "databases": len(represented_databases) >= int(gate["minimum_databases"]),
        "all_train": all(row.get("split") == "train" for row in prepared_examples),
        "all_verified": bool(records)
        and all(
            row["target_status"] == "ok"
            and (
                row["kind"] == "keep"
                or row["initial_status"] in {"execution_error", "syntax_error"}
            )
            for row in records
        ),
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "pass" if all(checks.values()) else "revise",
        "evidence_scope": evidence_scope,
        "source_examples": len(prepared_examples),
        "repair_examples": len(repairs),
        "keep_examples": len(keeps),
        "databases": len(represented_databases),
        "mutation_counts": dict(sorted(mutation_counts.items())),
        "gate_checks": checks,
        "real_training_yield_known": real_training_yield_known,
        "optimizer_updates": 0,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
        "privacy": {
            "sql_included": False,
            "questions_included": False,
            "sample_ids_included": False,
            "database_ids_included": False,
        },
    }
    return records, receipt
