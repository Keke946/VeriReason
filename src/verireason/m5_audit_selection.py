"""Deterministic private sampling for M5 semantic error auditing."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sql_complexity


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def select_audit_rows(
    evaluation_examples: list[dict[str, Any]],
    prepared_examples: list[dict[str, Any]],
    *,
    seed: str,
    wrong_per_complexity: int = 8,
    execution_per_complexity: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = {str(row["sample_id"]): row for row in prepared_examples}
    if len(prepared) != len(prepared_examples):
        raise ValueError("prepared examples contain duplicate sample_id values")
    if {str(row["sample_id"]) for row in evaluation_examples} != set(prepared):
        raise ValueError("prepared and evaluation sample_id sets differ")

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    tail = {"syntax_error", "timeout", "resource_limit", "unsafe"}
    for result in evaluation_examples:
        sample_id = str(result["sample_id"])
        source = prepared[sample_id]
        complexity = sql_complexity(str(source["target_sql"]))
        category = str(result["failure_category"])
        if category in {"wrong_result", "execution_error"}:
            strata[(category, complexity.label)].append(result)
        elif category in tail:
            strata[("tail_all", category)].append(result)

    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for (category, stratum), rows in sorted(strata.items()):
        limit = (
            wrong_per_complexity
            if category == "wrong_result"
            else execution_per_complexity
            if category == "execution_error"
            else len(rows)
        )
        ranked = sorted(rows, key=lambda row: _rank(seed, str(row["sample_id"])))
        chosen = ranked[:limit]
        stratum_counts[f"{category}:{stratum}"] = {
            "population": len(rows),
            "selected": len(chosen),
        }
        for result in chosen:
            sample_id = str(result["sample_id"])
            source = prepared[sample_id]
            extraction = result.get("extraction")
            selected.append(
                {
                    "sample_id": sample_id,
                    "db_id": source["db_id"],
                    "question": source["question"],
                    "gold_sql": source["target_sql"],
                    "prediction_sql": (
                        extraction.get("sql") if isinstance(extraction, dict) else None
                    ),
                    "failure_category": result["failure_category"],
                    "prediction_status": result.get("prediction_status"),
                    "strict_comparison_status": result.get("strict_comparison_status"),
                    "gold_complexity": sql_complexity(str(source["target_sql"])).label,
                    "audit_label": None,
                    "audit_secondary_labels": [],
                    "audit_note": None,
                }
            )

    selected.sort(key=lambda row: _rank(seed, str(row["sample_id"])))
    fingerprints = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row in selected
    ]
    selection_sha = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    public = {
        "schema_version": 1,
        "seed": seed,
        "selection_sha256": selection_sha,
        "selected_examples": len(selected),
        "strata": stratum_counts,
        "selected_failure_categories": dict(
            sorted(Counter(str(row["failure_category"]) for row in selected).items())
        ),
        "privacy": {
            "private_rows_published": False,
            "public_row_identifiers_included": False,
        },
    }
    return selected, public


def write_private_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
