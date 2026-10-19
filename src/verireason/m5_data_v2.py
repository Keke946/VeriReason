"""Outcome-blind, training-only candidate construction for M5 data-v2."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from verireason.data_pipeline import sql_complexity

_JOIN = re.compile(r"\bJOIN\b", flags=re.IGNORECASE)
_TARGET_FEATURES = frozenset({"subquery", "group_by", "having"})


def build_data_v2_candidates(
    prepared_examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select hard-structure candidates without model outcomes or validation rows."""
    if not prepared_examples:
        raise ValueError("prepared training examples must not be empty")

    private_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in prepared_examples:
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError("prepared examples contain duplicate sample_id values")
        seen_ids.add(sample_id)
        if row.get("split") != "train":
            raise ValueError("data-v2 candidate construction accepts training rows only")

        parsed = sql_complexity(str(row["target_sql"]))
        join_count = len(_JOIN.findall(str(row["target_sql"])))
        reasons = sorted(set(parsed.features) & _TARGET_FEATURES)
        if join_count >= 2:
            reasons.append("multi_join_gte_2")
        if not reasons:
            continue
        private_rows.append(
            {
                "sample_id": sample_id,
                "db_id": str(row["db_id"]),
                "selection_reasons": reasons,
                "gold_complexity": parsed.label,
                "join_count": join_count,
                "prompt_sha256": str(row["prompt_sha256"]),
            }
        )

    private_rows.sort(key=lambda row: row["sample_id"])
    identities = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in private_rows]
    selection_sha256 = hashlib.sha256("\n".join(identities).encode()).hexdigest()
    reason_counts = Counter(reason for row in private_rows for reason in row["selection_reasons"])
    public = {
        "schema_version": 1,
        "policy": {
            "allowed_split": "train",
            "minimum_join_count": 2,
            "target_features": sorted(_TARGET_FEATURES),
            "uses_model_outcomes": False,
            "uses_validation_rows": False,
            "uses_heldout_rows": False,
        },
        "source_examples": len(prepared_examples),
        "candidate_examples": len(private_rows),
        "candidate_ratio": len(private_rows) / len(prepared_examples),
        "candidate_databases": len({row["db_id"] for row in private_rows}),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "complexity_distribution": dict(
            sorted(Counter(row["gold_complexity"] for row in private_rows).items())
        ),
        "selection_sha256": selection_sha256,
        "privacy": {
            "sample_ids_included": False,
            "database_ids_included": False,
            "questions_included": False,
            "sql_included": False,
        },
    }
    return private_rows, public
