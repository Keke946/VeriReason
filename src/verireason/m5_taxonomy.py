"""Validation and SQL-free aggregation for the private M5 error audit."""

from __future__ import annotations

from collections import Counter
from typing import Any

AUDIT_LABELS = frozenset(
    {
        "schema_grounding",
        "projection",
        "predicate",
        "aggregation",
        "composition",
        "cardinality",
        "calculation",
        "execution_syntax",
        "efficiency_resource",
        "safety_generation",
        "ambiguous_or_equivalent",
    }
)


def validate_annotations(
    audit_rows: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join annotations to frozen rows and reject incomplete or invalid audit state."""
    row_ids = [str(row["sample_id"]) for row in audit_rows]
    annotation_ids = [str(row["sample_id"]) for row in annotations]
    if len(set(row_ids)) != len(row_ids) or len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("audit rows and annotations must have unique sample_id values")
    if set(row_ids) != set(annotation_ids):
        raise ValueError("annotation sample_id set differs from frozen audit selection")

    by_id = {str(row["sample_id"]): row for row in annotations}
    validated: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        annotation = by_id[str(audit_row["sample_id"])]
        primary = annotation.get("primary_label")
        secondary = annotation.get("secondary_labels", [])
        if primary not in AUDIT_LABELS:
            raise ValueError(f"invalid primary label: {primary!r}")
        if not isinstance(secondary, list) or any(label not in AUDIT_LABELS for label in secondary):
            raise ValueError("secondary_labels must contain only frozen taxonomy labels")
        if primary in secondary or len(secondary) != len(set(secondary)):
            raise ValueError("secondary_labels must be unique and exclude the primary label")
        confidence = annotation.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        if not isinstance(annotation.get("review_required"), bool):
            raise ValueError("review_required must be boolean")
        validated.append(
            {
                "failure_category": str(audit_row["failure_category"]),
                "gold_complexity": str(audit_row["gold_complexity"]),
                "primary_label": primary,
                "secondary_labels": tuple(secondary),
                "confidence": confidence,
                "review_required": annotation["review_required"],
            }
        )
    return validated


def summarize_annotations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a public aggregate containing no row identifiers or SQL text."""
    primary = Counter(row["primary_label"] for row in rows)
    secondary = Counter(label for row in rows for label in row["secondary_labels"])
    cross = Counter((row["failure_category"], row["primary_label"]) for row in rows)
    return {
        "schema_version": 1,
        "audited_examples": len(rows),
        "primary_labels": dict(sorted(primary.items())),
        "secondary_labels": dict(sorted(secondary.items())),
        "failure_category_by_primary_label": {
            category: {
                label: count
                for (observed_category, label), count in sorted(cross.items())
                if observed_category == category
            }
            for category in sorted({category for category, _ in cross})
        },
        "confidence": dict(sorted(Counter(row["confidence"] for row in rows).items())),
        "review_required": sum(row["review_required"] for row in rows),
        "privacy": {
            "row_level_content_included": False,
            "sample_ids_included": False,
            "sql_included": False,
        },
    }
