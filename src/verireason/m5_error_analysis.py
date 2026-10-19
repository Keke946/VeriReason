"""Deterministic, SQL-free aggregate analysis of M4 validation failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from verireason.data_pipeline import sql_complexity


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    strict = sum(int(row["strict_execution_accuracy"]) for row in rows)
    behavior = sum(int(row["bird_ex_behavior_compatible"]) for row in rows)
    failures = Counter(str(row["failure_category"]) for row in rows)
    return {
        "examples": total,
        "strict_correct": strict,
        "strict_accuracy": _rate(strict, total),
        "behavior_correct": behavior,
        "behavior_accuracy": _rate(behavior, total),
        "failure_categories": dict(sorted(failures.items())),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}

    def nearest(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p25": nearest(0.25),
        "median": nearest(0.50),
        "p75": nearest(0.75),
        "max": ordered[-1],
    }


def analyze_m4_errors(
    evaluation_examples: list[dict[str, Any]],
    prepared_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return aggregate diagnostics without copying sensitive row-level text."""
    prepared = {str(row["sample_id"]): row for row in prepared_examples}
    if len(prepared) != len(prepared_examples):
        raise ValueError("prepared examples contain duplicate sample_id values")
    evaluated = {str(row["sample_id"]): row for row in evaluation_examples}
    if len(evaluated) != len(evaluation_examples):
        raise ValueError("evaluation examples contain duplicate sample_id values")
    if set(prepared) != set(evaluated):
        raise ValueError("prepared and evaluation sample_id sets differ")

    enriched: list[dict[str, Any]] = []
    feature_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for sample_id in sorted(evaluated):
        result = evaluated[sample_id]
        source = prepared[sample_id]
        gold = sql_complexity(str(source["target_sql"]))
        row = {**result, "gold_complexity": gold.label, "gold_features": gold.features}
        enriched.append(row)

        extraction = result.get("extraction")
        predicted_sql = extraction.get("sql") if isinstance(extraction, dict) else None
        predicted_features = (
            set(sql_complexity(predicted_sql).features)
            if isinstance(predicted_sql, str) and predicted_sql.strip()
            else set()
        )
        for feature in (
            "join",
            "aggregate",
            "subquery",
            "group_by",
            "set_operation",
            "order_by",
            "having",
            "window",
        ):
            gold_has = feature in gold.features
            predicted_has = feature in predicted_features
            label = (
                "true_positive"
                if gold_has and predicted_has
                else "false_negative"
                if gold_has
                else "false_positive"
                if predicted_has
                else "true_negative"
            )
            feature_confusion[feature][label] += 1

    complexity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    database_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        complexity_groups[row["gold_complexity"]].append(row)
        for feature in row["gold_features"]:
            feature_groups[feature].append(row)
        database_groups[str(prepared[str(row["sample_id"])]["db_id"])].append(row)

    database_accuracies = [
        _rate(sum(int(row["strict_execution_accuracy"]) for row in rows), len(rows))
        for rows in database_groups.values()
    ]
    confusion_output: dict[str, Any] = {}
    for feature, counts in sorted(feature_confusion.items()):
        tp = counts["true_positive"]
        fn = counts["false_negative"]
        fp = counts["false_positive"]
        confusion_output[feature] = {
            **{
                name: counts[name]
                for name in (
                    "true_positive",
                    "false_negative",
                    "false_positive",
                    "true_negative",
                )
            },
            "recall": _rate(tp, tp + fn),
            "precision": _rate(tp, tp + fp),
        }

    return {
        "schema_version": 1,
        "privacy": {
            "row_level_content_included": False,
            "excluded_fields": [
                "sample_id",
                "question",
                "gold_sql",
                "prediction_sql",
                "database_id",
                "database_path",
            ],
        },
        "overall": _group_summary(enriched),
        "by_gold_complexity": {
            name: _group_summary(rows) for name, rows in sorted(complexity_groups.items())
        },
        "by_gold_feature": {
            name: _group_summary(rows) for name, rows in sorted(feature_groups.items())
        },
        "gold_prediction_feature_confusion": confusion_output,
        "database_aggregate": {
            "database_count": len(database_groups),
            "examples_per_database": {
                "min": min(map(len, database_groups.values())),
                "max": max(map(len, database_groups.values())),
            },
            "strict_accuracy_quantiles": _quantiles(database_accuracies),
        },
    }
