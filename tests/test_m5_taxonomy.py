import pytest

from verireason.m5_taxonomy import summarize_annotations, validate_annotations


def test_taxonomy_validation_and_summary_are_sql_free() -> None:
    audit = [
        {"sample_id": "a", "failure_category": "wrong_result", "gold_complexity": "simple"},
        {
            "sample_id": "b",
            "failure_category": "execution_error",
            "gold_complexity": "complex",
        },
    ]
    annotations = [
        {
            "sample_id": "a",
            "primary_label": "predicate",
            "secondary_labels": ["calculation"],
            "confidence": "high",
            "review_required": False,
        },
        {
            "sample_id": "b",
            "primary_label": "schema_grounding",
            "secondary_labels": [],
            "confidence": "medium",
            "review_required": True,
        },
    ]

    summary = summarize_annotations(validate_annotations(audit, annotations))

    assert summary["audited_examples"] == 2
    assert summary["primary_labels"] == {"predicate": 1, "schema_grounding": 1}
    assert summary["review_required"] == 1
    assert summary["privacy"] == {
        "row_level_content_included": False,
        "sample_ids_included": False,
        "sql_included": False,
    }


def test_taxonomy_rejects_missing_ids_and_unknown_labels() -> None:
    audit = [{"sample_id": "a", "failure_category": "wrong_result", "gold_complexity": "simple"}]
    with pytest.raises(ValueError, match="set differs"):
        validate_annotations(audit, [])
    with pytest.raises(ValueError, match="invalid primary"):
        validate_annotations(
            audit,
            [
                {
                    "sample_id": "a",
                    "primary_label": "invented",
                    "secondary_labels": [],
                    "confidence": "high",
                    "review_required": False,
                }
            ],
        )
