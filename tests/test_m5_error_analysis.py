import pytest

from verireason.m5_error_analysis import analyze_m4_errors


def test_error_analysis_aggregates_without_row_level_content() -> None:
    prepared = [
        {"sample_id": "a", "db_id": "db1", "target_sql": "SELECT * FROM t"},
        {
            "sample_id": "b",
            "db_id": "db2",
            "target_sql": "SELECT COUNT(*) FROM t JOIN u ON t.id=u.id",
        },
        {
            "sample_id": "c",
            "db_id": "db2",
            "target_sql": "SELECT id FROM t WHERE id IN (SELECT id FROM u)",
        },
    ]
    evaluation = [
        {
            "sample_id": "a",
            "strict_execution_accuracy": 1,
            "bird_ex_behavior_compatible": 1,
            "failure_category": "strict_correct",
            "extraction": {"sql": "SELECT * FROM t"},
        },
        {
            "sample_id": "b",
            "strict_execution_accuracy": 0,
            "bird_ex_behavior_compatible": 0,
            "failure_category": "wrong_result",
            "extraction": {"sql": "SELECT COUNT(*) FROM t"},
        },
        {
            "sample_id": "c",
            "strict_execution_accuracy": 0,
            "bird_ex_behavior_compatible": 0,
            "failure_category": "execution_error",
            "extraction": {"sql": "SELECT id FROM t"},
        },
    ]

    report = analyze_m4_errors(evaluation, prepared)

    assert report["overall"]["examples"] == 3
    assert report["overall"]["strict_accuracy"] == pytest.approx(1 / 3)
    assert report["by_gold_feature"]["join"]["strict_correct"] == 0
    assert report["gold_prediction_feature_confusion"]["join"]["false_negative"] == 1
    assert report["gold_prediction_feature_confusion"]["subquery"]["false_negative"] == 1
    assert report["database_aggregate"]["database_count"] == 2
    serialized = str(report)
    assert "SELECT" not in serialized
    assert "db1" not in serialized


def test_error_analysis_requires_identical_unique_ids() -> None:
    row = {
        "sample_id": "a",
        "strict_execution_accuracy": 1,
        "bird_ex_behavior_compatible": 1,
        "failure_category": "strict_correct",
        "extraction": {"sql": "SELECT 1"},
    }
    with pytest.raises(ValueError, match="sets differ"):
        analyze_m4_errors([row], [])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_m4_errors([row, row], [{"sample_id": "a", "db_id": "d", "target_sql": "SELECT 1"}])
