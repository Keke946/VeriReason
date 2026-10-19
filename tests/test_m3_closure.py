import pytest

from verireason.m3_closure import _money_summary, recompute_evaluation


def _evaluation() -> dict:
    rows = [
        {
            "sample_id": str(index),
            "prompt_sha256": f"{index:064x}",
            "extraction": {"strict_sql_only": strict},
            "bird_ex_behavior_compatible": correct,
            "strict_execution_accuracy": correct,
            "prediction_status": "ok",
            "failure_category": "strict_correct" if correct else "wrong_result",
            "error": None,
        }
        for index, (correct, strict) in enumerate([(1, True), (0, True), (1, True), (0, False)] * 2)
    ]
    return {
        "examples": rows,
        "metrics": {
            "example_count": 8,
            "bird_ex_behavior_compatible": 0.5,
            "strict_execution_accuracy": 0.5,
            "strict_sql_only_rate": 0.75,
            "generated_tokens": 80,
            "generation_seconds": 10.0,
            "completion_tokens_per_second": 8.0,
            "failure_categories": {"strict_correct": 4, "wrong_result": 4},
        },
    }


def test_closure_recomputes_rows_and_labels_end_to_end_throughput() -> None:
    summary = recompute_evaluation(_evaluation())
    assert summary["bird_ex_behavior_compatible"] == 0.5
    assert summary["strict_execution_accuracy"] == 0.5
    assert summary["strict_sql_only_rate"] == 0.75
    assert summary["throughput_scope"] == "generation_plus_evaluator_end_to_end"


def test_closure_rejects_evaluator_errors() -> None:
    evaluation = _evaluation()
    evaluation["examples"][0]["failure_category"] = "evaluator_error"
    with pytest.raises(ValueError, match="evaluator_error"):
        recompute_evaluation(evaluation)


def test_closure_records_exact_provider_invoice_arithmetic() -> None:
    summary = _money_summary(
        (
            ("2026-08-24T04:00:17+08:00", "1.43241388"),
            ("2026-08-24T04:56:46+08:00", "2.45042777"),
        ),
        "9.64484428",
        "24.18879004",
    )
    assert summary["actual_provider_cost_cny"] == "3.88284165"
    assert summary["post_m3_cumulative_cost_cny"] == "13.52768593"
    assert summary["program_wall_estimate_is_actual_cost"] is False
