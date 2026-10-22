import pytest

from verireason.repair_probe import (
    execution_feedback,
    load_repair_probe_config,
    repair_messages,
    resolve_repair_action,
    summarize_repair_probe,
)
from verireason.sql_evaluator import QueryResult


def config():
    return load_repair_probe_config("configs/e6_execution_repair_probe.toml")


def test_config_is_inference_only_and_training_only():
    value = config()
    assert value["evidence"]["optimizer_updates"] == 0
    assert value["source"]["validation_payloads_read"] == 0
    assert value["source"]["heldout_payloads_read"] == 0


def test_feedback_is_bounded_and_never_mentions_correctness_or_gold():
    result = QueryResult(
        "ok",
        rows=((1, "x" * 200), (2, "b"), (3, "c"), (4, "d")),
        column_count=2,
        row_count=4,
    )
    feedback = execution_feedback(result, max_rows=3, max_characters=180)
    assert len(feedback) <= 180 and "Rows returned: 4" in feedback
    assert "correct" not in feedback.lower() and "gold" not in feedback.lower()
    assert "(4" not in feedback


@pytest.mark.parametrize(
    ("status", "error", "fragment"),
    [
        ("execution_error", "no such column: t.bad", "unknown column: t.bad"),
        ("execution_error", "no such table: missing", "unknown table: missing"),
        ("syntax_error", 'near "FROM": syntax error', "syntax error near: FROM"),
        ("unsafe", "DROP rejected", "read-only safety policy"),
        ("timeout", "private/path/query", "wall-clock limit"),
    ],
)
def test_failure_feedback_is_normalized(status, error, fragment):
    text = execution_feedback(QueryResult(status, error=error), max_rows=3, max_characters=300)
    assert fragment in text and "private/path/query" not in text


def test_feedback_and_control_prompts_differ_only_in_observation_contract():
    source = [{"role": "system", "content": "SQL only"}, {"role": "user", "content": "q"}]
    feedback = repair_messages(source, "SELECT bad", arm="feedback", feedback="unknown column")
    control = repair_messages(source, "SELECT bad", arm="control", feedback="unknown column")
    assert feedback[:-1] == control[:-1]
    assert "unknown column" in feedback[-1]["content"]
    assert "unknown column" not in control[-1]["content"]
    assert "gold" not in str(feedback).lower()


def test_action_contract_accepts_keep_or_strict_sql_only():
    assert resolve_repair_action(" KEEP ", "SELECT 1") == ("SELECT 1", "keep", True)
    assert resolve_repair_action("SELECT 2;", "SELECT 1") == ("SELECT 2", "rewrite", True)
    assert resolve_repair_action("```sql\nSELECT 2\n```", "SELECT 1")[0] is None


def records(*, feedback_repairs=(0, 4), control_repairs=(), feedback_damage=(), control_damage=()):
    rows = []
    for state in range(32):
        initial = int(state >= 19)
        for arm in ("feedback", "control"):
            for candidate in range(2):
                if initial:
                    damaged = state in (feedback_damage if arm == "feedback" else control_damage)
                    final = int(not damaged)
                else:
                    repaired = state in (feedback_repairs if arm == "feedback" else control_repairs)
                    final = int(repaired and candidate == 0)
                rows.append(
                    {
                        "state_index": state,
                        "example_id": str(state // 4),
                        "arm": arm,
                        "candidate": candidate,
                        "initial_reward": initial,
                        "final_reward": final,
                        "strict_output": True,
                        "final_status": "ok",
                    }
                )
    return rows


def test_gate_requires_feedback_specific_repairs_across_examples():
    result = summarize_repair_probe(records(), config())
    assert result["gate_passed"] is True
    assert result["feedback_only_repaired_states"] == 2
    assert result["distinct_feedback_only_examples"] == 2
    assert result["optimizer_updates"] == 0


def test_gate_fails_when_extra_attempts_explain_repairs_or_feedback_damages_more():
    tied = summarize_repair_probe(records(control_repairs=(0, 4)), config())
    assert tied["gate_passed"] is False
    damaged = summarize_repair_probe(records(feedback_damage=(19,), control_damage=()), config())
    assert damaged["gate_passed"] is False
    assert damaged["gate_checks"]["damage_control"] is False


def test_incomplete_result_is_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        summarize_repair_probe(records()[:-1], config())
