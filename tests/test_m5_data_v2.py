import json

import pytest

from verireason.m5_data_v2 import build_data_v2_candidates


def _row(sample_id: str, sql: str, split: str = "train") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "db_id": "private_db",
        "target_sql": sql,
        "prompt_sha256": sample_id * 64,
    }


def test_candidate_policy_is_training_only_and_outcome_blind() -> None:
    rows = [
        _row("a", "SELECT * FROM t"),
        _row("b", "SELECT * FROM t JOIN u ON t.id=u.id JOIN v ON u.id=v.id"),
        _row("c", "SELECT x, COUNT(*) FROM t GROUP BY x HAVING COUNT(*) > 1"),
        _row("d", "SELECT * FROM t WHERE id IN (SELECT id FROM u)"),
    ]

    private, public = build_data_v2_candidates(rows)

    assert {row["sample_id"] for row in private} == {"b", "c", "d"}
    assert public["candidate_examples"] == 3
    assert public["policy"]["uses_model_outcomes"] is False
    assert public["policy"]["uses_validation_rows"] is False
    serialized = json.dumps(public)
    assert "private_db" not in serialized
    assert "SELECT" not in serialized


def test_candidate_policy_rejects_non_train_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="training rows only"):
        build_data_v2_candidates([_row("a", "SELECT 1", split="validation")])
    with pytest.raises(ValueError, match="duplicate"):
        build_data_v2_candidates(
            [
                _row("a", "SELECT * FROM t JOIN u ON t.id=u.id JOIN v ON u.id=v.id"),
                _row("a", "SELECT x FROM t GROUP BY x"),
            ]
        )
