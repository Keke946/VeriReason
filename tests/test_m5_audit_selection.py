import json
import stat

import pytest

from verireason.m5_audit_selection import select_audit_rows, write_private_json


def _fixture() -> tuple[list[dict], list[dict]]:
    prepared = []
    evaluated = []
    sqls = [
        "SELECT 1",
        "SELECT COUNT(*) FROM t JOIN u ON t.id=u.id",
        "SELECT id FROM t WHERE id IN (SELECT id FROM u)",
    ]
    categories = ["wrong_result", "execution_error", "syntax_error"]
    for index, (sql, category) in enumerate(zip(sqls, categories, strict=True)):
        sample_id = str(index)
        prepared.append(
            {
                "sample_id": sample_id,
                "db_id": "private_db",
                "question": "private question",
                "target_sql": sql,
            }
        )
        evaluated.append(
            {
                "sample_id": sample_id,
                "failure_category": category,
                "prediction_status": "ok",
                "strict_comparison_status": "not_equal",
                "extraction": {"sql": "SELECT 2"},
            }
        )
    return evaluated, prepared


def test_selection_is_deterministic_and_public_receipt_is_private_free() -> None:
    evaluated, prepared = _fixture()
    first, receipt = select_audit_rows(evaluated, prepared, seed="fixed")
    second, repeated = select_audit_rows(evaluated, prepared, seed="fixed")

    assert first == second
    assert receipt == repeated
    assert receipt["selected_examples"] == 3
    public = json.dumps(receipt)
    assert "private question" not in public
    assert "private_db" not in public
    assert "SELECT" not in public


def test_private_writer_forces_owner_only_permissions(tmp_path) -> None:
    output = tmp_path / "private" / "audit.json"
    write_private_json(output, [{"gold_sql": "SELECT 1"}])

    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_selection_rejects_identity_mismatch() -> None:
    evaluated, prepared = _fixture()
    with pytest.raises(ValueError, match="sets differ"):
        select_audit_rows(evaluated[:-1], prepared, seed="fixed")
