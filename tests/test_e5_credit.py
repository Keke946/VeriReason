"""E5 tests execute only disposable, hand-authored SQLite training fixtures."""

import math
import sqlite3

import pytest

from verireason.e5_credit import (
    ClauseSpan,
    ExecutionCredit,
    clause_spans,
    localize_execution,
    token_credit,
)
from verireason.e5_diagnostics import CASES, create_database
from verireason.m6_component_scorer import SQLiteRewardComponentScorer
from verireason.reward_contract import RewardComponents, group_rewards
from verireason.trl_reward_adapter import RewardInput


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    create_database(path)
    return path


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_execution_credit(database, case):
    result = localize_execution(database, case.gold, case.prediction, split="train")
    assert result.terminal_reward == case.expected_terminal
    assert (result.span.clause if result.span else None) == case.expected_clause
    if result.span:
        assert case.prediction[result.span.start : result.span.end].strip()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales").fetchone() == (4,)


@pytest.mark.parametrize("split", ["validation", "heldout", "Mini-Dev", "", "TRAIN"])
def test_split_rejected_before_any_io(split):
    with pytest.raises(ValueError, match="training-only"):
        localize_execution("nonexistent", "SELECT 1", "SELECT 0", split=split)


def test_gold_failure_is_not_reward(database):
    with pytest.raises(ValueError, match="gold preflight"):
        localize_execution(database, "SELECT missing FROM sales", "SELECT 1", split="train")


def test_clause_tokenizer_quotes_and_parentheses():
    sql = "  SELECT 'WHERE', (1+2) FROM sales WHERE amount>10 ORDER BY id;"
    spans = clause_spans(sql)
    assert [s.clause for s in spans] == ["SELECT", "FROM", "WHERE", "ORDER_BY"]
    assert sql[spans[-1].start : spans[-1].end] == "ORDER BY id"


SQL = "SELECT id FROM sales WHERE amount>20"
SPAN = ClauseSpan("WHERE", SQL.index("WHERE"), len(SQL))
OFFSETS = tuple((i, i + 1) for i in range(len(SQL))) + ((0, 0),)
CREDIT = ExecutionCredit(0, "single_clause_execution_repair", SPAN, SQL)


def test_negative_advantage_only_hits_target_and_not_eos():
    output = token_credit(CREDIT, SQL, OFFSETS, terminal_advantage=-0.75)
    assert output.reason == "localized"
    assert output.advantages[: SPAN.start] == (0,) * SPAN.start
    assert output.advantages[SPAN.start : -1] == (-0.75,) * (len(SQL) - SPAN.start)
    assert output.advantages[-1] == 0


@pytest.mark.parametrize("advantage", [-1.0, 0.0, 1.0])
def test_abstention_exact_e2_fallback(advantage):
    output = token_credit(
        ExecutionCredit(0, "ambiguous"), SQL, OFFSETS, terminal_advantage=advantage
    )
    assert output.advantages == (advantage,) * len(OFFSETS)
    assert output.mask == (1,) * len(OFFSETS)


@pytest.mark.parametrize("advantage", [0.0, 1.0])
def test_never_invent_signal_or_localize_positive(advantage):
    assert token_credit(CREDIT, SQL, OFFSETS, terminal_advantage=advantage).advantages == (
        advantage,
    ) * len(OFFSETS)


def test_correct_completion_keeps_full_credit():
    output = token_credit(
        ExecutionCredit(1, "terminal_correct"), SQL, OFFSETS, terminal_advantage=1.0
    )
    assert output.mask == (1,) * len(OFFSETS)


@pytest.mark.parametrize(
    "offsets",
    [
        (),
        ((0, len(SQL)),),
        ((-1, 3),),
        ((0, 2), (1, 3)),
        ((0, len(SQL) + 1),),
        OFFSETS[1:],
        OFFSETS[:-2],
    ],
)
def test_bad_or_cross_boundary_offsets_fall_back(offsets):
    output = token_credit(CREDIT, SQL, offsets, terminal_advantage=-1)
    assert output.reason == "e2_terminal_only"
    assert output.advantages == (-1,) * len(offsets)


def test_nonfinite_advantage_rejected():
    with pytest.raises(ValueError, match="finite"):
        token_credit(CREDIT, SQL, OFFSETS, terminal_advantage=math.nan)


def test_stale_completion_credit_falls_back():
    assert (
        token_credit(CREDIT, SQL.replace("20", "21"), OFFSETS, terminal_advantage=-1).reason
        == "e2_terminal_only"
    )


@pytest.mark.parametrize("case", [CASES[i] for i in (0, 1, 4, 12, 20)])
def test_terminal_matches_existing_e2_scorer(database, case):
    scorer = SQLiteRewardComponentScorer(database.parent)
    existing = scorer(
        RewardInput(
            "synthetic prompt",
            case.prediction,
            (),
            {"gold_sql": case.gold, "database_path": database.name},
        )
    )
    result = localize_execution(database, case.gold, case.prediction, split="train")
    assert group_rewards([existing], arm="E2") == (float(result.terminal_reward),)


@pytest.mark.parametrize("rewards", [(1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1)])
def test_terminal_group_centering_is_preserved(rewards):
    raw = group_rewards([RewardComponents(r, 1, 0.0, 0) for r in rewards], arm="E2")
    # Existing E2 uses scale_rewards=none: center only, no additional z-score.
    advantages = tuple(r - sum(raw) / len(raw) for r in raw)
    for reward, advantage in zip(rewards, advantages, strict=True):
        credit = CREDIT if not reward else ExecutionCredit(1, "terminal_correct")
        output = token_credit(credit, SQL, OFFSETS, terminal_advantage=advantage)
        assert all(a == advantage * m for a, m in zip(output.advantages, output.mask, strict=True))
        if len(set(rewards)) == 1:
            assert not any(output.advantages)


def test_unicode_byte_offsets_do_not_silently_misalign():
    sql = "SELECT '猫' FROM sales"
    credit = ExecutionCredit(0, "repair", ClauseSpan("SELECT", 0, 11), sql)
    byte_offsets = tuple((i, i + 1) for i in range(len(sql.encode("utf-8"))))
    assert (
        token_credit(credit, sql, byte_offsets, terminal_advantage=-1).reason == "e2_terminal_only"
    )
