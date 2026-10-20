"""CPU analytical loss tests plus optional real Qwen tokenizer tests."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from verireason.e5_credit import (
    ClauseSpan,
    ExecutionCredit,
    TokenCredit,
    clause_spans,
    localize_execution,
)
from verireason.e5_diagnostics import CASES, create_database
from verireason.e5_interface import (
    advantage_matrix,
    attach_token_advantages,
    mock_dapo_loss,
    qwen_token_credit,
    terminal_advantages,
    verified_offsets,
)
from verireason.e5_interface_audit import MAPPING_SQL, TRAINING_SUMMARIES, training_source_audit

TOKENIZER = Path(
    os.environ.get("E5_TOKENIZER_JSON", "artifacts/e5-interface/qwen3-tokenizer/tokenizer.json")
)


@pytest.fixture
def tokenizer():
    package = pytest.importorskip("tokenizers", reason="optional E5 real-tokenizer CPU gate")
    if not TOKENIZER.is_file():
        pytest.skip("fixed-revision Qwen tokenizer absent; run the mandatory E5 interface gate")
    return package.Tokenizer.from_file(str(TOKENIZER))


def encoded_row(tokenizer, sql):
    eos = tokenizer.token_to_id("<|endoftext|>")
    ids = tuple(tokenizer.encode(sql, add_special_tokens=False).ids) + (eos, eos)
    return ids, (1,) * (len(ids) - 1) + (0,), eos


@pytest.mark.parametrize("sql", MAPPING_SQL)
def test_real_qwen_original_offsets_and_repeated_text(tokenizer, sql):
    span = next(s for s in clause_spans(sql) if s.clause == "WHERE")
    ids, valid, eos = encoded_row(tokenizer, sql)
    offsets, reason = verified_offsets(tokenizer, sql, ids, valid, eos_id=eos)
    assert reason == "verified_ids_and_text"
    assert offsets is not None
    # Geometry-only fixture: nested/alias cases do NOT authorize execution localization.
    output = qwen_token_credit(
        ExecutionCredit(0, "geometry_fixture", span, sql),
        sql,
        ids,
        valid,
        tokenizer,
        eos_id=eos,
        terminal_advantage=-0.25,
    )
    assert output.reason == "localized"
    assert not output.mask[-1] and not output.mask[-2]  # pad and EOS
    for selected, (start, end) in zip(output.mask, offsets, strict=True):
        if selected:
            assert start < span.end and end > span.start
        elif end > start:
            assert not sql[max(start, span.start) : min(end, span.end)].strip()


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_text",
        "missing_eos",
        "truncated",
        "bad_mask",
        "interior_eos",
        "ambiguous_offsets",
        "no_offset_api",
    ],
)
def test_real_qwen_uncertainty_falls_back(tokenizer, failure):
    sql = MAPPING_SQL[0]
    span = next(s for s in clause_spans(sql) if s.clause == "WHERE")
    ids, valid, eos = encoded_row(tokenizer, sql)
    kw = {}
    if failure == "wrong_text":
        sql = sql.replace("20", "21")
    elif failure == "missing_eos":
        ids, valid = ids[:-2], valid[:-2]
    elif failure == "truncated":
        kw["truncated"] = True
    elif failure == "bad_mask":
        valid = (0,) + valid[1:]
    elif failure == "interior_eos":
        ids = (eos,) + ids[1:]
    elif failure in {"ambiguous_offsets", "no_offset_api"}:
        base = tokenizer
        encoded = base.encode(sql, add_special_tokens=False)
        offsets = [(0, 1)] * len(encoded.ids)
        tokenizer = SimpleNamespace(decode=base.decode)
        if failure == "ambiguous_offsets":
            tokenizer.encode = lambda *args, **kwargs: SimpleNamespace(
                ids=encoded.ids, offsets=offsets
            )
    result = qwen_token_credit(
        ExecutionCredit(0, "fixture", span, sql),
        sql,
        ids,
        valid,
        tokenizer,
        eos_id=eos,
        terminal_advantage=-0.25,
        **kw,
    )
    assert result.reason != "localized"
    assert result.advantages == (-0.25,) * len(ids)


def test_noncanonical_generated_ids_same_decoded_text_abstain(tokenizer):
    sql = "SELECT"
    # Same text, alternative generation segmentation: re-encoded offsets are not its offsets.
    ids = tuple(tokenizer.token_to_id(c) for c in sql)
    eos = tokenizer.token_to_id("<|endoftext|>")
    assert tokenizer.decode(list(ids), skip_special_tokens=False) == sql
    offsets, reason = verified_offsets(
        tokenizer, sql, ids + (eos,), (1,) * (len(ids) + 1), eos_id=eos
    )
    assert offsets is None and reason == "generated_segmentation_mismatch"


def test_boundary_inside_one_bpe_token_abstains(tokenizer):
    sql = "SELECT id FROM sales"
    ids, valid, eos = encoded_row(tokenizer, sql)
    credit = ExecutionCredit(0, "geometry_fixture", ClauseSpan("SELECT", 2, 4), sql)
    assert (
        qwen_token_credit(
            credit, sql, ids, valid, tokenizer, eos_id=eos, terminal_advantage=-1
        ).reason
        == "e2_terminal_only"
    )


def loss_fixture(mode="local_unscaled"):
    a = terminal_advantages((1, 0, 0, 0))
    masks = ((1, 1, 1, 1, 1, 1),) * 4
    credits = (TokenCredit((a[0],) * 6, (1,) * 6, "e2_terminal_only"),) + tuple(
        TokenCredit((0, 0, 0, 0, x, x), (0, 0, 0, 0, 1, 1), "localized") for x in a[1:]
    )
    adv = advantage_matrix(a, credits, masks, mode=mode)
    logits = (((0.2, -0.1, 0.4),) * 6,) * 4
    labels = ((0,) * 6,) * 4
    logp = 0.2 - math.log(sum(math.exp(x) for x in logits[0][0]))
    return logits, labels, ((logp,) * 6,) * 4, adv, masks


def test_local_loss_only_changes_target_token_gradients_and_positive_anchor():
    baseline = mock_dapo_loss(*loss_fixture("terminal"), num_items_in_batch=24)
    local = mock_dapo_loss(*loss_fixture(), num_items_in_batch=24)
    assert local.logits_gradient[0] == baseline.logits_gradient[0]
    for row in range(1, 4):
        assert local.logits_gradient[row][:4] == ((0.0,) * 3,) * 4
        assert local.logits_gradient[row][4:] == baseline.logits_gradient[row][4:]
    assert local.negative_credit_mass == baseline.negative_credit_mass / 3


def test_mass_control_restores_credit_mass_but_increases_local_scale():
    baseline = mock_dapo_loss(*loss_fixture("terminal"), num_items_in_batch=24)
    matched = mock_dapo_loss(*loss_fixture("local_mass_matched"), num_items_in_batch=24)
    assert matched.negative_credit_mass == baseline.negative_credit_mass
    assert matched.logits_gradient[1][-1] == pytest.approx(
        tuple(3 * x for x in baseline.logits_gradient[1][-1])
    )


@pytest.mark.parametrize("ratio", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("a", [-0.25, 0.0, 0.75])
def test_mock_loss_gradient_matches_finite_difference(ratio, a):
    z = (0.2, -0.1, 0.4)
    logp = z[0] - math.log(sum(math.exp(x) for x in z))
    old = ((logp - math.log(ratio),),)

    def evaluate(values):
        return mock_dapo_loss(((values,),), ((0,),), old, ((a,),), ((1,),), num_items_in_batch=7)

    result = evaluate(z)
    for i in range(3):
        up, down = list(z), list(z)
        up[i] += 1e-6
        down[i] -= 1e-6
        numerical = (evaluate(tuple(up)).loss - evaluate(tuple(down)).loss) / 2e-6
        assert result.logits_gradient[0][0][i] == pytest.approx(numerical, abs=1e-9)


def test_accumulation_and_padding_do_not_change_dapo_denominator():
    args = loss_fixture()
    full = mock_dapo_loss(*args, num_items_in_batch=24)
    micro = [mock_dapo_loss(*(x[i : i + 1] for x in args), num_items_in_batch=24) for i in range(4)]
    assert sum(x.loss for x in micro) == pytest.approx(full.loss)
    assert tuple(x.logits_gradient[0] for x in micro) == full.logits_gradient
    logits, labels, old, adv, masks = args
    result = mock_dapo_loss(
        tuple(r + ((900.0, -900.0, 0.0),) for r in logits),
        tuple(r + (0,) for r in labels),
        tuple(r + (0.0,) for r in old),
        tuple(r + (100.0,) for r in adv),
        tuple(r + (0,) for r in masks),
        num_items_in_batch=24,
    )
    assert result.loss == pytest.approx(full.loss)
    assert all(r[-1] == (0.0, 0.0, 0.0) for r in result.logits_gradient)


@pytest.mark.parametrize("rewards", [(0, 0, 0, 0), (1, 1, 1, 1)])
def test_constant_terminal_groups_stay_zero(rewards):
    assert terminal_advantages(rewards) == (0.0,) * 4


def test_uncertain_mapping_keeps_exact_e2_loss():
    args = loss_fixture("terminal")
    a = terminal_advantages((1, 0, 0, 0))
    credits = tuple(TokenCredit((x,) * 6, (1,) * 6, "offset_failure") for x in a)
    matrix = advantage_matrix(a, credits, args[-1])
    assert matrix == args[3]
    assert mock_dapo_loss(*args, num_items_in_batch=24) == mock_dapo_loss(
        args[0], args[1], args[2], matrix, args[4], num_items_in_batch=24
    )


def test_nonfinite_loss_and_advantages_fail():
    args = list(loss_fixture())
    args[3] = ((math.nan,) * 6,) * 4
    with pytest.raises(ValueError, match="nonfinite"):
        mock_dapo_loss(*args, num_items_in_batch=24)


@pytest.mark.parametrize("case", [CASES[i] for i in (4, 6, 15, 17)], ids=lambda c: c.name)
def test_real_tokenizer_does_not_override_execution_gate(tokenizer, tmp_path, case):
    database = tmp_path / "synthetic.sqlite"
    create_database(database)
    credit = localize_execution(database, case.gold, case.prediction, split="train")
    ids, valid, eos = encoded_row(tokenizer, case.prediction)
    mapped = qwen_token_credit(
        credit, case.prediction, ids, valid, tokenizer, eos_id=eos, terminal_advantage=-0.25
    )
    assert (mapped.reason == "localized") == (case.expected_clause is not None)


def test_training_audit_reads_only_allowlisted_training_aggregates(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = Path.read_text
    opened = []

    def record(path, *args, **kwargs):
        opened.append(path.relative_to(root).as_posix())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record)
    result = training_source_audit(root)
    assert tuple(opened) == TRAINING_SUMMARIES
    assert result["real_localization_coverage"] is None
    assert result["real_token_mapping_coverage"] is None
    assert result["real_training_rows_examined"] == 0


def test_fully_truncated_completion_mask_keeps_zero_loss():
    a = (-0.25,)
    credit = (TokenCredit(a, (1,), "truncated_or_shape"),)
    matrix = advantage_matrix(a, credit, ((0,),))
    result = mock_dapo_loss(
        (((100.0, -100.0),),), ((0,),), ((0.0,),), matrix, ((0,),), num_items_in_batch=1
    )
    assert result.loss == 0 and result.logits_gradient == (((0.0, 0.0),),)


def test_scored_batch_seam_changes_only_advantages():
    a = terminal_advantages((1, 0, 0, 0))
    original = {
        "advantages": a,
        "completion_mask": ((1,) * 6,) * 4,
        "completion_ids": ((10,) * 6,) * 4,
        "num_items_in_batch": 24,
        "old_per_token_logps": object(),
    }
    credits = tuple(TokenCredit((x,) * 6, (1,) * 6, "e2_terminal_only") for x in a)
    patched = attach_token_advantages(original, credits)
    assert original["advantages"] == a
    assert patched["advantages"] == tuple((x,) * 6 for x in a)
    assert all(patched[k] is v for k, v in original.items() if k != "advantages")


def test_training_audit_rejects_contaminated_scope(tmp_path):
    import json

    first = tmp_path / TRAINING_SUMMARIES[0]
    first.parent.mkdir(parents=True)
    first.write_text(
        json.dumps(
            {"scope": "validation", "heldout_examples_loaded": 0, "heldout_examples_evaluated": 0}
        )
    )
    with pytest.raises(ValueError, match="training-only"):
        training_source_audit(tmp_path)


def test_loss_rejects_overflow_ratio():
    with pytest.raises(ValueError, match="nonfinite importance"):
        mock_dapo_loss(
            (((0.0, 0.0),),), ((0,),), ((-1000.0,),), ((-0.25,),), ((1,),), num_items_in_batch=1
        )
