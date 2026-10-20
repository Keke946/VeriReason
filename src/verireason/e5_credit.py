"""CPU-only E5 prototype: conservative, reference-relative execution localization.

No trainer dispatch, dataset loader, model import, or scalar shaping reward lives here.
Offsets refer to the original completion, not SQL regenerated from its AST.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from sqlglot import exp, parse, tokenize
from sqlglot.errors import SqlglotError

from verireason.sql_evaluator import (
    DEFAULT_LIMITS,
    EvaluatorLimits,
    analyze_sql,
    compare_strict_results,
    execute_read_only,
)
from verireason.sql_features import extract_sql_features
from verireason.sqlite_schema import load_sqlite_schema

CLAUSES = {
    "SELECT": ("expressions", "distinct"),
    "FROM": ("from_", "joins"),
    "WHERE": ("where",),
    "GROUP_BY": ("group",),
    "HAVING": ("having",),
    "ORDER_BY": ("order",),
    "LIMIT": ("limit",),
}


@dataclass(frozen=True)
class ClauseSpan:
    clause: str
    start: int
    end: int  # exclusive Python character offset


@dataclass(frozen=True)
class ExecutionCredit:
    terminal_reward: int
    reason: str
    span: ClauseSpan | None = None
    source_completion: str | None = None


@dataclass(frozen=True)
class TokenCredit:
    advantages: tuple[float, ...]
    mask: tuple[int, ...]
    reason: str


def _flat_query(sql: str) -> tuple[exp.Select | None, str]:
    if not analyze_sql(sql).accepted:
        return None, "unsafe_or_invalid"
    try:
        trees = parse(sql, read="sqlite")
    except (SqlglotError, ValueError, RecursionError):
        return None, "parse_failure"
    if len(trees) != 1 or not isinstance(trees[0], exp.Select):
        return None, "unsupported_query"
    tree = trees[0]
    if len(list(tree.find_all(exp.Select))) != 1 or tree.args.get("with_"):
        return None, "nested_query"
    if any(isinstance(n, exp.Alias | exp.TableAlias) for n in tree.walk()):
        return None, "alias_abstention"
    allowed = {key for keys in CLAUSES.values() for key in keys}
    if any(value and key not in allowed for key, value in tree.args.items()):
        return None, "unsupported_modifier"
    if any(isinstance(n, exp.Window | exp.Subquery) for n in tree.walk()):
        return None, "unsupported_scope"
    deterministic_functions = {
        "SUM",
        "COUNT",
        "MIN",
        "MAX",
        "AVG",
        "ABS",
        "LOWER",
        "UPPER",
        "COALESCE",
        "NULLIF",
        "LENGTH",
        "ROUND",
        "CAST",
    }
    if any(f.sql_name() not in deterministic_functions for f in tree.find_all(exp.Func)):
        return None, "function_abstention"
    # SQLite allows arbitrary bare columns in aggregate queries; execution is not evidence
    # that such a reference or repair has deterministic semantics.
    group = tree.args.get("group")
    if group is not None or any(tree.find_all(exp.AggFunc)):
        group_columns = {
            c.sql() for c in (group.expressions if group else []) if isinstance(c, exp.Column)
        }
        for selected in tree.expressions:
            for column in selected.find_all(exp.Column):
                parent = column.parent
                inside_aggregate = False
                while parent is not None and parent is not tree:
                    inside_aggregate |= isinstance(parent, exp.AggFunc)
                    parent = parent.parent
                if not inside_aggregate and column.sql() not in group_columns:
                    return None, "bare_aggregate_column"
    tables = [t.name.casefold() for t in tree.find_all(exp.Table)]
    if not tables or len(tables) != len(set(tables)):
        return None, "ambiguous_sources"
    # LIMIT needs a proven total ordering, not just the presence of ORDER BY.
    if tree.args.get("limit"):
        return None, "limit_total_order_unproven"
    return tree, "supported"


def clause_spans(sql: str) -> tuple[ClauseSpan, ...]:
    """Tokenize clause boundaries; quoted keywords and nested expressions are not boundaries."""
    tokens = tokenize(sql, read="sqlite")
    depth = 0
    starts: list[tuple[str, int]] = []
    end = len(sql.rstrip())
    for token in tokens:
        kind = token.token_type.name
        if kind == "L_PAREN":
            depth += 1
        elif kind == "R_PAREN":
            depth -= 1
        elif depth == 0 and kind in CLAUSES:
            starts.append((kind, token.start))
        elif depth == 0 and kind == "SEMICOLON":
            end = token.start
    if not starts or starts[0][0] != "SELECT" or len({c for c, _ in starts}) != len(starts):
        raise ValueError("ambiguous clause boundaries")
    return tuple(
        ClauseSpan(clause, start, starts[i + 1][1] if i + 1 < len(starts) else end)
        for i, (clause, start) in enumerate(starts)
    )


def _signature(tree: exp.Select, keys: tuple[str, ...]) -> tuple[str, ...]:
    result = []
    for key in keys:
        value = tree.args.get(key)
        values = value if isinstance(value, list) else [value]
        result.extend(
            v.sql(dialect="sqlite", comments=False) if isinstance(v, exp.Expression) else str(v)
            for v in values
        )
    return tuple(result)


def localize_execution(
    database: str | Path,
    gold_sql: str,
    completion: str,
    *,
    split: str,
    limits: EvaluatorLimits = DEFAULT_LIMITS,
) -> ExecutionCredit:
    """Only a training caller may execute. A split label is not a provenance proof.

    The shipped runner constructs its own synthetic DB and cannot accept external data paths.
    Future real-data use must enter through the existing frozen train-manifest loader.
    """
    if split != "train":
        raise ValueError("E5 requires training-only inputs")
    gold = execute_read_only(database, gold_sql, limits)
    if not gold.successful:
        raise ValueError("gold preflight failed; terminal credit cannot be trusted")
    prediction = execute_read_only(database, completion, limits)
    comparison = compare_strict_results(prediction, gold, limits)
    terminal = int(comparison.equal)
    if terminal:
        return ExecutionCredit(1, "terminal_correct")
    if prediction.status != "ok":
        return ExecutionCredit(0, "execution_failure_abstention")
    if comparison.status == "comparator_resource_limit":
        return ExecutionCredit(0, "comparison_resource_abstention")
    trees = []
    for sql in (gold_sql, completion):
        tree, reason = _flat_query(sql)
        if tree is None:
            return ExecutionCredit(0, reason)
        trees.append(tree)
    gold_tree, pred_tree = trees
    schema = load_sqlite_schema(database)
    for sql in (gold_sql, completion):
        if not extract_sql_features(sql, schema).supported:
            return ExecutionCredit(0, "schema_or_determinism_abstention")
    changed = [
        c
        for c, keys in CLAUSES.items()
        if _signature(gold_tree, keys) != _signature(pred_tree, keys)
    ]
    if len(changed) != 1:
        return ExecutionCredit(0, "ambiguous_clause_difference")
    clause = changed[0]
    try:
        pred_spans = {s.clause: s for s in clause_spans(completion)}
        gold_spans = {s.clause: s for s in clause_spans(gold_sql)}
    except (SqlglotError, ValueError, RecursionError):
        return ExecutionCredit(0, "span_abstention")
    if clause not in pred_spans or clause not in gold_spans:
        return ExecutionCredit(0, "missing_clause_no_token_target")
    span, replacement = pred_spans[clause], gold_spans[clause]
    repaired = (
        completion[: span.start]
        + gold_sql[replacement.start : replacement.end].rstrip()
        + " "
        + completion[span.end :]
    )
    repaired_tree, _ = _flat_query(repaired)
    # Reparse the actual text splice: offsets must describe the same AST intervention.
    if repaired_tree is None or any(
        _signature(repaired_tree, keys) != _signature(gold_tree, keys) for keys in CLAUSES.values()
    ):
        return ExecutionCredit(0, "repair_parse_abstention")
    repaired_result = execute_read_only(database, repaired, limits)
    if not compare_strict_results(repaired_result, gold, limits).equal:
        return ExecutionCredit(0, "repair_not_strict_correct")
    return ExecutionCredit(0, "single_clause_execution_repair", span, completion)


def token_credit(
    credit: ExecutionCredit,
    completion: str,
    offsets: tuple[tuple[int, int], ...],
    *,
    terminal_advantage: float,
) -> TokenCredit:
    """Mask only negative E2 advantages; never create reward in constant-zero groups.

    The caller supplies the SAME group advantage E2 would use, and subsequently applies the
    ordinary padding/EOS/truncation mask and the unchanged DAPO loss denominator. No rescaling by
    selected-token count is allowed. (0, 0) denotes a special token. Real tokenizer alignment and
    the TRL loss seam remain a separate CPU integration gate.
    """
    if not math.isfinite(terminal_advantage):
        raise ValueError("terminal advantage must be finite")
    fallback = TokenCredit(
        (terminal_advantage,) * len(offsets), (1,) * len(offsets), "e2_terminal_only"
    )
    if (
        credit.span is None
        or credit.terminal_reward
        or terminal_advantage >= 0
        or credit.source_completion != completion
    ):
        return fallback
    span = credit.span
    if not 0 <= span.start < span.end <= len(completion):
        return fallback
    covered = [False] * len(completion)
    mask = []
    previous = 0
    for start, end in offsets:
        if (start, end) == (0, 0):
            mask.append(0)
            continue
        if not 0 <= previous <= start < end <= len(completion):
            return fallback
        previous = end
        covered[start:end] = [True] * (end - start)
        intersects = start < span.end and end > span.start
        outside = completion[start : min(end, span.start)] + completion[max(start, span.end) : end]
        if intersects and outside.strip():
            return fallback
        mask.append(int(intersects))
    if not any(mask) or any(
        not seen and not c.isspace() for c, seen in zip(completion, covered, strict=True)
    ):
        return fallback
    return TokenCredit(tuple(terminal_advantage * m for m in mask), tuple(mask), "localized")
