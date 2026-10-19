"""Conservative, schema-aware SQLite feature extraction for M6 dense rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ErrorLevel, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from verireason.reward_contract import active_dimension_structural_reward
from verireason.sql_evaluator import analyze_sql

FEATURE_DIMENSIONS = (
    "select_signatures",
    "tables",
    "columns",
    "join_signatures",
    "predicate_signatures",
    "group_by",
    "having",
    "order_by",
    "limit",
    "set_operations",
    "subquery_profile",
)

FEATURE_WEIGHTS = {
    "select_signatures": 0.20,
    "tables": 0.10,
    "columns": 0.10,
    "join_signatures": 0.15,
    "predicate_signatures": 0.15,
    "group_by": 0.08,
    "having": 0.05,
    "order_by": 0.07,
    "limit": 0.03,
    "set_operations": 0.04,
    "subquery_profile": 0.03,
}


@dataclass(frozen=True)
class SqlFeatureExtraction:
    supported: bool
    reason: str | None
    features: dict[str, frozenset[str]]


@dataclass(frozen=True)
class StructuralRewardResult:
    r_valid: int
    r_struct: float
    g_conf: int
    prediction_reason: str | None


class UnsupportedGoldStructureError(ValueError):
    """Gold SQL executed but cannot safely receive dense structural feedback."""


def _unsupported(reason: str) -> SqlFeatureExtraction:
    return SqlFeatureExtraction(False, reason, {})


def _typed_literal(node: exp.Literal) -> exp.Var:
    return exp.Var(this="<text>" if node.is_string else "<number>")


def _canonical(node: exp.Expression) -> str:
    copy = node.copy().transform(
        lambda item: _typed_literal(item) if isinstance(item, exp.Literal) else item
    )

    def sort_boolean(item: exp.Expression) -> exp.Expression:
        if isinstance(item, exp.And | exp.Or):
            terms = sorted((item.left, item.right), key=lambda term: term.sql(dialect="sqlite"))
            return item.__class__(this=terms[0], expression=terms[1])
        return item

    previous = None
    while previous != copy.sql(dialect="sqlite"):
        previous = copy.sql(dialect="sqlite")
        copy = copy.transform(sort_boolean)
    return copy.sql(dialect="sqlite", normalize=True, pretty=False).casefold()


def _schema_for_sqlglot(schema: dict[str, set[str]]) -> dict[str, dict[str, str]]:
    if not schema or any(not table or not columns for table, columns in schema.items()):
        raise ValueError("schema must contain named tables with columns")
    return {
        table.casefold(): {column.casefold(): "UNKNOWN" for column in columns}
        for table, columns in schema.items()
    }


def _has_unsupported_construct(expression: exp.Expression) -> str | None:
    if any(with_clause.args.get("recursive") for with_clause in expression.find_all(exp.With)):
        return "unsupported_recursive_cte"
    if any(isinstance(node, exp.Placeholder) for node in expression.walk()):
        return "unsupported_parameter"
    for join in expression.find_all(exp.Join):
        if join.args.get("method") and str(join.args["method"]).casefold() == "natural":
            return "unsupported_natural_join"
    for function in expression.find_all(exp.Func):
        name = function.sql_name().casefold()
        if isinstance(function, exp.Anonymous):
            name = function.name.casefold()
        if name in {
            "rand",
            "random",
            "randomblob",
            "current_time",
            "current_date",
            "current_timestamp",
        }:
            return "unsupported_nondeterministic_function"
        if name in {"date", "datetime", "julianday", "strftime", "time", "unixepoch"}:
            arguments = list(function.iter_expressions())
            if not arguments or any(
                isinstance(argument, exp.Literal)
                and argument.is_string
                and argument.this.casefold() in {"now", "localtime", "subsec", "subsecond"}
                for argument in arguments
            ):
                return "unsupported_nondeterministic_function"
    for table in expression.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            return "unsupported_table_function"
    return None


def _physical_table_map(scopes: list[Scope]) -> tuple[dict[str, str], set[str]]:
    aliases: dict[str, str] = {}
    tables: set[str] = set()
    for scope in scopes:
        for alias, (_, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table):
                physical = source.name.casefold()
                aliases[alias.casefold()] = physical
                tables.add(physical)
    return aliases, tables


def _scope_table_map(scope: Scope) -> dict[str, str]:
    physical_sources = [
        (alias.casefold(), source.name.casefold())
        for alias, (_, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    ]
    totals = {
        name: sum(other == name for _, other in physical_sources) for _, name in physical_sources
    }
    seen: dict[str, int] = {}
    result: dict[str, str] = {}
    for alias, name in physical_sources:
        seen[name] = seen.get(name, 0) + 1
        result[alias] = name if totals[name] == 1 else f"{name}#role={seen[name]}"
    return result


def _rewrite_physical_columns(node: exp.Expression, aliases: dict[str, str]) -> exp.Expression:
    copy = node.copy()
    for column in copy.find_all(exp.Column):
        table = column.table.casefold()
        if table in aliases:
            column.set("table", exp.to_identifier(aliases[table]))
    return copy


def _output_expressions(expression: exp.Expression) -> list[exp.Expression]:
    current = expression
    while isinstance(current, exp.SetOperation):
        current = current.this
    return list(current.expressions) if isinstance(current, exp.Select) else []


def _rewrite_with_lineage(node: exp.Expression, scope: Scope) -> exp.Expression:
    outputs = _output_expressions(scope.expression)
    output_aliases = {
        selected.alias.casefold(): selected.unalias() for selected in outputs if selected.alias
    }
    roles = _scope_table_map(scope)

    def rewrite(item: exp.Expression) -> exp.Expression:
        if not isinstance(item, exp.Column):
            return item
        table = item.table.casefold()
        if not table and item.name.casefold() in output_aliases:
            return _rewrite_with_lineage(output_aliases[item.name.casefold()], scope)
        selected = scope.selected_sources.get(table)
        if selected is None:
            return item
        source = selected[1]
        if isinstance(source, exp.Table):
            replacement = item.copy()
            replacement.set("table", exp.to_identifier(roles[table]))
            return replacement
        if isinstance(source, Scope):
            matches = [
                projected.unalias()
                for projected in _output_expressions(source.expression)
                if projected.alias_or_name.casefold() == item.name.casefold()
            ]
            if len(matches) != 1:
                raise ValueError("unresolved or duplicate derived-column lineage")
            return _rewrite_with_lineage(matches[0], source)
        raise ValueError("unsupported selected source")

    return node.copy().transform(rewrite)


def _add_occurrence(bucket: set[str], signature: str) -> None:
    occurrence = 1
    while f"{signature}#occurrence={occurrence}" in bucket:
        occurrence += 1
    bucket.add(f"{signature}#occurrence={occurrence}")


def _clause_items(expression: exp.Expression, key: str) -> list[exp.Expression]:
    clause = expression.args.get(key)
    if clause is None:
        return []
    expressions = list(clause.expressions)
    if expressions:
        return expressions
    return [clause.this] if clause.this is not None else []


def _boolean_signature(node: exp.Expression, aliases: dict[str, str]) -> str:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And | exp.Or):
        children: list[exp.Expression] = []

        def collect(current: exp.Expression) -> None:
            while isinstance(current, exp.Paren):
                current = current.this
            if isinstance(current, type(node)):
                collect(current.left)
                collect(current.right)
            else:
                children.append(current)

        collect(node)
        signatures = sorted(_boolean_signature(child, aliases) for child in children)
        return f"{node.key}({','.join(signatures)})"
    return _canonical(_rewrite_physical_columns(node, aliases))


def _predicate_tokens(node: exp.Expression, aliases: dict[str, str]) -> set[str]:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And | exp.Or):
        return {
            f"logic:{_boolean_signature(node, aliases)}",
            *_predicate_tokens(node.left, aliases),
            *_predicate_tokens(node.right, aliases),
        }
    return {_canonical(_rewrite_physical_columns(node, aliases))}


def extract_sql_features(sql: str, schema: dict[str, set[str]]) -> SqlFeatureExtraction:
    """Extract the frozen M6 feature contract, failing closed on uncertain semantics."""
    analysis = analyze_sql(sql)
    if not analysis.accepted or analysis.sql is None:
        return _unsupported("unsafe_or_invalid_sql")
    try:
        expressions = parse(analysis.sql, dialect="sqlite", error_level=ErrorLevel.RAISE)
    except (ParseError, ValueError):
        return _unsupported("parse_error")
    if len(expressions) != 1 or expressions[0] is None:
        return _unsupported("not_single_query")
    parsed = expressions[0]
    if any(isinstance(node, exp.Select) and not node.expressions for node in parsed.walk()):
        return _unsupported("empty_select_list")
    reason = _has_unsupported_construct(parsed)
    if reason:
        return _unsupported(reason)

    ctes = list(parsed.find_all(exp.CTE))
    cte_names = {cte.alias_or_name.casefold() for cte in ctes}
    try:
        qualified = qualify(
            parsed,
            dialect="sqlite",
            schema=_schema_for_sqlglot(schema),
            expand_alias_refs=True,
            expand_stars=True,
            infer_schema=False,
            allow_partial_qualification=False,
            validate_qualify_columns=True,
            quote_identifiers=False,
            canonicalize_table_aliases=True,
        )
        scopes = list(traverse_scope(qualified))
    except Exception as exc:  # SQLGlot exposes several optimizer error subclasses.
        return _unsupported(f"schema_resolution_error:{type(exc).__name__}")

    # SQLGlot reports only the CTE consumer as a Table node, so use scope sources to detect use.
    consumed_ctes = {
        source.expression.parent.alias_or_name.casefold()
        for scope in scopes
        for _, (_, source) in scope.selected_sources.items()
        if isinstance(source, Scope)
        and isinstance(source.expression.parent, exp.CTE)
        and source.expression.parent.alias_or_name
    }
    if cte_names - consumed_ctes:
        return _unsupported("unused_cte")

    _, physical_tables = _physical_table_map(scopes)
    buckets: dict[str, set[str]] = {dimension: set() for dimension in FEATURE_DIMENSIONS}
    buckets["tables"].update(physical_tables)

    try:
        for scope in scopes:
            expression = scope.expression
            for column in scope.columns:
                rewritten = _rewrite_with_lineage(column, scope)
                for physical_column in rewritten.find_all(exp.Column):
                    if physical_column.table.partition("#role=")[0] in physical_tables:
                        buckets["columns"].add(_canonical(physical_column))
            if isinstance(expression, exp.Select):
                for selected in expression.expressions:
                    _add_occurrence(
                        buckets["select_signatures"],
                        _canonical(_rewrite_with_lineage(selected.unalias(), scope)),
                    )
                for join in expression.args.get("joins") or []:
                    join_copy = join.copy()
                    on = join_copy.args.pop("on", None)
                    parts = [_canonical(_rewrite_with_lineage(join_copy, scope))]
                    if on is not None:
                        parts.extend(
                            sorted(_predicate_tokens(_rewrite_with_lineage(on, scope), {}))
                        )
                    buckets["join_signatures"].add("|".join(parts))
                where = expression.args.get("where")
                if where is not None:
                    buckets["predicate_signatures"].update(
                        _predicate_tokens(_rewrite_with_lineage(where.this, scope), {})
                    )
                for item in _clause_items(expression, "group"):
                    buckets["group_by"].add(_canonical(_rewrite_with_lineage(item, scope)))
                having = expression.args.get("having")
                if having is not None:
                    buckets["having"].update(
                        _predicate_tokens(_rewrite_with_lineage(having.this, scope), {})
                    )
            order = expression.args.get("order")
            if order is not None:
                for item in order.expressions:
                    buckets["order_by"].add(_canonical(_rewrite_with_lineage(item, scope)))
            limit = expression.args.get("limit")
            if limit is not None:
                value = limit.expression
                buckets["limit"].add(
                    "limit:<number>" if isinstance(value, exp.Literal) else "limit:expression"
                )
    except ValueError:
        return _unsupported("unresolved_derived_lineage")
    for operation in qualified.walk():
        if isinstance(operation, exp.Union | exp.Intersect | exp.Except):
            qualifier = (
                " all"
                if isinstance(operation, exp.Union) and not operation.args.get("distinct", True)
                else ""
            )
            _add_occurrence(buckets["set_operations"], f"{operation.key}{qualifier}")

    subqueries = list(qualified.find_all(exp.Subquery))
    if subqueries:
        for subquery in subqueries:
            depth = 1
            parent: Any = subquery.parent
            while parent is not None:
                if isinstance(parent, exp.Subquery):
                    depth += 1
                parent = parent.parent
            _add_occurrence(buckets["subquery_profile"], f"depth:{depth}")
    return SqlFeatureExtraction(
        supported=True,
        reason=None,
        features={key: frozenset(values) for key, values in buckets.items()},
    )


def score_sql_structure(
    gold_sql: str,
    prediction_sql: str,
    schema: dict[str, set[str]],
    *,
    prediction_execution_status: str,
) -> StructuralRewardResult:
    """Apply the dense confidence gate without changing terminal execution correctness."""
    gold = extract_sql_features(gold_sql, schema)
    if not gold.supported:
        raise UnsupportedGoldStructureError(f"gold SQL failed structural preflight: {gold.reason}")
    r_valid = int(prediction_execution_status == "ok")
    prediction = extract_sql_features(prediction_sql, schema)
    if not r_valid or not prediction.supported:
        return StructuralRewardResult(
            r_valid=r_valid,
            r_struct=0.0,
            g_conf=0,
            prediction_reason=prediction.reason,
        )
    return StructuralRewardResult(
        r_valid=1,
        r_struct=active_dimension_structural_reward(
            gold.features, prediction.features, FEATURE_WEIGHTS
        ),
        g_conf=1,
        prediction_reason=None,
    )
