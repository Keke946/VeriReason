import pytest

from verireason.reward_contract import active_dimension_structural_reward
from verireason.sql_features import (
    SqlFeatureExtraction,
    extract_sql_features,
    score_sql_structure,
)

SCHEMA = {
    "users": {"id", "name", "age", "team_id"},
    "teams": {"id", "name"},
    "orders": {"id", "user_id", "amount", "status"},
}
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


def _extract(sql: str) -> SqlFeatureExtraction:
    result = extract_sql_features(sql, SCHEMA)
    assert isinstance(result, SqlFeatureExtraction)
    return result


def _supported(sql: str) -> SqlFeatureExtraction:
    result = _extract(sql)
    assert result.supported, result.reason
    assert result.reason is None
    return result


def _unsupported(sql: str) -> None:
    result = _extract(sql)
    assert not result.supported
    assert result.reason
    assert result.features == {}


def test_complete_feature_contract_covers_all_eleven_dimensions() -> None:
    result = _supported(
        """
        SELECT u.id, SUM(o.amount) AS total
        FROM users AS u INNER JOIN orders AS o ON u.id = o.user_id
        WHERE u.age > 18 AND (o.status = 'paid' OR o.status = 'open')
        GROUP BY u.id
        HAVING SUM(o.amount) > 0
        UNION ALL
        SELECT id, 0 AS total FROM users
        WHERE id IN (SELECT user_id FROM orders)
        ORDER BY total DESC LIMIT 5
        """
    )

    assert set(result.features) == set(FEATURE_WEIGHTS)
    assert all(result.features[dimension] for dimension in FEATURE_WEIGHTS)
    assert result.features["tables"] >= frozenset({"users", "orders"})
    assert any("union all" in token.casefold() for token in result.features["set_operations"])
    assert any("desc" in token.casefold() for token in result.features["order_by"])


def test_alias_renaming_is_canonical_but_physical_table_and_columns_remain_visible() -> None:
    first = _supported(
        "SELECT u.id FROM users AS u JOIN orders AS o ON u.id = o.user_id WHERE o.status = 'paid'"
    )
    second = _supported(
        "SELECT person.id FROM users AS person JOIN orders AS purchase "
        "ON person.id = purchase.user_id WHERE purchase.status = 'paid'"
    )

    assert first.features == second.features
    assert first.features["tables"] == frozenset({"users", "orders"})
    assert any("users" in token and "id" in token for token in first.features["columns"])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM missing_table",
        "SELECT missing_column FROM users",
        "SELECT id FROM users JOIN orders ON users.id = orders.user_id",
    ],
)
def test_schema_resolution_fails_closed_for_unknown_or_ambiguous_identifiers(sql: str) -> None:
    _unsupported(sql)


def test_cte_name_is_not_an_entity_table_and_its_projected_column_resolves() -> None:
    result = _supported(
        "WITH adults AS (SELECT id FROM users WHERE age >= 18) SELECT id FROM adults"
    )

    assert result.features["tables"] == frozenset({"users"})
    assert all("adults" not in token.casefold() for token in result.features["tables"])
    assert any("users" in token and "id" in token for token in result.features["columns"])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT FROM users",
        "WITH RECURSIVE tree(id) AS (SELECT id FROM users UNION ALL "
        "SELECT id FROM tree) SELECT id FROM tree",
        "SELECT * FROM users NATURAL JOIN orders",
        "WITH unused AS (SELECT id FROM users) SELECT id FROM orders",
        "SELECT id FROM users WHERE id = ?",
        "SELECT random() FROM users",
        "SELECT datetime('now') FROM users",
        "SELECT strftime('%Y') FROM users",
    ],
)
def test_malformed_or_intentionally_unsupported_constructs_fail_closed(sql: str) -> None:
    _unsupported(sql)


def test_literals_are_value_independent_but_type_sensitive() -> None:
    integer_three = _supported("SELECT id FROM users WHERE age > 3")
    integer_four = _supported("SELECT id FROM users WHERE age > 4")
    text_three = _supported("SELECT id FROM users WHERE age > '3'")

    assert (
        integer_three.features["predicate_signatures"]
        == integer_four.features["predicate_signatures"]
    )
    assert (
        integer_three.features["predicate_signatures"]
        != text_three.features["predicate_signatures"]
    )


def test_conjunct_order_is_canonical() -> None:
    first = _supported("SELECT id FROM users WHERE age > 18 AND team_id = 2")
    second = _supported("SELECT id FROM users WHERE team_id = 2 AND age > 18")

    assert first.features["predicate_signatures"] == second.features["predicate_signatures"]


def test_extra_predicate_lowers_precision_without_erasing_shared_predicate() -> None:
    gold = _supported("SELECT id FROM users WHERE age > 18")
    prediction = _supported("SELECT id FROM users WHERE age > 21 AND team_id = 2")
    shared = gold.features["predicate_signatures"] & prediction.features["predicate_signatures"]

    assert shared
    assert len(prediction.features["predicate_signatures"]) > len(shared)


def test_mixed_boolean_topology_is_preserved() -> None:
    first = _supported("SELECT id FROM users WHERE age > 18 AND (team_id = 2 OR name = 'a')")
    second = _supported("SELECT id FROM users WHERE (age > 18 AND team_id = 2) OR name = 'a'")
    assert first.features["predicate_signatures"] != second.features["predicate_signatures"]


def test_extra_projection_and_join_lower_structural_f1() -> None:
    gold = _supported("SELECT u.id FROM users AS u JOIN orders AS o ON u.id = o.user_id")
    prediction = _supported(
        "SELECT u.id, o.amount, t.name FROM users AS u "
        "JOIN orders AS o ON u.id = o.user_id "
        "JOIN teams AS t ON u.team_id = t.id"
    )

    score = active_dimension_structural_reward(gold.features, prediction.features, FEATURE_WEIGHTS)
    assert 0.0 < score < 1.0


def test_repeated_projection_and_union_branch_cannot_receive_full_score() -> None:
    projection_gold = _supported("SELECT id FROM users")
    projection_attack = _supported("SELECT id, id, id FROM users")
    assert (
        active_dimension_structural_reward(
            projection_gold.features, projection_attack.features, FEATURE_WEIGHTS
        )
        < 1.0
    )

    union_gold = _supported("SELECT id FROM users UNION ALL SELECT id FROM orders")
    union_attack = _supported(
        "SELECT id FROM users UNION ALL SELECT id FROM orders UNION ALL SELECT id FROM orders"
    )
    assert (
        active_dimension_structural_reward(
            union_gold.features, union_attack.features, FEATURE_WEIGHTS
        )
        < 1.0
    )


def test_output_aliases_having_order_and_parentheses_are_canonical() -> None:
    order_x = _supported("SELECT age AS x FROM users ORDER BY x")
    order_y = _supported("SELECT age AS y FROM users ORDER BY y")
    assert order_x.features == order_y.features

    cte_x = _supported("WITH a AS (SELECT id AS x FROM users) SELECT x FROM a")
    cte_y = _supported("WITH b AS (SELECT id AS y FROM users) SELECT y FROM b")
    assert cte_x.features == cte_y.features

    having_first = _supported(
        "SELECT team_id FROM users GROUP BY team_id HAVING COUNT(*) > 1 AND MAX(age) > 20"
    )
    having_second = _supported(
        "SELECT team_id FROM users GROUP BY team_id HAVING MAX(age) > 20 AND COUNT(*) > 1"
    )
    assert having_first.features == having_second.features
    assert (
        _supported("SELECT id FROM users WHERE age > 18").features
        == _supported("SELECT id FROM users WHERE (age > 18)").features
    )


def test_self_join_roles_are_not_collapsed() -> None:
    gold = _supported(
        "SELECT a.name FROM users a JOIN users b ON a.team_id = b.id WHERE a.age > 18"
    )
    wrong_projection = _supported(
        "SELECT b.name FROM users a JOIN users b ON a.team_id = b.id WHERE a.age > 18"
    )
    wrong_edge = _supported(
        "SELECT a.name FROM users a JOIN users b ON b.team_id = a.id WHERE a.age > 18"
    )
    wrong_role = _supported(
        "SELECT a.name FROM users a JOIN users b ON a.team_id = b.id WHERE b.age > 18"
    )
    for prediction in (wrong_projection, wrong_edge, wrong_role):
        assert prediction.features != gold.features
        assert (
            active_dimension_structural_reward(gold.features, prediction.features, FEATURE_WEIGHTS)
            < 1.0
        )


def test_set_operation_subquery_group_having_order_and_limit_have_public_signals() -> None:
    result = _supported(
        """
        SELECT u.team_id, COUNT(*) AS n
        FROM users AS u
        WHERE u.id IN (SELECT o.user_id FROM orders AS o WHERE o.amount > 1.5)
        GROUP BY u.team_id HAVING COUNT(*) >= 2
        UNION SELECT team_id, 0 AS n FROM users
        ORDER BY n DESC LIMIT 10
        """
    )

    for dimension in (
        "set_operations",
        "subquery_profile",
        "group_by",
        "having",
        "order_by",
        "limit",
    ):
        assert result.features[dimension]


def test_confidence_gate_scores_supported_execution_and_closes_on_failure() -> None:
    supported = score_sql_structure(
        "SELECT id FROM users WHERE age > 18",
        "SELECT id FROM users WHERE age > 21",
        SCHEMA,
        prediction_execution_status="ok",
    )
    assert supported.g_conf == supported.r_valid == 1
    assert supported.r_struct == 1.0

    failed_execution = score_sql_structure(
        "SELECT id FROM users",
        "SELECT id FROM users",
        SCHEMA,
        prediction_execution_status="timeout",
    )
    assert (
        failed_execution.g_conf,
        failed_execution.r_valid,
        failed_execution.r_struct,
    ) == (0, 0, 0)

    unsupported_dense = score_sql_structure(
        "SELECT id FROM users",
        "SELECT random() FROM users",
        SCHEMA,
        prediction_execution_status="ok",
    )
    assert (unsupported_dense.g_conf, unsupported_dense.r_valid, unsupported_dense.r_struct) == (
        0,
        1,
        0,
    )
