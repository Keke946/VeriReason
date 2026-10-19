from verireason.smoke_sql import extract_sql, is_strict_sql_only, score_completion

SETUP = """
CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
INSERT INTO items VALUES (1, 'safe'), (2, 'other');
"""


def test_positive_read_only_query_scores_correctly() -> None:
    score = score_completion(SETUP, "```sql\nSELECT name FROM items WHERE id = 1;\n```", [["safe"]])

    assert score.valid_read_only_sql
    assert score.execution_correct
    assert score.rows == [["safe"]]


def test_negative_wrong_query_is_valid_but_incorrect() -> None:
    score = score_completion(SETUP, "SELECT name FROM items WHERE id = 2", [["safe"]])

    assert score.valid_read_only_sql
    assert not score.execution_correct


def test_malformed_output_is_rejected() -> None:
    score = score_completion(SETUP, "I cannot write SQL.", [["safe"]])

    assert not score.valid_read_only_sql
    assert score.extracted_sql is None


def test_adversarial_second_statement_is_rejected() -> None:
    score = score_completion(SETUP, "SELECT name FROM items; DROP TABLE items", [["safe"]])

    assert not score.valid_read_only_sql
    assert "multiple statements" in (score.error or "")


def test_adversarial_mutation_cte_is_rejected() -> None:
    score = score_completion(
        SETUP,
        "WITH changed AS (DELETE FROM items RETURNING name) SELECT * FROM changed",
        [["safe"]],
    )

    assert not score.valid_read_only_sql


def test_extract_sql_uses_last_fenced_query() -> None:
    completion = "analysis ```sql\nSELECT 1\n``` correction ```sql\nSELECT 2;\n```"

    assert extract_sql(completion) == "SELECT 2"


def test_strict_sql_only_allows_whitespace_and_one_trailing_semicolon() -> None:
    assert is_strict_sql_only("  SELECT name FROM items;\n")
    assert is_strict_sql_only("WITH chosen AS (SELECT 1) SELECT * FROM chosen")


def test_strict_sql_only_rejects_fences_and_explanatory_prose() -> None:
    assert not is_strict_sql_only("```sql\nSELECT name FROM items;\n```")
    assert not is_strict_sql_only("Here is the query: SELECT name FROM items;")
    assert not is_strict_sql_only("SELECT name FROM items; explanation")
