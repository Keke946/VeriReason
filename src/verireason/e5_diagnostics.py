"""Hand-authored training-only challenge suite, not a sampled accuracy benchmark."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from verireason.e5_credit import localize_execution


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    gold: str
    prediction: str
    expected_clause: str | None
    expected_terminal: int = 0


BASE = "SELECT id FROM sales WHERE amount > 10"
GROUP = "SELECT dept, SUM(amount) FROM sales GROUP BY dept"
CASES = (
    Case("correct", "positive", BASE, BASE, None, 1),
    Case(
        "equivalent_predicate",
        "equivalent",
        BASE,
        "SELECT id FROM sales WHERE NOT amount <= 10",
        None,
        1,
    ),
    Case(
        "equivalent_alias",
        "equivalent",
        BASE,
        "SELECT s.id FROM sales AS s WHERE s.amount > 10",
        None,
        1,
    ),
    Case(
        "equivalent_nested",
        "equivalent",
        BASE,
        "SELECT id FROM (SELECT * FROM sales) WHERE amount > 10",
        None,
        1,
    ),
    Case("select", "negative", BASE, BASE.replace("SELECT id", "SELECT amount"), "SELECT"),
    Case("from", "negative", BASE, BASE.replace("sales", "other"), "FROM"),
    Case("where", "negative", BASE, BASE.replace("> 10", "> 20"), "WHERE"),
    Case("group", "negative", GROUP, GROUP + ", id", "GROUP_BY"),
    Case(
        "having",
        "negative",
        GROUP + " HAVING SUM(amount)>25",
        GROUP + " HAVING SUM(amount)>50",
        "HAVING",
    ),
    Case(
        "order",
        "negative",
        "SELECT id FROM sales ORDER BY id ASC",
        "SELECT id FROM sales ORDER BY id DESC",
        "ORDER_BY",
    ),
    Case(
        "join",
        "negative",
        "SELECT sales.id FROM sales JOIN other ON sales.id=other.id",
        "SELECT sales.id FROM sales JOIN other ON sales.id<other.id",
        "FROM",
    ),
    Case(
        "limit",
        "ambiguous",
        "SELECT id FROM sales ORDER BY id LIMIT 2",
        "SELECT id FROM sales ORDER BY id LIMIT 1",
        None,
    ),
    Case("two_errors", "ambiguous", BASE, "SELECT amount FROM sales WHERE amount>20", None),
    Case("missing_clause", "ambiguous", BASE, "SELECT id FROM sales", None),
    Case("extra_clause", "ambiguous", "SELECT id FROM sales", BASE, None),
    Case(
        "nested_wrong", "nested", BASE, "SELECT id FROM (SELECT * FROM sales) WHERE amount>20", None
    ),
    Case(
        "cte",
        "nested",
        BASE,
        "WITH s AS (SELECT * FROM sales) SELECT id FROM s WHERE amount>20",
        None,
    ),
    Case("safe_alias_wrong", "alias", BASE, "SELECT s.id FROM sales AS s WHERE s.amount>20", None),
    Case(
        "duplicate_alias", "alias", BASE, "SELECT s.id FROM sales s JOIN other s ON s.id=s.id", None
    ),
    Case(
        "ambiguous_column",
        "alias",
        BASE,
        "SELECT id FROM sales JOIN other ON sales.id=other.id",
        None,
    ),
    Case("drop", "unsafe", BASE, "DROP TABLE sales", None),
    Case("multi_statement", "unsafe", BASE, BASE + "; DELETE FROM sales", None),
    Case("file_function", "unsafe", BASE, "SELECT readfile('/etc/passwd')", None),
    Case("syntax", "negative", BASE, "SELECT FROM sales WHERE", None),
    Case("unknown_column", "negative", BASE, "SELECT missing FROM sales", None),
    Case("constant", "adversarial", BASE, "SELECT 999", None),
    Case("empty_result", "adversarial", BASE, "SELECT id FROM sales WHERE 0", "WHERE"),
    Case("duplicate_padding", "adversarial", BASE, BASE + " UNION ALL " + BASE, None),
    Case("random", "adversarial", "SELECT id FROM sales", "SELECT random() FROM sales", None),
    Case(
        "comment_keyword",
        "adversarial",
        BASE,
        "SELECT id FROM sales /* SELECT FROM WHERE */ WHERE amount > 20",
        "WHERE",
    ),
    Case(
        "quoted_keyword",
        "adversarial",
        "SELECT 'WHERE' FROM sales",
        "SELECT 'FROM' FROM sales",
        "SELECT",
    ),
    Case("unicode", "negative", "SELECT '猫' FROM sales", "SELECT '狗' FROM sales", "SELECT"),
    Case(
        "offset",
        "ambiguous",
        "SELECT id FROM sales ORDER BY id LIMIT 2",
        "SELECT id FROM sales ORDER BY id LIMIT 2 OFFSET 1",
        None,
    ),
    Case("select_alias", "alias", "SELECT id FROM sales", "SELECT amount AS id FROM sales", None),
    Case("window", "nested", "SELECT id FROM sales", "SELECT SUM(amount) OVER () FROM sales", None),
    Case(
        "bare_aggregate",
        "ambiguous",
        GROUP,
        "SELECT dept, SUM(amount) FROM sales GROUP BY amount",
        None,
    ),
    Case(
        "connection_function",
        "adversarial",
        "SELECT id FROM sales",
        "SELECT changes() FROM sales",
        None,
    ),
)


def create_database(path: Path) -> None:
    """Write only our disposable fixture; production queries use the existing read-only worker."""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE sales(id INTEGER PRIMARY KEY, dept TEXT, amount INTEGER);"
            "INSERT INTO sales VALUES (1,'a',5),(2,'a',15),(3,'b',20),(4,'b',30);"
            "CREATE TABLE other(id INTEGER PRIMARY KEY, dept TEXT, amount INTEGER);"
            "INSERT INTO other VALUES (1,'a',40),(2,'b',5);"
        )


def diagnose() -> dict:
    reasons: Counter[str] = Counter()
    clauses: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    mismatches = []
    localized = correct = failures = 0
    with TemporaryDirectory(prefix="verireason-e5-synthetic-") as directory:
        database = Path(directory) / "synthetic.sqlite"
        create_database(database)
        for case in CASES:
            result = localize_execution(database, case.gold, case.prediction, split="train")
            clause = result.span.clause if result.span else None
            reasons[result.reason] += 1
            categories[case.category] += 1
            correct += result.terminal_reward
            localized += int(clause is not None)
            failures += int(result.terminal_reward == 0)
            if clause:
                clauses[clause] += 1
            if (clause, result.terminal_reward) != (case.expected_clause, case.expected_terminal):
                mismatches.append(case.name)
    abstained = failures - localized
    return {
        "experiment_id": "e5_cpu_feasibility_001",
        "dataset": "hand_authored_synthetic_train_v1",
        "case_count": len(CASES),
        "terminal_correct": correct,
        "incorrect_count": failures,
        "localized_count": localized,
        "abstained_incorrect_count": abstained,
        "localization_coverage_all": localized / len(CASES),
        "localization_coverage_incorrect": localized / failures,
        "abstention_rate_incorrect": abstained / failures,
        "terminal_only_rate_all": (len(CASES) - localized) / len(CASES),
        "clause_counts": dict(sorted(clauses.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "category_counts": dict(sorted(categories.items())),
        "expectation_mismatches": mismatches,
        "validation_examples_loaded": 0,
        "heldout_examples_loaded": 0,
        "gpu_runs": 0,
        "cloud_cost_cny": 0,
        "decision": "Revise",
        "decision_reason": "Synthetic feasibility only; real-tokenizer/loss integration and "
        "representative training-only coverage remain unverified. No GPU smoke.",
    }
