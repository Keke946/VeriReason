"""Deterministic M2 data identities, split prechecks, and leakage audits."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

_WORD = re.compile(r"[\w]+", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")
_SQL_COMMENT = re.compile(r"--[^\n]*(?:\n|$)|/\*.*?\*/", flags=re.DOTALL)
_AGGREGATE = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Example:
    row_id: str
    db_id: str
    question: str
    sql: str
    source: str = "training"
    database_sha256: str | None = None
    schema_fingerprint: str | None = None
    schema_tokens: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Complexity:
    label: str
    score: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class SplitPrecheck:
    seed: str
    validation_database_count_requested: int
    database_counts: dict[str, int]
    example_counts: dict[str, int]
    example_ratios: dict[str, float]
    examples_per_database: dict[str, dict[str, int]]
    complexity_distribution: dict[str, dict[str, int]]
    feature_distribution: dict[str, dict[str, int]]
    validation_gate_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.validation_gate_errors


@dataclass(frozen=True)
class DuplicateCounts:
    within_split_pairs: int
    cross_split_pairs: int


@dataclass(frozen=True)
class DatabaseIdentity:
    db_id: str
    path: str
    size_bytes: int
    sha256: str
    schema_fingerprint: str
    schema_tokens: tuple[str, ...]
    integrity_check: str


@dataclass(frozen=True)
class LeakageReport:
    hard_failures: dict[str, DuplicateCounts]
    review_findings: dict[str, DuplicateCounts]
    pending_checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(
            counts.cross_split_pairs == 0
            and (name != "heldout_gold_interface_exposure" or counts.within_split_pairs == 0)
            for name, counts in self.hard_failures.items()
        )


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_observed_identity(
    *,
    expected_sha256: str | None,
    observed_sha256: str,
    previous_headers: dict[str, str | int | None] | None = None,
    observed_headers: dict[str, str | int | None] | None = None,
) -> tuple[str, ...]:
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"artifact SHA-256 changed: expected {expected_sha256}, observed {observed_sha256}"
        )
    warnings: list[str] = []
    if previous_headers and observed_headers:
        for name in ("content_length", "etag", "last_modified"):
            if previous_headers.get(name) != observed_headers.get(name):
                warnings.append(
                    f"provenance header changed with stable SHA-256: {name}: "
                    f"{previous_headers.get(name)!r} -> {observed_headers.get(name)!r}"
                )
    return tuple(warnings)


def resolve_database_path(root: str | Path, db_id: str) -> Path:
    base = Path(root)
    candidates = (
        base / db_id / f"{db_id}.sqlite",
        base / db_id / f"{db_id}.db",
        base / f"{db_id}.sqlite",
        base / f"{db_id}.db",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"database {db_id!r} must resolve exactly once under {base}: {matches}")
    return matches[0]


def inspect_database(
    root: str | Path, db_id: str, *, logical_root: str | Path | None = None
) -> DatabaseIdentity:
    path = resolve_database_path(root, db_id)
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        integrity = "\n".join(str(row[0]) for row in integrity_rows)
        if integrity != "ok":
            raise ValueError(f"database {db_id!r} failed integrity_check: {integrity}")
        schema_rows = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE type IN ('table', 'view', 'index', 'trigger') ORDER BY type, name"
        ).fetchall()
    finally:
        connection.close()
    schema_payload = json.dumps(schema_rows, ensure_ascii=False, separators=(",", ":"))
    schema_tokens = tuple(sorted(set(_WORD.findall(normalize_text(schema_payload)))))
    logical_base = Path(logical_root).resolve() if logical_root else Path(root).resolve()
    try:
        logical_path = path.resolve().relative_to(logical_base).as_posix()
    except ValueError as error:
        raise ValueError(f"database path escapes logical root {logical_base}: {path}") from error
    return DatabaseIdentity(
        db_id=db_id,
        path=logical_path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        schema_fingerprint=hashlib.sha256(schema_payload.encode()).hexdigest(),
        schema_tokens=schema_tokens,
        integrity_check=integrity,
    )


def normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def normalize_sql(value: str) -> str:
    without_comments = _SQL_COMMENT.sub(" ", unicodedata.normalize("NFKC", value))
    normalized = _SPACE.sub(" ", without_comments).strip().rstrip(";").strip()
    return normalized.casefold()


def question_tokens(value: str) -> frozenset[str]:
    return frozenset(_WORD.findall(normalize_text(value)))


def stable_database_rank(seed: str, db_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{db_id}".encode()).hexdigest()


def split_database_ids(
    db_ids: Iterable[str], seed: str, validation_database_count: int
) -> dict[str, str]:
    unique = sorted(set(db_ids))
    if not 0 < validation_database_count < len(unique):
        raise ValueError("validation database count must leave both splits non-empty")
    ranked = sorted(unique, key=lambda db_id: (stable_database_rank(seed, db_id), db_id))
    validation = set(ranked[:validation_database_count])
    return {db_id: "validation" if db_id in validation else "train" for db_id in unique}


def sql_complexity(sql: str) -> Complexity:
    normalized = f" {normalize_sql(sql).upper()} "
    features: list[str] = []
    rules = (
        ("join", bool(re.search(r"\bJOIN\b", normalized))),
        ("aggregate", bool(_AGGREGATE.search(normalized))),
        ("subquery", len(re.findall(r"\bSELECT\b", normalized)) > 1),
        ("group_by", bool(re.search(r"\bGROUP\s+BY\b", normalized))),
        ("set_operation", bool(re.search(r"\b(?:UNION|INTERSECT|EXCEPT)\b", normalized))),
        ("order_by", bool(re.search(r"\bORDER\s+BY\b", normalized))),
        ("having", bool(re.search(r"\bHAVING\b", normalized))),
        ("window", bool(re.search(r"\bOVER\s*\(", normalized))),
    )
    for name, present in rules:
        if present:
            features.append(name)
    points = {
        "join": 1,
        "aggregate": 1,
        "subquery": 2,
        "group_by": 1,
        "set_operation": 2,
        "order_by": 1,
        "having": 1,
        "window": 2,
    }
    score = sum(points[name] for name in features)
    label = "simple" if score <= 1 else "moderate" if score <= 3 else "complex"
    if score >= 6:
        label = "highly_complex"
    return Complexity(label=label, score=score, features=tuple(features))


def sample_fingerprint(example: Example) -> str:
    database_identity = example.database_sha256 or f"db_id:{example.db_id}"
    payload = "\0".join(
        (database_identity, normalize_text(example.question), normalize_sql(example.sql))
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_split_precheck(
    examples: Sequence[Example],
    *,
    seed: str,
    validation_database_count: int,
    minimum_validation_ratio: float,
    maximum_validation_ratio: float,
    critical_features: Sequence[str],
) -> tuple[dict[str, str], SplitPrecheck]:
    assignment = split_database_ids(
        (example.db_id for example in examples), seed, validation_database_count
    )
    examples_by_split: Counter[str] = Counter()
    databases_by_split: dict[str, set[str]] = defaultdict(set)
    per_database: dict[str, Counter[str]] = defaultdict(Counter)
    complexity: dict[str, Counter[str]] = defaultdict(Counter)
    features: dict[str, Counter[str]] = defaultdict(Counter)

    for example in examples:
        split = assignment[example.db_id]
        examples_by_split[split] += 1
        databases_by_split[split].add(example.db_id)
        per_database[split][example.db_id] += 1
        parsed = sql_complexity(example.sql)
        complexity[split][parsed.label] += 1
        features[split].update(parsed.features)

    total = len(examples)
    ratios = {split: examples_by_split[split] / total for split in ("train", "validation")}
    errors: list[str] = []
    validation_ratio = ratios["validation"]
    if not minimum_validation_ratio <= validation_ratio <= maximum_validation_ratio:
        errors.append(
            "validation example ratio "
            f"{validation_ratio:.6f} is outside [{minimum_validation_ratio}, "
            f"{maximum_validation_ratio}]"
        )
    for feature in critical_features:
        if features["validation"][feature] == 0:
            errors.append(f"validation is missing critical SQL feature: {feature}")

    report = SplitPrecheck(
        seed=seed,
        validation_database_count_requested=validation_database_count,
        database_counts={
            split: len(databases_by_split[split]) for split in ("train", "validation")
        },
        example_counts={split: examples_by_split[split] for split in ("train", "validation")},
        example_ratios=ratios,
        examples_per_database={
            split: dict(sorted(per_database[split].items())) for split in ("train", "validation")
        },
        complexity_distribution={
            split: dict(sorted(complexity[split].items())) for split in ("train", "validation")
        },
        feature_distribution={
            split: {feature: features[split][feature] for feature in critical_features}
            for split in ("train", "validation")
        },
        validation_gate_errors=tuple(errors),
    )
    return assignment, report


def _pair_counts(values: Sequence[tuple[str, str]]) -> DuplicateCounts:
    by_value: dict[str, Counter[str]] = defaultdict(Counter)
    for split, value in values:
        if value:
            by_value[value][split] += 1
    within = 0
    cross = 0
    for split_counts in by_value.values():
        within += sum(count * (count - 1) // 2 for count in split_counts.values())
        counts = list(split_counts.values())
        cross += sum(
            counts[i] * counts[j] for i in range(len(counts)) for j in range(i + 1, len(counts))
        )
    return DuplicateCounts(within, cross)


def _near_question_pairs(rows: Sequence[tuple[str, frozenset[str]]]) -> DuplicateCounts:
    within = 0
    cross = 0
    for index, (left_split, left) in enumerate(rows):
        if not left:
            continue
        for right_split, right in rows[index + 1 :]:
            if not right:
                continue
            smaller, larger = sorted((len(left), len(right)))
            if smaller / larger < 0.90:
                continue
            score = len(left & right) / len(left | right)
            if score >= 0.90:
                if left_split == right_split:
                    within += 1
                else:
                    cross += 1
    return DuplicateCounts(within, cross)


def audit_leakage(examples: Sequence[tuple[str, Example]]) -> LeakageReport:
    split_values = [(split, example) for split, example in examples]
    database_rows = {
        (split, example.db_id, example.database_sha256 or "") for split, example in split_values
    }
    database_hashes = [(split, database_hash) for split, _db_id, database_hash in database_rows]
    schema_rows = {
        (
            split,
            example.db_id,
            example.schema_fingerprint or "",
            tuple(example.schema_tokens or ()),
        )
        for split, example in split_values
    }
    schemas = [(split, fingerprint) for split, _db_id, fingerprint, _tokens in schema_rows]
    pending: list[str] = []
    if any(not example.database_sha256 for _, example in split_values):
        pending.append("database_sha256_cross_split")
        pending.append("renamed_identical_database_cross_split")
    if any(
        not example.schema_fingerprint or not example.schema_tokens for _, example in split_values
    ):
        pending.append("schema_similarity")

    heldout_exposure_count = sum(
        1
        for split, example in split_values
        if split in {"train", "validation"} and example.source == "heldout"
    )
    hard = {
        "db_id": _pair_counts([(split, db_id) for split, db_id, _database_hash in database_rows]),
        "database_sha256": _pair_counts(database_hashes),
        "joint_sample_fingerprint": _pair_counts(
            [(split, sample_fingerprint(example)) for split, example in split_values]
        ),
        "heldout_gold_interface_exposure": DuplicateCounts(heldout_exposure_count, 0),
        "renamed_identical_database": _pair_counts(database_hashes),
    }
    review = {
        "normalized_question": _pair_counts(
            [(split, normalize_text(example.question)) for split, example in split_values]
        ),
        "normalized_sql": _pair_counts(
            [(split, normalize_sql(example.sql)) for split, example in split_values]
        ),
        "schema_fingerprint": _pair_counts(schemas),
        "schema_jaccard_gte_0_90": _near_question_pairs(
            [
                (split, frozenset(tokens))
                for split, _db_id, _fingerprint, tokens in schema_rows
                if tokens
            ]
        ),
        "question_jaccard_gte_0_90": _near_question_pairs(
            [(split, question_tokens(example.question)) for split, example in split_values]
        ),
    }
    return LeakageReport(hard_failures=hard, review_findings=review, pending_checks=tuple(pending))


def _records_from_json(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        yield from raw
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                yield from value
                return
        raise ValueError("JSON object contains no record list")
    else:
        raise ValueError("label file must contain a JSON list/object or JSONL records")


def load_examples(path: str | Path, source: str = "training") -> list[Example]:
    result: list[Example] = []
    for index, row in enumerate(_records_from_json(Path(path))):
        db_id = row.get("db_id") or row.get("database_id")
        question = row.get("question") or row.get("Question")
        sql = row.get("SQL") or row.get("query") or row.get("sql")
        if not all(isinstance(value, str) and value.strip() for value in (db_id, question, sql)):
            raise ValueError(f"row {index} lacks non-empty db_id, question, or SQL")
        row_id = str(row.get("id", row.get("question_id", index)))
        result.append(Example(row_id, db_id, question, sql, source=source))
    return result


def report_as_dict(report: SplitPrecheck | LeakageReport) -> dict[str, Any]:
    raw = asdict(report)
    if isinstance(report, LeakageReport):
        raw["passed"] = report.passed
    else:
        raw["passed"] = report.passed
    return raw
