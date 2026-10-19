import json
import sqlite3
from pathlib import Path

import pytest

from verireason.m3_contract import (
    IGNORE_INDEX,
    AnswerOnlyCollator,
    SchemaTable,
    build_messages,
    comparison_contract,
    extract_generated_sql,
    format_target,
    inspect_schema_tables,
    load_m3_config,
    normalize_single_token_ids,
    prepare_example,
    select_tiny_slice,
    serialize_schema,
    tokenize_answer_only,
    validate_completed_result,
    verify_m2_manifest_pair,
    verify_manifest_identity,
    verify_update_evidence,
)
from verireason.m3_runtime import read_slice

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(
        self, conversation, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize is True
        assert enable_thinking is False
        text = "|".join(f"{row['role']}:{row['content']}" for row in conversation)
        if add_generation_prompt:
            text += "|assistant:"
        return self.encode(text, add_special_tokens=True)

    def encode(self, text, *, add_special_tokens):
        values = [10 + index for index, _part in enumerate(text.split())]
        return ([1] if add_special_tokens else []) + values


class ArrayLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "company.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE employees (
          id INTEGER PRIMARY KEY,
          department_id INTEGER NOT NULL,
          active INTEGER,
          FOREIGN KEY (department_id) REFERENCES departments(id)
        );
        INSERT INTO departments VALUES (1, 'secret sample value');
        """
    )
    connection.close()
    return path


def config() -> dict:
    return load_m3_config(ROOT / "configs/m3_vertical_slice.toml", ROOT / "configs/m2_data.toml")


def test_m3_config_freezes_model_heldout_and_evaluator() -> None:
    loaded = config()
    assert loaded["model"]["revision"] == "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    assert loaded["data"]["heldout_examples_loaded"] == 0
    assert loaded["evaluator"]["maximum_result_rows"] == 100_000
    schema = json.loads((ROOT / "results/schemas/m3-result.schema.json").read_text())
    assert schema["title"] == "VeriReason M3 Vertical Slice Result"


def test_schema_serializer_is_stable_complete_and_excludes_values(database: Path) -> None:
    tables = inspect_schema_tables(database, "company")
    rendered = serialize_schema("company", tables)

    assert [table.name for table in tables] == ["departments", "employees"]
    assert 'Table "departments"' in rendered
    assert '"name" TEXT NOT NULL' in rendered
    assert 'FOREIGN KEY ("department_id") REFERENCES "departments"("id")' in rendered
    assert "secret sample value" not in rendered
    assert rendered == serialize_schema("company", inspect_schema_tables(database, "company"))


def test_schema_serializer_supports_implicit_and_explicit_foreign_key_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "foreign_keys.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child_implicit (
          parent_id INTEGER,
          FOREIGN KEY (parent_id) REFERENCES parent
        );
        CREATE TABLE child_explicit (
          parent_id INTEGER,
          FOREIGN KEY (parent_id) REFERENCES parent(id)
        );
        """
    )
    implicit_pragma = list(connection.execute("PRAGMA foreign_key_list(child_implicit)"))
    connection.close()

    assert implicit_pragma[0][4] is None
    first = serialize_schema("foreign_keys", inspect_schema_tables(path, "foreign_keys"))
    second = serialize_schema("foreign_keys", inspect_schema_tables(path, "foreign_keys"))
    lines = first.splitlines()
    assert '- FOREIGN KEY ("parent_id") REFERENCES "parent"' in lines
    assert '- FOREIGN KEY ("parent_id") REFERENCES "parent"("id")' in lines
    assert "None" not in first
    assert first == second


def test_target_and_prompt_are_sql_only_without_reasoning() -> None:
    target = format_target("  SELECT COUNT(*) FROM employees; \n")
    messages = build_messages("Return SQL only.", "Database: company", "How many?", target)

    assert target == "SELECT COUNT(*) FROM employees"
    assert messages[-1] == {"role": "assistant", "content": target}
    assert "reason" not in messages[-1]["content"].lower()


def test_answer_only_masking_includes_sql_and_eos() -> None:
    tokenizer = FakeTokenizer()
    prompt = build_messages("Return SQL only.", "Database: company", "How many?")
    target = "SELECT COUNT(*) FROM employees"
    input_ids, labels, prompt_length = tokenize_answer_only(tokenizer, prompt, target)

    assert labels[:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert labels[prompt_length:-1] == tokenizer.encode(target, add_special_tokens=False)
    assert input_ids[-1] == labels[-1] == tokenizer.eos_token_id
    assert sum(label != IGNORE_INDEX for label in labels) == len(labels) - prompt_length


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        (1, 2, 3),
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]},
        ArrayLike([[1, 2, 3]]),
        {"input_ids": ArrayLike([[1, 2, 3]])},
    ],
)
def test_tokenizer_output_normalizes_supported_single_example_shapes(value) -> None:
    assert normalize_single_token_ids(value) == [1, 2, 3]


@pytest.mark.parametrize(
    "value",
    [
        {"attention_mask": [[1, 1]]},
        {"input_ids": [[1], [2]]},
        [],
        "input_ids",
        [1, "attention_mask"],
        [1, True],
        [[1, 2], 3],
    ],
)
def test_tokenizer_output_rejects_ambiguous_or_non_integer_shapes(value) -> None:
    with pytest.raises(ValueError):
        normalize_single_token_ids(value)


def test_batch_encoding_like_chat_template_produces_answer_only_integer_contract() -> None:
    class BatchEncodingTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [[11, 12, 13]], "attention_mask": [[1, 1, 1]]}

    tokenizer = BatchEncodingTokenizer()
    prompt = build_messages("Return SQL only.", "Database: company", "How many?")
    input_ids, labels, prompt_length = tokenize_answer_only(tokenizer, prompt, "SELECT 1")

    assert input_ids[:prompt_length] == [11, 12, 13]
    assert all(type(token) is int for token in input_ids)
    assert labels[:prompt_length] == [IGNORE_INDEX] * prompt_length


def test_read_slice_rejects_non_integer_tokens_before_model_loading(tmp_path: Path) -> None:
    example = {
        "input_ids": ["input_ids", "attention_mask", 1],
        "labels": [-100, -100, 1],
        "prompt_tokens": 2,
        "answer_tokens": 1,
        "supervised_tokens": 1,
    }
    payload = {
        "schema_version": 1,
        "synthetic": False,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "slices": {
            "train": {"examples": [example]},
            "validation": {"examples": [example]},
        },
    }
    from verireason.m3_contract import sha256_json

    payload["slice_bundle_sha256"] = sha256_json(payload)
    path = tmp_path / "slice.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integers only"):
        read_slice(path)


def test_prepare_example_drops_only_complete_trailing_tables() -> None:
    loaded = config()
    loaded["prompt"]["max_sequence_tokens"] = 70
    tables = (
        SchemaTable("a", "Table a\n- id INTEGER"),
        SchemaTable("b", "Table b\n- id INTEGER\n- very_long_column_name TEXT"),
        SchemaTable("c", "Table c\n- id INTEGER\n- another_long_column_name TEXT"),
    )
    row = {
        "row_id": "one",
        "sample_fingerprint": "f" * 64,
        "source": "training",
        "split": "train",
        "operational_eligible": True,
        "db_id": "db",
        "question": "Return ids",
        "sql": "SELECT id FROM a",
        "database_path": "/synthetic/db.sqlite",
    }
    prepared = prepare_example(row, tables, FakeTokenizer(), loaded)

    assert "Table a" in prepared.schema_text
    assert prepared.supervised_tokens == prepared.answer_tokens
    assert prepared.input_ids[-1] == FakeTokenizer.eos_token_id


@pytest.mark.parametrize(
    "change",
    [
        {"split": "heldout"},
        {"operational_eligible": False},
        {"source": "heldout"},
    ],
)
def test_prepare_example_rejects_non_training_interfaces(change: dict) -> None:
    row = {
        "row_id": "one",
        "source": "training",
        "split": "train",
        "operational_eligible": True,
        "db_id": "db",
        "question": "question",
        "sql": "SELECT 1",
        **change,
    }
    with pytest.raises(ValueError, match="eligible train/validation|heldout"):
        prepare_example(
            row, (SchemaTable("a", "Table a\n- id INTEGER"),), FakeTokenizer(), config()
        )


def _rows() -> list[dict]:
    sql = [
        "SELECT * FROM a JOIN b ON a.id=b.id",
        "SELECT COUNT(*) FROM a",
        "SELECT * FROM a WHERE id IN (SELECT id FROM b)",
        "SELECT category, COUNT(*) FROM a GROUP BY category",
        "SELECT * FROM a ORDER BY id",
        "SELECT 1",
    ]
    return [
        {
            "row_id": str(index),
            "sample_fingerprint": f"{index:064x}",
            "source": "training",
            "split": "train",
            "operational_eligible": True,
            "db_id": f"db{index % 2}",
            "question": f"q{index}",
            "sql": statement,
        }
        for index, statement in enumerate(sql)
    ]


def test_tiny_selector_is_deterministic_hashed_and_feature_covered() -> None:
    kwargs = {
        "split": "train",
        "count": 4,
        "seed": "fixed",
        "required_features": ("join", "aggregate", "subquery"),
        "source_manifest_sha256": "a" * 64,
    }
    first = select_tiny_slice(_rows(), **kwargs)
    second = select_tiny_slice(list(reversed(_rows())), **kwargs)

    assert first["sample_ids"] == second["sample_ids"]
    assert first["slice_sha256"] == second["slice_sha256"]
    assert all(first["feature_counts"][feature] >= 1 for feature in kwargs["required_features"])


def test_tiny_selector_never_uses_ineligible_or_heldout() -> None:
    rows = _rows()
    rows.extend(
        [
            {
                **rows[0],
                "row_id": "bad",
                "sample_fingerprint": "b" * 64,
                "operational_eligible": False,
            },
            {**rows[1], "row_id": "secret", "sample_fingerprint": "c" * 64, "source": "heldout"},
        ]
    )
    selected = select_tiny_slice(
        rows,
        split="train",
        count=4,
        seed="fixed",
        required_features=("join", "aggregate", "subquery"),
        source_manifest_sha256="a" * 64,
    )
    assert "bad" not in selected["sample_ids"]
    assert "secret" not in selected["sample_ids"]


def test_collator_masks_padding_and_rejects_zero_loss() -> None:
    collator = AnswerOnlyCollator(0)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
    )
    assert batch["input_ids"][1] == [4, 5, 0]
    assert batch["attention_mask"][1] == [1, 1, 0]
    assert batch["labels"][1] == [-100, 5, -100]
    with pytest.raises(ValueError, match="supervised"):
        collator([{"input_ids": [1], "labels": [-100]}])


@pytest.mark.parametrize(
    ("text", "status", "strict"),
    [
        ("SELECT 1;", "strict", True),
        ("```sql\nSELECT 1;\n```", "recovered", False),
        ("Here is SQL: SELECT 1", "recovered", False),
        ("explanation only", "no_query", False),
    ],
)
def test_generation_extraction_preserves_strict_vs_recovered(
    text: str, status: str, strict: bool
) -> None:
    result = extract_generated_sql(text)
    assert result["status"] == status
    assert result["strict_sql_only"] is strict


def test_comparison_contract_changes_on_prompt_or_decoding() -> None:
    loaded = config()
    first = comparison_contract("a" * 64, ["b" * 64], loaded)
    changed_prompt = comparison_contract("a" * 64, ["c" * 64], loaded)
    loaded["decoding"]["max_new_tokens"] += 1
    changed_decode = comparison_contract("a" * 64, ["b" * 64], loaded)
    assert (
        len(
            {
                first["comparison_contract_sha256"],
                changed_prompt["comparison_contract_sha256"],
                changed_decode["comparison_contract_sha256"],
            }
        )
        == 3
    )


def test_update_evidence_requires_step_gradient_and_parameter_delta() -> None:
    evidence = {
        "trainable_parameter_count": 10,
        "supervised_token_count": 5,
        "initial_loss": 2.0,
        "gradient_norm": 0.7,
        "optimizer_steps": 1,
        "adapter_parameter_delta_detected": True,
        "adapter_delta_l2_norm": 0.01,
    }
    verify_update_evidence(evidence)
    for key, value in (
        ("optimizer_steps", 0),
        ("gradient_norm", 0),
        ("adapter_parameter_delta_detected", False),
    ):
        broken = {**evidence, key: value}
        with pytest.raises(ValueError):
            verify_update_evidence(broken)


def test_manifest_identity_rejects_heldout(tmp_path: Path) -> None:
    manifest = tmp_path / "training.json"
    payload = {
        "schema_version": 2,
        "databases": {},
        "examples": [{"source": "heldout"}],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="heldout"):
        verify_manifest_identity(manifest, digest)


def test_manifest_pair_binds_index_to_training_manifest(tmp_path: Path) -> None:
    import hashlib

    training = tmp_path / "training.json"
    training.write_text(
        json.dumps({"schema_version": 2, "databases": {}, "examples": []}), encoding="utf-8"
    )
    training_sha = hashlib.sha256(training.read_bytes()).hexdigest()
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"training_manifest_sha256": training_sha}), encoding="utf-8")
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()

    assert (
        verify_m2_manifest_pair(
            index,
            training,
            expected_index_sha256=index_sha,
            expected_training_sha256=training_sha,
        )["schema_version"]
        == 2
    )
    with pytest.raises(ValueError, match="index"):
        verify_m2_manifest_pair(
            index,
            training,
            expected_index_sha256="0" * 64,
            expected_training_sha256=training_sha,
        )


def _completed_result() -> dict:
    evaluation = {
        "metrics": {
            "bird_ex_behavior_compatible": 0.5,
            "strict_execution_accuracy": 0.5,
        },
        "examples": [{"sample_id": "one", "prompt_sha256": "a" * 64}],
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "training": {
            "trainable_parameter_count": 10,
            "supervised_token_count": 5,
            "initial_loss": 2.0,
            "gradient_norm": 0.7,
            "optimizer_steps": 1,
            "adapter_parameter_delta_detected": True,
            "adapter_delta_l2_norm": 0.01,
        },
        "fresh_reload": {"different_process": True},
        "comparison_contract": {"comparison_contract_sha256": "b" * 64},
        "base_evaluation": evaluation,
        "sft_evaluation": json.loads(json.dumps(evaluation)),
        "memory": {},
        "timing": {},
        "cost": {},
    }


def test_completed_result_requires_identical_base_sft_exam() -> None:
    result = _completed_result()
    validate_completed_result(result)
    result["sft_evaluation"]["examples"][0]["prompt_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identical"):
        validate_completed_result(result)


def test_completed_result_rejects_any_heldout_consumption() -> None:
    result = _completed_result()
    result["heldout_examples_loaded"] = 1
    with pytest.raises(ValueError, match="heldout"):
        validate_completed_result(result)
