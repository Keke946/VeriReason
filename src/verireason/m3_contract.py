"""Frozen Day 4 M3 data, prompt, masking, and comparison contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from verireason.data_pipeline import sha256_file, sql_complexity
from verireason.sql_evaluator import EvaluatorLimits

IGNORE_INDEX = -100
_QUERY_START = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class TokenizerLike(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Any: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class SchemaTable:
    name: str
    text: str


@dataclass(frozen=True)
class PreparedExample:
    sample_id: str
    split: str
    db_id: str
    database_path: str
    question: str
    target_sql: str
    schema_text: str
    messages: tuple[dict[str, str], ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    answer_tokens: int
    supervised_tokens: int
    prompt_sha256: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Durably replace one JSON file without exposing a half-written final path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_m3_config(path: str | Path, m2_path: str | Path | None = None) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"model", "data", "prompt", "decoding", "qlora", "evaluator", "evidence"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M3 config must be schema v1 with every frozen contract section")
    if (
        config["model"]["model_id"] != "Qwen/Qwen3-4B-Base"
        or config["model"]["revision"] != "906bfd4b4dc7f14ee4320094d8b41684abff8539"
    ):
        raise ValueError("M3 model identity differs from the frozen M1 checkpoint")
    if (
        config["data"]["heldout_examples_loaded"] != 0
        or config["data"]["heldout_examples_evaluated"] != 0
    ):
        raise ValueError("M3 heldout consumption must remain exactly zero")
    if config["data"]["selector_version"] != "m3_tiny_selector_v1":
        raise ValueError("M3 selector contract version changed")
    if config["data"]["train_count"] != 8 or config["data"]["validation_count"] != 8:
        raise ValueError("M3 tiny slice counts must remain fixed at train=8 and validation=8")
    if (
        config["prompt"]["include_sample_values"]
        or config["prompt"]["include_reasoning"]
        or config["prompt"]["enable_thinking"]
    ):
        raise ValueError("M3 prompt cannot include sample values or reasoning traces")
    if (
        config["prompt"]["append_eos"] is not True
        or config["prompt"]["schema_overflow_policy"] != "drop_complete_tables_from_end"
        or config["decoding"]["do_sample"] is not False
        or config["decoding"]["num_beams"] != 1
        or config["decoding"]["max_new_tokens"] != config["prompt"]["max_answer_tokens"]
    ):
        raise ValueError("M3 target, overflow, or deterministic decoding contract changed")
    if config["evaluator"]["metrics"] != [
        "bird_ex_behavior_compatible",
        "strict_execution_accuracy",
    ]:
        raise ValueError("M3 required evaluator metrics changed")
    qlora = config["qlora"]
    validate_qlora_config(qlora)
    if (
        config["data"]["train_count"]
        != qlora["per_device_train_batch_size"]
        * qlora["gradient_accumulation_steps"]
        * qlora["maximum_optimizer_steps"]
    ):
        raise ValueError("M3 train count must equal two complete accumulation updates")
    if (
        config["evidence"]["maximum_session_wall_seconds"] != 14_400
        or config["evidence"]["maximum_session_cost_cny"] != 10.36
        or config["evidence"]["minimum_output_free_bytes"] != 20 * 1024**3
        or config["evidence"]["terminate_grace_seconds"] < 1
    ):
        raise ValueError("M3 cumulative wall, cost, disk, or termination hard stop changed")
    if m2_path is not None:
        with Path(m2_path).open("rb") as handle:
            m2 = tomllib.load(handle)
        keys = (
            "timeout_seconds",
            "maximum_cpu_seconds",
            "maximum_cpu_hard_seconds",
            "maximum_result_rows",
            "maximum_result_bytes",
            "maximum_vm_bytes",
            "maximum_float_pair_checks",
            "float_absolute_tolerance",
            "float_relative_tolerance",
        )
        drift = [key for key in keys if config["evaluator"][key] != m2["evaluator"][key]]
        if drift:
            raise ValueError(f"M3 evaluator drifted from M2: {', '.join(drift)}")
    return config


def validate_qlora_config(config: Mapping[str, Any]) -> None:
    if config.get("load_in_4bit") is not True or config.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("M3 QLoRA must use 4-bit NF4")
    if config.get("bnb_4bit_compute_dtype") != "bfloat16":
        raise ValueError("M3 QLoRA compute dtype must remain bfloat16")
    if not 1 <= int(config.get("lora_rank", 0)) <= 256:
        raise ValueError("LoRA rank is outside the bounded smoke range")
    if int(config.get("lora_alpha", 0)) < int(config["lora_rank"]):
        raise ValueError("LoRA alpha must be at least its rank")
    if not config.get("target_modules") or int(config.get("maximum_optimizer_steps", 0)) < 1:
        raise ValueError("QLoRA requires target modules and at least one possible optimizer step")
    if not 0 < float(config.get("learning_rate", 0)) <= 1e-3:
        raise ValueError("M3 learning rate is outside the smoke safety bound")


def evaluator_limits(config: Mapping[str, Any]) -> EvaluatorLimits:
    raw = config["evaluator"]
    return EvaluatorLimits(
        timeout_seconds=raw["timeout_seconds"],
        maximum_vm_bytes=raw["maximum_vm_bytes"],
        maximum_cpu_seconds=raw["maximum_cpu_seconds"],
        maximum_cpu_hard_seconds=raw["maximum_cpu_hard_seconds"],
        maximum_result_rows=raw["maximum_result_rows"],
        maximum_result_bytes=raw["maximum_result_bytes"],
        maximum_float_pair_checks=raw["maximum_float_pair_checks"],
        float_absolute_tolerance=raw["float_absolute_tolerance"],
        float_relative_tolerance=raw["float_relative_tolerance"],
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_schema_tables(database_path: str | Path, db_id: str) -> tuple[SchemaTable, ...]:
    path = Path(database_path).resolve()
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name COLLATE BINARY"
            )
        ]
        tables: list[SchemaTable] = []
        for name in names:
            columns = list(connection.execute(f"PRAGMA table_info({_quote_identifier(name)})"))
            foreign_keys = list(
                connection.execute(f"PRAGMA foreign_key_list({_quote_identifier(name)})")
            )
            lines = [f"Table {_quote_identifier(name)}"]
            for _cid, column, declared_type, not_null, _default, primary_key in columns:
                attributes = [declared_type or ""]
                if primary_key:
                    attributes.append("PRIMARY KEY")
                if not_null:
                    attributes.append("NOT NULL")
                suffix = " ".join(item for item in attributes if item)
                lines.append(f"- {_quote_identifier(column)}{(' ' + suffix) if suffix else ''}")
            for row in sorted(
                foreign_keys, key=lambda item: (item[2], item[3], item[4] or "", item[0])
            ):
                reference = (
                    f"{_quote_identifier(row[2])}({_quote_identifier(row[4])})"
                    if row[4] is not None
                    else _quote_identifier(row[2])
                )
                lines.append(f"- FOREIGN KEY ({_quote_identifier(row[3])}) REFERENCES {reference}")
            tables.append(SchemaTable(name=name, text="\n".join(lines)))
        if not tables:
            raise ValueError(f"database has no user tables: {db_id}")
        return tuple(tables)
    finally:
        connection.close()


def serialize_schema(db_id: str, tables: Sequence[SchemaTable]) -> str:
    if not tables:
        raise ValueError("schema serialization requires at least one complete table")
    return f"Database: {db_id}\n\n" + "\n\n".join(table.text for table in tables)


def format_target(sql: str, *, strip_trailing_semicolon: bool = True) -> str:
    target = sql.strip()
    if strip_trailing_semicolon and target.endswith(";"):
        target = target[:-1].rstrip()
    if not target:
        raise ValueError("gold SQL target cannot be empty")
    return target


def build_messages(
    system: str, schema: str, question: str, target: str | None = None
) -> tuple[dict[str, str], ...]:
    user = f"Database schema:\n{schema}\n\nQuestion:\n{question.strip()}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return tuple(messages)


def normalize_single_token_ids(value: Any) -> list[int]:
    """Normalize supported tokenizer outputs to one non-empty, integer-only sequence."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer mapping output is missing input_ids")
        value = value["input_ids"]
    if not isinstance(value, list | tuple) and hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple) or isinstance(value, str | bytes):
        raise ValueError("tokenizer output must be a token sequence or single-batch input_ids")
    values = list(value)
    if not values:
        raise ValueError("tokenizer produced an empty token sequence")
    nested = [isinstance(item, list | tuple) or hasattr(item, "tolist") for item in values]
    if any(nested):
        if len(values) != 1 or not all(nested):
            raise ValueError("tokenizer output must contain exactly one batch")
        batch = values[0]
        if not isinstance(batch, list | tuple) and hasattr(batch, "tolist"):
            batch = batch.tolist()
        if not isinstance(batch, list | tuple) or isinstance(batch, str | bytes):
            raise ValueError("tokenizer single batch is not a token sequence")
        values = list(batch)
        if not values:
            raise ValueError("tokenizer produced an empty token sequence")
    if any(type(token) is not int for token in values):
        raise ValueError("token sequence must contain integers only (bool is forbidden)")
    return values


def validate_answer_only_token_contract(
    input_ids: Sequence[Any],
    labels: Sequence[Any],
    prompt_tokens: Any,
    *,
    answer_tokens: Any | None = None,
    supervised_tokens: Any | None = None,
) -> None:
    """Fail closed on malformed or internally inconsistent answer-only token records."""
    ids = list(input_ids)
    targets = list(labels)
    if not ids or len(ids) != len(targets):
        raise ValueError("input_ids and labels must be non-empty and equal length")
    if any(type(token) is not int for token in ids + targets):
        raise ValueError("input_ids and labels must contain integers only")
    if type(prompt_tokens) is not int or not 0 < prompt_tokens < len(ids):
        raise ValueError("prompt_tokens must identify a non-empty prompt and answer")
    observed_answer = len(ids) - prompt_tokens
    observed_supervised = sum(label != IGNORE_INDEX for label in targets)
    if targets[:prompt_tokens] != [IGNORE_INDEX] * prompt_tokens:
        raise ValueError("prompt labels must all use the ignore index")
    if targets[prompt_tokens:] != ids[prompt_tokens:]:
        raise ValueError("answer labels must equal answer input tokens")
    if answer_tokens is not None and (
        type(answer_tokens) is not int or answer_tokens != observed_answer
    ):
        raise ValueError("answer_tokens is inconsistent with the token arrays")
    if supervised_tokens is not None and (
        type(supervised_tokens) is not int or supervised_tokens != observed_supervised
    ):
        raise ValueError("supervised_tokens is inconsistent with labels")
    if observed_answer != observed_supervised:
        raise ValueError("every answer token, including EOS, must be supervised")


def _chat_ids(
    tokenizer: TokenizerLike,
    messages: Sequence[Mapping[str, str]],
    *,
    generation_prompt: bool,
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
    )
    return normalize_single_token_ids(ids)


def tokenize_answer_only(
    tokenizer: TokenizerLike,
    prompt_messages: Sequence[Mapping[str, str]],
    target: str,
) -> tuple[list[int], list[int], int]:
    prompt_ids = _chat_ids(tokenizer, prompt_messages, generation_prompt=True)
    target_ids = normalize_single_token_ids(tokenizer.encode(target, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    input_ids = prompt_ids + target_ids + [eos]
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids + [eos]
    validate_answer_only_token_contract(input_ids, labels, len(prompt_ids))
    return input_ids, labels, len(prompt_ids)


def prepare_example(
    row: Mapping[str, Any],
    schema_tables: Sequence[SchemaTable],
    tokenizer: TokenizerLike,
    config: Mapping[str, Any],
) -> PreparedExample:
    if (
        row.get("split") not in {"train", "validation"}
        or row.get("operational_eligible") is not True
    ):
        raise ValueError("M3 accepts only operationally eligible train/validation rows")
    if row.get("source") == "heldout":
        raise ValueError("heldout cannot enter M3 preparation")
    prompt = config["prompt"]
    target = format_target(
        str(row["sql"]), strip_trailing_semicolon=prompt["strip_target_trailing_semicolon"]
    )
    answer_ids = tokenizer.encode(target, add_special_tokens=False)
    answer_ids = normalize_single_token_ids(answer_ids)
    if len(answer_ids) + 1 > prompt["max_answer_tokens"]:
        raise ValueError("gold SQL exceeds the frozen answer-token budget")
    kept = list(schema_tables)
    while kept:
        schema = serialize_schema(str(row["db_id"]), kept)
        messages = build_messages(prompt["system"], schema, str(row["question"]))
        input_ids, labels, prompt_tokens = tokenize_answer_only(tokenizer, messages, target)
        if len(input_ids) <= prompt["max_sequence_tokens"]:
            break
        kept.pop()
    else:
        raise ValueError("question plus minimum complete schema and gold SQL exceed token budget")
    sample_id = str(row.get("row_id"))
    database_path = str(row.get("database_path", ""))
    answer_tokens = len(input_ids) - prompt_tokens
    supervised_tokens = sum(label != IGNORE_INDEX for label in labels)
    validate_answer_only_token_contract(
        input_ids,
        labels,
        prompt_tokens,
        answer_tokens=answer_tokens,
        supervised_tokens=supervised_tokens,
    )
    return PreparedExample(
        sample_id=sample_id,
        split=str(row["split"]),
        db_id=str(row["db_id"]),
        database_path=database_path,
        question=str(row["question"]),
        target_sql=target,
        schema_text=schema,
        messages=messages,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        supervised_tokens=supervised_tokens,
        prompt_sha256=sha256_json(messages),
    )


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("row_id")
    if not isinstance(value, str) or not value:
        raise ValueError("every selectable row requires a stable sample identity")
    return value


def select_tiny_slice(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    count: int,
    seed: str,
    required_features: Sequence[str],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    if split not in {"train", "validation"} or count < 1:
        raise ValueError("tiny slice split/count is invalid")
    eligible = [
        row
        for row in rows
        if row.get("split") == split
        and row.get("operational_eligible") is True
        and row.get("source") != "heldout"
    ]
    if len(eligible) < count:
        raise ValueError("not enough eligible rows for requested tiny slice")
    identities = [_row_id(row) for row in eligible]
    if len(set(identities)) != len(identities):
        raise ValueError(f"eligible {split} rows contain duplicate stable row IDs")
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{split}\0{row['db_id']}\0{_row_id(row)}".encode()
        ).hexdigest(),
    )
    selected: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for feature in required_features:
        match = next(
            (
                row
                for row in ranked
                if _row_id(row) not in used and feature in sql_complexity(str(row["sql"])).features
            ),
            None,
        )
        if match is None:
            raise ValueError(f"eligible {split} rows cannot cover required feature: {feature}")
        selected.append(match)
        used.add(_row_id(match))
    for row in ranked:
        if len(selected) >= count:
            break
        if _row_id(row) not in used:
            selected.append(row)
            used.add(_row_id(row))
    if len(selected) != count:
        raise ValueError("required feature coverage exceeds requested slice count")
    ids = [_row_id(row) for row in selected]
    payload = {
        "selector_version": "m3_tiny_selector_v1",
        "seed": seed,
        "split": split,
        "source_manifest_sha256": source_manifest_sha256,
        "sample_ids": ids,
    }
    return {
        **payload,
        "slice_sha256": sha256_json(payload),
        "database_ids": sorted({str(row["db_id"]) for row in selected}),
        "feature_counts": dict(
            sorted(
                Counter(
                    feature
                    for row in selected
                    for feature in sql_complexity(str(row["sql"])).features
                ).items()
            )
        ),
        "rows": selected,
    }


class AnswerOnlyCollator:
    """Pad pre-tokenized examples while preserving answer-only labels."""

    def __init__(
        self, pad_token_id: int, tensor_factory: Callable[[list[list[int]]], Any] | None = None
    ):
        self.pad_token_id = pad_token_id
        self.tensor_factory = tensor_factory

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for feature in features:
            ids = list(feature["input_ids"])
            target = list(feature["labels"])
            if len(ids) != len(target) or not any(label != IGNORE_INDEX for label in target):
                raise ValueError("collator requires aligned inputs with supervised answer tokens")
            padding = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            labels.append(target + [IGNORE_INDEX] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        result: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
        }
        if self.tensor_factory:
            result = {key: self.tensor_factory(value) for key, value in result.items()}
        return result


def extract_generated_sql(completion: str) -> dict[str, Any]:
    raw = completion.strip()
    strict = bool(raw) and "```" not in raw
    normalized = raw[:-1].rstrip() if raw.endswith(";") else raw
    if ";" in normalized:
        strict = False
    match = _QUERY_START.search(normalized)
    if not match or not normalized:
        return {"raw": completion, "sql": None, "strict_sql_only": False, "status": "no_query"}
    strict = strict and match.start() == 0
    candidate = normalized[match.start() :].strip()
    if strict and candidate != normalized:
        strict = False
    if not strict:
        fenced = _FENCE.findall(raw)
        source = fenced[-1].strip() if fenced else raw
        recovery = _QUERY_START.search(source)
        candidate = source[recovery.start() :].strip() if recovery else candidate
        if candidate.endswith(";"):
            candidate = candidate[:-1].rstrip()
    return {
        "raw": completion,
        "sql": candidate or None,
        "strict_sql_only": strict,
        "status": "strict" if strict else "recovered",
    }


def comparison_contract(
    validation_slice_sha256: str,
    prompt_hashes: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "validation_slice_sha256": validation_slice_sha256,
        "prompt_hashes": list(prompt_hashes),
        "decoding": config["decoding"],
        "evaluator": config["evaluator"],
        "model": config["model"],
    }
    return {**payload, "comparison_contract_sha256": sha256_json(payload)}


def verify_update_evidence(evidence: Mapping[str, Any]) -> None:
    numeric = ("initial_loss", "gradient_norm", "adapter_delta_l2_norm")
    if any(not math.isfinite(float(evidence.get(key, math.nan))) for key in numeric):
        raise ValueError("optimizer evidence contains a non-finite diagnostic")
    if int(evidence.get("trainable_parameter_count", 0)) <= 0:
        raise ValueError("no trainable adapter parameters were observed")
    if int(evidence.get("supervised_token_count", 0)) <= 0:
        raise ValueError("no answer tokens participated in loss")
    if float(evidence["gradient_norm"]) <= 0 or int(evidence.get("optimizer_steps", 0)) < 1:
        raise ValueError("no real optimizer update was demonstrated")
    if (
        evidence.get("adapter_parameter_delta_detected") is not True
        or float(evidence["adapter_delta_l2_norm"]) <= 0
    ):
        raise ValueError("adapter parameters did not change")


def verify_manifest_identity(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("training manifest SHA-256 differs from frozen M2 identity")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or not isinstance(payload.get("databases"), dict):
        raise ValueError("training manifest schema is not frozen M2 schema v2")
    if any(row.get("source") == "heldout" for row in payload.get("examples", [])):
        raise ValueError("heldout content appeared in the training manifest")
    return payload


def verify_m2_manifest_pair(
    index_path: str | Path,
    training_path: str | Path,
    *,
    expected_index_sha256: str,
    expected_training_sha256: str,
) -> dict[str, Any]:
    if sha256_file(index_path) != expected_index_sha256:
        raise ValueError("manifest index SHA-256 differs from frozen M2 identity")
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if index.get("training_manifest_sha256") != expected_training_sha256:
        raise ValueError("manifest index does not reference the frozen training manifest")
    return verify_manifest_identity(training_path, expected_training_sha256)


def validate_completed_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != 1 or result.get("status") != "completed":
        raise ValueError("M3 result is not a completed schema-v1 artifact")
    if result.get("heldout_examples_loaded") != 0 or result.get("heldout_examples_evaluated") != 0:
        raise ValueError("completed M3 result consumed heldout data")
    for section in ("memory", "timing", "cost"):
        if not isinstance(result.get(section), Mapping):
            raise ValueError(f"completed M3 result is missing {section} evidence")
    verify_update_evidence(result["training"])
    reload = result.get("fresh_reload")
    if not isinstance(reload, Mapping) or reload.get("different_process") is not True:
        raise ValueError("adapter was not demonstrated in a fresh process")
    contract = result.get("comparison_contract")
    if not isinstance(contract, Mapping) or not contract.get("comparison_contract_sha256"):
        raise ValueError("Base/SFT comparison contract identity is missing")
    base = result.get("base_evaluation")
    sft = result.get("sft_evaluation")
    if not isinstance(base, Mapping) or not isinstance(sft, Mapping):
        raise ValueError("Base and SFT evaluations are both required")
    for evaluation in (base, sft):
        metrics = evaluation.get("metrics")
        rows = evaluation.get("examples")
        if not isinstance(metrics, Mapping) or not isinstance(rows, list) or not rows:
            raise ValueError("evaluation result is empty")
        for metric in ("bird_ex_behavior_compatible", "strict_execution_accuracy"):
            value = metrics.get(metric)
            if not isinstance(value, int | float) or not 0 <= value <= 1:
                raise ValueError(f"invalid required evaluation metric: {metric}")
    base_ids = [row["sample_id"] for row in base["examples"]]
    sft_ids = [row["sample_id"] for row in sft["examples"]]
    base_prompts = [row["prompt_sha256"] for row in base["examples"]]
    sft_prompts = [row["prompt_sha256"] for row in sft["examples"]]
    if base_ids != sft_ids or base_prompts != sft_prompts:
        raise ValueError("Base and SFT did not evaluate identical samples and prompt bytes")
