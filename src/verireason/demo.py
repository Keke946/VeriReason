"""Run the selected VeriReason E2 adapter on an authorized SQLite database."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verireason.m3_contract import (
    build_messages,
    extract_generated_sql,
    inspect_schema_tables,
    normalize_single_token_ids,
    serialize_schema,
)
from verireason.m3_runtime import artifact_manifest, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config
from verireason.sql_evaluator import execute_read_only

E2_ADAPTER_MANIFEST_SHA256 = "3492a3ec870f7c6e8e672459f025fc96a9a8cba543b27b7df4fc5f6c6bfbd3b7"


def prepare_demo_prompt(
    database: str | Path,
    question: str,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the measured prompt contract, dropping only whole trailing tables if needed."""
    if not question.strip():
        raise ValueError("question cannot be empty")
    path = Path(database)
    tables = list(inspect_schema_tables(path, path.stem))
    maximum_prompt_tokens = int(config["prompt"]["max_sequence_tokens"]) - int(
        config["decoding"]["max_new_tokens"]
    )
    dropped: list[str] = []
    while tables:
        schema = serialize_schema(path.stem, tables)
        messages = build_messages(config["prompt"]["system"], schema, question)
        raw_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = normalize_single_token_ids(raw_ids)
        if len(input_ids) <= maximum_prompt_tokens:
            return {
                "messages": messages,
                "input_ids": input_ids,
                "schema": schema,
                "prompt_tokens": len(input_ids),
                "dropped_tables": dropped,
            }
        if len(tables) == 1:
            break
        dropped.append(tables.pop().name)
    raise ValueError("question plus the smallest complete schema exceeds the 2,048-token contract")


def _jsonable_rows(rows: tuple[tuple[Any, ...], ...], limit: int = 20) -> list[list[Any]]:
    return [
        [
            value if value is None or isinstance(value, str | int | float) else str(value)
            for value in row
        ]
        for row in rows[:limit]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="SQLite file to inspect read-only")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--adapter", required=True, help="Extracted VeriReason E2 adapter directory"
    )
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Permit download of the fixed Qwen base revision; otherwise require a local cache",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated query with the frozen read-only resource limits",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON")
    args = parser.parse_args()

    database = Path(args.database).resolve()
    adapter = Path(args.adapter).resolve()
    if not database.is_file():
        raise SystemExit(f"SQLite database does not exist: {database}")
    if not adapter.is_dir():
        raise SystemExit(f"adapter directory does not exist: {adapter}")
    observed = artifact_manifest(adapter)["manifest_sha256"]
    if observed != E2_ADAPTER_MANIFEST_SHA256:
        raise SystemExit(
            "adapter identity does not match the selected E2 release: "
            f"expected {E2_ADAPTER_MANIFEST_SHA256}, observed {observed}"
        )

    config = load_m4_config(args.config, args.m2_config)
    tokenizer = load_tokenizer(config, args.allow_model_download)
    prompt = prepare_demo_prompt(database, args.question, tokenizer, config)

    import torch
    from peft import PeftModel

    base = load_base(config, trainable=False, allow_download=args.allow_model_download)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True)
    model.eval()
    inputs = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cuda:0")
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=False,
            num_beams=1,
            max_new_tokens=config["decoding"]["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=True)
    extraction = extract_generated_sql(completion)
    payload: dict[str, Any] = {
        "model": f"{config['model']['model_id']}@{config['model']['revision']}",
        "adapter_manifest_sha256": observed,
        "prompt_tokens": prompt["prompt_tokens"],
        "dropped_tables": prompt["dropped_tables"],
        "sql": extraction["sql"],
        "strict_sql_only": extraction["strict_sql_only"],
    }
    if args.execute and extraction["sql"]:
        result = execute_read_only(database, extraction["sql"])
        payload["execution"] = {
            "status": result.status,
            "row_count": result.row_count,
            "column_count": result.column_count,
            "duration_seconds": result.duration_seconds,
            "rows_preview": _jsonable_rows(result.rows),
            "error": result.error,
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["sql"] or completion.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
