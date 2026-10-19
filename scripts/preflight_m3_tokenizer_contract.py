#!/usr/bin/env python3
"""Verify the real offline tokenizer's single-example answer-only token contract."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from verireason.m3_contract import (
    atomic_write_json,
    build_messages,
    load_m3_config,
    normalize_single_token_ids,
    tokenize_answer_only,
    validate_answer_only_token_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    config = load_m3_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    messages = build_messages(
        config["prompt"]["system"],
        'Database: synthetic\n\nTable "items"\n- "id" INTEGER PRIMARY KEY',
        "Return every item id.",
    )
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    normalized = normalize_single_token_ids(raw)
    input_ids, labels, prompt_tokens = tokenize_answer_only(
        tokenizer, messages, 'SELECT "id" FROM "items"'
    )
    validate_answer_only_token_contract(input_ids, labels, prompt_tokens)
    raw_input = raw.get("input_ids") if isinstance(raw, Mapping) else raw
    raw_shape = getattr(raw_input, "shape", None)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "synthetic": True,
        "model_id": config["model"]["model_id"],
        "revision": config["model"]["revision"],
        "raw_type": type(raw).__name__,
        "raw_mapping_keys": sorted(raw.keys()) if isinstance(raw, Mapping) else [],
        "raw_input_ids_shape": list(raw_shape) if raw_shape is not None else None,
        "normalized_token_count": len(normalized),
        "normalized_all_exact_int": all(type(token) is int for token in normalized),
        "prompt_tokens": prompt_tokens,
        "answer_tokens": len(input_ids) - prompt_tokens,
        "supervised_tokens": sum(label != -100 for label in labels),
    }
    if prompt_tokens <= 2:
        raise ValueError("synthetic prompt unexpectedly produced at most two tokens")
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
