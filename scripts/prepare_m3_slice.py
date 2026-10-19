#!/usr/bin/env python3
"""Freeze real M3 tiny slices and token contracts from the private M2 training manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from verireason.m3_contract import (
    atomic_write_json,
    inspect_schema_tables,
    load_m3_config,
    prepare_example,
    select_tiny_slice,
    sha256_file,
    sha256_json,
    verify_m2_manifest_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-index", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--length-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer

    config = load_m3_config(args.config, args.m2_config)
    manifest = verify_m2_manifest_pair(
        args.manifest_index,
        args.training_manifest,
        expected_index_sha256=config["data"]["manifest_index_sha256"],
        expected_training_sha256=config["data"]["training_manifest_sha256"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=config["model"]["trust_remote_code"],
        local_files_only=not args.allow_model_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    rows = manifest["examples"]
    length_report = json.loads(Path(args.length_report).read_text(encoding="utf-8"))
    claimed_report_sha = length_report.pop("report_sha256", None)
    observed_report_sha = sha256_json(length_report)
    length_report["report_sha256"] = claimed_report_sha
    if (
        claimed_report_sha != observed_report_sha
        or length_report.get("source_manifest_sha256") != config["data"]["training_manifest_sha256"]
        or length_report.get("frozen_slice_counts")
        != {
            "train": config["data"]["train_count"],
            "validation": config["data"]["validation_count"],
        }
        or length_report.get("heldout_examples_loaded") != 0
        or length_report.get("heldout_examples_evaluated") != 0
    ):
        raise SystemExit("real token-length report identity or heldout contract is invalid")
    dataset_root = Path(args.dataset_root).resolve()
    slices = {}
    verified_databases = set()
    for split, count in (
        ("train", config["data"]["train_count"]),
        ("validation", config["data"]["validation_count"]),
    ):
        token_eligible = set(length_report["token_eligible_sample_ids"][split])
        if len(token_eligible) < count:
            raise SystemExit(
                f"{split} has only {len(token_eligible)} token-eligible rows; "
                f"fixed count is {count}"
            )
        selected = select_tiny_slice(
            rows,
            split=split,
            count=count,
            seed=config["data"]["selector_seed"],
            required_features=config["data"]["required_features"],
            source_manifest_sha256=config["data"]["training_manifest_sha256"],
        )
        prepared = []
        exclusions = []
        for row in selected.pop("rows"):
            if row["row_id"] not in token_eligible:
                exclusions.append(
                    {
                        "sample_id": row["row_id"],
                        "reason": "fixed sample is not token-eligible under the frozen contract",
                    }
                )
                continue
            identity = manifest["databases"][row["db_id"]]
            database = (dataset_root / identity["path"]).resolve()
            if not database.is_relative_to(dataset_root):
                raise SystemExit("manifest database path escapes the published dataset root")
            if row["db_id"] not in verified_databases:
                if sha256_file(database) != identity["sha256"]:
                    raise SystemExit(f"selected database identity changed: {row['db_id']}")
                verified_databases.add(row["db_id"])
            enriched = {**row, "database_path": str(database)}
            try:
                tables = inspect_schema_tables(database, row["db_id"])
                prepared.append(asdict(prepare_example(enriched, tables, tokenizer, config)))
            except ValueError as error:
                exclusions.append({"sample_id": row["row_id"], "reason": str(error)})
        if exclusions:
            raise SystemExit(
                f"selected {split} slice has token/schema exclusions; do not replace samples: "
                + json.dumps(exclusions, ensure_ascii=False)
            )
        selected["examples"] = prepared
        selected["selected_count"] = len(prepared)
        selected["prompt_hashes"] = [row["prompt_sha256"] for row in prepared]
        selected["token_lengths"] = {
            "prompt": [row["prompt_tokens"] for row in prepared],
            "answer": [row["answer_tokens"] for row in prepared],
            "total": [len(row["input_ids"]) for row in prepared],
        }
        slices[split] = selected
    train_databases = set(slices["train"]["database_ids"])
    validation_databases = set(slices["validation"]["database_ids"])
    if train_databases & validation_databases:
        raise SystemExit("M2 database-level train/validation boundary was violated")
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "synthetic": False,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "model": config["model"],
        "prompt_contract": config["prompt"],
        "source_manifest_sha256": config["data"]["training_manifest_sha256"],
        "token_length_report_sha256": claimed_report_sha,
        "slices": slices,
    }
    payload["slice_bundle_sha256"] = sha256_json(payload)
    output = Path(args.output)
    atomic_write_json(output, payload)
    print(
        json.dumps({"output": str(output), "slice_bundle_sha256": payload["slice_bundle_sha256"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
