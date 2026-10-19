#!/usr/bin/env python3
"""Measure real eligible M3 prompt/token lengths before freezing tiny-slice counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verireason.m3_contract import (
    atomic_write_json,
    inspect_schema_tables,
    load_m3_config,
    prepare_example,
    sha256_json,
    verify_m2_manifest_pair,
)


def _summary(values: list[int]) -> dict[str, int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "minimum": None, "p50": None, "p95": None, "maximum": None}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest-index", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
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
        trust_remote_code=False,
        local_files_only=not args.allow_model_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    schema_cache = {}
    dataset_root = Path(args.dataset_root).resolve()
    lengths = {
        split: {kind: [] for kind in ("prompt", "answer", "total")}
        for split in ("train", "validation")
    }
    exclusions = Counter()
    counts = Counter()
    token_eligible_ids = {"train": [], "validation": []}
    for row in manifest["examples"]:
        split = row.get("split")
        if split not in {"train", "validation"} or row.get("operational_eligible") is not True:
            continue
        counts[split] += 1
        db_id = row["db_id"]
        identity = manifest["databases"][db_id]
        database = (dataset_root / identity["path"]).resolve()
        if not database.is_relative_to(dataset_root):
            raise SystemExit("manifest database path escapes the published dataset root")
        if db_id not in schema_cache:
            schema_cache[db_id] = inspect_schema_tables(database, db_id)
        try:
            prepared = prepare_example(
                {**row, "database_path": str(database)}, schema_cache[db_id], tokenizer, config
            )
        except ValueError as error:
            exclusions[f"{split}:{error}"] += 1
            continue
        lengths[split]["prompt"].append(prepared.prompt_tokens)
        lengths[split]["answer"].append(prepared.answer_tokens)
        lengths[split]["total"].append(len(prepared.input_ids))
        token_eligible_ids[split].append(row["row_id"])
    payload = {
        "schema_version": 1,
        "synthetic": False,
        "source_manifest_sha256": config["data"]["training_manifest_sha256"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "eligible_counts": dict(sorted(counts.items())),
        "token_lengths": {
            split: {kind: _summary(values) for kind, values in kinds.items()}
            for split, kinds in lengths.items()
        },
        "exclusion_counts": dict(sorted(exclusions.items())),
        "token_eligible_sample_ids": token_eligible_ids,
        "frozen_slice_counts": {
            "train": config["data"]["train_count"],
            "validation": config["data"]["validation_count"],
        },
    }
    payload["report_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
