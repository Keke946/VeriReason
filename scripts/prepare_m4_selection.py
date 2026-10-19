#!/usr/bin/env python3
"""Freeze the full M4 train epoch and the dev256/final953/full1209 partition."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from verireason.m3_contract import (
    atomic_write_json,
    inspect_schema_tables,
    prepare_example,
    sha256_file,
    sha256_json,
    verify_m2_manifest_pair,
)
from verireason.m4_contract import load_m4_config, select_formal_rows, validate_validation_partition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--manifest-index", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--token-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-dir", required=True)
    args = parser.parse_args()
    config = load_m4_config(args.config, args.m2_config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    manifest = verify_m2_manifest_pair(
        args.manifest_index,
        args.training_manifest,
        expected_index_sha256=config["data"]["manifest_index_sha256"],
        expected_training_sha256=config["data"]["training_manifest_sha256"],
    )
    report = json.loads(Path(args.token_report).read_text())
    if (
        report.get("heldout_examples_loaded") != 0
        or report.get("source_manifest_sha256") != config["data"]["training_manifest_sha256"]
    ):
        raise SystemExit("token report identity/heldout contract failed")
    eligible = {
        split: set(report["token_eligible_sample_ids"][split]) for split in ("train", "validation")
    }
    rows = [
        {**row, "token_eligible": row["row_id"] in eligible[row["split"]]}
        for row in manifest["examples"]
        if row.get("split") in eligible
    ]
    train = select_formal_rows(
        rows,
        split="train",
        count=config["data"]["train_count"],
        seed=config["data"]["selector_seed"],
        required_features=config["data"]["required_features"],
        source_manifest_sha256=config["data"]["training_manifest_sha256"],
        selector_version=config["data"]["selector_version"],
    )
    if len(train["sample_ids"]) != len(eligible["train"]):
        raise SystemExit("full M4 train route would omit a token-eligible row")
    dev = select_formal_rows(
        rows,
        split="validation",
        count=config["data"]["development_validation_count"],
        seed=config["data"]["selector_seed"],
        required_features=config["data"]["required_features"],
        source_manifest_sha256=config["data"]["training_manifest_sha256"],
        selector_version=config["data"]["selector_version"],
    )
    dev_ids = set(dev["sample_ids"])
    remaining = [
        row for row in rows if row.get("split") != "validation" or row["row_id"] not in dev_ids
    ]
    final = select_formal_rows(
        remaining,
        split="validation",
        count=config["data"]["final_validation_count"],
        seed=config["data"]["selector_seed"],
        required_features=[],
        source_manifest_sha256=config["data"]["training_manifest_sha256"],
        selector_version=config["data"]["selector_version"],
    )
    full = {
        "selector_version": config["data"]["selector_version"],
        "seed": config["data"]["selector_seed"],
        "split": "validation",
        "count": len(dev["sample_ids"]) + len(final["sample_ids"]),
        "source_manifest_sha256": config["data"]["training_manifest_sha256"],
        "sample_ids": [*dev["sample_ids"], *final["sample_ids"]],
        "database_ids": sorted(set(dev["database_ids"]) | set(final["database_ids"])),
        "feature_counts": {},
    }
    full["selection_sha256"] = sha256_json(
        {
            key: full[key]
            for key in (
                "selector_version",
                "seed",
                "split",
                "count",
                "source_manifest_sha256",
                "sample_ids",
            )
        }
    )
    validate_validation_partition(dev, final, full)
    dataset_root, prepared_dir, schema_cache = (
        Path(args.dataset_root).resolve(),
        Path(args.prepared_dir),
        {},
    )
    prepared_dir.mkdir(parents=True, exist_ok=True)
    selections = (("train", train), ("dev", dev), ("final", final), ("full-validation", full))
    for name, selection in selections:
        lookup = {
            row["row_id"]: row
            for row in (
                [*train.get("rows", [])]
                if name == "train"
                else [*dev.get("rows", []), *final.get("rows", [])]
            )
        }
        prepared = []
        for sample_id in selection["sample_ids"]:
            row = lookup[sample_id]
            identity = manifest["databases"][row["db_id"]]
            database = (dataset_root / identity["path"]).resolve()
            if (
                not database.is_relative_to(dataset_root)
                or sha256_file(database) != identity["sha256"]
            ):
                raise SystemExit("selected database identity/path failed")
            schema_cache.setdefault(row["db_id"], inspect_schema_tables(database, row["db_id"]))
            prepared.append(
                asdict(
                    prepare_example(
                        {**row, "database_path": str(database)},
                        schema_cache[row["db_id"]],
                        tokenizer,
                        config,
                    )
                )
            )
        bundle = {
            "schema_version": 1,
            "split": name,
            "selection_sha256": selection["selection_sha256"],
            "sample_ids": selection["sample_ids"],
            "prompt_hashes": [item["prompt_sha256"] for item in prepared],
            "examples": prepared,
            "heldout_examples_loaded": 0,
            "heldout_examples_evaluated": 0,
        }
        bundle["prepared_bundle_sha256"] = sha256_json(bundle)
        atomic_write_json(prepared_dir / f"prepared-{name}.json", bundle)
    for selection in (train, dev, final):
        selection.pop("rows")
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "train": train,
        "development_validation": dev,
        "final_validation": final,
        "full_validation": full,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    payload["selection_bundle_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps({"selection_bundle_sha256": payload["selection_bundle_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
