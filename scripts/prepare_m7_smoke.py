#!/usr/bin/env python3
"""Select frozen post-training prompts and join them to immutable M4 prepared inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json, sha256_file, sha256_json
from verireason.m7_contract import select_smoke_rows
from verireason.posttraining_contract import load_posttraining_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m7_e2_smoke.toml")
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--prepared-train", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_posttraining_config(args.config)
    if sha256_file(args.training_manifest) != config["data"]["training_manifest_sha256"]:
        raise SystemExit("training manifest identity mismatch")
    manifest = json.loads(Path(args.training_manifest).read_text())
    prepared = json.loads(Path(args.prepared_train).read_text())
    if prepared.get("heldout_examples_loaded") or prepared.get("heldout_examples_evaluated"):
        raise SystemExit("M4 prepared bundle reports heldout use")

    rows = select_smoke_rows(
        manifest["examples"],
        count=config["data"]["prompt_count"],
        seed=config["data"]["selector_seed"],
        required_features=config["data"]["required_features"],
    )
    prepared_by_id = {row["sample_id"]: row for row in prepared["examples"]}
    root = Path(args.database_root).resolve()
    examples = []
    for row in rows:
        source = prepared_by_id.get(row["row_id"])
        if source is None or source.get("split") != "train":
            raise SystemExit("selected row is missing from the immutable M4 train bundle")
        logical_path = manifest["databases"][row["db_id"]]["path"]
        database = (root / logical_path).resolve()
        if not database.is_relative_to(root) or not database.is_file():
            raise SystemExit("selected database path is unavailable")
        examples.append(
            {
                "reward_group_id": row["row_id"],
                "messages": source["messages"],
                "gold_sql": row["sql"],
                "database_path": logical_path,
            }
        )
    payload = {
        "schema_version": 1,
        "selection_policy": config["data"]["selector_version"],
        "sample_ids": [row["row_id"] for row in rows],
        "examples": examples,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    payload["bundle_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps({"bundle_sha256": payload["bundle_sha256"], "prompts": len(examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
