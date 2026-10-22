#!/usr/bin/env python3
"""Build private E7 repair-SFT rows from a prepared M4 training bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.repair_sft_data import build_repair_sft_examples, load_repair_sft_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e7_repair_sft_data.toml")
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.prepared_train.read_text())
    if (
        source.get("split") != "train"
        or source.get("heldout_examples_loaded") != 0
        or source.get("heldout_examples_evaluated") != 0
    ):
        raise ValueError("prepared source is not the training-only M4 bundle")
    records, receipt = build_repair_sft_examples(
        source["examples"],
        args.database_root,
        load_repair_sft_config(args.config),
        evidence_scope="training_only_materialization",
        real_training_yield_known=True,
    )
    atomic_write_json(args.private_output, {"schema_version": 1, "records": records})
    os.chmod(args.private_output, 0o600)
    atomic_write_json(args.public_output, receipt)


if __name__ == "__main__":
    main()
