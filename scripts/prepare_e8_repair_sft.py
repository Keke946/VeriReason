#!/usr/bin/env python3
"""Create the private E8 DB-disjoint split and a public SQL-free receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.repair_sft_smoke import (
    load_repair_sft_smoke_config,
    prepare_repair_sft_split,
    write_private_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e8_repair_sft_smoke.toml")
    parser.add_argument("--e7-private", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.e7_private.read_text())
    records = source.get("records")
    if source.get("schema_version") != 1 or not isinstance(records, list):
        raise ValueError("E7 private dataset is malformed")
    private, public = prepare_repair_sft_split(records, load_repair_sft_smoke_config(args.config))
    write_private_split(args.private_output, private)
    atomic_write_json(args.public_output, public)


if __name__ == "__main__":
    main()
