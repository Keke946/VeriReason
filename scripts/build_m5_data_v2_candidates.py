#!/usr/bin/env python3
"""Build private data-v2 candidates and a public SQL-free receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m5_audit_selection import write_private_json
from verireason.m5_data_v2 import build_data_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.prepared_train.read_text())
    if (
        payload.get("heldout_examples_loaded") != 0
        or payload.get("heldout_examples_evaluated") != 0
    ):
        raise ValueError("prepared training source must prove zero heldout use")
    candidates, receipt = build_data_v2_candidates(payload["examples"])
    receipt["prepared_bundle_sha256"] = payload.get("prepared_bundle_sha256")
    write_private_json(args.private_output, candidates)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
