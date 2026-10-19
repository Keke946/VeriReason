#!/usr/bin/env python3
"""Create a private M5 audit worksheet and a SQL-free public selection receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m5_audit_selection import select_audit_rows, write_private_json


def _examples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain an examples list")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--seed", default="m5-audit-v1")
    args = parser.parse_args()

    prepared = [row for path in args.prepared for row in _examples(path)]
    private, public = select_audit_rows(_examples(args.evaluation), prepared, seed=args.seed)
    write_private_json(args.private_output, private)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
