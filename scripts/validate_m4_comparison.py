#!/usr/bin/env python3
"""Validate the predeclared E0/E1 same-exam contract."""

import argparse
import json
from pathlib import Path

from verireason.m4_contract import validate_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("sft")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text())
    sft = json.loads(Path(args.sft).read_text())
    validate_comparison(base, sft)
    print(
        json.dumps(
            {"status": "passed", "validation_selection_sha256": base["validation_selection_sha256"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
