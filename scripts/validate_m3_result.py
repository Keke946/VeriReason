#!/usr/bin/env python3
"""Validate a completed real M3 result against the frozen vertical-slice contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m3_contract import load_m3_config, validate_completed_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    args = parser.parse_args()
    load_m3_config(args.config, args.m2_config)
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    validate_completed_result(result)
    print("M3 result contract passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
