#!/usr/bin/env python3
"""Publish a SQL-free E1/E2 common-evaluation comparison."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m8_contract import (
    build_public_comparison,
    load_m8_config,
    validate_e2_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m8_common_eval.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--e2-evaluation", required=True)
    parser.add_argument("--e1-raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_m8_config(args.config)
    bundle = json.loads(Path(args.prepared).read_text())
    e2 = json.loads(Path(args.e2_evaluation).read_text())
    validate_e2_evaluation(e2, bundle, config)
    m4 = json.loads(Path(config["sources"]["m4_public_summary"]).read_text())
    e1_raw = json.loads(Path(args.e1_raw).read_text()) if args.e1_raw else None
    result = build_public_comparison(config=config, m4_summary=m4, e2_evaluation=e2, e1_raw=e1_raw)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
