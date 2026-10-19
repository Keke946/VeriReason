#!/usr/bin/env python3
"""Publish a SQL-free E1/E2/E3/E4 common-evaluation comparison."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m8_contract import (
    build_e4_public_comparison,
    load_m8_config,
    validate_candidate_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m8_e4_common_eval.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--e4-evaluation", required=True)
    parser.add_argument("--e1-raw")
    parser.add_argument("--e2-raw")
    parser.add_argument("--e3-raw")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_m8_config(args.config)
    bundle = json.loads(Path(args.prepared).read_text())
    e4 = json.loads(Path(args.e4_evaluation).read_text())
    validate_candidate_evaluation(e4, bundle, config)
    common = json.loads(Path(config["sources"]["m8_e3_common_summary"]).read_text())
    raw = {
        arm: json.loads(Path(path).read_text()) if path else None
        for arm, path in (("E1", args.e1_raw), ("E2", args.e2_raw), ("E3", args.e3_raw))
    }
    result = build_e4_public_comparison(
        config=config,
        existing_e3_common=common,
        e4_evaluation=e4,
        e1_raw=raw["E1"],
        e2_raw=raw["E2"],
        e3_raw=raw["E3"],
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
