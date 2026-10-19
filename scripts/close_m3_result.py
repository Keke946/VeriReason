#!/usr/bin/env python3
"""Build or verify the curated M3 closure summary from returned private evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m3_closure import build_closure_summary


def _provider_charge(value: str) -> tuple[str, str]:
    try:
        timestamp, amount = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("charge must be TIMESTAMP=AMOUNT") from error
    if not timestamp or not amount:
        raise argparse.ArgumentTypeError("charge must be TIMESTAMP=AMOUNT")
    return timestamp, amount


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--pin-commit", required=True)
    parser.add_argument("--provider-charge", action="append", type=_provider_charge, required=True)
    parser.add_argument("--previous-cumulative-cost-cny", required=True)
    parser.add_argument("--provider-balance-cny", required=True)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_closure_summary(
        args.evidence_dir,
        config_path=args.config,
        m2_config_path=args.m2_config,
        execution_commit=args.execution_commit,
        pin_commit=args.pin_commit,
        charges=args.provider_charge,
        previous_cumulative_cost_cny=args.previous_cumulative_cost_cny,
        provider_balance_cny=args.provider_balance_cny,
    )
    output = Path(args.output)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("M3 closure summary is missing or differs from the independent audit")
        print("M3 closure evidence and curated summary passed independent validation.")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote independently audited M3 closure summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
