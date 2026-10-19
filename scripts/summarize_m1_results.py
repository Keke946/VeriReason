#!/usr/bin/env python3
"""Derive deterministic M1 summary metrics from saved raw result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.smoke_sql import is_strict_sql_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        cases = result["metrics"]["cases"]
        summary = {
            "experiment_id": result["experiment_id"],
            "case_count": len(cases),
            "strict_sql_only_count": sum(is_strict_sql_only(case["completion"]) for case in cases),
        }
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
