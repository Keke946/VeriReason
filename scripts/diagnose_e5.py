#!/usr/bin/env python3
"""Run only the built-in synthetic E5 suite; no external dataset or database input."""

import argparse
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path

import sqlglot

from verireason.e5_diagnostics import diagnose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = diagnose()
    result["wall_seconds"] = time.perf_counter() - started
    result["runtime"] = {
        "python": platform.python_version(),
        "sqlglot": sqlglot.__version__,
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.system(),
        "gpu_packages_imported": sorted(
            name for name in ("torch", "trl", "transformers", "vllm") if name in sys.modules
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return int(bool(result["expectation_mismatches"]))


if __name__ == "__main__":
    raise SystemExit(main())
