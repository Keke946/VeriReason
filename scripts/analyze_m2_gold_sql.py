#!/usr/bin/env python3
"""Run the static SQL safety analyzer over fixed M2 gold labels."""

import argparse
import json
from collections import Counter
from pathlib import Path

from verireason.data_pipeline import load_examples, sha256_file
from verireason.sql_evaluator import analyze_sql


def analyze(path: Path, source: str) -> dict[str, object]:
    examples = load_examples(path, source=source)
    rejected: Counter[str] = Counter()
    failure_ids: dict[str, list[str]] = {}
    for example in examples:
        result = analyze_sql(example.sql)
        if not result.accepted:
            category = result.error or "unknown"
            rejected[category] += 1
            failure_ids.setdefault(category, []).append(example.row_id)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "total": len(examples),
        "accepted": len(examples) - sum(rejected.values()),
        "rejected": sum(rejected.values()),
        "rejected_by_category": dict(sorted(rejected.items())),
        "rejected_example_ids": {key: sorted(value) for key, value in sorted(failure_ids.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--heldout-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "training": analyze(args.training_labels, "training"),
        "heldout": analyze(args.heldout_labels, "heldout"),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(payload["training"]["rejected"] or payload["heldout"]["rejected"])


if __name__ == "__main__":
    raise SystemExit(main())
