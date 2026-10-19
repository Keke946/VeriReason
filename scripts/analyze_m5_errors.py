#!/usr/bin/env python3
"""Build the curated M5 aggregate error-analysis artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.data_pipeline import sha256_file
from verireason.m5_error_analysis import analyze_m4_errors


def _read_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain an object")
    return payload


def _examples(payload: dict, path: Path) -> list[dict]:
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"{path} does not contain an examples list")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_payload = _read_payload(args.evaluation)
    prepared = [row for path in args.prepared for row in _examples(_read_payload(path), path)]
    report = analyze_m4_errors(_examples(evaluation_payload, args.evaluation), prepared)
    report["source_contract"] = {
        "evaluation_sha256": sha256_file(args.evaluation),
        "prepared_sha256": [sha256_file(path) for path in args.prepared],
        "condition": evaluation_payload.get("condition"),
        "validation_selection_sha256": evaluation_payload.get("validation_selection_sha256"),
        "heldout_examples_loaded": evaluation_payload.get("heldout_examples_loaded"),
        "heldout_examples_evaluated": evaluation_payload.get("heldout_examples_evaluated"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
