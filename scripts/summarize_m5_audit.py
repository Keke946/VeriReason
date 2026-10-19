#!/usr/bin/env python3
"""Validate private M5 labels and emit an SQL-free aggregate draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m5_taxonomy import summarize_annotations, validate_annotations


def _read_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validated = validate_annotations(_read_list(args.audit), _read_list(args.annotations))
    summary = summarize_annotations(validated)
    if summary["review_required"]:
        summary["status"] = "draft_pending_independent_review"
    else:
        summary["status"] = "review_complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
