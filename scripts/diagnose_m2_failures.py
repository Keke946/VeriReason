#!/usr/bin/env python3
"""Rerun only prior M2 gold-preflight failures under diagnostic-only profiles."""

import argparse
import json
import os
from pathlib import Path

from verireason.data_pipeline import load_examples
from verireason.m2_diagnostics import (
    build_public_report,
    load_diagnostic_config,
    require_expected_selection,
    require_private_output_directory,
    run_diagnostics,
    select_failed_queries,
    verify_input_evidence,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_public(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    os.chmod(path, 0o644)


def _partial_root(private_preflight: Path) -> Path:
    resolved = private_preflight.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if candidate.name.endswith(".partial"):
            return candidate
    raise ValueError("private preflight must be located below materialization .partial")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-preflight", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--heldout-labels", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, default=Path("configs/m2_data_v1_3s.toml"))
    parser.add_argument("--training-database-root", type=Path, required=True)
    parser.add_argument("--heldout-database-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/m2_diagnostic.toml"))
    args = parser.parse_args()

    config = load_diagnostic_config(args.config)
    source_identity = verify_input_evidence(
        config,
        {
            "private_preflight": args.private_preflight,
            "training_labels": args.training_labels,
            "heldout_labels": args.heldout_labels,
            "formal_config": args.formal_config,
            "source_receipt": args.source_receipt,
        },
    )
    output_dir = require_private_output_directory(
        args.output_dir,
        git_worktree=REPOSITORY_ROOT,
        materialization_partial=_partial_root(args.private_preflight),
    )
    private_preflight = json.loads(args.private_preflight.read_text(encoding="utf-8"))
    items = select_failed_queries(
        private_preflight,
        {
            "training": load_examples(args.training_labels),
            "heldout": load_examples(args.heldout_labels, source="heldout"),
        },
    )
    require_expected_selection(items, config)
    checkpoint_path = output_dir / "diagnostic-checkpoint-private.json"
    checkpoint = run_diagnostics(
        items,
        {
            "training": args.training_database_root,
            "heldout": args.heldout_database_root,
        },
        config,
        checkpoint_path,
        source_identity=source_identity,
    )
    report = build_public_report(checkpoint, items)
    _write_public(output_dir / "diagnostic-summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
