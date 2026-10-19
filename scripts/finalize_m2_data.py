#!/usr/bin/env python3
"""Finalize M2 manifests from frozen preflight and diagnostic evidence without SQL execution."""

import argparse
import json
from pathlib import Path

from verireason.m2_finalizer import finalize_m2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization-partial", type=Path, required=True)
    parser.add_argument("--diagnostic-checkpoint", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--heldout-labels", type=Path, required=True)
    parser.add_argument(
        "--historical-formal-config", type=Path, default=Path("configs/m2_data_v1_3s.toml")
    )
    parser.add_argument("--amended-formal-config", type=Path, default=Path("configs/m2_data.toml"))
    parser.add_argument("--config", type=Path, default=Path("configs/m2_finalizer.toml"))
    parser.add_argument("--final-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_m2(
        materialization_partial=args.materialization_partial,
        diagnostic_checkpoint=args.diagnostic_checkpoint,
        training_labels=args.training_labels,
        heldout_labels=args.heldout_labels,
        historical_formal_config=args.historical_formal_config,
        amended_formal_config=args.amended_formal_config,
        finalizer_config=args.config,
        final_dir=args.final_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
