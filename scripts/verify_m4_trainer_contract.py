#!/usr/bin/env python3
"""Verify the exact Transformers branch required by M4's custom loss contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help=(
            "report a missing local torch/Transformers runtime without claiming the regression ran"
        ),
    )
    parser.add_argument("--output", help="optional atomic JSON evidence path")
    args = parser.parse_args()

    def publish(payload: dict) -> None:
        if args.output:
            atomic_write_json(Path(args.output), payload)
        print(json.dumps(payload, sort_keys=True))

    try:
        import torch
        import transformers
        from transformers import Trainer
    except ModuleNotFoundError as error:
        payload = {
            "status": "unavailable_not_verified",
            "reason": f"{type(error).__name__}: {error}",
            "required_transformers_version": "5.14.1",
        }
        publish(payload)
        return 0 if args.allow_unavailable else 2
    if transformers.__version__ != "5.14.1":
        raise SystemExit(
            f"M4 requires transformers==5.14.1, found {transformers.__version__}; do not train"
        )
    source = inspect.getsource(Trainer.training_step)
    required_source_fragments = (
        "model_accepts_loss_kwargs",
        "num_items_in_batch",
        "current_gradient_accumulation_steps",
        "not self.model_accepts_loss_kwargs or num_items_in_batch is None",
    )
    missing = [fragment for fragment in required_source_fragments if fragment not in source]
    if missing:
        raise SystemExit(f"M4 Trainer 5.14.1 loss/GAS source contract drifted: {missing}")
    publish(
        {
            "status": "verified",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "training_step_source_fragments": required_source_fragments,
            "training_step_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
