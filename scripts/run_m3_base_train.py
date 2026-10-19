#!/usr/bin/env python3
"""Run M3 Base validation and tiny QLoRA, then stop before fresh-process reload."""

from __future__ import annotations

import argparse
import json

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import base_and_train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m3_vertical_slice.toml")
    parser.add_argument("--m2-config", default="configs/m2_data.toml")
    parser.add_argument("--slice", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()
    try:
        result = base_and_train(
            config_path=args.config,
            m2_config_path=args.m2_config,
            slice_path=args.slice,
            adapter_dir=args.adapter_dir,
            output=args.output,
            allow_download=args.allow_model_download,
        )
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        atomic_write_json(f"{args.output}.failure.json", failure)
        raise
    print(json.dumps({"status": result["status"], "output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
