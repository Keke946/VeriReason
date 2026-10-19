#!/usr/bin/env python3
"""Load the M7 adapter in a distinct process and record SQL-free evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import artifact_manifest, load_base
from verireason.m4_contract import load_m4_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from peft import PeftModel

    config = load_m4_config(args.m4_config, "configs/m2_data.toml")
    base = load_base(config, trainable=False, allow_download=False)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    if not model.peft_config:
        raise RuntimeError("reloaded model has no PEFT adapter")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "adapter_manifest_sha256": artifact_manifest(Path(args.adapter))["manifest_sha256"],
        "adapter_names": sorted(model.peft_config),
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
