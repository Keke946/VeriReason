#!/usr/bin/env python3
"""Validate the M9 source, prepared selection, and adapter before starting vLLM."""

import argparse
import json
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import artifact_manifest
from verireason.m9_serving import load_m9_config, select_benchmark_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m9_serving.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit("M9 preflight output already exists")
    config = load_m9_config(args.config)
    bundle = json.loads(Path(args.prepared).read_text())
    examples, selection_sha = select_benchmark_examples(bundle, config)
    adapter = artifact_manifest(args.adapter_dir)
    if adapter["manifest_sha256"] != config["model"]["adapter_manifest_sha256"]:
        raise SystemExit("M9 E2 adapter manifest drifted")
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "status": "passed",
            "experiment_id": config["experiment_id"],
            "adapter_manifest_sha256": adapter["manifest_sha256"],
            "benchmark_selection_sha256": selection_sha,
            "benchmark_example_count": len(examples),
            "heldout_examples_loaded": 0,
            "heldout_examples_evaluated": 0,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
