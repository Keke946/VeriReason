#!/usr/bin/env python3
"""Require real tokenizer files and the pinned TRL source; never fetch data or weights."""

import argparse
import json
import platform
import time
from pathlib import Path

from verireason.e5_interface_audit import (
    loss_audit,
    mapping_audit,
    rollout_export_audit,
    synthetic_execution_mapping_audit,
    training_source_audit,
    trl_source_audit,
)


def main():
    from tokenizers import Tokenizer, __version__

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--trl-wheel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    mapping = mapping_audit(tokenizer)
    execution_mapping = synthetic_execution_mapping_audit(tokenizer)
    loss = loss_audit()
    source = trl_source_audit(Path(args.trl_wheel))
    provenance = training_source_audit(Path(__file__).resolve().parents[1])
    checks_pass = (
        mapping["mapped_geometry_cases"] == len(mapping["geometry_cases"])
        and mapping["uncertain_exact_fallback_count"] == len(mapping["uncertain_cases"])
        and loss["all_values_finite"]
        and loss["maximum_gradient_absolute_error"] < 1e-9
        and execution_mapping["abstention_overrides"] == 0
    )
    result = {
        "experiment_id": "e5_cpu_interface_001",
        "decision": "Revise",
        "interface_checks_passed": checks_pass,
        "mapping": mapping,
        "synthetic_execution_mapping": execution_mapping,
        "loss": loss,
        "trl_source": source,
        "training_provenance": provenance,
        "rollout_export_policy": rollout_export_audit(Path(__file__).resolve().parents[1]),
        "tokenizer": {
            "repository": "Qwen/Qwen3-4B-Base",
            "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "tokenizers_version": __version__,
            "tokenizer_json_bytes": Path(args.tokenizer_json).stat().st_size,
        },
        "python": platform.python_version(),
        "wall_seconds": time.perf_counter() - started,
        "gpu_runs": 0,
        "paid_resource_starts": 0,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
        "decision_reason": "CPU geometry/loss mock is feasible, but no real training rows, "
        "generated SQL/IDs or SQLite databases are local; real coverage remains unknown.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
