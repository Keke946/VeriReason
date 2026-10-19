#!/usr/bin/env python3
"""Create the SQL-free M9 serving summary from raw benchmark and GPU samples."""

import argparse
import json
from pathlib import Path

from verireason.data_pipeline import sha256_file
from verireason.m3_contract import atomic_write_json
from verireason.m9_serving import build_public_summary, load_m9_config


def peak_gpu_memory(path: str | Path) -> float:
    values = []
    for line in Path(path).read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                values.append(float(parts[-2]))
            except ValueError:
                continue
    if not values:
        raise ValueError("M9 GPU sampler has no memory observations")
    return max(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m9_serving.toml")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--gpu-samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = load_m9_config(args.config)
    raw = json.loads(Path(args.raw).read_text())
    summary = build_public_summary(
        raw,
        config,
        raw_result_sha256=sha256_file(args.raw),
        ready_gpu_memory_mib=float(raw["ready_gpu_memory_mib"]),
        peak_gpu_memory_mib=peak_gpu_memory(args.gpu_samples),
    )
    output = Path(args.output)
    if args.check:
        if not output.is_file() or json.loads(output.read_text()) != summary:
            raise SystemExit("M9 public summary differs from reconstructed evidence")
    else:
        if output.exists():
            raise SystemExit("M9 public summary output already exists")
        atomic_write_json(output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
