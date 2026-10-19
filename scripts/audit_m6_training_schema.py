#!/usr/bin/env python3
"""Run the aggregate-only M6 real-schema audit over the frozen training manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from verireason.m6_audit import audit_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = audit_training_manifest(
        args.training_manifest,
        expected_manifest_sha256=args.manifest_sha256,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
