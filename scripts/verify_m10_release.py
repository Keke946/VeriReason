#!/usr/bin/env python3
"""Write or verify the deterministic M10 public-evidence manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from verireason.release import (
    ReleaseVerificationError,
    build_release_manifest,
    render_manifest,
    verify_release_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.write:
            manifest = build_release_manifest(ROOT, args.config)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(render_manifest(manifest), encoding="utf-8")
            print(f"Wrote {args.manifest} with {manifest['artifact_count']} artifacts.")
        else:
            verify_release_manifest(ROOT, args.config, args.manifest)
            print(f"Verified {args.manifest} against all configured public artifacts.")
    except ReleaseVerificationError as error:
        print(f"M10 release verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
