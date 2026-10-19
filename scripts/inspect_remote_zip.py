#!/usr/bin/env python3
"""Print remote ZIP size evidence using HEAD and Range requests only."""

import argparse

from verireason.remote_zip import inspect_remote_zip, summary_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    args = parser.parse_args()
    print(summary_json(inspect_remote_zip(args.url)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
