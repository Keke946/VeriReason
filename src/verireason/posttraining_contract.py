"""Dispatch immutable E2/E3/E4 post-training configurations."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_posttraining_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        experiment_id = tomllib.load(handle).get("experiment_id", "")
    if experiment_id.startswith("e2_m7_"):
        from verireason.m7_contract import load_m7_config

        return load_m7_config(path)
    if experiment_id.startswith(("e3_m8_", "e4_m8_")):
        from verireason.m8_rl_contract import load_m8_rl_config

        return load_m8_rl_config(path)
    raise ValueError(f"unsupported post-training experiment: {experiment_id!r}")
