"""Inspect a local or CUDA environment against a checked-in profile."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verireason.config import load_toml


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _base_distribution_version(version: str | None) -> str | None:
    """Ignore a PEP 440 local build label such as PyTorch's ``+cu130``."""
    return version.partition("+")[0] if version else None


def _nvidia_smi(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(number) for number in numbers[:3])


def collect_environment(profile: str, config_path: str | Path) -> dict[str, Any]:
    config = load_toml(config_path)
    expected = config["profiles"][profile]
    report: dict[str, Any] = {
        "profile": profile,
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "expected": expected,
        "imports": {},
        "packages": {},
        "cuda": {"available": False},
    }

    for package in expected.get("packages", {}):
        report["packages"][package] = _package_version(package)
        if profile == "gpu":
            try:
                importlib.import_module(package)
                report["imports"][package] = {"ok": True, "error": None}
            except Exception as error:  # Import compatibility is part of the GPU gate.
                report["imports"][package] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }

    if profile == "gpu":
        report["nvidia_smi"] = _nvidia_smi([])
        report["topology"] = _nvidia_smi(["topo", "-m"])
        try:
            import torch

            available = torch.cuda.is_available()
            report["cuda"] = {
                "available": available,
                "runtime": torch.version.cuda,
                "device_count": torch.cuda.device_count() if available else 0,
                "devices": [],
            }
            if available:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    report["cuda"]["devices"].append(
                        {
                            "index": index,
                            "name": properties.name,
                            "total_memory_gib": properties.total_memory / (1024**3),
                            "compute_capability": f"{properties.major}.{properties.minor}",
                        }
                    )
        except ImportError:
            report["cuda"]["import_error"] = "torch is not installed"

        driver = _nvidia_smi(["--query-gpu=driver_version", "--format=csv,noheader"])
        report["cuda"]["driver"] = driver.splitlines()[0] if driver else None

    return report


def validate_environment(report: dict[str, Any]) -> list[str]:
    expected = report["expected"]
    errors: list[str] = []
    if not report["python"].startswith(f"{expected['python_major_minor']}."):
        errors.append(
            f"Python {report['python']} does not match {expected['python_major_minor']}.x"
        )

    for field in ("system", "machine"):
        if field in expected and report[field] != expected[field]:
            errors.append(f"{field} {report[field]!r} does not match {expected[field]!r}")

    for package, version in expected.get("packages", {}).items():
        actual = report["packages"].get(package)
        if _base_distribution_version(actual) != version:
            errors.append(f"{package} {actual or 'missing'} does not match {version}")
        import_result = report.get("imports", {}).get(package)
        if import_result and not import_result["ok"]:
            errors.append(f"{package} import failed: {import_result['error']}")

    if report["profile"] == "gpu":
        cuda = report["cuda"]
        if not cuda.get("available"):
            errors.append("CUDA is not available")
            return errors
        runtime = cuda.get("runtime") or ""
        if not runtime.startswith(expected["cuda_runtime_major_minor"]):
            errors.append(
                f"CUDA runtime {runtime or 'missing'} does not match "
                f"{expected['cuda_runtime_major_minor']}.x"
            )
        driver = cuda.get("driver")
        if not driver or _version_tuple(driver) < _version_tuple(expected["minimum_nvidia_driver"]):
            errors.append(
                f"NVIDIA driver {driver or 'missing'} is below {expected['minimum_nvidia_driver']}"
            )
        if cuda.get("device_count", 0) < expected["minimum_cuda_devices"]:
            errors.append("not enough CUDA devices")
        for device in cuda.get("devices", []):
            if device["total_memory_gib"] < expected["minimum_device_memory_gib"]:
                errors.append(
                    f"GPU {device['index']} has {device['total_memory_gib']:.2f} GiB; "
                    f"requires at least {expected['minimum_device_memory_gib']:.2f} GiB"
                )

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("local", "gpu"), required=True)
    parser.add_argument("--config", default="configs/environment.toml")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_environment(args.profile, args.config)
    report["errors"] = validate_environment(report)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
