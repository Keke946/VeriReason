"""Run the bounded M1 candidate-model inference benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verireason.config import ModelSpec, SmokeConfig, load_smoke_config
from verireason.envcheck import collect_environment, validate_environment
from verireason.smoke_sql import is_strict_sql_only, score_completion


class GpuMemorySampler:
    """Sample whole-device memory so vLLM subprocesses are visible too."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if len(self.peak_mib) < len(values):
                    self.peak_mib.extend([0] * (len(values) - len(self.peak_mib)))
                self.peak_mib = [
                    max(old, new) for old, new in zip(self.peak_mib, values, strict=True)
                ]
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> GpuMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _prompt(config: SmokeConfig, question: str) -> str:
    schema = config.benchmark["schema_description"].strip()
    return (
        "Given this SQLite schema:\n"
        f"{schema}\n\n"
        f"Question: {question}\n"
        "Return exactly one read-only SQLite SELECT query and no explanation."
    )


def _package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "trl", "peft", "vllm", "accelerate", "bitsandbytes")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_source_identity() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    head = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain=v1")
    diff = run_git("diff", "--binary", "HEAD")
    source_paths = [
        Path("pyproject.toml"),
        Path("Makefile"),
        Path("configs"),
        Path("scripts"),
        Path("src"),
    ]
    source_files = sorted(
        path
        for root in source_paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
    )
    manifest = hashlib.sha256()
    for path in source_files:
        manifest.update(path.as_posix().encode())
        manifest.update(b"\0")
        manifest.update(path.read_bytes())
        manifest.update(b"\0")
    return {
        "git_head": head.strip() if head else None,
        "git_dirty": bool(status and status.strip()),
        "git_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "source_manifest_sha256": manifest.hexdigest(),
        "source_file_count": len(source_files),
    }


def _run_transformers(
    config: SmokeConfig, model_spec: ModelSpec, allow_download: bool
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.benchmark["seed"])
    torch.manual_seed(config.benchmark["seed"])
    torch.cuda.manual_seed_all(config.benchmark["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        local_files_only=not allow_download,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=not allow_download,
        trust_remote_code=False,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    first_prompt = _prompt(config, config.cases[0].question)
    warmup_inputs = tokenizer(
        first_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.benchmark["max_input_tokens"],
    ).to("cuda:0")
    warmup_start = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    case_results: list[dict[str, Any]] = []
    total_completion_tokens = 0
    generation_start = time.perf_counter()
    for case in config.cases:
        prompt = _prompt(config, case.question)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.benchmark["max_input_tokens"],
        ).to("cuda:0")
        input_tokens = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.benchmark["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, input_tokens:]
        completion_tokens = int(completion_ids.shape[-1])
        total_completion_tokens += completion_tokens
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        score = score_completion(config.benchmark["setup_sql"], completion, case.expected_rows)
        case_results.append(
            {
                "case_id": case.case_id,
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "completion": completion,
                "strict_sql_only": is_strict_sql_only(completion),
                "extracted_sql": score.extracted_sql,
                "valid_read_only_sql": score.valid_read_only_sql,
                "execution_correct": score.execution_correct,
                "rows": score.rows,
                "error": score.error,
            }
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    return {
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "generation_seconds": generation_seconds,
        "completion_tokens": total_completion_tokens,
        "completion_tokens_per_second": (
            total_completion_tokens / generation_seconds if generation_seconds else None
        ),
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "valid_sql_count": sum(row["valid_read_only_sql"] for row in case_results),
        "execution_correct_count": sum(row["execution_correct"] for row in case_results),
        "strict_sql_only_count": sum(row["strict_sql_only"] for row in case_results),
        "case_count": len(case_results),
        "cases": case_results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_smoke.toml")
    parser.add_argument("--environment-config", default="configs/environment.toml")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly-rate-cny", required=True, type=float)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads; omit to require a populated local cache.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_smoke_config(args.config)
    model_spec = config.model(args.model_key)
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "suite_id": config.benchmark["suite_id"],
        "status": "running",
        "started_at": started_at.isoformat(),
        "model": {
            "key": model_spec.key,
            "model_id": model_spec.model_id,
            "revision": model_spec.revision,
            "role": model_spec.role,
            "license": model_spec.license,
        },
        "backend": config.benchmark["backend"],
        "allow_download": args.allow_download,
        "packages": _package_versions(),
        "source": _git_source_identity(),
        "actual_cost_cny": None,
    }
    exit_code = 0
    sampler = GpuMemorySampler(config.benchmark["gpu_poll_interval_seconds"])
    try:
        environment = collect_environment("gpu", args.environment_config)
        environment_errors = validate_environment(environment)
        result["environment"] = environment
        if environment_errors:
            raise RuntimeError("environment gate failed: " + "; ".join(environment_errors))
        with sampler:
            result["metrics"] = _run_transformers(config, model_spec, args.allow_download)
        result["metrics"]["nvidia_smi_peak_used_mib"] = sampler.peak_mib
        result["status"] = "completed"
    except Exception as error:  # Preserve failed-run evidence before returning non-zero.
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        wall_seconds = time.perf_counter() - wall_start
        result["completed_at"] = datetime.now(UTC).isoformat()
        result["wall_seconds"] = wall_seconds
        result["hourly_rate_cny"] = args.hourly_rate_cny
        result["estimated_command_cost_cny"] = args.hourly_rate_cny * wall_seconds / 3600
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
