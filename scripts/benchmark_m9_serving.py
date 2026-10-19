#!/usr/bin/env python3
"""Benchmark the static E2 vLLM endpoint on real Text-to-SQL validation prompts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verireason.config import load_m2_config
from verireason.m3_contract import atomic_write_json, evaluator_limits, extract_generated_sql
from verireason.m9_serving import (
    load_m9_config,
    scenario_metrics,
    select_benchmark_examples,
    validate_raw_result,
)
from verireason.sql_evaluator import evaluate_prediction


def _json_request(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def _stream_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    chunks: list[str] = []
    usage = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if text:
                    first_token = first_token or time.perf_counter()
                    chunks.append(text)
        ended = time.perf_counter()
        if first_token is None or not usage or int(usage.get("completion_tokens", 0)) <= 0:
            raise ValueError("stream omitted first-token or completion-usage evidence")
        return {
            "success": True,
            "ttft_seconds": first_token - started,
            "e2e_seconds": ended - started,
            "completion_tokens": int(usage["completion_tokens"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion": "".join(chunks),
            "error": None,
        }
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as error:
        return {
            "success": False,
            "ttft_seconds": None,
            "e2e_seconds": time.perf_counter() - started,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "completion": "",
            "error": f"{type(error).__name__}: {error}",
        }


def _failure_category(extracted: dict[str, Any], score: Any, error: str | None) -> str:
    if not extracted["sql"]:
        return "no_query"
    if error:
        return "evaluator_error"
    if score.prediction_status != "ok":
        return score.prediction_status
    if not extracted["strict_sql_only"]:
        return "format_noncompliant"
    return "strict_correct" if score.strict_execution_accuracy else "wrong_result"


def _correctness(
    observations: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    m2_config: dict[str, Any],
) -> dict[str, Any]:
    limits = evaluator_limits(m2_config)
    rows = []
    for observation, example in zip(observations, examples, strict=True):
        extracted = extract_generated_sql(observation["completion"])
        score, error = None, None
        if extracted["sql"]:
            try:
                score = evaluate_prediction(
                    example["database_path"],
                    extracted["sql"],
                    example["target_sql"],
                    limits,
                )
            except Exception as failure:  # Evidence is retained and scored zero.
                error = f"{type(failure).__name__}: {failure}"
        rows.append(
            {
                "sample_id": example["sample_id"],
                "prompt_sha256": example["prompt_sha256"],
                "completion": observation["completion"],
                "extraction": extracted,
                "bird_ex_behavior_compatible": (
                    score.bird_ex_behavior_compatible if score else 0
                ),
                "strict_execution_accuracy": score.strict_execution_accuracy if score else 0,
                "prediction_status": score.prediction_status if score else "not_evaluated",
                "failure_category": _failure_category(extracted, score, error),
                "error": error,
            }
        )
    count = len(rows)
    return {
        "example_count": count,
        "bird_ex_behavior_compatible": sum(
            row["bird_ex_behavior_compatible"] for row in rows
        )
        / count,
        "strict_execution_accuracy": sum(row["strict_execution_accuracy"] for row in rows)
        / count,
        "strict_sql_only_rate": sum(row["extraction"]["strict_sql_only"] for row in rows)
        / count,
        "valid_sql_rate": sum(row["prediction_status"] == "ok" for row in rows) / count,
        "failure_categories": dict(
            sorted(Counter(row["failure_category"] for row in rows).items())
        ),
        "examples": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m9_serving.toml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--server-startup-seconds", required=True, type=float)
    parser.add_argument("--ready-gpu-memory-mib", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit("M9 raw output already exists")
    config = load_m9_config(args.config)
    bundle = json.loads(Path(args.prepared).read_text())
    examples, selection_sha = select_benchmark_examples(bundle, config)

    models = _json_request(f"{args.endpoint}/v1/models").get("data", [])
    model_ids = [row.get("id") for row in models]
    served_name = config["model"]["served_model_name"]
    if served_name not in model_ids:
        raise SystemExit("M9 E2 adapter is absent from /v1/models")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    prompts = [
        tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for example in examples
    ]
    request_kwargs = {
        "endpoint": args.endpoint,
        "model": served_name,
        "max_tokens": config["decoding"]["max_tokens"],
        "timeout_seconds": config["benchmark"]["request_timeout_seconds"],
    }
    for prompt in prompts[: config["benchmark"]["warmup_requests"]]:
        if not _stream_completion(prompt=prompt, **request_kwargs)["success"]:
            raise SystemExit("M9 warmup request failed")

    scenarios = []
    for concurrency in config["benchmark"]["concurrencies"]:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            observations = list(
                pool.map(
                    lambda prompt: _stream_completion(prompt=prompt, **request_kwargs),
                    prompts,
                )
            )
        wall = time.perf_counter() - started
        scenarios.append(
            {
                "concurrency": concurrency,
                "metrics": scenario_metrics(observations, wall),
                "observations": observations,
            }
        )

    m2_config = load_m2_config(config["correctness"]["formal_evaluator_config"])
    correctness = _correctness(scenarios[0]["observations"], examples, m2_config)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "served_model_name": served_name,
        "adapter_experiment_id": config["model"]["adapter_experiment_id"],
        "adapter_manifest_sha256": config["model"]["adapter_manifest_sha256"],
        "benchmark_selection_sha256": selection_sha,
        "server_startup_seconds": args.server_startup_seconds,
        "ready_gpu_memory_mib": args.ready_gpu_memory_mib,
        "server_models": model_ids,
        "warmup_requests": config["benchmark"]["warmup_requests"],
        "scenarios": scenarios,
        "correctness": correctness,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    atomic_write_json(output, payload)
    validate_raw_result(payload, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
