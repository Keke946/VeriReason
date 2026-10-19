"""Contracts, selection, and SQL-free reporting for the M9 serving benchmark."""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sha256_file
from verireason.m3_contract import sha256_json


def load_m9_config(path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "sources",
        "model",
        "data",
        "server",
        "decoding",
        "benchmark",
        "correctness",
        "evidence",
        "stops",
    }
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M9 config must be schema v1 with every contract section")
    if config.get("status") != "local_preregistered_pending_control_approval":
        raise ValueError("M9 serving must remain unapproved during local preparation")
    if config.get("experiment_id") != "m9_e2_vllm_serving_001":
        raise ValueError("unexpected M9 experiment identity")

    model, data, server = config["model"], config["data"], config["server"]
    if (
        model.get("model_id") != "Qwen/Qwen3-4B-Base"
        or model.get("revision") != "906bfd4b4dc7f14ee4320094d8b41684abff8539"
        or model.get("adapter_experiment_id") != "e2_m7_outcome_grpo_formal_001"
        or model.get("adapter_manifest_sha256")
        != "3492a3ec870f7c6e8e672459f025fc96a9a8cba543b27b7df4fc5f6c6bfbd3b7"
    ):
        raise ValueError("M9 must serve the accepted best E2 model")
    if model.get("lora_rank") != 16 or server.get("max_lora_rank") != 16:
        raise ValueError("M9 vLLM LoRA allocation must match the rank-16 adapter")
    if server.get("dynamic_lora_loading") or server.get("backend") != "vllm_openai_compatible":
        raise ValueError("M9 must statically load one LoRA into the vLLM API server")
    if not 0 < float(server.get("gpu_memory_utilization", 0)) < 1:
        raise ValueError("M9 GPU memory utilization must be a fraction")
    if server.get("max_num_seqs") != max(config["benchmark"].get("concurrencies", [0])):
        raise ValueError("M9 maximum sequence concurrency drifted from the benchmark")
    if (
        data.get("split") != "validation"
        or data.get("source_example_count") != 1209
        or data.get("benchmark_example_count") != 64
        or data.get("heldout_examples_loaded")
        or data.get("heldout_examples_evaluated")
    ):
        raise ValueError("M9 data scope must be fixed validation hash64 with zero heldout")
    if config["benchmark"].get("concurrencies") != [1, 4, 8]:
        raise ValueError("M9 benchmark must preserve concurrency 1/4/8")
    decoding = config["decoding"]
    if (
        decoding.get("temperature") != 0.0
        or decoding.get("top_p") != 1.0
        or decoding.get("max_tokens") != 384
        or not decoding.get("stream")
        or not decoding.get("include_usage")
        or decoding.get("enable_thinking")
    ):
        raise ValueError("M9 deterministic streaming decoding contract drifted")
    evidence = config["evidence"]
    if evidence.get("maximum_session_wall_seconds") != 3600:
        raise ValueError("M9 one-hour session ceiling drifted")

    repository = Path(root)
    for name in ("m4_config", "m2_config", "e2_training_summary", "e2_evaluation_summary"):
        source = repository / config["sources"][name]
        if not source.is_file() or sha256_file(source) != config["sources"][f"{name}_sha256"]:
            raise ValueError(f"M9 frozen source identity drifted: {name}")
    return config


def select_benchmark_examples(
    bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    data = config["data"]
    examples, sample_ids, prompt_hashes = (
        bundle.get("examples"),
        bundle.get("sample_ids"),
        bundle.get("prompt_hashes"),
    )
    if (
        bundle.get("split") != "full-validation"
        or bundle.get("selection_sha256") != data["source_selection_sha256"]
        or bundle.get("heldout_examples_loaded")
        or bundle.get("heldout_examples_evaluated")
        or not all(
            isinstance(rows, list) and len(rows) == 1209
            for rows in (examples, sample_ids, prompt_hashes)
        )
    ):
        raise ValueError("M9 requires the frozen full1209 prepared validation bundle")
    if [row.get("sample_id") for row in examples] != sample_ids:
        raise ValueError("M9 prepared sample order drifted")
    if [row.get("prompt_sha256") for row in examples] != prompt_hashes:
        raise ValueError("M9 prepared prompt identity drifted")

    seed = data["selector_seed"]
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest(),
    )[: data["benchmark_example_count"]]
    selection = {
        "selector_version": data["selector_version"],
        "selector_seed": seed,
        "count": len(ranked),
        "source_selection_sha256": data["source_selection_sha256"],
        "sample_ids": ranked,
    }
    selection_sha = sha256_json(selection)
    if selection_sha != data["benchmark_selection_sha256"]:
        raise ValueError("M9 deterministic benchmark selection drifted")
    lookup = {row["sample_id"]: dict(row) for row in examples}
    return [lookup[sample_id] for sample_id in ranked], selection_sha


def percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def scenario_metrics(
    observations: Sequence[Mapping[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    if not observations or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("M9 scenario has invalid observations or wall time")
    successes = [row for row in observations if row.get("success")]
    ttft = [float(row["ttft_seconds"]) for row in successes]
    e2e = [float(row["e2e_seconds"]) for row in successes]
    tokens = sum(int(row["completion_tokens"]) for row in successes)
    return {
        "request_count": len(observations),
        "successful_requests": len(successes),
        "request_success_rate": len(successes) / len(observations),
        "wall_seconds": wall_seconds,
        "ttft_p50_seconds": percentile(ttft, 0.50) if ttft else None,
        "ttft_p95_seconds": percentile(ttft, 0.95) if ttft else None,
        "e2e_latency_p50_seconds": percentile(e2e, 0.50) if e2e else None,
        "e2e_latency_p95_seconds": percentile(e2e, 0.95) if e2e else None,
        "completion_tokens": tokens,
        "request_throughput_per_second": len(successes) / wall_seconds,
        "completion_token_throughput_per_second": tokens / wall_seconds,
    }


def validate_raw_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("experiment_id") != config["experiment_id"]
        or result.get("served_model_name") != config["model"]["served_model_name"]
        or result.get("adapter_manifest_sha256")
        != config["model"]["adapter_manifest_sha256"]
        or result.get("benchmark_selection_sha256")
        != config["data"]["benchmark_selection_sha256"]
        or result.get("heldout_examples_loaded")
        or result.get("heldout_examples_evaluated")
    ):
        raise ValueError("M9 raw result identity or heldout contract failed")
    if (
        result.get("served_model_name") not in result.get("server_models", [])
        or result.get("warmup_requests") != config["benchmark"]["warmup_requests"]
        or not math.isfinite(float(result.get("server_startup_seconds", math.nan)))
        or result["server_startup_seconds"] <= 0
    ):
        raise ValueError("M9 endpoint, warmup, or startup evidence failed")
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, list) or [row.get("concurrency") for row in scenarios] != [
        1,
        4,
        8,
    ]:
        raise ValueError("M9 concurrency scenarios are missing or reordered")
    for scenario in scenarios:
        metrics = scenario.get("metrics", {})
        if (
            len(scenario.get("observations", [])) != config["benchmark"]["requests_per_scenario"]
            or metrics.get("request_success_rate")
            < config["correctness"]["minimum_request_success_rate"]
        ):
            raise ValueError("M9 request-count or success-rate gate failed")
        required = (
            "ttft_p50_seconds",
            "ttft_p95_seconds",
            "e2e_latency_p50_seconds",
            "e2e_latency_p95_seconds",
            "request_throughput_per_second",
            "completion_token_throughput_per_second",
        )
        if any(
            not math.isfinite(float(metrics.get(key, math.nan))) or metrics[key] <= 0
            for key in required
        ):
            raise ValueError("M9 performance metric is missing, nonfinite, or nonpositive")
    correctness = result.get("correctness", {})
    if (
        correctness.get("example_count") != config["data"]["benchmark_example_count"]
        or len(correctness.get("examples", [])) != config["data"]["benchmark_example_count"]
        or correctness.get("strict_sql_only_rate", 0)
        < config["correctness"]["minimum_strict_sql_only_rate"]
    ):
        raise ValueError("M9 serving correctness gate failed")
    for key in ("strict_execution_accuracy", "bird_ex_behavior_compatible", "valid_sql_rate"):
        value = float(correctness.get(key, math.nan))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"M9 correctness metric is invalid: {key}")


def build_public_summary(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    raw_result_sha256: str,
    ready_gpu_memory_mib: float,
    peak_gpu_memory_mib: float,
) -> dict[str, Any]:
    validate_raw_result(result, config)
    if not all(
        math.isfinite(value) and value > 0
        for value in (ready_gpu_memory_mib, peak_gpu_memory_mib)
    ) or peak_gpu_memory_mib < ready_gpu_memory_mib:
        raise ValueError("M9 GPU memory evidence is invalid")
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "backend": config["server"]["backend"],
        "served_model_name": config["model"]["served_model_name"],
        "adapter_experiment_id": config["model"]["adapter_experiment_id"],
        "adapter_manifest_sha256": config["model"]["adapter_manifest_sha256"],
        "benchmark_selection_sha256": config["data"]["benchmark_selection_sha256"],
        "benchmark_example_count": config["data"]["benchmark_example_count"],
        "server_startup_seconds": result["server_startup_seconds"],
        "ready_gpu_memory_mib": ready_gpu_memory_mib,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "scenarios": [
            {"concurrency": row["concurrency"], **row["metrics"]}
            for row in result["scenarios"]
        ],
        "correctness": {
            key: result["correctness"][key]
            for key in (
                "example_count",
                "bird_ex_behavior_compatible",
                "strict_execution_accuracy",
                "strict_sql_only_rate",
                "valid_sql_rate",
                "failure_categories",
            )
        },
        "raw_result_sha256": raw_result_sha256,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
