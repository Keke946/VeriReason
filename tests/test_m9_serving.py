import copy
from pathlib import Path

import pytest

from verireason.m3_contract import sha256_json
from verireason.m9_serving import (
    build_public_summary,
    load_m9_config,
    percentile,
    scenario_metrics,
    select_benchmark_examples,
    validate_raw_result,
)


def _config():
    return load_m9_config("configs/m9_serving.toml")


def _observations(count=64):
    return [
        {
            "success": True,
            "ttft_seconds": 0.1 + index / 1000,
            "e2e_seconds": 0.5 + index / 100,
            "completion_tokens": 10 + index % 3,
            "completion": "SELECT 1",
            "error": None,
        }
        for index in range(count)
    ]


def _raw(config):
    scenarios = []
    for concurrency in (1, 4, 8):
        observations = _observations()
        scenarios.append(
            {
                "concurrency": concurrency,
                "metrics": scenario_metrics(observations, 8.0 / concurrency),
                "observations": observations,
            }
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "served_model_name": config["model"]["served_model_name"],
        "adapter_manifest_sha256": config["model"]["adapter_manifest_sha256"],
        "benchmark_selection_sha256": config["data"]["benchmark_selection_sha256"],
        "server_startup_seconds": 42.0,
        "server_models": [config["model"]["served_model_name"]],
        "warmup_requests": 2,
        "scenarios": scenarios,
        "correctness": {
            "example_count": 64,
            "strict_execution_accuracy": 0.5,
            "bird_ex_behavior_compatible": 0.55,
            "strict_sql_only_rate": 1.0,
            "valid_sql_rate": 0.9,
            "failure_categories": {"strict_correct": 32, "wrong_result": 32},
            "examples": [{"completion": "SELECT 1"} for _ in range(64)],
        },
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }


def test_load_m9_config_binds_best_e2_and_static_lora():
    config = _config()
    assert config["model"]["served_model_name"] == "verireason-e2"
    assert config["server"]["dynamic_lora_loading"] is False
    assert config["benchmark"]["concurrencies"] == [1, 4, 8]


def test_cloud_runner_explicitly_disables_prefix_caching():
    runner = Path("scripts/run_m9_cloud.sh").read_text()
    assert "--no-enable-prefix-caching" in runner
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in runner


def test_deterministic_selection_and_heldout_gate():
    config = _config()
    sample_ids = [str(index) for index in range(1209)]
    examples = [
        {"sample_id": sample_id, "prompt_sha256": f"prompt-{sample_id}"}
        for sample_id in sample_ids
    ]
    bundle = {
        "split": "full-validation",
        "selection_sha256": config["data"]["source_selection_sha256"],
        "sample_ids": sample_ids,
        "prompt_hashes": [row["prompt_sha256"] for row in examples],
        "examples": examples,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: __import__("hashlib").sha256(
            f"{config['data']['selector_seed']}\0{sample_id}".encode()
        ).hexdigest(),
    )[:64]
    selection = {
        "selector_version": config["data"]["selector_version"],
        "selector_seed": config["data"]["selector_seed"],
        "count": 64,
        "source_selection_sha256": config["data"]["source_selection_sha256"],
        "sample_ids": ranked,
    }
    synthetic = copy.deepcopy(config)
    synthetic["data"]["benchmark_selection_sha256"] = sha256_json(selection)
    selected, selection_sha = select_benchmark_examples(bundle, synthetic)
    assert [row["sample_id"] for row in selected] == ranked
    assert selection_sha == sha256_json(selection)

    bundle["heldout_examples_loaded"] = 1
    with pytest.raises(ValueError, match="full1209"):
        select_benchmark_examples(bundle, synthetic)


def test_percentile_and_scenario_metrics_are_measured_from_requests():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    metrics = scenario_metrics(_observations(4), 2.0)
    assert metrics["request_success_rate"] == 1.0
    assert metrics["request_throughput_per_second"] == 2.0
    assert metrics["completion_tokens"] == 43


def test_raw_result_rejects_missing_request_and_nonzero_heldout():
    config = _config()
    result = _raw(config)
    validate_raw_result(result, config)
    result["scenarios"][1]["observations"].pop()
    with pytest.raises(ValueError, match="request-count"):
        validate_raw_result(result, config)

    result = _raw(config)
    result["heldout_examples_evaluated"] = 1
    with pytest.raises(ValueError, match="identity or heldout"):
        validate_raw_result(result, config)


def test_public_summary_strips_request_rows_and_requires_memory():
    config = _config()
    result = _raw(config)
    summary = build_public_summary(
        result,
        config,
        raw_result_sha256="a" * 64,
        ready_gpu_memory_mib=9000.0,
        peak_gpu_memory_mib=12000.0,
    )
    assert summary["peak_gpu_memory_mib"] == 12000.0
    assert "observations" not in summary["scenarios"][0]
    assert "examples" not in summary["correctness"]
    assert summary["heldout_examples_evaluated"] == 0

    with pytest.raises(ValueError, match="memory"):
        build_public_summary(
            result,
            config,
            raw_result_sha256="a" * 64,
            ready_gpu_memory_mib=12000.0,
            peak_gpu_memory_mib=9000.0,
        )
