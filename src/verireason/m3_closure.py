"""Independent local acceptance audit for returned M3 vertical-slice evidence."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sql_complexity
from verireason.m3_contract import (
    IGNORE_INDEX,
    load_m3_config,
    sha256_json,
    validate_answer_only_token_contract,
    validate_completed_result,
    verify_update_evidence,
)

MODEL_ID = "Qwen/Qwen3-4B-Base"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
M2_MANIFEST_SHA256 = "d90ffab32e296248376c542b736aef5ce7f547acc702a9eab41073789cf5b72a"
EXPERIMENT_ID = "m3_vertical_slice_001"
ATTEMPT_ID = "m3_vertical_slice_001_attempt_003"

REQUIRED_ARTIFACTS = (
    "adapter-manifest.json",
    "adapter/README.md",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter/chat_template.jinja",
    "adapter/tokenizer.json",
    "adapter/tokenizer_config.json",
    "base-train.json",
    "environment.json",
    "gpu-pip-freeze.txt",
    "launch.log",
    "m3-result.json",
    "nvidia-smi.txt",
    "preflight.json",
    "session-completion.json",
    "session-progress.json",
    "slice.json",
    "stage-base-and-train.log",
    "stage-environment-evidence.log",
    "stage-freeze-slice.log",
    "stage-fresh-reload-eval.log",
    "stage-gpu-environment.log",
    "stage-m3-verify.log",
    "stage-nvidia-smi.log",
    "stage-pip-freeze.log",
    "stage-token-lengths.log",
    "stage-validate-result.log",
    "token-length-report.json",
    "tokenizer-contract-preflight.json",
    "training-progress.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _claimed_json_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    claimed = body.pop(field, None)
    _require(isinstance(claimed, str) and claimed == sha256_json(body), f"invalid {field}")
    return claimed


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    _require(root.is_dir(), f"evidence directory is missing: {root}")
    _require(stat.S_IMODE(root.stat().st_mode) == 0o700, "evidence directory must be mode 0700")
    adapter = root / "adapter"
    _require(adapter.is_dir(), "adapter directory is missing")
    _require(stat.S_IMODE(adapter.stat().st_mode) == 0o700, "adapter directory must be mode 0700")

    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing = sorted(set(REQUIRED_ARTIFACTS) - observed)
    _require(not missing, f"required M3 artifacts are missing: {missing}")

    inventory: list[dict[str, Any]] = []
    for relative in sorted(observed):
        path = root / relative
        mode = stat.S_IMODE(path.stat().st_mode)
        _require(mode == 0o600, f"returned evidence must be private mode 0600: {relative}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "mode": "0600",
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _adapter_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in (root / "adapter").rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root / "adapter").as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {"files": files, "manifest_sha256": sha256_json(files)}


def _validate_tokenizer_preflight(
    payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == 1 and payload.get("status") == "passed",
        "tokenizer preflight did not pass",
    )
    _require(payload.get("synthetic") is True, "tokenizer preflight prompt must be synthetic")
    _require(payload.get("model_id") == MODEL_ID, "tokenizer preflight model changed")
    _require(payload.get("revision") == MODEL_REVISION, "tokenizer preflight revision changed")
    _require(payload.get("raw_type") == "BatchEncoding", "unexpected real tokenizer return type")
    keys = payload.get("raw_mapping_keys")
    _require(isinstance(keys, list) and "input_ids" in keys, "tokenizer mapping lacks input_ids")
    _require(
        payload.get("normalized_all_exact_int") is True, "normalized tokens are not exact integers"
    )
    normalized = payload.get("normalized_token_count")
    prompt = payload.get("prompt_tokens")
    answer = payload.get("answer_tokens")
    supervised = payload.get("supervised_tokens")
    _require(
        type(normalized) is int and normalized == prompt and prompt > 2,
        "invalid prompt token count",
    )
    _require(
        type(answer) is int and answer > 0 and answer == supervised,
        "invalid answer-only masking count",
    )
    _require(config["prompt"]["append_eos"] is True, "EOS supervision is not frozen")
    return {
        "status": "passed",
        "synthetic_prompt": True,
        "model_id": payload["model_id"],
        "revision": payload["revision"],
        "raw_type": payload["raw_type"],
        "raw_mapping_keys": keys,
        "raw_input_ids_shape": payload.get("raw_input_ids_shape"),
        "normalized_token_count": normalized,
        "normalized_all_exact_int": True,
        "prompt_tokens": prompt,
        "answer_tokens": answer,
        "supervised_tokens": supervised,
    }


def _validate_length_report(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    report_sha = _claimed_json_hash(payload, "report_sha256")
    _require(payload.get("synthetic") is False, "real token-length report is marked synthetic")
    _require(
        payload.get("source_manifest_sha256") == M2_MANIFEST_SHA256, "token report manifest changed"
    )
    _require(payload.get("heldout_examples_loaded") == 0, "token report loaded heldout")
    _require(payload.get("heldout_examples_evaluated") == 0, "token report evaluated heldout")
    _require(
        payload.get("eligible_counts") == {"train": 5364, "validation": 1209},
        "eligible counts changed",
    )
    _require(payload.get("exclusion_counts") == {}, "unexpected token/schema exclusions")
    _require(
        payload.get("frozen_slice_counts") == {"train": 8, "validation": 8}, "frozen counts changed"
    )
    for split in ("train", "validation"):
        ids = payload.get("token_eligible_sample_ids", {}).get(split)
        expected = payload["eligible_counts"][split]
        _require(
            isinstance(ids, list) and len(ids) == expected,
            f"{split} token-eligible IDs are incomplete",
        )
        _require(len(ids) == len(set(ids)), f"{split} token-eligible IDs contain duplicates")
        for name in ("prompt", "answer", "total"):
            summary = payload.get("token_lengths", {}).get(split, {}).get(name)
            _require(isinstance(summary, Mapping), f"missing {split} {name} token summary")
            _require(summary.get("count") == expected, f"wrong {split} {name} token count")
            _require(summary.get("minimum", 0) > 0, f"invalid {split} {name} token minimum")
    _require(
        config["data"]["training_manifest_sha256"] == M2_MANIFEST_SHA256,
        "config manifest identity changed",
    )
    return report_sha


def _validate_slice(
    payload: Mapping[str, Any],
    length_report: Mapping[str, Any],
    report_sha: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_sha = _claimed_json_hash(payload, "slice_bundle_sha256")
    _require(
        payload.get("status") == "frozen" and payload.get("synthetic") is False,
        "real slice is not frozen",
    )
    _require(payload.get("source_manifest_sha256") == M2_MANIFEST_SHA256, "slice manifest changed")
    _require(
        payload.get("token_length_report_sha256") == report_sha,
        "slice is not bound to token report",
    )
    _require(payload.get("heldout_examples_loaded") == 0, "slice loaded heldout")
    _require(payload.get("heldout_examples_evaluated") == 0, "slice evaluated heldout")
    _require(payload.get("model") == config["model"], "slice model identity changed")
    _require(payload.get("prompt_contract") == config["prompt"], "slice prompt contract changed")

    summaries: dict[str, Any] = {}
    database_sets: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        selected = payload.get("slices", {}).get(split)
        _require(isinstance(selected, Mapping), f"missing {split} slice")
        examples = selected.get("examples")
        _require(
            isinstance(examples, list) and len(examples) == 8, f"{split} slice is not exactly 8"
        )
        sample_ids = selected.get("sample_ids")
        _require(sample_ids == [row.get("sample_id") for row in examples], f"{split} IDs drifted")
        _require(len(set(sample_ids)) == 8, f"{split} IDs contain duplicates")
        eligible_ids = set(length_report["token_eligible_sample_ids"][split])
        _require(
            all(sample_id in eligible_ids for sample_id in sample_ids),
            f"{split} contains token-ineligible IDs",
        )
        prompt_hashes = [row.get("prompt_sha256") for row in examples]
        _require(selected.get("prompt_hashes") == prompt_hashes, f"{split} prompt hashes drifted")
        selector_payload = {
            "selector_version": selected.get("selector_version"),
            "seed": selected.get("seed"),
            "split": selected.get("split"),
            "source_manifest_sha256": selected.get("source_manifest_sha256"),
            "sample_ids": sample_ids,
        }
        _require(
            selected.get("slice_sha256") == sha256_json(selector_payload),
            f"{split} selector hash changed",
        )
        _require(
            selected.get("selector_version") == config["data"]["selector_version"],
            "selector version changed",
        )
        _require(selected.get("seed") == config["data"]["selector_seed"], "selector seed changed")
        _require(selected.get("selected_count") == 8, f"{split} selected_count changed")

        feature_counts = dict(
            sorted(
                Counter(
                    feature
                    for row in examples
                    for feature in sql_complexity(str(row["target_sql"])).features
                ).items()
            )
        )
        _require(
            selected.get("feature_counts") == feature_counts, f"{split} feature counts drifted"
        )
        for feature in config["data"]["required_features"]:
            _require(
                feature_counts.get(feature, 0) > 0, f"{split} lacks required {feature} coverage"
            )

        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        total_lengths: list[int] = []
        for row in examples:
            input_ids = row.get("input_ids")
            labels = row.get("labels")
            prompt_tokens = row.get("prompt_tokens")
            answer_tokens = row.get("answer_tokens")
            supervised_tokens = row.get("supervised_tokens")
            validate_answer_only_token_contract(
                input_ids,
                labels,
                prompt_tokens,
                answer_tokens=answer_tokens,
                supervised_tokens=supervised_tokens,
            )
            _require(
                all(type(token) is int for token in input_ids),
                "slice input_ids are not exact integers",
            )
            _require(
                all(type(token) is int for token in labels), "slice labels are not exact integers"
            )
            _require(
                labels[:prompt_tokens] == [IGNORE_INDEX] * prompt_tokens,
                "prompt labels are not masked",
            )
            _require(
                labels[prompt_tokens:] == input_ids[prompt_tokens:],
                "answer labels differ from answer tokens",
            )
            _require(row.get("split") == split, f"{split} example has wrong split")
            _require(
                row.get("prompt_sha256") == sha256_json(row.get("messages")), "prompt hash changed"
            )
            prompt_lengths.append(prompt_tokens)
            answer_lengths.append(answer_tokens)
            total_lengths.append(len(input_ids))
        expected_lengths = {
            "prompt": prompt_lengths,
            "answer": answer_lengths,
            "total": total_lengths,
        }
        _require(
            selected.get("token_lengths") == expected_lengths, f"{split} token lengths drifted"
        )
        database_ids = set(selected.get("database_ids", []))
        _require(
            database_ids == {row.get("db_id") for row in examples}, f"{split} database IDs drifted"
        )
        database_sets[split] = database_ids
        summaries[split] = {
            "count": 8,
            "sample_ids": sample_ids,
            "slice_sha256": selected["slice_sha256"],
            "feature_counts": feature_counts,
            "token_lengths": expected_lengths,
            "prompt_hashes": prompt_hashes,
        }
    _require(
        not database_sets["train"] & database_sets["validation"],
        "train/validation database boundary changed",
    )
    return {"slice_bundle_sha256": bundle_sha, "splits": summaries}


def recompute_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = evaluation.get("examples")
    _require(isinstance(rows, list) and len(rows) == 8, "evaluation must contain exactly 8 rows")
    _require(all(row.get("error") is None for row in rows), "evaluator_error is present")
    _require(
        all(row.get("failure_category") != "evaluator_error" for row in rows),
        "evaluator_error category is present",
    )
    _require(
        all(row.get("prediction_status") != "gold_preflight_failure" for row in rows),
        "gold preflight failed",
    )
    count = len(rows)
    recomputed = {
        "example_count": count,
        "bird_ex_behavior_compatible": sum(row["bird_ex_behavior_compatible"] for row in rows)
        / count,
        "strict_execution_accuracy": sum(row["strict_execution_accuracy"] for row in rows) / count,
        "strict_sql_only_rate": sum(bool(row["extraction"]["strict_sql_only"]) for row in rows)
        / count,
        "failure_categories": dict(
            sorted(Counter(row["failure_category"] for row in rows).items())
        ),
    }
    reported = evaluation.get("metrics")
    _require(isinstance(reported, Mapping), "evaluation metrics are missing")
    for key, value in recomputed.items():
        _require(reported.get(key) == value, f"reported {key} differs from row recomputation")
    generated = reported.get("generated_tokens")
    seconds = reported.get("generation_seconds")
    throughput = reported.get("completion_tokens_per_second")
    _require(type(generated) is int and generated >= 0, "invalid generated-token count")
    _require(isinstance(seconds, int | float) and seconds > 0, "invalid generation duration")
    _require(
        isinstance(throughput, int | float) and math.isfinite(throughput), "invalid throughput"
    )
    _require(
        math.isclose(throughput, generated / seconds, rel_tol=1e-12),
        "throughput arithmetic changed",
    )
    return {
        **recomputed,
        "generated_tokens": generated,
        "generation_plus_evaluator_seconds": seconds,
        "completion_tokens_per_second": throughput,
        "throughput_scope": "generation_plus_evaluator_end_to_end",
    }


def _validate_training(
    result: Mapping[str, Any], base_train: Mapping[str, Any], progress: Mapping[str, Any]
) -> dict[str, Any]:
    training = result.get("training")
    _require(
        isinstance(training, Mapping) and training == base_train.get("training"),
        "training evidence drifted",
    )
    verify_update_evidence(training)
    for name in (
        "initial_loss",
        "final_observed_loss",
        "gradient_norm",
        "adapter_delta_l2_norm",
        "training_seconds",
    ):
        _require(
            math.isfinite(float(training.get(name, math.nan))), f"non-finite training field: {name}"
        )
    _require(training.get("micro_steps") == 8, "training did not run 8 microsteps")
    _require(training.get("optimizer_steps") == 2, "training did not run 2 optimizer updates")
    _require(training.get("gradient_norm") > 0, "gradient norm is not positive")
    _require(training.get("adapter_delta_l2_norm") > 0, "adapter delta is not positive")
    _require(
        progress.get("micro_steps") == 8 and progress.get("optimizer_steps") == 2,
        "progress counters drifted",
    )
    _require(
        progress.get("latest_loss") == training.get("final_observed_loss"), "progress loss drifted"
    )
    _require(
        progress.get("gradient_norm") == training.get("gradient_norm"), "progress gradient drifted"
    )
    _require(
        progress.get("supervised_token_count") == training.get("supervised_token_count"),
        "progress supervised tokens drifted",
    )
    return dict(training)


def _validate_comparison(
    result: Mapping[str, Any], base_train: Mapping[str, Any], slice_summary: Mapping[str, Any]
) -> dict[str, Any]:
    contract = result.get("comparison_contract")
    _require(
        isinstance(contract, Mapping) and contract == base_train.get("comparison_contract"),
        "comparison contract drifted",
    )
    body = dict(contract)
    claimed = body.pop("comparison_contract_sha256", None)
    _require(claimed == sha256_json(body), "comparison contract hash is invalid")
    validation = slice_summary["splits"]["validation"]
    _require(
        contract.get("validation_slice_sha256") == validation["slice_sha256"],
        "validation slice hash drifted",
    )
    _require(
        contract.get("prompt_hashes") == validation["prompt_hashes"],
        "comparison prompt hashes drifted",
    )
    base_rows = base_train.get("base_evaluation", {}).get("examples", [])
    final_base_rows = result.get("base_evaluation", {}).get("examples", [])
    sft_rows = result.get("sft_evaluation", {}).get("examples", [])
    _require(base_rows == final_base_rows, "Base rows changed after training artifact publication")
    expected_ids = validation["sample_ids"]
    expected_prompts = validation["prompt_hashes"]
    for label, rows in (("Base", final_base_rows), ("SFT", sft_rows)):
        _require(
            [row.get("sample_id") for row in rows] == expected_ids, f"{label} IDs/order drifted"
        )
        _require(
            [row.get("prompt_sha256") for row in rows] == expected_prompts,
            f"{label} prompts drifted",
        )
    return dict(contract)


def _validate_adapter(
    root: Path, result: Mapping[str, Any], base_train: Mapping[str, Any]
) -> dict[str, Any]:
    observed = _adapter_manifest(root)
    published = _load_json(root / "adapter-manifest.json")
    _require(observed == published, "saved adapter manifest does not match adapter files")
    _require(observed == base_train.get("adapter"), "base/train adapter manifest drifted")
    reload = result.get("fresh_reload")
    _require(
        isinstance(reload, Mapping) and observed == reload.get("adapter"),
        "fresh reload adapter manifest drifted",
    )
    prior = reload.get("prior_process_id")
    fresh = reload.get("reload_process_id")
    _require(prior == base_train.get("process_id"), "recorded prior PID drifted")
    _require(
        type(prior) is int and type(fresh) is int and prior != fresh,
        "adapter was not loaded in a fresh process",
    )
    _require(reload.get("different_process") is True, "fresh-process flag is false")
    return {
        "manifest_sha256": observed["manifest_sha256"],
        "adapter_model_bytes": next(
            row["bytes"] for row in observed["files"] if row["path"] == "adapter_model.safetensors"
        ),
        "adapter_model_sha256": next(
            row["sha256"] for row in observed["files"] if row["path"] == "adapter_model.safetensors"
        ),
        "prior_process_id": prior,
        "reload_process_id": fresh,
        "different_process": True,
    }


def _validate_session(
    session: Mapping[str, Any], preflight: Mapping[str, Any], execution_commit: str
) -> dict[str, Any]:
    _require(session.get("status") == "completed", "session did not complete")
    _require(session.get("approved_commit") == execution_commit, "session execution commit changed")
    _require(
        preflight.get("status") == "passed" and preflight.get("git_dirty") is False,
        "preflight identity failed",
    )
    _require(
        preflight.get("approved_commit") == execution_commit, "preflight approved commit changed"
    )
    _require(
        preflight.get("observed_commit") == execution_commit, "preflight observed commit changed"
    )
    elapsed = session.get("elapsed_seconds")
    maximum = session.get("maximum_session_wall_seconds")
    _require(isinstance(elapsed, int | float) and elapsed > 0, "invalid full session duration")
    _require(maximum == 14_400 and elapsed < maximum, "session wall contract failed")
    _require(
        preflight.get("fixed_train_count") == 8 and preflight.get("fixed_validation_count") == 8,
        "preflight counts changed",
    )
    _require(
        preflight.get("output_filesystem_free_bytes", 0)
        >= preflight.get("minimum_output_free_bytes", 1),
        "preflight disk gate failed",
    )
    return {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "maximum_session_wall_seconds": maximum,
        "started_utc": preflight.get("started_utc"),
    }


def _money_summary(
    charges: Sequence[tuple[str, str]],
    previous_cumulative_cost_cny: str,
    provider_balance_cny: str,
) -> dict[str, Any]:
    _require(len(charges) == 2, "M3 closure requires the two provider invoice transactions")
    transactions = []
    total = Decimal("0")
    for timestamp, amount_text in charges:
        amount = Decimal(amount_text)
        _require(amount > 0, "provider charge must be positive")
        total += amount
        transactions.append(
            {
                "timestamp": timestamp,
                "description": "container_usage_billing",
                "amount_cny": format(amount, "f"),
            }
        )
    previous = Decimal(previous_cumulative_cost_cny)
    cumulative = previous + total
    _require(total == Decimal("3.88284165"), "provider invoice total differs from accepted M3 bill")
    _require(previous == Decimal("9.64484428"), "pre-M3 cumulative cost changed")
    _require(cumulative == Decimal("13.52768593"), "post-M3 cumulative cost arithmetic changed")
    return {
        "source": "user_provided_provider_screenshot",
        "billing_kind": "provider_actual_container_usage",
        "transactions": transactions,
        "actual_provider_cost_cny": format(total, "f"),
        "pre_m3_cumulative_cost_cny": format(previous, "f"),
        "post_m3_cumulative_cost_cny": format(cumulative, "f"),
        "provider_balance_after_transactions_cny": format(Decimal(provider_balance_cny), "f"),
        "program_wall_estimate_is_actual_cost": False,
    }


def build_closure_summary(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    m2_config_path: str | Path,
    execution_commit: str,
    pin_commit: str,
    charges: Sequence[tuple[str, str]],
    previous_cumulative_cost_cny: str,
    provider_balance_cny: str,
) -> dict[str, Any]:
    root = Path(evidence_dir)
    config = load_m3_config(config_path, m2_config_path)
    inventory = _artifact_inventory(root)
    tokenizer = _validate_tokenizer_preflight(
        _load_json(root / "tokenizer-contract-preflight.json"), config
    )
    length_report = _load_json(root / "token-length-report.json")
    report_sha = _validate_length_report(length_report, config)
    slice_payload = _load_json(root / "slice.json")
    slice_summary = _validate_slice(slice_payload, length_report, report_sha, config)
    base_train = _load_json(root / "base-train.json")
    progress = _load_json(root / "training-progress.json")
    result = _load_json(root / "m3-result.json")
    session = _load_json(root / "session-completion.json")
    preflight = _load_json(root / "preflight.json")

    validate_completed_result(result)
    for payload in (length_report, slice_payload, base_train, result):
        _require(payload.get("heldout_examples_loaded") == 0, "heldout was loaded")
        _require(payload.get("heldout_examples_evaluated") == 0, "heldout was evaluated")
    _require(result.get("experiment_id") == EXPERIMENT_ID, "experiment identity changed")
    _require(
        result.get("source") == {"commit": execution_commit, "dirty": False},
        "result source identity changed",
    )
    _require(base_train.get("source") == result.get("source"), "base/train source identity changed")
    _require(
        slice_summary["slice_bundle_sha256"] == base_train.get("slice_bundle_sha256"),
        "base/train slice bundle drifted",
    )
    _require(
        slice_summary["slice_bundle_sha256"] == result.get("slice_bundle_sha256"),
        "final slice bundle drifted",
    )

    comparison = _validate_comparison(result, base_train, slice_summary)
    training = _validate_training(result, base_train, progress)
    adapter = _validate_adapter(root, result, base_train)
    session_summary = _validate_session(session, preflight, execution_commit)
    _require(
        session_summary["elapsed_seconds"] >= result["timing"]["measured_total_wall_seconds"],
        "full session is shorter than measured phases",
    )
    base_metrics = recompute_evaluation(result["base_evaluation"])
    sft_metrics = recompute_evaluation(result["sft_evaluation"])
    money = _money_summary(charges, previous_cumulative_cost_cny, provider_balance_cny)

    hashes = {row["path"]: row["sha256"] for row in inventory}
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": ATTEMPT_ID,
        "source": {
            "execution_commit": execution_commit,
            "pin_commit": pin_commit,
            "commits_are_distinct": execution_commit != pin_commit,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "m2_training_manifest_sha256": M2_MANIFEST_SHA256,
        },
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "tokenizer_preflight": tokenizer,
        "slice": slice_summary,
        "comparison_contract": comparison,
        "training": training,
        "adapter_reload": adapter,
        "base_evaluation": base_metrics,
        "sft_evaluation": sft_metrics,
        "memory": result["memory"],
        "timing": {
            **result["timing"],
            "complete_session_elapsed_seconds": session_summary["elapsed_seconds"],
            "complete_session_source": "session-completion.json",
        },
        "cost": money,
        "evidence": {
            "local_directory": "artifacts/m3-cloud-attempt-003",
            "directory_mode": "0700",
            "file_mode": "0600",
            "required_artifact_count": len(REQUIRED_ARTIFACTS),
            "observed_artifact_count": len(inventory),
            "artifact_inventory_sha256": sha256_json(inventory),
            "artifact_inventory": inventory,
            "key_file_sha256": {
                name: hashes[name]
                for name in (
                    "tokenizer-contract-preflight.json",
                    "token-length-report.json",
                    "slice.json",
                    "base-train.json",
                    "training-progress.json",
                    "adapter-manifest.json",
                    "m3-result.json",
                    "session-completion.json",
                )
            },
        },
        "acceptance": {
            "independent_row_metric_recomputation": "passed",
            "tokenizer_integer_and_masking_contract": "passed",
            "slice_identity_and_coverage": "passed",
            "same_exam_contract": "passed",
            "optimizer_update_and_adapter_delta": "passed",
            "fresh_process_adapter_reload": "passed",
            "evaluator_and_gold_preflight_errors_absent": "passed",
            "heldout_zero": "passed",
            "artifact_hash_and_permission_audit": "passed",
            "actual_provider_invoice_recorded": "passed",
        },
    }
