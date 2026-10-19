"""Contracts and SQL-free reporting for M8 common evaluations."""

from __future__ import annotations

import math
import tomllib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verireason.m3_contract import sha256_file

_EVALUATION_IDENTITIES = {
    "e2_m8_common_validation_001": {
        "condition": "E2",
        "adapter_experiment_id": "e2_m7_outcome_grpo_formal_001",
        "source_names": ("m4_public_summary", "m7_public_summary", "m4_config", "m2_config"),
    },
    "e3_m8_common_validation_001": {
        "condition": "E3",
        "adapter_experiment_id": "e3_m8_dense_grpo_formal_001",
        "source_names": (
            "m4_public_summary",
            "m8_common_summary",
            "m8_e3_formal_summary",
            "m4_config",
            "m2_config",
        ),
    },
    "e4_m8_common_validation_001": {
        "condition": "E4",
        "adapter_experiment_id": "e4_m8_gdpo_formal_001",
        "source_names": (
            "m4_public_summary",
            "m8_common_summary",
            "m8_e3_common_summary",
            "m8_e4_formal_summary",
            "m4_config",
            "m2_config",
        ),
    },
}


def load_m8_config(path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"sources", "model", "data", "comparison", "execution", "stops"}
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M8 config must be schema v1 with every contract section")
    if config.get("status") != "local_preregistered_pending_control_approval":
        raise ValueError("M8 common evaluation must remain unapproved during local preparation")
    identity = _EVALUATION_IDENTITIES.get(config.get("experiment_id"))
    if identity is None:
        raise ValueError("unexpected M8 experiment identity")
    model, data, comparison, execution = (
        config["model"],
        config["data"],
        config["comparison"],
        config["execution"],
    )
    if (
        model.get("condition") != identity["condition"]
        or model.get("adapter_experiment_id") != identity["adapter_experiment_id"]
    ):
        raise ValueError("M8 adapter identity does not match the registered evaluation")
    if data.get("split") != "validation" or data.get("example_count") != 1209:
        raise ValueError("M8 must reuse the frozen full1209 validation exam")
    if data.get("heldout_examples_loaded") or data.get("heldout_examples_evaluated"):
        raise ValueError("M8 common evaluation cannot access heldout")
    if comparison.get("primary_metric") != "strict_execution_accuracy":
        raise ValueError("M8 primary metric drifted")
    if not comparison.get("e1_result_reused") or comparison.get("e1_regeneration_allowed"):
        raise ValueError("M8 must reuse, not regenerate, the accepted E1 result")
    if model["condition"] in {"E3", "E4"} and (
        not comparison.get("e2_result_reused") or comparison.get("e2_regeneration_allowed")
    ):
        raise ValueError(
            f"M8 {model['condition']} must reuse, not regenerate, the accepted E2 result"
        )
    if model["condition"] == "E4" and (
        not comparison.get("e3_result_reused") or comparison.get("e3_regeneration_allowed")
    ):
        raise ValueError("M8 E4 must reuse, not regenerate, the accepted E3 result")
    if execution.get("maximum_session_wall_seconds") != 14400:
        raise ValueError("M8 common-evaluation wall ceiling changed")
    if execution.get("maximum_session_cost_cny") != 10.36:
        raise ValueError("M8 common-evaluation cost ceiling changed")

    repository = Path(root)
    for name in identity["source_names"]:
        source = repository / config["sources"][name]
        if not source.is_file() or sha256_file(source) != config["sources"][f"{name}_sha256"]:
            raise ValueError(f"M8 frozen source identity drifted: {name}")
    with (repository / config["sources"]["m4_config"]).open("rb") as handle:
        frozen_m4 = tomllib.load(handle)
    if (
        frozen_m4.get("model", {}).get("model_id") != model["model_id"]
        or frozen_m4.get("model", {}).get("revision") != model["revision"]
    ):
        raise ValueError("M8 model identity differs from the frozen M4 protocol")
    config["_frozen_m4"] = frozen_m4
    config["_identity"] = identity
    return config


def validate_prepared_bundle(bundle: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    data = config["data"]
    count = data["example_count"]
    examples = bundle.get("examples")
    sample_ids = bundle.get("sample_ids")
    prompt_hashes = bundle.get("prompt_hashes")
    if bundle.get("split") != "full-validation":
        raise ValueError("M8 requires the M4 full-validation prepared bundle")
    if bundle.get("selection_sha256") != data["selection_sha256"]:
        raise ValueError("M8 prepared selection drifted")
    if not all(
        isinstance(rows, list) and len(rows) == count
        for rows in (examples, sample_ids, prompt_hashes)
    ):
        raise ValueError("M8 prepared bundle must contain exactly 1209 aligned rows")
    if len(set(sample_ids)) != count:
        raise ValueError("M8 prepared bundle contains duplicate sample IDs")
    if [row.get("sample_id") for row in examples] != sample_ids:
        raise ValueError("M8 prepared examples and sample IDs are misaligned")
    if [row.get("prompt_sha256") for row in examples] != prompt_hashes:
        raise ValueError("M8 prepared examples and prompt hashes are misaligned")
    if bundle.get("heldout_examples_loaded") or bundle.get("heldout_examples_evaluated"):
        raise ValueError("heldout entered the M8 prepared bundle")


def validate_candidate_evaluation(
    evaluation: Mapping[str, Any], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    validate_prepared_bundle(bundle, config)
    condition = config["model"]["condition"]
    if evaluation.get("condition") != condition:
        raise ValueError(f"M8 evaluation is not labeled {condition}")
    if (
        evaluation.get("adapter_experiment_id") != config["model"]["adapter_experiment_id"]
        or evaluation.get("adapter_manifest_sha256") != config["model"]["adapter_manifest_sha256"]
    ):
        raise ValueError(f"M8 {condition} evaluation adapter identity drifted")
    for key in ("validation_selection_sha256", "sample_ids", "prompt_hashes"):
        expected_key = "selection_sha256" if key == "validation_selection_sha256" else key
        if evaluation.get(key) != bundle.get(expected_key):
            raise ValueError(f"M8 {condition} evaluation drifted: {key}")
    if len(evaluation.get("examples", [])) != config["data"]["example_count"]:
        raise ValueError(f"M8 {condition} evaluation row count drifted")
    if [row.get("sample_id") for row in evaluation["examples"]] != bundle["sample_ids"]:
        raise ValueError(f"M8 {condition} evaluation rows are not in frozen sample order")
    if [row.get("prompt_sha256") for row in evaluation["examples"]] != bundle["prompt_hashes"]:
        raise ValueError(f"M8 {condition} evaluation rows have prompt drift")
    frozen_m4 = config["_frozen_m4"]
    if evaluation.get("decoding") != frozen_m4["decoding"]:
        raise ValueError(f"M8 {condition} decoding protocol drifted")
    if evaluation.get("evaluator") != frozen_m4["evaluator"]:
        raise ValueError(f"M8 {condition} evaluator protocol drifted")
    if evaluation.get("heldout_examples_loaded") or evaluation.get("heldout_examples_evaluated"):
        raise ValueError("heldout entered M8 evaluation")


def validate_e2_evaluation(
    evaluation: Mapping[str, Any], bundle: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Backward-compatible E2-only wrapper used by the accepted Phase-A summarizer."""
    if config["model"]["condition"] != "E2":
        raise ValueError("E2 validator requires an E2 evaluation config")
    validate_candidate_evaluation(evaluation, bundle, config)


def public_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = evaluation.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation has no example rows")
    count = len(rows)
    return {
        "example_count": count,
        "bird_ex_behavior_compatible": sum(
            float(row["bird_ex_behavior_compatible"]) for row in rows
        )
        / count,
        "strict_execution_accuracy": sum(float(row["strict_execution_accuracy"]) for row in rows)
        / count,
        "strict_sql_only_rate": sum(bool(row["extraction"]["strict_sql_only"]) for row in rows)
        / count,
        "valid_sql_rate": sum(row["prediction_status"] == "ok" for row in rows) / count,
        "failure_categories": dict(
            sorted(Counter(row["failure_category"] for row in rows).items())
        ),
    }


def _exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_transitions(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metric: str,
    *,
    reference_label: str = "e1",
    candidate_label: str = "e2",
) -> dict[str, Any]:
    reference_rows = {row["sample_id"]: row for row in reference.get("examples", [])}
    candidate_rows = {row["sample_id"]: row for row in candidate.get("examples", [])}
    if not reference_rows or reference_rows.keys() != candidate_rows.keys():
        raise ValueError("paired E1/E2 rows are absent or misaligned")
    b = sum(
        bool(reference_rows[key][metric]) and not bool(candidate_rows[key][metric])
        for key in reference_rows
    )
    c = sum(
        not bool(reference_rows[key][metric]) and bool(candidate_rows[key][metric])
        for key in reference_rows
    )
    return {
        f"{reference_label}_correct_{candidate_label}_wrong": b,
        f"{reference_label}_wrong_{candidate_label}_correct": c,
        "discordant_pairs": b + c,
        "exact_two_sided_mcnemar_p": _exact_mcnemar(b, c),
    }


def build_public_comparison(
    *,
    config: Mapping[str, Any],
    m4_summary: Mapping[str, Any],
    e2_evaluation: Mapping[str, Any],
    e1_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_selection = config["data"]["selection_sha256"]
    if (
        m4_summary.get("status") != "completed"
        or m4_summary.get("selection", {}).get("full_selection_sha256") != expected_selection
    ):
        raise ValueError("accepted M4 E1 summary identity drifted")
    e1 = m4_summary["sft_evaluation"]["full1209_primary"]
    e2 = public_metrics(e2_evaluation)
    metrics = [config["comparison"]["primary_metric"], *config["comparison"]["secondary_metrics"]]
    deltas = {name: e2[name] - e1[name] for name in metrics}
    paired: dict[str, Any] = {"status": "not_available_without_existing_raw_e1"}
    if e1_raw is not None:
        for key in (
            "validation_selection_sha256",
            "sample_ids",
            "prompt_hashes",
            "decoding",
            "evaluator",
        ):
            if e1_raw.get(key) != e2_evaluation.get(key):
                raise ValueError(f"paired E1/E2 evidence drifted: {key}")
        paired = {
            "status": "completed_from_existing_raw_e1",
            "strict_execution_accuracy": paired_transitions(
                e1_raw, e2_evaluation, "strict_execution_accuracy"
            ),
            "bird_ex_behavior_compatible": paired_transitions(
                e1_raw, e2_evaluation, "bird_ex_behavior_compatible"
            ),
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "validation_selection_sha256": expected_selection,
        "example_count": config["data"]["example_count"],
        "primary_metric": config["comparison"]["primary_metric"],
        "E1": e1,
        "E2": e2,
        "E2_minus_E1": deltas,
        "paired_analysis": paired,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }


def _validate_paired_identity(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], label: str
) -> None:
    for key in (
        "validation_selection_sha256",
        "sample_ids",
        "prompt_hashes",
        "decoding",
        "evaluator",
    ):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"paired {label} evidence drifted: {key}")


def _paired_metrics(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_label: str,
    candidate_label: str = "E3",
) -> dict[str, Any]:
    return {
        "status": "completed_from_existing_raw_evidence",
        "strict_execution_accuracy": paired_transitions(
            reference,
            candidate,
            "strict_execution_accuracy",
            reference_label=reference_label.lower(),
            candidate_label=candidate_label.lower(),
        ),
        "bird_ex_behavior_compatible": paired_transitions(
            reference,
            candidate,
            "bird_ex_behavior_compatible",
            reference_label=reference_label.lower(),
            candidate_label=candidate_label.lower(),
        ),
        "reference_condition": reference_label,
        "candidate_condition": candidate_label,
    }


def build_e3_public_comparison(
    *,
    config: Mapping[str, Any],
    m4_summary: Mapping[str, Any],
    existing_common: Mapping[str, Any],
    e3_evaluation: Mapping[str, Any],
    e1_raw: Mapping[str, Any] | None = None,
    e2_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the SQL-free E1/E2/E3 comparison without regenerating prior conditions."""
    expected_selection = config["data"]["selection_sha256"]
    if config["model"]["condition"] != "E3":
        raise ValueError("E3 comparison requires an E3 evaluation config")
    if (
        m4_summary.get("status") != "completed"
        or m4_summary.get("selection", {}).get("full_selection_sha256") != expected_selection
    ):
        raise ValueError("accepted M4 E1 summary identity drifted")
    if (
        existing_common.get("status") != "completed"
        or existing_common.get("experiment_id") != "e2_m8_common_validation_001"
        or existing_common.get("validation_selection_sha256") != expected_selection
        or existing_common.get("example_count") != config["data"]["example_count"]
    ):
        raise ValueError("accepted E1/E2 common summary identity drifted")

    e1 = existing_common["E1"]
    e2 = existing_common["E2"]
    e3 = public_metrics(e3_evaluation)
    metrics = [config["comparison"]["primary_metric"], *config["comparison"]["secondary_metrics"]]
    paired: dict[str, Any] = {
        "E1_vs_E3": {"status": "not_available_without_existing_raw_e1"},
        "E2_vs_E3": {"status": "not_available_without_existing_raw_e2"},
    }
    if e1_raw is not None:
        _validate_paired_identity(e1_raw, e3_evaluation, "E1/E3")
        paired["E1_vs_E3"] = _paired_metrics(e1_raw, e3_evaluation, "E1")
    if e2_raw is not None:
        _validate_paired_identity(e2_raw, e3_evaluation, "E2/E3")
        paired["E2_vs_E3"] = _paired_metrics(e2_raw, e3_evaluation, "E2")

    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "validation_selection_sha256": expected_selection,
        "example_count": config["data"]["example_count"],
        "primary_metric": config["comparison"]["primary_metric"],
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "E3_minus_E1": {name: e3[name] - e1[name] for name in metrics},
        "E3_minus_E2": {name: e3[name] - e2[name] for name in metrics},
        "paired_analysis": paired,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }


def build_e4_public_comparison(
    *,
    config: Mapping[str, Any],
    existing_e3_common: Mapping[str, Any],
    e4_evaluation: Mapping[str, Any],
    e1_raw: Mapping[str, Any] | None = None,
    e2_raw: Mapping[str, Any] | None = None,
    e3_raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the SQL-free E1/E2/E3/E4 comparison without regenerating prior arms."""
    expected_selection = config["data"]["selection_sha256"]
    if config["model"]["condition"] != "E4":
        raise ValueError("E4 comparison requires an E4 evaluation config")
    if (
        existing_e3_common.get("status") != "completed"
        or existing_e3_common.get("experiment_id") != "e3_m8_common_validation_001"
        or existing_e3_common.get("validation_selection_sha256") != expected_selection
        or existing_e3_common.get("example_count") != config["data"]["example_count"]
    ):
        raise ValueError("accepted E1/E2/E3 common summary identity drifted")

    prior = {arm: existing_e3_common[arm] for arm in ("E1", "E2", "E3")}
    e4 = public_metrics(e4_evaluation)
    metrics = [config["comparison"]["primary_metric"], *config["comparison"]["secondary_metrics"]]
    paired: dict[str, Any] = {
        f"{arm}_vs_E4": {"status": f"not_available_without_existing_raw_{arm.lower()}"}
        for arm in ("E1", "E2", "E3")
    }
    for arm, raw in (("E1", e1_raw), ("E2", e2_raw), ("E3", e3_raw)):
        if raw is not None:
            _validate_paired_identity(raw, e4_evaluation, f"{arm}/E4")
            paired[f"{arm}_vs_E4"] = _paired_metrics(raw, e4_evaluation, arm, "E4")

    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "validation_selection_sha256": expected_selection,
        "example_count": config["data"]["example_count"],
        "primary_metric": config["comparison"]["primary_metric"],
        **prior,
        "E4": e4,
        **{
            f"E4_minus_{arm}": {name: e4[name] - prior[arm][name] for name in metrics}
            for arm in ("E1", "E2", "E3")
        },
        "paired_analysis": paired,
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
    }
