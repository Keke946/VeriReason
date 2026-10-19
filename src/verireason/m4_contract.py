"""Pre-registered M4 primary-epoch selection and evidence contracts."""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verireason.data_pipeline import sql_complexity
from verireason.m3_contract import sha256_json

MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
EVALUATOR_KEYS = (
    "timeout_seconds",
    "maximum_cpu_seconds",
    "maximum_cpu_hard_seconds",
    "maximum_result_rows",
    "maximum_result_bytes",
    "maximum_vm_bytes",
    "maximum_float_pair_checks",
    "float_absolute_tolerance",
    "float_relative_tolerance",
)


def load_m4_config(path: str | Path, m2_path: str | Path | None = None) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {
        "model",
        "data",
        "prompt",
        "decoding",
        "qlora",
        "checkpoint",
        "epoch_extension_diagnostic",
        "evaluator",
        "success",
        "evidence",
    }
    if config.get("schema_version") != 1 or required - config.keys():
        raise ValueError("M4 config must be schema v1 with all frozen sections")
    if config.get("status") != "precloud_preregistered_pending_control_approval":
        raise ValueError("M4 must remain pending before the cloud run")
    if config["model"]["revision"] != MODEL_REVISION:
        raise ValueError("M4 model revision drifted")
    data, q, checkpoint, diagnostic = (
        config[name] for name in ("data", "qlora", "checkpoint", "epoch_extension_diagnostic")
    )
    if data["heldout_examples_loaded"] != 0 or data["heldout_examples_evaluated"] != 0:
        raise ValueError("M4 heldout use must remain zero")
    if (
        data["train_count"],
        data["development_validation_count"],
        data["final_validation_count"],
        data["full_validation_count"],
    ) != (5364, 256, 953, 1209):
        raise ValueError("M4 primary scale or validation partition changed")
    preregistration = {
        "selector_version": data["selector_version"],
        "selector_seed": data["selector_seed"],
        "required_features": data["required_features"],
        "train_count": data["train_count"],
        "development_validation_count": data["development_validation_count"],
        "final_validation_count": data["final_validation_count"],
        "full_validation_count": data["full_validation_count"],
        "source_manifest_sha256": data["training_manifest_sha256"],
    }
    if data["selection_preregistration_sha256"] != sha256_json(preregistration):
        raise ValueError("M4 selection preregistration identity changed")
    if config["prompt"]["contract_version"] != "m3_sqlite_sql_only_v1":
        raise ValueError("M4 must reuse the M3 prompt contract")
    if config["prompt"]["overlength_policy"] != "fail_if_any_full_train_row_is_token_ineligible":
        raise ValueError("M4 full route cannot silently exclude token-ineligible training rows")
    if (
        q["effective_batch_size"]
        != q["per_device_train_batch_size"] * q["gradient_accumulation_steps"]
    ):
        raise ValueError("M4 effective batch is inconsistent")
    if q["gradient_accumulation_steps"] != 8 or q["effective_batch_size"] != 8:
        raise ValueError("M4 freezes batch-1 x accumulation-8")
    if (
        q["partial_accumulation_policy"],
        q["final_real_microbatches"],
        q["final_zero_gradient_padding_microbatches"],
        q["final_real_loss_multiplier"],
    ) != ("scale_real_losses_and_zero_gradient_pad", 4, 4, 2):
        raise ValueError("M4 final four-example accumulation contract changed")
    expected_steps = math.ceil(data["train_count"] / q["effective_batch_size"])
    if (
        q["primary_epochs"],
        q["updates_per_epoch"],
        q["primary_updates"],
        q["maximum_optimizer_steps"],
        q["scheduler_total_optimizer_steps"],
    ) != (1, expected_steps, expected_steps, expected_steps, expected_steps):
        raise ValueError("M4 primary epoch/schedule must end at update 671")
    if q["epoch2_extension"] != "control_review_required" or q["automatic_extension"]:
        raise ValueError("M4 cannot automatically enter epoch 2")
    if q["transformers_version"] != "5.14.1":
        raise ValueError("M4 trainer semantics are pinned to Transformers 5.14.1")
    if q["warmup_optimizer_steps"] != math.ceil(expected_steps * q["warmup_ratio"]):
        raise ValueError("M4 explicit primary warmup drifted")
    if (
        checkpoint["early_stopping"]
        or checkpoint["selection_rule"] != "final_primary_update_671"
        or checkpoint["save_optimizer_steps"] != 64
        or checkpoint["save_total_limit"] != 4
        or checkpoint["required_loss_checkpoint_updates"] != [512, 671]
    ):
        raise ValueError("M4 primary checkpoint evidence contract changed")
    if (
        diagnostic["validation_selection"],
        diagnostic["loss"],
        diagnostic["loss_checkpoint_updates"],
        diagnostic["advisory_relative_dev_loss_drop"],
        diagnostic["automatic_extension"],
        diagnostic["decision"],
    ) != (
        "dev256",
        "teacher_forced_answer_only_token_weighted_mean",
        [512, 671],
        0.01,
        False,
        "control_review_required_after_primary",
    ):
        raise ValueError("M4 epoch-extension diagnostic must remain advisory only")
    if (
        config["evidence"]["expected_session_wall_seconds"],
        config["evidence"]["maximum_session_wall_seconds"],
        config["evidence"]["maximum_session_cost_cny"],
    ) != (43_200, 50_400, 36.26):
        raise ValueError("M4 deduplicated time/cost preregistration changed")
    if m2_path:
        with Path(m2_path).open("rb") as handle:
            m2 = tomllib.load(handle)
        drift = [key for key in EVALUATOR_KEYS if config["evaluator"][key] != m2["evaluator"][key]]
        if drift:
            raise ValueError(f"M4 evaluator drifted from M2: {', '.join(drift)}")
    return config


def _stable_id(row: Mapping[str, Any]) -> str:
    value = row.get("row_id")
    if not isinstance(value, str) or not value:
        raise ValueError("every M4 row needs a stable row_id")
    return value


def _rank(row: Mapping[str, Any], seed: str, split: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{split}\0{row['db_id']}\0{_stable_id(row)}".encode()
    ).hexdigest()


def select_formal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    count: int,
    seed: str,
    required_features: Sequence[str],
    source_manifest_sha256: str,
    selector_version: str = "m4_full_primary_epoch_v3",
) -> dict[str, Any]:
    """Outcome-blind DB/feature/hash selection; count=all proves full coverage."""
    eligible = [
        r
        for r in rows
        if r.get("split") == split
        and r.get("operational_eligible") is True
        and r.get("source") != "heldout"
        and r.get("token_eligible", True) is True
    ]
    if len(eligible) < count:
        raise ValueError("insufficient token-eligible M4 rows")
    if len({_stable_id(r) for r in eligible}) != len(eligible):
        raise ValueError("duplicate M4 row identity")
    ranked = sorted(eligible, key=lambda r: _rank(r, seed, split))
    by_db: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ranked:
        by_db[str(row["db_id"])].append(row)
    chosen = [by_db[db][0] for db in sorted(by_db)]
    if len(chosen) > count:
        raise ValueError("count cannot preserve the database boundary")
    used = {_stable_id(r) for r in chosen}
    for feature in required_features:
        if any(feature in sql_complexity(str(r["sql"])).features for r in chosen):
            continue
        match = next(
            (
                r
                for r in ranked
                if _stable_id(r) not in used and feature in sql_complexity(str(r["sql"])).features
            ),
            None,
        )
        if match is None:
            raise ValueError(f"cannot cover required feature: {feature}")
        chosen.append(match)
        used.add(_stable_id(match))
    for row in ranked:
        if len(chosen) == count:
            break
        if _stable_id(row) not in used:
            chosen.append(row)
            used.add(_stable_id(row))
    ids = [_stable_id(r) for r in chosen]
    identity = {
        "selector_version": selector_version,
        "seed": seed,
        "split": split,
        "count": count,
        "source_manifest_sha256": source_manifest_sha256,
        "sample_ids": ids,
    }
    return {
        **identity,
        "selection_sha256": sha256_json(identity),
        "database_ids": sorted(by_db),
        "feature_counts": dict(
            sorted(
                Counter(f for r in chosen for f in sql_complexity(str(r["sql"])).features).items()
            )
        ),
        "rows": chosen,
    }


def validate_validation_partition(
    dev: Mapping[str, Any], final: Mapping[str, Any], full: Mapping[str, Any]
) -> None:
    dev_ids, final_ids, full_ids = (
        set(dev["sample_ids"]),
        set(final["sample_ids"]),
        full["sample_ids"],
    )
    if dev_ids & final_ids or len(dev_ids) != 256 or len(final_ids) != 953 or len(full_ids) != 1209:
        raise ValueError("M4 dev256/final953/full1209 partition is invalid")
    if set(full_ids) != dev_ids | final_ids:
        raise ValueError("M4 full1209 must be exactly dev256 union final953")


def extension_decision(
    loss_512: float, loss_671: float, *, advisory_threshold: float = 0.01
) -> dict[str, Any]:
    finite = math.isfinite(loss_512) and math.isfinite(loss_671) and loss_512 != 0.0
    relative = (loss_512 - loss_671) / loss_512 if finite else None
    return {
        "formula": "(loss_512-loss_671)/loss_512",
        "losses_finite": finite,
        "relative_dev_loss_drop": relative,
        "advisory_reference_threshold": advisory_threshold,
        "advisory_threshold_met": bool(
            finite and relative is not None and relative >= advisory_threshold
        ),
        "automatic_extension": False,
        "epoch2_extension": "control_review_required",
        "decision": "primary_epoch_stopped_pending_control_review",
    }


def validate_comparison(base: Mapping[str, Any], sft: Mapping[str, Any]) -> None:
    for key in (
        "validation_selection_sha256",
        "sample_ids",
        "prompt_hashes",
        "decoding",
        "evaluator",
    ):
        if base.get(key) != sft.get(key):
            raise ValueError(f"E0/E1 comparison drift: {key}")
    if base.get("heldout_examples_loaded") != 0 or sft.get("heldout_examples_loaded") != 0:
        raise ValueError("heldout entered M4 comparison")


def validate_resume_contract(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if dict(expected) != dict(observed):
        raise ValueError("M4 resume checkpoint contract drifted")
