#!/usr/bin/env python3
"""Fresh-process E8 adapter reload and fixed four-cell training-only audit."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from verireason.m3_contract import atomic_write_json
from verireason.m3_runtime import artifact_manifest, load_base, load_tokenizer
from verireason.m4_contract import load_m4_config
from verireason.repair_probe import resolve_repair_action
from verireason.repair_sft_runtime import (
    audit_messages,
    validate_private_split,
    validate_reload_evidence,
    validate_training_evidence,
)
from verireason.repair_sft_smoke import (
    EXPERIMENT_ID,
    load_repair_sft_smoke_config,
    summarize_repair_sft_evaluation,
    write_private_split,
)
from verireason.sql_evaluator import QueryResult, compare_strict_results, execute_read_only


def evaluate_condition(model, tokenizer, audit, condition, config, database_root):
    import torch

    rows = []
    generation = config["evaluation"]
    for source in audit:
        initial_sql = str(source["messages"][-2]["content"])
        gold_sql = str(source["target"]) if source["kind"] == "repair" else initial_sql
        database = (database_root / str(source["database_path"])).resolve()
        if not database.is_relative_to(database_root) or not database.is_file():
            raise ValueError("E8 audit database path escaped its root")
        gold = execute_read_only(database, gold_sql)
        if not gold.successful:
            raise RuntimeError("E8 audit gold SQL is no longer executable")
        for arm in generation["arms"]:
            messages = audit_messages(source, arm)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            prompt_tokens = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(generation["maximum_new_tokens"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0, prompt_tokens:]
            token_ids = generated.tolist()
            if tokenizer.eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
            completion = tokenizer.decode(
                token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            action_sql, action_kind, strict_output = resolve_repair_action(completion, initial_sql)
            result = (
                execute_read_only(database, action_sql)
                if action_sql is not None
                else QueryResult("invalid_format", error="repair output contract failed")
            )
            rows.append(
                {
                    "condition": condition,
                    "arm": arm,
                    "record_id": source["record_id"],
                    "source_sample_id": source["source_sample_id"],
                    "db_id": source["db_id"],
                    "kind": source["kind"],
                    "completion": completion,
                    "action_sql": action_sql,
                    "action_kind": action_kind,
                    "strict_output": strict_output,
                    "final_correct": bool(compare_strict_results(result, gold).equal),
                    "final_status": result.status,
                }
            )
    return rows


def load_adapter(m4, adapter):
    from peft import PeftModel

    base = load_base(m4, trainable=False, allow_download=False)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e8_repair_sft_smoke.toml")
    parser.add_argument("--m4-config", default="configs/m4_formal_sft.toml")
    parser.add_argument("--private-split", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--e1-adapter", type=Path, required=True)
    parser.add_argument("--repair-adapter", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_repair_sft_smoke_config(args.config)
    m4 = load_m4_config(args.m4_config, "configs/m2_data.toml")
    payload = json.loads(args.private_split.read_text())
    train, audit = validate_private_split(payload, config)
    training = json.loads(args.training_result.read_text())
    validate_training_evidence(training, training_examples=len(train), config=config)
    if (
        m4["model"]["model_id"] != config["model"]["model_id"]
        or m4["model"]["revision"] != config["model"]["revision"]
        or artifact_manifest(args.e1_adapter)["manifest_sha256"]
        != config["model"]["parent_adapter_manifest"]
    ):
        raise ValueError("E8 evaluation model or immutable E1 identity drifted")
    adapter_manifest = artifact_manifest(args.repair_adapter)["manifest_sha256"]
    reload = {
        "adapter_manifest_sha256": adapter_manifest,
        "training_process_id": training.get("process_id"),
        "reload_process_id": os.getpid(),
        "adapter_loaded": False,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("E8 evaluation requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    database_root = args.database_root.resolve()
    tokenizer = load_tokenizer(m4, False)
    rows = []
    baseline = load_adapter(m4, args.e1_adapter)
    rows.extend(
        evaluate_condition(baseline, tokenizer, audit, "e1_baseline", config, database_root)
    )
    write_private_split(
        output / "e8-evaluation-private.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "complete": False,
            "completed_conditions": ["e1_baseline"],
            "records": rows,
            "validation_payloads_read": 0,
            "heldout_payloads_read": 0,
        },
    )
    del baseline
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    repaired = load_adapter(m4, args.repair_adapter)
    reload["adapter_loaded"] = bool(repaired.peft_config)
    validate_reload_evidence(training, reload)
    rows.extend(evaluate_condition(repaired, tokenizer, audit, "repair_sft", config, database_root))
    del repaired
    gc.collect()
    torch.cuda.empty_cache()

    summary = summarize_repair_sft_evaluation(rows, config)
    summary["reload"] = {**reload, "independent_reload_passed": True}
    raw = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "completed_conditions": ["e1_baseline", "repair_sft"],
        "records": rows,
        "validation_payloads_read": 0,
        "heldout_payloads_read": 0,
    }
    write_private_split(output / "e8-evaluation-private.json", raw)
    atomic_write_json(output / "e8-repair-sft-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
