#!/usr/bin/env python3
"""Run the fixed, inference-only execution-feedback repair probe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-commit", required=True)
    parser.add_argument("--config", default="configs/e6_execution_repair_probe.toml")
    parser.add_argument("--source-replay", required=True)
    parser.add_argument("--prepared-smoke", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--e1-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if head != args.approved_commit or dirty:
        parser.error("approved clean commit required")

    from verireason.m3_contract import atomic_write_json
    from verireason.repair_probe import (
        execution_feedback,
        load_repair_probe_config,
        repair_messages,
        resolve_repair_action,
        summarize_repair_probe,
    )

    config = load_repair_probe_config(args.config)
    source = json.loads(Path(args.source_replay).read_text())
    prepared = json.loads(Path(args.prepared_smoke).read_text())
    if (
        source.get("experiment_id") != config["source"]["experiment_id"]
        or len(source.get("records", ())) != config["source"]["initial_states"]
        or len(source.get("selection", ())) != config["source"]["training_examples"]
        or source.get("optimizer_steps") != 0
        or source.get("gate", {}).get("passed") is not False
    ):
        raise ValueError("accepted stopped E5 source replay required")
    examples = prepared.get("examples", ())
    if len(examples) != 8 or prepared.get("heldout_examples_loaded") != 0:
        raise ValueError("retained training-only prepared smoke required")
    for index, (selection, example) in enumerate(zip(source["selection"], examples, strict=True)):
        if (
            selection["example_id"] != example["reward_group_id"]
            or selection["gold_sql"] != example["gold_sql"]
            or selection["database_path"] != example["database_path"]
        ):
            raise ValueError(f"source/prepared identity mismatch at group {index}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("fresh empty output directory required")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    raw_path = output_dir / "repair-probe-private.json"
    summary_path = output_dir / "repair-probe-summary.json"

    from peft import PeftModel
    from transformers import set_seed

    from verireason.m3_runtime import (
        DeviceMemorySampler,
        artifact_manifest,
        load_base,
        load_tokenizer,
    )
    from verireason.m4_contract import load_m4_config
    from verireason.sql_evaluator import (
        QueryResult,
        compare_strict_results,
        execute_read_only,
    )

    if artifact_manifest(args.e1_adapter)["manifest_sha256"] != config["model"]["adapter_manifest"]:
        raise ValueError("immutable E1 adapter identity mismatch")
    m4 = load_m4_config("configs/m4_formal_sft.toml", "configs/m2_data.toml")
    tokenizer = load_tokenizer(m4, False)
    base = load_base(m4, trainable=False, allow_download=False)
    model = PeftModel.from_pretrained(
        base,
        args.e1_adapter,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()

    generation = config["generation"]
    feedback_config = config["feedback"]
    records: list[dict] = []
    started = time.time()
    import torch

    torch.cuda.reset_peak_memory_stats()
    with DeviceMemorySampler(interval_seconds=0.2) as memory:
        for state_index, initial in enumerate(source["records"]):
            if time.time() - started >= config["evidence"]["maximum_session_wall_seconds"]:
                raise TimeoutError("repair probe wall limit reached")
            group = int(initial["group"])
            selection = source["selection"][group]
            example = examples[group]
            database = (Path(args.database_root) / selection["database_path"]).resolve()
            initial_result = execute_read_only(database, initial["text"])
            gold_result = execute_read_only(database, selection["gold_sql"])
            initial_reward = int(compare_strict_results(initial_result, gold_result).equal)
            if (
                initial_reward != initial["reward"]
                or initial_result.status != initial["terminal_status"]
            ):
                raise ValueError("source replay terminal evidence drifted")
            feedback = execution_feedback(
                initial_result,
                max_rows=feedback_config["maximum_preview_rows"],
                max_characters=feedback_config["maximum_preview_characters"],
            )
            # Matched random draws: reset the same per-state seed for each arm.
            for arm in generation["arms"]:
                set_seed(generation["seed"] + state_index)
                messages = repair_messages(
                    example["messages"], initial["text"], arm=arm, feedback=feedback
                )
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                prompt_length = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        num_return_sequences=generation["completions_per_arm"],
                        max_new_tokens=generation["maximum_new_tokens"],
                        temperature=generation["temperature"],
                        top_p=generation["top_p"],
                        top_k=generation["top_k"],
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )[:, prompt_length:]
                for candidate, token_ids in enumerate(generated.tolist()):
                    if tokenizer.eos_token_id in token_ids:
                        token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
                    completion = tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    action_sql, action_kind, strict_output = resolve_repair_action(
                        completion, initial["text"]
                    )
                    result = (
                        execute_read_only(database, action_sql)
                        if action_sql is not None
                        else QueryResult("invalid_format", error="repair output contract failed")
                    )
                    final_reward = int(compare_strict_results(result, gold_result).equal)
                    records.append(
                        {
                            "state_index": state_index,
                            "example_id": initial["example_id"],
                            "group": group,
                            "source_rollout": initial["rollout"],
                            "arm": arm,
                            "candidate": candidate,
                            "initial_reward": initial_reward,
                            "initial_status": initial_result.status,
                            "feedback": feedback,
                            "completion": completion,
                            "action_sql": action_sql,
                            "action_kind": action_kind,
                            "strict_output": strict_output,
                            "final_reward": final_reward,
                            "final_status": result.status,
                        }
                    )
            atomic_write_json(
                raw_path,
                {
                    "schema_version": 1,
                    "experiment_id": config["experiment_id"],
                    "complete": False,
                    "states_completed": state_index + 1,
                    "records": records,
                },
            )
            raw_path.chmod(0o600)

    summary = summarize_repair_probe(records, config)
    summary.update(
        {
            "source_experiment_id": source["experiment_id"],
            "source_commit": head,
            "initial_statuses": dict(Counter(row["initial_status"] for row in records[::4])),
            "wall_seconds": time.time() - started,
            "peak_gpu_memory_mib": memory.peak_mib,
        }
    )
    atomic_write_json(
        raw_path,
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "complete": True,
            "states_completed": config["source"]["initial_states"],
            "records": records,
        },
    )
    raw_path.chmod(0o600)
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
