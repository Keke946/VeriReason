"""GPU-only M3 baseline, tiny QLoRA, reload, and evaluator runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verireason.m3_contract import (
    AnswerOnlyCollator,
    atomic_write_json,
    comparison_contract,
    evaluator_limits,
    extract_generated_sql,
    load_m3_config,
    sha256_json,
    validate_answer_only_token_contract,
    verify_update_evidence,
)
from verireason.sql_evaluator import evaluate_prediction


class DeviceMemorySampler:
    """Sample whole-device memory, including allocations outside torch."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mib = 0
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
                self.peak_mib = max(
                    self.peak_mib,
                    *(int(line.strip()) for line in output.splitlines() if line.strip()),
                )
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def read_slice(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.pop("slice_bundle_sha256", None)
    observed = sha256_json(payload)
    payload["slice_bundle_sha256"] = claimed
    if claimed != observed or payload.get("synthetic") is not False:
        raise ValueError("real M3 slice identity is missing or invalid")
    if (
        payload.get("heldout_examples_loaded") != 0
        or payload.get("heldout_examples_evaluated") != 0
    ):
        raise ValueError("M3 slice reports heldout consumption")
    for split in ("train", "validation"):
        examples = payload.get("slices", {}).get(split, {}).get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"M3 {split} slice must contain examples")
        for example in examples:
            validate_answer_only_token_contract(
                example.get("input_ids", ()),
                example.get("labels", ()),
                example.get("prompt_tokens"),
                answer_tokens=example.get("answer_tokens"),
                supervised_tokens=example.get("supervised_tokens"),
            )
    return payload


def git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=15
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def package_versions() -> dict[str, str]:
    import importlib.metadata

    packages = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "trl")
    return {name: importlib.metadata.version(name) for name in packages}


def load_tokenizer(config: Mapping[str, Any], allow_download: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base(config: Mapping[str, Any], *, trainable: bool, allow_download: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = config["qlora"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=False,
        local_files_only=not allow_download,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if trainable:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=quant["gradient_checkpointing"]
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=quant["lora_rank"],
                lora_alpha=quant["lora_alpha"],
                lora_dropout=quant["lora_dropout"],
                target_modules=quant["target_modules"],
                bias=quant["bias"],
                task_type=quant["task_type"],
            ),
        )
        model.config.use_cache = False
    return model


def _gpu_snapshot() -> dict[str, Any]:
    import torch

    return {
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def evaluate_model(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    model.eval()
    limits = evaluator_limits(config)
    rows = []
    generated_tokens = 0
    started = time.perf_counter()
    for example in examples:
        prompt_ids = list(example["input_ids"][: example["prompt_tokens"]])
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=config["decoding"]["do_sample"],
                num_beams=config["decoding"]["num_beams"],
                max_new_tokens=config["decoding"]["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0, inputs.shape[1] :]
        generated_tokens += int(completion_ids.shape[0])
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        extracted = extract_generated_sql(completion)
        score = None
        error = None
        if extracted["sql"]:
            try:
                score = evaluate_prediction(
                    example["database_path"], extracted["sql"], example["target_sql"], limits
                )
            except Exception as failure:
                if isinstance(failure, ValueError) and str(failure).startswith(
                    "gold SQL preflight failed"
                ):
                    raise RuntimeError(str(failure)) from failure
                error = f"{type(failure).__name__}: {failure}"
        if not extracted["sql"]:
            failure_category = "no_query"
        elif error:
            failure_category = "evaluator_error"
        elif score.prediction_status != "ok":
            failure_category = score.prediction_status
        elif not extracted["strict_sql_only"]:
            failure_category = "format_noncompliant"
        elif not score.strict_execution_accuracy:
            failure_category = "wrong_result"
        else:
            failure_category = "strict_correct"
        rows.append(
            {
                "sample_id": example["sample_id"],
                "prompt_sha256": example["prompt_sha256"],
                "completion": completion,
                "extraction": extracted,
                "bird_ex_behavior_compatible": score.bird_ex_behavior_compatible if score else 0,
                "strict_execution_accuracy": score.strict_execution_accuracy if score else 0,
                "prediction_status": score.prediction_status if score else "not_evaluated",
                "strict_comparison_status": score.strict_comparison_status if score else None,
                "failure_category": failure_category,
                "error": error,
            }
        )
    seconds = time.perf_counter() - started
    count = len(rows)
    return {
        "examples": rows,
        "metrics": {
            "example_count": count,
            "bird_ex_behavior_compatible": sum(row["bird_ex_behavior_compatible"] for row in rows)
            / count,
            "strict_execution_accuracy": sum(row["strict_execution_accuracy"] for row in rows)
            / count,
            "strict_sql_only_rate": sum(row["extraction"]["strict_sql_only"] for row in rows)
            / count,
            "generated_tokens": generated_tokens,
            "generation_seconds": seconds,
            "completion_tokens_per_second": generated_tokens / seconds,
            "failure_categories": dict(
                sorted(Counter(row["failure_category"] for row in rows).items())
            ),
        },
    }


def train_adapter(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Mapping[str, Any],
    adapter_dir: str | Path,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    qlora = config["qlora"]
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    before = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}
    collator = AnswerOnlyCollator(
        tokenizer.pad_token_id, tensor_factory=lambda values: torch.tensor(values, dtype=torch.long)
    )
    loader = DataLoader(
        [{"input_ids": row["input_ids"], "labels": row["labels"]} for row in examples],
        batch_size=qlora["per_device_train_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for _name, parameter in trainable), lr=qlora["learning_rate"]
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    steps = 0
    micro_steps = 0
    losses = []
    gradient_norm = math.nan
    supervised_tokens = sum(label != -100 for example in examples for label in example["labels"])
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is not finite")
        losses.append(float(loss.detach()))
        (loss / qlora["gradient_accumulation_steps"]).backward()
        micro_steps += 1
        boundary = micro_steps % qlora["gradient_accumulation_steps"] == 0
        final_batch = batch_index + 1 == len(loader)
        if boundary or final_batch:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _name, parameter in trainable], qlora["maximum_gradient_norm"]
                )
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if progress_path is not None:
                atomic_write_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "status": "training_in_progress",
                        "micro_steps": micro_steps,
                        "optimizer_steps": steps,
                        "latest_loss": losses[-1],
                        "gradient_norm": gradient_norm,
                        "supervised_token_count": supervised_tokens,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
            if steps >= qlora["maximum_optimizer_steps"]:
                break
    adapter_dir = Path(adapter_dir)
    temporary_adapter = adapter_dir.with_name(f".{adapter_dir.name}.partial-{os.getpid()}")
    if adapter_dir.exists() or temporary_adapter.exists():
        raise FileExistsError("adapter destination or partial destination already exists")
    temporary_adapter.mkdir(parents=True)
    model.save_pretrained(temporary_adapter, safe_serialization=True)
    tokenizer.save_pretrained(temporary_adapter)
    delta_squared = 0.0
    changed = False
    for name, parameter in trainable:
        delta = parameter.detach().float().cpu() - before[name]
        norm = float(torch.linalg.vector_norm(delta))
        delta_squared += norm * norm
        changed = changed or bool(norm > 0)
    evidence = {
        "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
        "supervised_token_count": supervised_tokens,
        "initial_loss": losses[0] if losses else math.nan,
        "final_observed_loss": losses[-1] if losses else math.nan,
        "gradient_norm": gradient_norm,
        "micro_steps": micro_steps,
        "optimizer_steps": steps,
        "adapter_parameter_delta_detected": changed,
        "adapter_delta_l2_norm": math.sqrt(delta_squared),
        "training_seconds": time.perf_counter() - started,
    }
    verify_update_evidence(evidence)
    os.replace(temporary_adapter, adapter_dir)
    return evidence


def artifact_manifest(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"files": rows, "manifest_sha256": sha256_json(rows)}


def base_and_train(
    *,
    config_path: str,
    m2_config_path: str,
    slice_path: str,
    adapter_dir: str,
    output: str,
    allow_download: bool,
) -> dict[str, Any]:
    import torch

    wall_started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    config = load_m3_config(config_path, m2_config_path)
    source = git_identity()
    if config["evidence"]["require_clean_git"] and source["dirty"]:
        raise RuntimeError("M3 cloud run requires a clean committed source tree")
    slices = read_slice(slice_path)
    validation = slices["slices"]["validation"]
    contract = comparison_contract(validation["slice_sha256"], validation["prompt_hashes"], config)
    torch.manual_seed(config["qlora"]["seed"])
    torch.cuda.manual_seed_all(config["qlora"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    with DeviceMemorySampler() as memory_sampler:
        tokenizer = load_tokenizer(config, allow_download)
        model = load_base(config, trainable=False, allow_download=allow_download)
        base = evaluate_model(model, tokenizer, validation["examples"], config)
        atomic_write_json(
            output,
            {
                "schema_version": 1,
                "status": "base_completed_training_pending",
                "experiment_id": config["experiment_id"],
                "started_at": started_at,
                "source": source,
                "slice_bundle_sha256": slices["slice_bundle_sha256"],
                "heldout_examples_loaded": 0,
                "heldout_examples_evaluated": 0,
                "comparison_contract": contract,
                "base_evaluation": base,
                "elapsed_seconds": time.perf_counter() - wall_started,
                "process_id": os.getpid(),
            },
        )
        del model
        torch.cuda.empty_cache()
        model = load_base(config, trainable=True, allow_download=allow_download)
        training = train_adapter(
            model,
            slices["slices"]["train"]["examples"],
            tokenizer,
            config,
            adapter_dir,
            progress_path=Path(output).parent / "training-progress.json",
        )
    wall_seconds = time.perf_counter() - wall_started
    adapter = artifact_manifest(adapter_dir)
    atomic_write_json(Path(output).parent / "adapter-manifest.json", adapter)
    result = {
        "schema_version": 1,
        "status": "awaiting_fresh_reload",
        "experiment_id": config["experiment_id"],
        "started_at": started_at,
        "source": source,
        "packages": package_versions(),
        "slice_bundle_sha256": slices["slice_bundle_sha256"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "comparison_contract": contract,
        "base_evaluation": base,
        "training": training,
        "adapter": adapter,
        "memory": {**_gpu_snapshot(), "nvidia_smi_peak_used_mib": memory_sampler.peak_mib},
        "wall_seconds": wall_seconds,
        "process_id": os.getpid(),
    }
    atomic_write_json(output, result)
    return result


def reload_and_evaluate(
    *,
    config_path: str,
    m2_config_path: str,
    slice_path: str,
    adapter_dir: str,
    prior_result_path: str,
    output: str,
    allow_download: bool,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel

    wall_started = time.perf_counter()
    config = load_m3_config(config_path, m2_config_path)
    slices = read_slice(slice_path)
    prior = json.loads(Path(prior_result_path).read_text(encoding="utf-8"))
    source = git_identity()
    if source["dirty"] or source != prior["source"]:
        raise RuntimeError("source identity changed before fresh reload evaluation")
    if prior["process_id"] == os.getpid():
        raise RuntimeError("adapter reload must run in a fresh operating-system process")
    if prior["adapter"] != artifact_manifest(adapter_dir):
        raise RuntimeError("saved adapter identity changed before reload")
    validation = slices["slices"]["validation"]
    contract = comparison_contract(validation["slice_sha256"], validation["prompt_hashes"], config)
    if contract != prior["comparison_contract"]:
        raise RuntimeError("Base/SFT comparison contract changed before reload evaluation")
    torch.cuda.reset_peak_memory_stats()
    with DeviceMemorySampler() as memory_sampler:
        tokenizer = load_tokenizer(config, allow_download)
        base = load_base(config, trainable=False, allow_download=allow_download)
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        sft = evaluate_model(model, tokenizer, validation["examples"], config)
    reload_wall_seconds = time.perf_counter() - wall_started
    total_wall_seconds = prior["wall_seconds"] + reload_wall_seconds
    rate = config["evidence"]["planning_rate_cny_per_hour"]
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "completed_at": datetime.now(UTC).isoformat(),
        "source": source,
        "slice_bundle_sha256": slices["slice_bundle_sha256"],
        "heldout_examples_loaded": 0,
        "heldout_examples_evaluated": 0,
        "comparison_contract": contract,
        "fresh_reload": {
            "prior_process_id": prior["process_id"],
            "reload_process_id": os.getpid(),
            "different_process": prior["process_id"] != os.getpid(),
            "adapter": artifact_manifest(adapter_dir),
        },
        "base_evaluation": prior["base_evaluation"],
        "sft_evaluation": sft,
        "training": prior["training"],
        "memory": {
            "base_and_train": prior["memory"],
            "reload": {
                **_gpu_snapshot(),
                "nvidia_smi_peak_used_mib": memory_sampler.peak_mib,
            },
        },
        "timing": {
            "base_and_train_wall_seconds": prior["wall_seconds"],
            "reload_wall_seconds": reload_wall_seconds,
            "measured_total_wall_seconds": total_wall_seconds,
        },
        "cost": {
            "planning_rate_cny_per_hour": rate,
            "estimated_from_measured_wall_cny": total_wall_seconds / 3600 * rate,
            "actual_provider_cost_cny": None,
        },
    }
    atomic_write_json(output, result)
    return result
