#!/usr/bin/env python3
"""Prospective E5 runner. No resource provisioning; explicit approved local commit required.

Not invoked during CPU development. --help never imports or loads the model stack.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--approved-commit", required=True)
    p.add_argument("--training-manifest", required=True)
    p.add_argument("--prepared-train", required=True)
    p.add_argument("--database-root", required=True)
    p.add_argument("--e1-adapter", required=True)
    p.add_argument("--e1-adapter-manifest", required=True)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--review", help="Private review JSON; omit for capture/audit only")
    p.add_argument("--paid-started-at", type=float, required=True,
                   help="Original paid-session Unix timestamp, including resumed review")
    a = p.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if head != a.approved_commit or dirty or branch != "codex/e5-cpu-feasibility":
        p.error("approved clean E5 commit required")
    remaining = 3600 - (time.time() - a.paid_started_at)
    if not 0 < remaining <= 3600:
        p.error("invalid/exhausted cumulative paid-session deadline")

    def deadline(signum, frame):
        raise TimeoutError("cumulative deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    # Validate paths and session privacy before any model load.
    from verireason.e5_smoke import PrivateJournal, SmokeSession

    journal = PrivateJournal(Path(a.session_dir))
    if journal.path.exists():
        state = journal.read()
        if (state["pending"] or state["phase"] in {"updating", "finishing"}
                or (journal.directory / "replay.partial").exists()):
            journal.close()
            raise RuntimeError("uncertain session: stop without loading model")
    journal.close()

    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import GRPOConfig

    from verireason.e5_backend import HFReplayBackend, load_training_selection
    from verireason.e5_trl import E5ReplayTrainer
    from verireason.m3_runtime import artifact_manifest, load_base, load_tokenizer
    from verireason.m4_contract import load_m4_config
    from verireason.m7_contract import load_m7_smoke_config
    from verireason.m7_runtime import grpo_argument_values

    if artifact_manifest(a.e1_adapter)["manifest_sha256"] != a.e1_adapter_manifest:
        raise ValueError("immutable E1 adapter mismatch")
    config = load_m7_smoke_config("configs/m7_e2_smoke.toml")
    m4 = load_m4_config("configs/m4_formal_sft.toml", "configs/m2_data.toml")
    tokenizer = load_tokenizer(m4, False)
    rows = load_training_selection(a.training_manifest, a.prepared_train, tokenizer)
    context = dict(parent="e1_m4_formal_sft_001", adapter_manifest=a.e1_adapter_manifest,
                   source_commit=head, manifest=config["data"]["training_manifest_sha256"],
                   paid_started_at=a.paid_started_at)
    session = SmokeSession(Path(a.session_dir), rows, context)
    try:
        set_seed(23)
        base = load_base(m4, trainable=False, allow_download=False)
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(base, a.e1_adapter, is_trainable=True,
                                          local_files_only=True)
        model.config.use_cache = False
        args = grpo_argument_values(config, str(Path(a.session_dir) / "checkpoints"))
        args.update(save_strategy="no", disable_tqdm=True)
        trainer = E5ReplayTrainer(
            model=model, processing_class=tokenizer, args=GRPOConfig(**args),
            reward_funcs=lambda completions, **kw: [0.] * len(completions),
            train_dataset=Dataset.from_list([{"prompt": r["prompt"]} for r in rows]),
        )
        backend = HFReplayBackend(trainer, Path(a.database_root))
        review = json.loads(Path(a.review).read_text()) if a.review else None
        print(json.dumps(session.run(backend, review), allow_nan=False))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
