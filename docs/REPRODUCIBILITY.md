# Reproducibility Guide

VeriReason separates three meanings of “reproduce” so a cheap integrity check is never confused
with an expensive model rerun.

## Level 1 — Verify code and published evidence (CPU, one command)

```bash
git clone https://github.com/Keke946/VeriReason.git
cd VeriReason
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/m10-cpu-lock.txt
python -m pip install --no-build-isolation --no-deps .
make m10-verify PYTHON=python PIP_CHECK_PYTHON=python
```

This verifies the durable-state guard, tests, lint, bytecode, dependency health, and SHA-256 of every
allowlisted SQL-free result. It needs no GPU, model, database, or cloud account.

Container equivalent:

```bash
docker build -t verireason:v1 .
docker run --rm verireason:v1
```

## Level 2 — Run the selected model on your SQLite database

Download and extract the E2 adapter from the GitHub v1.0.0 release, then use one CUDA GPU:

```bash
shasum -a 256 verireason-e2-adapter.tar.gz
# 9e17b2a5acb5ecbe78c778465dd8920ea3fd643340847243ba3892fbd366bcf8
tar -xzf verireason-e2-adapter.tar.gz
```

```bash
python -m pip install -e '.[gpu]'
verireason-demo \
  --database /path/to/example.sqlite \
  --question 'Which five products have the highest revenue?' \
  --adapter /path/to/verireason-e2-adapter \
  --allow-model-download
```

Model download is explicit because the fixed base is about 8 GiB. Omit
`--allow-model-download` to require an already-populated Hugging Face cache. Add `--execute` only for
a database you are authorized to query; execution remains read-only and resource-bounded.

## Level 3 — Re-run data, SFT, RL, evaluation, and serving

1. Materialize BIRD using the immutable URLs and hashes in `configs/m2_data.toml` and the commands
   in `docs/M2_DAY3A_REPORT.md`.
2. Publish the eligible manifest with `make m2-finalize` and the accepted private evidence.
3. Run M4 QLoRA with `docs/M4_RUNBOOK.md` and `scripts/run_m4_cloud.sh`.
4. Run E2/E3/E4 with the fixed configs and `scripts/run_m7_cloud.sh`.
5. Run common validation via `scripts/run_m8_eval.py`; never use Mini-Dev for tuning.
6. Serve the selected E2 adapter via `scripts/run_m9_cloud.sh`.

The canonical measured environment is Ubuntu 22.04.5, Python 3.11.15, PyTorch 2.11.0+cu130,
Transformers 5.14.1, TRL 1.9.0, PEFT 0.19.1, and vLLM 0.25.1 on one 48-GB RTX 4090 D.

## What must be archived before deleting retained cloud storage

The public Git repository plus the E2 release adapter are sufficient to inspect the project, verify
all published aggregates, and run the selected model. A full no-retraining replay of every arm also
requires these private/large artifacts to be copied to durable storage first:

- final M2 training manifest, source receipt, and private heldout manifest;
- E1 formal SFT adapter and its contract;
- E3 and E4 formal adapters and their adapter manifests;
- any raw E1/E2 paired validation evidence not already backed up locally;
- provider invoice screenshots needed to close the spend ledger.

The E2 formal adapter and E3/E4 raw validation evidence already have local copies. BIRD databases and
the Qwen base can be reconstructed from frozen upstream revisions, so keeping a paid provider disk
is convenient but not logically required after the artifacts above have two verified copies.
