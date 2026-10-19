# M3 Day 4 Vertical-Slice Runbook

Last updated: 2026-08-24

## Scope and status

Day 4A implemented and locally verified the M3 vertical-slice contract. Attempt 003 subsequently
completed the real frozen run and M3 is closed. This document preserves the exact execution contract
and failure gates; it is not authorization to rerun the paid session.

The accepted real experiment is `m3_vertical_slice_001`. It consumed operationally eligible M2 train
and validation rows only. Heldout loading and evaluation must both remain exactly zero.

The curated closure is `results/m3/m3-vertical-slice-summary.json`. Re-audit the returned ignored
artifact directory with `make m3-closure-verify`. The successful execution commit is
`530a0f22f491882def5e13b8344d1310c1a7960d`; the distinct pin commit is
`f1a50c463c011153fbe6ea098e5c8a72e53af150`. Actual provider cost is CNY 3.88284165 from two
user-verified invoices, not from the program's wall-time estimate.

## Frozen contracts

- Model: `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Prompt: `m3_sqlite_sql_only_v1`; no explanation, reasoning trace, or sample values.
- Schema: deterministic table/column/type/PK/NOT NULL/FK serialization.
- Target: original gold SQL stripped only of outer whitespace and one trailing semicolon; EOS is
  supervised and gold SQL is never truncated.
- Masking: system, user, assistant prefix, and padding labels are `-100`; SQL and EOS participate
  in cross-entropy loss.
- Overflow: remove complete schema table blocks from the deterministic end; reject the sample if
  one complete table plus question and full gold target still do not fit.
- Selection: fixed seed, stable row hashes, join/aggregate/subquery coverage, and preregistered
  counts of 8 train plus 8 validation. No model output may inform selection.
- Comparison: Base and fresh-reloaded SFT use identical ordered validation IDs, prompt hashes,
  greedy decoding, and the frozen M2 evaluator.
- Training: 4-bit NF4 QLoRA, rank 16, all linear projection modules, at most two optimizer steps.

The runtime uses maintained Transformers `AutoModelForCausalLM`/`BitsAndBytesConfig` APIs and PEFT
`prepare_model_for_kbit_training`, `LoraConfig`, `get_peft_model`, and
`PeftModel.from_pretrained`. Expected versions are Transformers 5.14.1, PEFT 0.19.1,
bitsandbytes 0.49.2, PyTorch 2.11.0, and Accelerate 1.14.0. No upstream trainer source is copied.
Project-owned code supplies data identity, masking, selection, safe SQL evaluation, update evidence,
comparison identity, and result validation.

## Required cloud inputs

The retained cloud disk must contain:

- published M2 dataset root `m2-v1`;
- `training-manifest.json` with SHA-256
  `d90ffab32e296248376c542b736aef5ce7f547acc702a9eab41073789cf5b72a`;
- all SQLite paths referenced by that manifest;
- the fixed model/tokenizer revision in the local Hugging Face cache, or a separately approved
  one-time model download;
- a clean checkout of the Day 4A commit and the pinned GPU environment.

The private heldout manifest path must not be passed to any M3 command.

## Preflight and exact sequence

Use one RTX 4090 D 48 GB at a freshly verified rate. The approved execution commit is supplied from
outside that commit, avoiding a self-referential Git identity. Choose a new absolute output path
outside Git for every attempt; the orchestrator creates it only after commit and disk gates pass:

```bash
git checkout --detach 530a0f22f491882def5e13b8344d1310c1a7960d

export M3_APPROVED_COMMIT=530a0f22f491882def5e13b8344d1310c1a7960d
export M3_DATASET_ROOT=/root/verireason-m2-data/processed/m2-v1
export M3_MANIFEST_INDEX="$M3_DATASET_ROOT/manifest-index.json"
export M3_TRAINING_MANIFEST="$M3_DATASET_ROOT/training-manifest.json"
export M3_OUTPUT_DIR=/root/verireason-m3/runs/m3_vertical_slice_001_attempt_001

sh scripts/run_m3_cloud.sh
```

This is the only M3 cloud-script invocation. It performs, without an idle approval pause:

1. exact approved-commit, clean-tree, external fresh-output, and 20-GiB gates;
2. local and GPU environment verification plus provenance evidence;
3. eligible-only token-length inspection over the real manifest;
4. deterministic publication of exactly 8 train and 8 validation examples with IDs and hashes;
5. Base inference on the frozen validation slice;
6. tiny QLoRA: batch 1 × accumulation 4 × two steps = all 8 training examples;
7. adapter save plus file hash manifest;
8. a separate Python process that reloads the base and adapter;
9. SFT inference on the identical validation prompts;
10. frozen safe SQL execution, both required metrics, and result validation.

The token report is still saved before selection. If either split has fewer than eight token-eligible
rows, the fixed selection lacks required feature coverage, or any selected row overflows/is excluded,
the same invocation fails. It never substitutes another row, seed, or count.

Do not add `--allow-model-download` unless the user separately approves the fixed 8.06 GB model
download. The retained M1 cache is expected but must be checked, not assumed.

## Expected time, cost, and capacity

- GPU: exactly one RTX 4090 D 48 GB.
- Additional free disk above retained dataset/model/environment: at least 20 GiB. Adapter and JSON
  evidence should be small, but this reserve covers caches and failed-run artifacts.
- Expected paid wall time after cache verification: 1.5–3.0 hours.
- Planning rate: CNY 2.59/hour, rechecked immediately before start.
- Expected cost: CNY 3.89–7.77 if the session remains within the honest 1.5–3.0-hour estimate.
- Script planning ceiling: 4 hours and CNY 10.36 if the freshly verified rate remains CNY
  2.59/hour. The larger ceiling is completion headroom, not permission to enlarge the experiment.
- The script enforces one cumulative 14,400-second monotonic deadline across all stages and
  terminates the active process
  group on expiry; it does not reset the allowance per stage.
- Project cumulative hard cap remains CNY 250; pre-M3 actual spend is CNY 9.64484428.

These are estimates. Provider actual cost must be copied back and appended to
`experiments/spend.csv`; measured wall-derived cost is not a substitute for the invoice.

Set the provider's automatic shutdown for about 4 hours 15 minutes after script start. This leaves
the 4-hour process budget plus a small buffer for final filesystem flush and evidence return. If the
provider accepts only whole-hour durations, 5 hours protects completion but increases runaway-cost
exposure; 4 hours is cheaper but may cut power exactly as the script reaches its own deadline. Record
the chosen tradeoff before launch. All outputs must be on the persistent cloud filesystem.

Evidence is saved incrementally: stage logs append continuously; preflight/progress/timeout,
token-report, slice, Base checkpoint, per-optimizer-step training progress, adapter manifest,
Base/train result, and final result use durable atomic JSON replacement. The adapter is first saved
to a same-filesystem partial directory and renamed only after save and update-evidence validation.
An interrupted run keeps completed evidence and any explicitly named partial adapter for diagnosis.

## Hard stops

Stop without changing seed, samples, prompt, evaluator, model, or limits if any of these occurs:

- provider price materially differs from CNY 2.59/hour;
- `M3_APPROVED_COMMIT` is absent/malformed, the source tree is dirty, or HEAD differs exactly;
- `make m3-verify`, GPU environment, or pinned package checks fail;
- model revision/cache identity or M2 manifest SHA-256 differs;
- any heldout path, row, prompt, or count enters the process;
- the output path exists, resolves inside the Git worktree (including through a symlink), is not
  absolute, or its actual filesystem has less than 20 GiB free;
- tokenizer has no working chat template or EOS token;
- either real token-eligible pool has fewer than 8 rows, or a fixed selected row has any
  token/schema overflow or exclusion;
- required join/aggregate/subquery coverage cannot be satisfied without changing seed;
- slice IDs/hashes drift between stages;
- Base/SFT comparison contract differs;
- loss or gradient norm is non-finite, supervised tokens are zero, optimizer steps remain zero, or
  adapter delta is absent;
- CUDA OOM, peak memory approaches device capacity without safe headroom, or training crashes;
- adapter file hash changes or fresh-process reload fails;
- evaluator reports a gold preflight failure or its limits differ from M2;
- cumulative wall time reaches 4 hours, projected/visible cost reaches CNY 10.36 at the verified
  CNY 2.59/hour rate, or provider billing cannot be inspected. A wall stop writes
  `session-timeout.json` before terminate/kill.

Do not retry with a different seed or easier slice in the same session. Preserve failure evidence,
record cost, and shut down.

## Return packet

Return these small artifacts; do not return databases, model cache, or heldout files:

- `token-length-report.json`;
- `slice.json`;
- `base-train.json`;
- `training-progress.json` and `adapter-manifest.json` when produced;
- `m3-result.json`;
- adapter directory or an access-controlled archive plus its file manifest;
- all `stage-*.log`, `preflight.json`, `session-completion.json` or `session-timeout.json`,
  environment, `nvidia-smi`, and package-freeze evidence;
- provider-visible actual cost and confirmation that the GPU was shut down.

The return report includes failure categories and explicitly states that heldout loaded/evaluated
counts are zero. Control validated that evidence, updated the registry/spend/state, and closed M3.
