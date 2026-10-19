# M4 Formal SFT Pre-cloud Runbook

Last updated: 2026-09-01. M4 is `pre-cloud / pending`: these are instructions for a later
control-approved GPU session, not authorization to create one. No heldout, cloud, GPU, download,
Day 5B, M5, or RL occurred in Day 5A.

## Frozen primary experiment

Run exactly all 5,364 operationally eligible train rows once: 670 full eight-microbatch windows
plus a separate final window of four real and four zero-gradient padding records, or 671 updates.
The existing one-epoch cosine schedule has 671 total steps and 21 warmup steps. There is no
automatic second epoch, early stopping, or validation checkpoint selection. Update 671 is the
atomically published primary E1 adapter.

Before token inspection, the cloud session runs `scripts/verify_m4_trainer_contract.py` in required
mode. It must import the exact `transformers==5.14.1` wheel, inspect the relevant `Trainer.training_step`
source branch, save its source SHA-256 and wheel versions in `trainer-contract.json`, and then the real
torch regression must pass. Qwen3's `**kwargs` would otherwise make
`model_accepts_loss_kwargs=True`, which skips the standard divide-by-eight path when M4's custom loss
does not consume `num_items_in_batch`. M4 fail-closed overrides that flag to false after asserting the
expected inference, so ordinary windows are 8-example means and the four-real/four-zero-pad window is
a 4-example mean. A missing/inexact local wheel is reported as `unavailable_not_verified`, never as a
passing regression; required cloud preflight treats it as fatal.

The CPU-only Trainer regression uses the same outer-output / inner-`checkpoints` helper as formal
training. A past closure failure was a test-fixture path mismatch, not a formal adapter path change;
the test still requires non-null Trainer log metrics, elapsed time, `m4-contract`, and checkpoint
manifest after that shared-path assertion.

At updates 512 and 671, retain complete checkpoints and calculate dev256 teacher-forced,
answer-only, token-weighted mean loss. `epoch-extension-decision.json` records those raw losses,
training loss window, gradient norm, LR, NaN/Inf state, elapsed time and checkpoint hash. It also
reports `(loss512-loss671)/loss512`; 1% is an advisory reference only. The program always stops at
671. Epoch 2 needs a later control review of that packet and a separately frozen extension
LR/scheduler/budget; it may not overwrite or be represented as this primary experiment.

`dev256` is diagnostic only. Generate E0 Base and E1 SFT once each on full1209. The report's
final953 view is the full result with dev256 IDs removed, not a third generation run. Full1209 is
the primary M4 comparison; all views use the same frozen prompts, greedy decoding and evaluator.

## Checkpoint and recovery

Save at normal 64-update boundaries and force saves at 512/671; retain four checkpoints so both
diagnostic checkpoints survive. A checkpoint includes adapter, optimizer, scheduler, RNG, Trainer
state and matching M4 contract. Resume only from a complete matching primary-epoch checkpoint. It
skips completed eight-microbatch windows, so no completed real row is repeated. The four final rows
never mix with a future epoch.

Stop for contract drift, heldout nonzero, NaN/Inf, repeated OOM after one safe fragmentation retry,
or wall/cost hard stop. Do not solve a failure by deleting rows, changing seed/hyperparameters, or
starting epoch 2. Preserve the output directory and resume the primary checkpoint.

## Time, storage, cost

M3 measured 0.762684 seconds per training microstep: primary training is a linear estimate of
4,091 s / 1.14 h. M3 measured 24.289 s/Base evaluation example and 5.407 s/SFT evaluation example:
the one unique full1209 E0/E1 generation is 8.16 + 1.82 = 9.97 h. Two teacher-forced dev256 passes
are not measured; using M3 training microstep time as a conservative forward-pass proxy gives at
most about 0.11 h. Thus the measured/linear raw total is about 11.22 h, plus unknown preparation,
checkpoint I/O, reload and shutdown overhead. Reserve 12 h; hard stop 14 h. At the historical
CNY 2.59/h this is CNY 31.08 expected and CNY 36.26 maximum. Recheck provider price and obtain
control approval because both exceed the CNY 30 gate. Keep ≥30 GiB persistent free space and set
provider auto-shutdown around 14 h 15 min.

Reuse the frozen M2 data disk, local model cache and M3 environment. Do not download replacements.

## Command after explicit control approval

```bash
git checkout --detach CONTROL_APPROVED_SHA
export M4_APPROVED_COMMIT=CONTROL_APPROVED_SHA
export M4_DATASET_ROOT=/root/verireason-m2-data/processed/m2-v1
export M4_MANIFEST_INDEX="$M4_DATASET_ROOT/manifest-index.json"
export M4_TRAINING_MANIFEST="$M4_DATASET_ROOT/training-manifest.json"
export M4_OUTPUT_DIR=/root/verireason-m4/e1_m4_formal_sft_001
sh scripts/run_m4_cloud.sh
```

The one orchestrated session verifies environment/token eligibility, freezes manifests, trains or
resumes the primary epoch, writes the decision packet, performs the two unique full1209 generations,
validates same-exam identity and publishes an SQL-free summary. Stop when it completes.
