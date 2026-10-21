# E5 prospective smoke runner: CPU implementation receipt

Owner: BiBo. Experiment: `e5_cpu_runner_001`. Date: 2026-10-21.
Scope: local CPU implementation/tests only. No GPU/cloud/paid resource, model weights, real
training payload, validation or Mini-Dev was accessed. Overall research decision remains Revise.
The reviewed branch was published by local control at `786a89a`; this change does not touch main.

## Real tokenizer and loss interface

`verified_offsets` accepts both the actual HF `PreTrainedTokenizerFast` wrapper and its underlying
tokenizers backend. With HF, it checks exact wrapper/backend decoding (cleanup disabled), encodes
only to recover offsets, and requires those encoded IDs to equal the ORIGINAL generated content
IDs. It never replaces original IDs. Alternative segmentations, stale text, EOS/truncation/mask
ambiguity, bad offsets or meaningful BPE boundary crossings use the exact E2 fallback. Character
offsets target absolute occurrences, not a text search. Nested/alias geometry grants no semantic
localization; the unchanged execution localizer still abstains on unsupported structure.

`E5ReplayTrainer` subclasses the installed TRL 1.9.0 `GRPOTrainer`; it calls the real inherited
`_compute_loss`, not an extracted/analytical loss copy. `replay_batches` converts the eight saved
four-rollout groups from terminal-centered `(B,)` advantages to `(B,T)` before `_prepare_inputs`
slices each microbatch. Every other tensor is preserved: original completion IDs/masks, frozen E1
old log probabilities, prompt IDs and the original four-rollout window token denominator D.
The capture stores the original log-prob dtype and replay restores it, including bfloat16;
it does not silently promote the GPU importance-ratio calculation to float32.
`_generate_and_score_completions` rejects any call during replay. No stock buffer shuffle or
regeneration can change the recorded order. Only one process, DAPO, beta=0, token ratios, fixed
clipping, no entropy bonus/auxiliary loss, and the reviewed eight-step/four-accumulation path are
supported. Existing E2/E3/E4 runner code/config/rewards/evidence are unchanged.

The actual pinned TRL constructor supplies a non-None `compute_loss_func` sentinel; this disables
Transformers' extra loss/GAS division, while its Accelerator uses GAS=1. The subclass asserts that
contract. A real Trainer parameter-update test compares eight accumulated four-microbatch updates
against eight whole-group SGD updates. Separate autograd tests check scalar-E2 versus token-fallback
loss/gradients exactly, clipping below/inside/above the interval, target-only local logit changes,
positive terminal credit and microbatch/full-batch equivalence. CPU training uses a randomly
initialized one-layer tiny Qwen3 with small LoRA adapters; it downloads no model checkpoint.

The training formula and unscaled choice remain as preregistered:

    A_i = r_i - mean_group(r)
    a_it = A_i*m_it  for reliable localization with A_i < 0; otherwise A_i
    L = -sum(v_it*min(rho_it*a_it, clip(rho_it,.8,1.28)*a_it))/D

N/K scaling is diagnostic only. The pre-update backend runs actual TRL autograd for terminal,
local-unscaled and mass-matched controls, checking finite loss/logps/ratios/parameter gradients
and unchanged E1 adapter weights. No optimizer is created by this check. Unscaled local credit
reduces total negative pressure; credit mass does not equal parameter-gradient norm.

## State machine and private evidence

`SmokeSession` owns preflight -> capturing -> captured -> awaiting_review -> gated -> updating ->
finishing -> complete (or stopped). All 32 original-ID records are private/durable before audit;
all audit/review/gate evidence precedes optimizer creation. The preregistered W/S/M/U counts and
coverage, mapping, abstention, mixed-group, distinct-useful-group and zero-attribution-error
thresholds in `E5_SMOKE_PLAN.md` are unchanged. Runtime threshold overrides are not exposed.

Capture persists an intent before each model-generation call, and fsyncs a complete group before
the next one. Update persists an intent immediately before optimizer.step. An uncertain intent,
partial write, or interrupted update phase blocks automatic resume; it never regenerates or
repeats a possibly completed update. A completed capture group can resume at the next group;
an awaiting-review journal can resume audit review with the captured IDs and stored diagnostics.
The new actor is still immutable E1 on that path; original old-policy logps remain attached.
The summary distinguishes confirmed optimizer count from an uncertain pending side effect.

An exclusive session lock prevents two writers. Private directories must be 0700, replay JSON
0600, and outputs inside this repository must be ignored; outside-repository private paths are
also allowed. Atomic JSON writes reject NaN/Inf, flush/fsync then rename, leaving partial evidence
on a failed write. Neither raw exception strings nor SQL/IDs enter the whitelisted public summary.
`results/schemas/e5-private-replay.schema.json` describes the private journal. The writer never
copies replay into Git; all committed SQL/ID examples in tests are authored synthetic fixtures.

Human review uses a private JSON object with the journal's exact `session_id`, nonempty `reviewer`,
the exact ordered `candidate_rows` list (all S candidates, including mapping abstentions), and empty
`wrong_rows` / `unresolved_rows` lists for promotion. Wrong/unresolved rows fail the gate. No
automatic reviewer or public raw-review export exists. Omitting review exits at awaiting_review
with zero updates; it does not wait indefinitely on a billed instance or imply resource shutdown.
Local control must stop or bound the retained resource as specified by the existing compute plan.

## Entry point and prerequisite boundary

`scripts/run_e5_smoke.py --help` is safe and loads no model stack. The actual entry requires a
clean exact approved commit on `codex/e5-cpu-feasibility`, original paid-start timestamp, frozen
training manifest/prepared-train inputs, retained base/adapter/databases, expected E1 adapter
manifest, and private session directory. It provisions nothing and allows no download. A POSIX
deadline alarm bounds the remaining time in the original 60-minute window, including resumed
review. Changing the timestamp is rejected by resume context equality.

The existing M7 outcome-blind selector and manifest identity are reused. Only eligible train rows
are selected; a prepared bundle containing validation/heldout rows is rejected. The upstream
training-source manifest is the existing mixed train/development source container, not a reason
to consume validation rows. No validation raw file or heldout manifest is opened.

The HF backend directly calls model.generate for four completions at a time, captures its returned
IDs before decoding, retains EOS/padding/truncation, executes only the training database, and
explicitly computes E1 old-policy logps before any update. The latter is mandatory for fixed-rollout
replay even though stock TRL may omit old logps for ordinary single-iteration online training.

After updates, the adapter is saved and rehydrated in an independent CPU process into its declared
module shapes. Only LoRA weights are materialized on CPU; the base architecture stays on meta.
All reloaded adapter tensors must be finite and exactly match the saved tensors/manifest. This is
an actual independent adapter load, not a base+adapter forward/inference test. The CPU result does
not certify full quantized GPU reload or forward parity; keep that limitation in the launch review.

## CPU commands and evidence

Install CPU PyTorch separately from the CPU wheel index, then the optional pinned test stack in
`requirements/e5-trainer-cpu.txt`. That stack is separate from the small M10 lock because Datasets
4.7.0 requires an older fsspec range. No CUDA wheel is installed for these tests.

    make e5-runner-verify PYTHON=.venv-e5/bin/python
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 make verify PYTHON=.venv-e5/bin/python
    .venv-e5/bin/python -m ruff check src tests scripts
    .venv-e5/bin/python -m compileall -q src scripts tests
    .venv-e5/bin/python -m pip check
    git diff --check

The focused target requires the actual pinned runtime and retained fixed-revision Qwen tokenizer
file before running tests, so optional-runtime skips cannot masquerade as its passing receipt.
`results/e5/e5-cpu-runner.json` is produced by the end-to-end synthetic SQLite/HF/TRL test using
explicitly authored captured records. It records zero updates before gate, eight CPU LoRA updates
after gate, measured adapter delta, independent adapter hydration and finite real-autograd controls.
A separate test calls actual tiny-HF model.generate and verifies original returned ID retention;
that short random generation is not a real E1/BIRD rollout or a coverage measurement.

## Remaining decision

CPU interface/state-machine evidence supports submitting this implementation for GPU-smoke launch
review. It does not authorize a launch. Real E1/training localization coverage remains unknown and
is obtained only in the prospective pre-update stage. Quantized GPU memory/performance, 8-bit
optimizer behavior and independent full-base functional reload remain unmeasured locally. The
eight fixed E1 rollout groups still differ from historical E2's online-generation schedule.

The unique next action is local-control review of this implementation/CPU receipt and the retained
input/launch packet, followed by explicit authorization before any GPU capture or conditional update.

Verification receipt: `make verify` passed 489 tests and `make e5-runner-verify` passed 165 with
no skips. The synthetic end-to-end adapter delta L2 was 0.0003939752272278129; all recorded real
TRL/autograd controls were finite. Two expected PEFT warnings disclose that the random tiny base
has no pretrained checkpoint config path. Ruff, compileall, pip check and diff checks also pass.
