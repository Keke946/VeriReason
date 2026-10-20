# Fifteen-Day Execution Plan

Status values: `pending`, `in_progress`, `blocked`, `completed`.

| Milestone | Day | Status | Primary output | Acceptance gate |
|---|---:|---|---|---|
| M0 Scope and operating system | 0 | completed | Frozen charter, research question, model envelope, compute budget | All Day 0 decisions recorded; `STATE.md` has no scope blocker |
| M1 Repository and environment | 1 | completed | Locked environment, project package, tests, model/hardware smoke gate | Clean install succeeds; candidate models complete tiny inference; 48 GB versus optional 2x24 GB route is decided from measurements |
| M2 Data and evaluation | 2–3 | completed | Versioned data pipeline, sandboxed verifier, frozen splits, leakage checks | Pipeline is deterministic; verifier positive/negative/adversarial tests pass; split manifest saved |
| M3 Vertical slice | 4 | completed | Base baseline and tiny SFT end-to-end run | Data -> train -> evaluate completes and produces a registry entry |
| M4 Formal SFT | 5–6 | completed | QLoRA/SFT checkpoint, logs, comparison | Full-5364×1 / 671-update primary run, exact Trainer preflight, and unique full1209 E0/E1 comparison are complete and audited; provider invoice remains an accounting gap |
| M5 Error analysis and data v2 | 7 | completed | Error taxonomy, audited examples, data revision | Failure classes quantified; data changes justified and versioned |
| M6 Reward system | 8 | completed | Terminal and reliability-gated dense rewards with adversarial tests | Positive, negative, malformed, unsafe, equivalent-query, and exploit cases pass tests |
| M7 GRPO post-training | 9–10 | completed | Outcome-only RL smoke and formal run | Stable run produces a checkpoint; reward hacking review completed |
| M8 Research comparison | 11–12 | completed | Dense-reward and multi-reward-normalization runs | The frozen E1–E4 ladder has measured same-exam accuracy, paired comparisons, sample-efficiency evidence, and reward diagnostics |
| M9 Serving and benchmark | 13 | completed | Efficient E2 API and serving report | Correctness checked; memory, TTFT/latency, and throughput recorded |
| M10 Reproduction and engineering | 14 | completed | Docker/Make/CI and clean-room smoke reproduction | A fresh environment follows README successfully |
| M11 Release and interview gate | 15 | completed | Report, cards, demo, resume bullets, mock-interview curriculum | Technical definition-of-done items 1–8 pass; outstanding personal teach-backs remain tracked separately in `docs/INTERVIEW.md` |

M2 completed its fixed-source materialization, failure-only diagnostic, versioned resource
amendment, and evidence-only publication. The finalizer published manifests under the 10-second
wall/12-second CPU and unchanged 100k-row/64-MiB/2-GiB limits, proving 6,496/28 training
eligibility, 498/2 heldout eligibility, 5,364/1,209 eligible train/validation examples across 59/10
databases, complete critical-feature coverage, 80/80 SQLite integrity, zero hard cross-split
leakage, private permissions, and unchanged source database identity. Gold SQL was not re-executed
during publication. M2 is complete; M3 subsequently completed as recorded below.

M3 completed its preregistered 8/8 engineering slice at execution commit
`530a0f22f491882def5e13b8344d1310c1a7960d`. Real token inspection preserved all 5,364/1,209
eligible train/validation candidates and the fixed slice covered join, aggregate, and subquery.
Tiny QLoRA completed eight microsteps and two optimizer updates with positive gradient norm and
adapter delta; the adapter was hash-verified and evaluated from a distinct process. Base and SFT
used the identical eight validation rows, prompt hashes, greedy decoding, and unchanged M2
evaluator. Base/SFT behavior-compatible and strict execution were 0/8 and 3/8; SQL-only rates were
0/8 and 8/8. Heldout use was zero. The complete session lasted 622.617 seconds; the actual provider
invoice was CNY 3.88284165. M4 remains pending and receives no authorization from this closure.

M4 completed its pre-registered primary epoch at execution commit
`f187167a882bf6b9fd2b152220fabbf770e92960`. The fixed exact-wheel preflight passed, then one
full eligible epoch completed 671 updates with final-four-window loss scaling and four retained
checkpoints. On the same unique 1,209 validation examples under greedy decoding and the unchanged
M2 evaluator, Base scored 0.00% strict execution accuracy and SFT scored 40.69%; SFT behavior-compatible
accuracy was 42.35%, valid-SQL rate 87.59%, and SQL-only rate 100%. Heldout use was 0/0. The 512→671
dev256 teacher-forced loss reduction was 0.18%; it is advisory evidence only, so epoch 2 is not run.
The archived session ran 41,804.54 seconds. Provider invoice cost is still pending as an explicit
accounting gap, but it no longer changes the already audited M4 technical result.

The user subsequently authorized local, no-cost M5 analysis while the provider invoice remains
temporarily unavailable. The first SQL-free aggregate pass is complete and identifies executable
wrong-result cases as the dominant failure class. M5 remains in progress until the deterministic
private audit and justified data-v2 revision are complete. The private audit is complete except for
three database-dependent equivalence checks, and a deterministic 1,678-row training-only candidate
pool is now versioned. Its consumer/training policy is not yet frozen. No M5 training, heldout
access, or paid resource has been authorized.

M5 subsequently froze the consumer roles without prematurely freezing training parameters. `E1T`
is a bounded targeted-SFT control starting from E1; E2–E4 independently start from the same E1;
Correction SFT is a trigger-only backup. The three unresolved database-dependent ambiguity cases
remain quarantined evidence debt and do not enter data-v2. M5 is technically complete and M6 begins
with local reward implementation only; cloud and heldout remain unauthorized.

M7 has completed its bounded E2 engineering smoke at execution commit `91a95ba`. Eight training
prompts produced 32 rollouts and eight optimizer updates; loss remained finite, maximum gradient
norm was 0.6484, LoRA delta L2 was 0.1334, peak memory was 11,736 MiB, and the saved adapter passed
a separate-process reload. Heldout use was 0/0. Only 2/8 groups were mixed-reward while 6/8 had zero
reward variance, so the smoke proves the training path but not an accuracy gain. M7 remains in
progress until the formal E2 run and reward-hacking review are complete.
The proposed formal E2 packet fixes 256 outcome-blind training prompts, 1,024 rollouts, checkpoints
every 64 updates, and a post-run 32-group/12.5% signal promotion gate. Its aggregate telemetry is
designed for a bounded reward-hacking review. This local preregistration is not cloud authorization.
Formal E2 subsequently completed all 256 updates and 1,024 rollouts at execution commit `98ea3c0`.
It produced 102 mixed groups (39.84%), passed the signal gate, changed and independently reloaded the
adapter, used 19,804 MiB peak GPU memory, and accessed zero validation/heldout examples. The bounded
reward-hacking review passed with one unsafe and one resource-limit completion safely blocked and
no truncation, entropy collapse, or length explosion. M7 is complete; quality comparison moves to M8.

M8 Phase A is locally preregistered. It reuses the accepted E1 full1209 result and evaluates E2 once
on the identical frozen validation selection, prompt contract, greedy decoding, and M2 evaluator;
it does not waste GPU time regenerating E1 or access heldout. Existing raw E1 evidence is used for
paired transitions if recoverable without regeneration. E3 and E4 remain unstarted until this common
comparison is complete and the E2-specific M7 runner has been deliberately generalized and tested.

M8 Phase A subsequently completed at execution commit `95da887`. On the identical full1209 exam,
E2 improved strict execution accuracy from 40.69% to 42.35%, behavior-compatible accuracy from
42.35% to 44.25%, and valid-SQL rate from 87.59% to 90.74%, while SQL-only remained 100%. Paired
strict transitions were 19 regressions versus 39 improvements (exact McNemar p=0.01193). Heldout
use remained zero. M8 stays in progress while E3/E4 are separately generalized, tested, and run.

The E3 execution packet is now locally preregistered. Its smoke and formal configurations reuse
E2's immutable E1 parent, exact hash-selected prompts, four-completion groups, decoding, optimizer,
seed, and update budgets; only the reward arm changes from terminal-only E2 to reliability-gated
sum-then-normalize E3. The next gate is the eight-prompt E3 engineering smoke, not formal training
or validation.

The smoke acceptance path is also locally closed: aggregate training evidence and the independent
reload result are reconciled automatically, including adapter identity, rollout/component counts,
reward semantics, update signal, resource bounds, and zero heldout use. A successful subprocess
exit without consistent evidence cannot publish a passing smoke summary.

The E3 engineering smoke subsequently passed at execution commit `5579446`: eight prompts, 32
rollouts, and eight updates completed with 7/8 mixed-reward groups, one zero-variance group, a
nonzero adapter delta, matching independent-reload identity, 11,736 MiB peak GPU memory, and zero
heldout use. An initial pre-model import failure exposed a missing `xgrammar` shared library and was
repaired by reinstalling the exact locked package only. The smoke establishes systems and signal
viability, not model-quality improvement. Formal E3 remains a separate approval and execution gate.

Formal E3 subsequently completed at execution commit `1732ea1` under the controlled E2-matched
256-prompt/1,024-rollout/256-update budget. It produced 239 mixed-reward groups and 17 zero-variance
groups, changed and independently reloaded its adapter, used 19,804 MiB peak memory, and accessed
zero validation/heldout examples. This establishes dense-signal training viability, not model
quality. The next M8 gate is the same-protocol full1209 E3 validation; E4 remains unstarted.

E3 common validation subsequently completed at execution commit `9a5f8de`. On the identical
full1209 exam, E3 reached 41.69% strict execution accuracy: +0.99 points versus E1 (paired p=0.08843)
and -0.66 points versus E2 (p=0.27996). Behavior-compatible accuracy was 43.42%, valid-SQL rate was
89.83%, SQL-only remained 100%, and heldout use was zero. The dense reward substantially increased
mixed training groups but did not outperform outcome-only E2. M8 remains in progress because E4 is
the preregistered test of whether per-component normalization changes that result.

The E4 engineering-smoke packet is now locally preregistered at eight prompts, 32 rollouts, and
eight optimizer updates. It starts from immutable E1 and mechanically preserves the E3 smoke
contract except for experiment identity, arm, and `normalize_then_sum` aggregation. It cannot
authorize formal E4 or heldout access.

The E4 smoke subsequently completed all eight prompts, 32 rollouts, and eight optimizer updates,
with seven mixed groups, a changed adapter, independent reload, and zero validation/heldout access.
Formal E4 then completed the parity-locked 256-prompt, 1,024-rollout, 256-update run from immutable
E1. It produced 235 mixed groups, changed and independently reloaded its adapter, used 19,804 MiB
peak memory, and accessed no validation or heldout examples. This closes E4 training only; the next
gate is the unchanged full1209 common-validation exam before any E3-versus-E4 quality claim.

The E4 common-validation packet is now locally preregistered. It generates only E4 predictions on
the existing 1,209-example selection, requires the accepted formal adapter identity, and reuses all
E1/E2/E3 results. Existing raw rows may add paired statistics but cannot be regenerated. Launch is
a separate approval; Mini-Dev remains sealed.

E4 common validation subsequently completed at execution commit `eb1e132`. On the identical
full1209 exam, E4 reached 41.4392% strict accuracy, 43.3416% behavior-compatible accuracy, 89.4955%
valid SQL, and 100% SQL-only output. It was +0.7444 points over E1, -0.9098 points versus E2, and
-0.2481 points versus E3 in strict accuracy; paired p-values were 0.2110, 0.1081, and 0.7283.
Heldout use remained 0/0. E2 is therefore the best validation arm, and the preregistered E3/E4
dense-feedback interventions are accepted as controlled negative results. M8 is technically
complete; its evaluation teach-back remains tracked as interview-readiness debt.

M9 local preparation is now authorized while the M8 teach-back remains tracked as interview debt.
The execution packet statically serves the best measured E2 adapter through vLLM, benchmarks a
deterministic 64-example validation workload at concurrency 1/4/8, measures client-observed TTFT,
latency, request/token throughput and GPU memory, and sends concurrency-1 outputs through the frozen
SQLite evaluator. Paid execution remains separately gated.

M9 subsequently completed from execution commit `b47fa46`: 192/192 measured endpoint requests
succeeded across concurrency 1/4/8, peak memory was 29,246 MiB, and the 64-example serving slice
scored 46.875% strict execution accuracy with heldout 0/0. The result is a serving/correctness
integration check, not a replacement for full1209 model selection. M10 now owns clean-room
installation, Docker/Make/CI entry points, and README-only smoke reproduction.

M10 subsequently completed at implementation commit `392b8a3`. A Git-only archive was installed in
a new exact Python 3.11.15 virtual environment and passed 314 tests, Ruff, compileall, the local
environment check, `pip check`, and deterministic validation of 19 allowlisted SQL-free evidence
files. Docker and CI entry points use the same CPU-only lock; local Docker execution was unavailable,
so the container build remains a first-public-CI check rather than an unreported local result. M11
now owns the technical report, cards, demo/release surface, resume packet, and mock-interview gate.

M11 published GitHub v1.0.0 with license/citation metadata, public README,
technical report, model/data cards, reproducibility guide, follow-up literature review,
release/interview packet, an E2 demo CLI, and a gate that locks headline documentation to accepted
result JSON. Public commit `dc133fc` passes exact Python 3.11.15 and Docker CI; the release contains
the selected E2 adapter with a GitHub-verified SHA-256. Final E1/E3/E4 adapters and the private M2
manifest/receipt evidence are now backed up locally without copying reconstructible databases,
base weights, caches, or intermediate checkpoints. M11 is technically complete. Outstanding
teach-backs remain explicit interview-readiness debt in `docs/INTERVIEW.md` and will be handled as
a separate learning task. No E5 experiment is authorized by publication or archival.

## Fifteen-day core experiment ladder

The exact step budget is frozen after M3. Each level leaves a complete artifact, so a later research failure cannot erase the earlier result.

1. `E0`: base model under the frozen evaluator.
2. `E1`: QLoRA SFT versus `E0`.
3. `E2`: outcome-only RL from `E1`, using terminal execution correctness.
4. `E3`: terminal plus reliability-gated dense structural rewards with standard sum-then-normalize aggregation.
5. `E4`: the same dense rewards with per-reward normalize-then-sum aggregation (GDPO-style).

This is a three-arm causal ladder, not a wasteful grid: `E2 -> E3` isolates the effect of dense feedback, while `E3 -> E4` isolates per-reward normalization. A one-reward GDPO cell would be mathematically redundant and will not be run. If the smoke budget shows three formal RL runs are infeasible, the predeclared minimum is `E2` versus `E4`, while `E3` becomes a short diagnostic run. Base, SFT, and outcome-only RL are never removed.

Primary outcome: held-out execution accuracy. Secondary outcomes: valid-SQL rate, robust/test-suite accuracy where available, performance by SQL difficulty, zero-reward-variance group fraction, entropy/KL/length diagnostics, and examples or GPU-hours required to reach a fixed validation threshold.

## Research upgrades after the core gate

Post-v1 E5 CPU-only feasibility gate (2026-10-20): implementation and authored synthetic audit
completed, **Revise** for promotion. `docs/E5_DESIGN.md` records the bounded support surface and
`results/e5/e5-cpu-feasibility.json` the measured 11/33 incorrect-case localization coverage,
22/33 abstention, and zero expectation mismatches. The next gate remains CPU-only tokenizer/loss
integration plus training provenance/coverage evidence. No 8-prompt GPU smoke is authorized.

These are sequential options, not promises hidden inside the 15-day schedule.

| Upgrade | Extra time | Entry gate | Research content | Failure-safe outcome |
|---|---:|---|---|---|
| U1 Clause-aware credit and informative-group sampling | 4–6 days | E2 and E4 are stable and the reward parser passes adversarial tests | EXPO-SQL-inspired clause/token credit plus DAPO/ReViSQL-inspired filtering of constant-reward groups | Core report remains complete; publish the failed/negative ablation |
| U2 7B confirmation | 2–4 days | At least one 4B intervention is positive or scientifically informative | Re-run only the decisive comparison at 7B | 4B result remains the main claim |
| U3 Multi-turn execution-correction agent | 5–8 days | Core release is already tagged | SQL-ASTRA/Progress-SQL/SERL-SQL-inspired progressive rewards and localized action credit | Release as a separate v2 branch/milestone |

## Time-compression rules

- Preserve stages; reduce model size and experiment breadth before deleting a stage.
- Run tiny smoke experiments before full jobs.
- Prepare evaluation and documentation while training jobs run.
- Stop low-signal runs using predefined criteria rather than waiting for full completion.
- Prefer four decisive ablations over a broad grid search.
- Do not implement U1–U3 before the Day 15 release path is green.

## Phase-task workflow

1. The control task scopes one milestone and confirms its acceptance gate.
2. A focused execution task implements only that milestone.
3. Independent read-heavy research or review may be delegated when explicitly requested.
4. The execution task verifies, records evidence, and updates project state.
5. The control task accepts or rejects the milestone before the next task starts.
