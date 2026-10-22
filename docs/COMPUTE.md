# Compute and Cost Policy

Last updated: 2026-10-21

## Known environment

- No local NVIDIA GPU is available.
- Cloud GPU rental is acceptable when a run has an explicit objective, estimate, stop condition, and experiment ID.
- Candidate provider options shown by the user include 24 GB and 48 GB 4090-class instances.

## Accepted Day 0 envelope

- Development/formal-training GPU: one 4090 D 48 GB instance at the reviewed rate of CNY 2.59/hour.
- Selected base model: `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Cheap fallback: `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Initial total cloud-compute target: CNY 180–250.
- Hard cap before a new decision: CNY 250.
- No single run estimated above CNY 30 starts without explicit user confirmation.
- The instance must be stopped when no training, evaluation, or serving benchmark is active.

The M1 session cost CNY 2.65115277, leaving CNY 247.34884723 under the CNY 250 core hard cap.
The provider price must still be rechecked before every deployment, and no paid restart is authorized
by this document alone.

After the completed M2 Day 3B work, measured M1+M2 cloud spend is CNY 9.64484428, leaving CNY
240.35515572 under the core hard cap before M3. This balance is accounting context, not automatic
authorization for a new paid session.

M3 Day 4A preregisters one vertical-slice session on the selected 1x RTX 4090 D 48 GB route. Its
planning range is 1.5–3.0 paid hours, CNY 3.89–7.77 at the previously observed CNY 2.59/hour, with
an automatic 4-hour planning ceiling of CNY 10.36 if that rate is reconfirmed. The extra hour is
completion headroom and does not expand the 8/8 experiment. It requires at least 20 GiB additional
free disk above retained
dataset/model/environment inputs. Counts are preregistered at 8 train and 8 validation: batch 1,
gradient accumulation 4, and two optimizer steps consume exactly two complete accumulation windows.
The single orchestrator first records real token lengths and fails rather than replacing a fixed
sample when eligibility, coverage, or overflow checks fail. It automatically enforces one cumulative
14,400-second wall deadline and the output-filesystem reserve. Persistent outputs are written
incrementally/atomically where practical, and provider auto-shutdown should be scheduled around
4 hours 15 minutes after launch to leave a small flush/return buffer. This paragraph records the plan and
does not authorize a cloud start. Exact commands and failure stops are in `docs/M3_RUNBOOK.md`.

M3 completed within the frozen 8/8 scope after two fail-closed compatibility attempts and one
successful attempt on the same paid cloud session. Attempt 003's orchestrator recorded 622.617
seconds from preflight through completion; that program time is not the provider's billed duration.
The user-verified provider screenshot contains two `container usage billing` transactions at
2026-08-24 04:00:17 Asia/Shanghai for CNY 1.43241388 and 2026-08-24 04:56:46 Asia/Shanghai for CNY
2.45042777. Their actual M3 total is CNY 3.88284165. The post-transaction provider balance was CNY
24.18879004; this account balance is not the project budget balance. Cumulative M1+M2+M3 actual
spend is CNY 13.52768593, leaving CNY 236.47231407 under the CNY 250 core hard cap. The raw result's
CNY 0.31852701 wall-derived estimate remains diagnostic only and is not used in accounting.

M7's completed E2 engineering smoke used 11,736 MiB peak GPU memory and 113.8 training seconds for
eight updates; its end-to-end orchestrated path took 313.6 seconds. Actual provider billing remains
pending, so no spend row is inferred from program time. The proposed formal E2 baseline scales to
256 prompts/updates and 1,024 rollouts. Linear training-only projection is 60.7 minutes; the planning
estimate is 1.5 hours and the hard stop is three hours/CNY 7.77 at the historical CNY 2.59/hour.
This is a preregistered ceiling, not launch authorization: current price and auto-shutdown must be
confirmed by the user before deployment.

Formal E2 subsequently completed 256 updates and 1,024 rollouts in 3,413.74 orchestrated seconds,
with 3,307.90 seconds inside the training stage and 19,804 MiB peak GPU memory. The final adapter and
evidence were downloaded before shutdown was requested. Program time is not substituted for billing;
the provider's actual charge remains pending and `experiments/spend.csv` is unchanged until supplied.

M8 Phase A reuses the accepted E1 full1209 evaluation and generates only the E2 condition. The
measured E1/SFT generation time was 6,437.54 seconds, so E2 plans two hours with a four-hour hard
wall ceiling. At the historical CNY 2.59/hour this is a CNY 10.36 maximum, not authorization. The
missing M7 invoice, current price, automatic shutdown, source identities, and explicit user approval
must be resolved before launch. E3/E4 training is not included in this Phase-A ceiling.

M8 Phase A subsequently completed its 1,209-example E2 evaluation with 57,638 generated tokens in
6,786.30 generation seconds (8.493 tokens/s). The successful process ran from approximately
2026-09-23 00:14:50 to 02:09:26 Asia/Shanghai, about 6,876 wall seconds including model loading and
report publication. One preceding attempt failed immediately before model computation because the
retained Hugging Face cache path was omitted; its log is preserved separately. Actual provider
billing is pending and is not inferred from either timing figure.

M8 E3 reuses E2's measured single-48-GB training topology. Its engineering smoke is eight prompts,
32 rollouts, and eight updates with a 30-minute/CNY 1.30 ceiling at the historical CNY 2.59/hour.
Only after that smoke passes may the separately preregistered 256-prompt/1,024-rollout formal E3 run
use the existing three-hour/CNY 7.77 ceiling. These are per-run bounds, not launch authorization;
the current provider rate, automatic shutdown, retained inputs, and fresh output path must be
confirmed first. Formal E3, E4, and validation are not bundled into the smoke session.

The E3 engineering smoke passed with 188.92 training seconds, 304.91 end-to-end session seconds,
and 11,736 MiB peak GPU memory. A first attempt failed before model loading because the retained
environment was missing the RECORD-listed `xgrammar 0.2.3` shared library; reinstalling that exact
package only restored the pinned stack. Program time is not provider billing. Actual smoke cost is
pending and `experiments/spend.csv` must not infer it from elapsed time. Formal E3 remains subject to
the separate three-hour/CNY 7.77 ceiling and a fresh launch confirmation.

Formal E3 subsequently completed 256 updates and 1,024 rollouts in 3,387.67 training seconds and
3,506.27 orchestrated seconds, with 19,804 MiB peak GPU memory. Its result JSONs were downloaded and
matched against their cloud SHA-256 identities; the adapter remains on retained cloud storage for
common validation. Program time is not billing, so the actual formal-E3 charge remains pending and
`experiments/spend.csv` is unchanged until the provider invoice is supplied.

E3 common validation generates only E3 on the already frozen full1209 exam. The measured E2 pass
used 6,786.30 generation seconds, so E3 plans two hours with the same four-hour/CNY 10.36 ceiling at
the historical CNY 2.59/hour. This is not launch authorization: the current rate, auto-shutdown,
clean execution commit, prepared selection, and retained adapter manifest must be confirmed first.

E3 common validation ran from 2026-10-01 20:07:04 to result publication at approximately 21:58:58
Asia/Shanghai, about 6,714 wall seconds including model loading and generation. It completed before
the provider's 22:30 automatic shutdown. Program wall time is not provider billing; actual cost
remains pending. The SQL-free summary is local and hash-verified. The 1.2 MiB raw result was retrieved
during the next already-approved start on 2026-10-02, without rerunning inference, and its local
SHA-256 matches the recorded cloud identity. The raw file remains ignored and outside Git.

E4 engineering smoke reuses E3 smoke's measured shape: eight prompts, 32 rollouts, eight updates,
and one 48 GB GPU. Its expected wall time remains about 10 minutes with a 30-minute hard wall; at
the historical CNY 2.59/hour the preregistered ceiling is CNY 1.30. This is not launch authorization:
the current price, automatic shutdown, clean accepted commit, retained inputs, exact E3 prompt-bundle
identity, and a fresh output directory must pass before model loading. Formal E4 is not included.

The E4 smoke completed in 314.72 end-to-end seconds, including 184.36 training seconds, and used
11,736 MiB peak GPU memory. One earlier attempt stopped before model load when the retained exact
`xgrammar` package lacked its native library; the same version was reinstalled without dependency
changes before the successful run. Program time is not provider billing, so actual cost remains
pending and `experiments/spend.csv` is unchanged.

E4 common validation reuses the same full1209 generation shape measured for E2 and E3. It plans two
hours and retains a four-hour/CNY 10.36 ceiling at the historical CNY 2.59/hour. Only E4 predictions
are generated; E1/E2/E3 results are reused. This is not launch authorization: current price,
automatic shutdown, accepted source commit, retained prepared bundle, formal E4 adapter identity,
and a fresh output directory must be confirmed first.

E4 common validation subsequently generated 56,662 tokens in 6,432.09 seconds at 8.809 tokens/s on
one RTX 4090 D 48 GB. The complete 1,209-example result was published before the provider's scheduled
shutdown, then hash-verified and retrieved during the next start without rerunning inference. E1/E2/E3
were reused rather than regenerated and heldout use was zero. Program time is not provider billing;
the actual charge remains pending and `experiments/spend.csv` is unchanged.

M9 serving is preregistered as one short session on the selected 1x RTX 4090 D 48 GB topology. It
statically serves the retained E2 rank-16 adapter, performs two warmups, then measures 64 fixed
validation requests at concurrency 1, 4, and 8. The session plans 30 minutes and has a one-hour hard
wall; at the historical CNY 2.59/hour the provisional session ceiling is CNY 2.59. This is not launch
authorization. The current provider price, automatic shutdown, clean approved commit, retained base
model/cache, E2 adapter identity, prepared full1209 bundle, at least 5 GiB output space, and a fresh
private output directory must all pass before vLLM starts. Mini-Dev remains sealed and heldout use
must stay zero.

M9 attempt 003 subsequently completed the endpoint benchmark. Server readiness took 155 seconds;
the three measured scenarios took 36.96, 14.20, and 9.83 seconds. Ready-state memory was 28,972 MiB
and peak memory was 29,246 MiB. Attempts 001 and 002 stopped before endpoint readiness and made no
benchmark requests; the successful attempt used explicit no-prefix-cache and Torch-sampler settings.
Program time is not provider billing, so the actual charge remains pending and is not inferred here.

M10 used CPU-only local verification. It created a fresh exact Python 3.11.15 environment from a
Git-only archive and loaded no GPU package, model, database, validation raw row, or heldout data.
Cloud/GPU cost for M10 is CNY 0. The Docker build is checked into CI but was not run locally because
Docker is unavailable on the Mac.

Formal E4 reuses formal E3's measured one-48-GB-GPU shape and 256-prompt/1,024-rollout budget;
formal E3 peaked at 19,804 MiB. E4 therefore plans 90 minutes and retains a three-hour/CNY 7.77
ceiling at the historical CNY 2.59/hour. This is not launch authorization. Current price,
automatic shutdown, clean accepted commit, immutable E1 adapter, retained M2 data, exact formal-E3
prepared-bundle identity, and a fresh output directory must be confirmed before model loading.

Formal E4 subsequently completed on one RTX 4090 D 48 GB with the same 19,804 MiB peak. Training
took 3,314.85 seconds and the complete successful session took 3,436.65 seconds. Two prior attempts
stopped before model load because of a database-root invocation error and a missing provider-managed
native-library symlink target. Program time is not provider billing; actual provider charges remain
pending and `experiments/spend.csv` is unchanged.

For M2 Day 3B specifically, CNY 12.95 is one cumulative hard stop across all attempts. It does not
reset when an instance is restarted. Before a second start, the provider's actual first-session
charge must be visible and recorded as gate evidence, and:

```text
remaining_hard_stop_cny = 12.95 - verified_first_session_actual_charge_cny
```

If the actual charge is unavailable or the planned second session cannot fit the remaining amount
at the freshly verified hourly rate, the instance must remain stopped. The final combined Day 2–3
spend row may be appended after closure, but delayed ledger entry does not waive this launch gate.

The first two verified Day 3B attempts used CNY 1.00987346 and CNY 1.78134210. The approved
failure-only diagnostic then used CNY 2.70870832 of container time; conservatively including CNY
0.00277268 of visible disk charges gives CNY 2.71148100 for that attempt and CNY 5.50269656
cumulative Day 3B cost. The remaining hard-stop allowance is CNY 7.44730344. The instance is shut
down after the diagnostic. Per control instruction, the combined Day 2–3 row is appended to
`experiments/spend.csv` only when M2 closes; these verified figures remain binding before any later
approval.

The final evidence-only publication session incurred CNY 1.48709165 in container charges and CNY
0.00390330 in visible retained-disk charges, totaling CNY 1.49099495. Across all M2 Day 3B attempts,
container charges were CNY 6.98580551 (2.697222 billed hours at CNY 2.59/hour) and conservatively
included visible disk charges were CNY 0.00788600, for CNY 6.99369151 total. This remained below the
CNY 12.95 Day 3B hard stop. The instance is confirmed shut down, M2 is closed, and the combined
session is recorded in `experiments/spend.csv`. No archive download, extraction, or gold-SQL
execution occurred during final publication.

## Topology gate: 48 GB versus multiple 24 GB cards

At the displayed prices, assuming the provider charges linearly per selected card (to be rechecked before deployment):

| Configuration | Displayed cost/hour | Planned role |
|---|---:|---|
| 1x4090 D 48 GB | CNY 2.59 | Default: simplest colocated 1.7B–4B QLoRA/GRPO path |
| 1x4090 24 GB | CNY 1.89 | Inference, evaluator, SFT smoke, or low-cost fallback |
| 2x4090 24 GB | CNY 3.78 | Candidate: one rollout/vLLM GPU plus one trainer GPU |
| 4x4090 24 GB | CNY 7.56 | Optional approved 7B confirmation only |
| 7x4090 24 GB | CNY 13.23 | Not planned; card count does not automatically create one 168 GB memory pool |

M1 selected one 48 GB card. The 4B inference smoke peaked at 8158 MiB, but inference memory does not
predict QLoRA or GRPO memory: training adds activations, gradients, adapters, optimizer state, multiple
completions, and potentially a colocated generation engine. A later topology change requires a
registered training/rollout measurement and a quantified bottleneck, not extrapolation from this
inference result.

Multi-GPU memory is useful only when the selected framework actually shards or parallelizes the relevant state. M1 must record whether the provider places the GPUs in one host and the available PCIe/NVLink topology before authorizing a multi-GPU run.

## Estimated usage plan

| Stage | Estimated GPU hours | Cost at CNY 2.59/hour | Spend gate |
|---|---:|---:|---|
| Environment and tiny inference | 2–3 | 5–8 | User deploys after image review |
| Baseline and vertical slice | 4–8 | 10–21 | Experiment ID required |
| Formal 1.7B–4B SFT and one retry | 6–14 | 16–36 | Confirm if a single run exceeds CNY 30 |
| Outcome-only RL smoke and formal run | 16–28 | 41–73 | Stop criteria and checkpoints required |
| Dense-reward/GDPO-style research runs | 12–24 | 31–62 | Freeze matrix and early-stop rules first |
| Evaluation, selected ablations, serving | 8–16 | 21–41 | Batch related work while instance is active |
| Contingency | 8 | 21 | Requires documented reason |

Expected core range before contingency: approximately 48–93 single-48-GB wall-hours, or CNY 124–241 at the displayed rate. Contingency must replace or compress other planned work rather than push spending above the cap. These are planning estimates, not measured facts. If M1 selects two 24 GB cards, the accounting uses the CNY 3.78 configuration-hour rate rather than pretending the second GPU is free.

Any post-core upgrade has a separate estimate and approval. A preliminary planning band is CNY 100–200 for U1/U2 together and CNY 100–250 for U3, but neither amount is approved by this document.

## Storage and environment cautions

- Do not deploy until a compatible PyTorch/CUDA image is selected in M1.
- The shown 100 GB system disk is tight for environments, multiple checkpoints, databases, and caches. M1 must either use the persistent cloud path with sufficient capacity or download only selected dataset shards; important logs and adapters live on persistent storage.
- Download caches, intermediate checkpoints, and raw generations must be pruned deliberately, never with broad destructive commands.
- Secrets belong in environment variables or an ignored `.env`, never Git.

## Spend accounting

Post-v1 E5 `e5_capture_audit_smoke_001` completed its real-rollout gate with zero optimizer updates
and is closed as a negative feasibility result. Its actual provider charge is still pending and
must be recorded from the provider invoice rather than inferred from program time.

E6 `e6_execution_repair_probe_001` is an inference-only paired probe on the retained 32 private E5
states: 64 feedback and 64 no-feedback repair completions, zero training. Planning estimate is
20-40 minutes on one retained 48-GB 4090 D, with a 60-minute program ceiling and CNY 2.59 reference
ceiling at the historical rate. Reconfirm price and retained inputs before model load. No restart,
sample replacement, or larger run is authorized by a failed or incomplete probe.

Every paid session is appended to `experiments/spend.csv` with provider, GPU, purpose, experiment IDs, start/end time, and actual cost. Planned and actual cost are separate fields.
