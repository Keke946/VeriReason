# E5 prospective capture, audit, then update smoke

Owner: BiBo. Proposed experiment: `e5_capture_audit_smoke_001`.
Status: **CPU implementation/tests completed; GPU launch not authorized**.
Current research decision: **Revise**. Implementation receipt: `docs/E5_RUNNER.md`.
Prepared 2026-10-21. No validation or Mini-Dev payload may be opened in any stage.

## Historical evidence limitation

On 2026-10-21 BiBo reported an actual check by the local client of the original Mac archive
`artifacts/cloud-backup` in the TextToSQL project. It contains the M2 training manifest (6,601
source records), final E1/E3/E4 adapters, and SQL-free E2/E3/E4 training aggregates. It contains
no historical per-completion rollout SQL, original completion IDs, or usable replay bundle.
The runner did not enable completion logging. Historical real training rollouts are therefore
not recoverable from the retained evidence. This is a user/local-client observation, not a new
independent read of the Mac archive by this executor. Do not restart cloud merely to search again.

The 6,601 count is the upstream training-source manifest size, not proof that every record has
`split=train`: the frozen project partitions that source into training and validation. Any new
selection must use only operationally eligible `split=train` rows (the accepted pool is 5,364).
Neither adapters nor aggregate statistics can reconstruct original generated IDs. Re-encoded SQL
is not historical generated-ID evidence. Validation raw outputs cannot fill the missing evidence.

Real historical E5 localization and token-mapping coverage remain **unknown/null**. The synthetic
11/33 (33.33%) result remains an authored challenge-suite result only. The historical CPU JSON
receipts are preserved rather than rewritten after this external archive report.

## Fixed scope and data

- Fixed base: Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Parent: immutable final E1 at primary update 671; verify its existing adapter identity once.
  Never initialize from E2/E3/E4 or alter accepted E1-E4 artifacts.
- Reuse the eight outcome-blind training prompts from `configs/m7_e2_smoke.toml`: selector
  `m7_e2_smoke_hash_v1`, seed `verireason-m7-e2-smoke-v1`, join/aggregate/subquery coverage.
  Preserve their ordered identity. No reward-based replacement, seed shopping, or resampling.
- Use the frozen M2 manifest identity and loader, then filter `split=train` and eligibility before
  loading prompt/SQL payloads or SQLite files. Do not open a heldout manifest or validation raw
  file. The training-source manifest may contain split metadata; no validation row is consumed.
- Generate exactly four completions per prompt, 32 total, with the E2 smoke settings: temperature
  1, top-p 1, top-k 0, seed 23, maximum completion length 384, direct generation, no vLLM.
- Retained selected training databases and base weights must already be available at launch.
  Missing inputs stop preflight; this plan does not authorize a dataset/model download.

## Ordered execution boundary

1. Run the CPU/source/trainer preflight and verify the approved clean execution commit, fixed E1,
   training selection, evaluator contract, retained assets, current rate and cumulative deadline.
   A narrow TRL integration and evidence writer must first be implemented and CPU-tested locally;
   the current E5 analytical mock is not a launchable trainer.
2. Keep the policy at E1 and generate all 32 rollouts. Save each original completion-ID sequence
   before decoding, plus the exact decoded completion, prompt/example/group/rollout identity,
   completion mask, EOS/truncation metadata, and frozen-evaluator terminal reward. Persist each
   complete four-rollout group privately before proceeding. No optimizer step is permitted.
3. With weights still E1, compute and retain old-policy per-token log probabilities over the
   original IDs and the terminal-centered advantages. Run the unchanged E5 localizer and verified
   tokenizer mapping on these saved records. Uncertain cases retain exact E2 credit. Persist the
   audit and a signed-off gate decision before constructing/enabling optimizer updates.
4. Check every gate below. A failed or unreviewed gate terminates with `optimizer_steps=0`, an
   explicit stop reason, and intact private replay evidence. Do not silently train terminal-only
   instead, modify thresholds, or generate replacements in this experiment.
5. Only if every gate passes, replay these SAME 32 saved completions in their original eight-group
   order through at most eight updates, one group per update and one use per completion. No new
   generation. Retain the E1 old-policy log probabilities and clipped ratios throughout. Save the
   changed adapter and independently reload it. No formal run or evaluation follows automatically.

Generating everything before any update differs from historical E2's interleaved on-policy smoke.
Later groups use a fixed E1 behavior policy after the actor has changed. This is a bounded
fixed-rollout engineering experiment with clipped importance ratios, not an exact E2 causal
comparison or a quality result. Do not label it parity-locked formal E5. A future formal protocol
must separately decide how audit-before-update coexists with online generation.

## Preregistered pre-update gates

These are prospective engineering minima, not estimates fitted to validation or to the synthetic
33.33%. They deliberately allow this small smoke to stop for insufficient signal. Report all
counts, including all-zero/all-one groups, unsupported syntax, unsafe SQL and execution failures.

Define W as all terminal-incorrect rollouts out of 32, including execution errors, unsafe and
truncated cases; S as semantic localization candidates among W; M as candidates with trustworthy
token mapping; and U as mapped candidates in mixed terminal-reward groups with negative centered
advantage and at least one original valid target token. Rates are null if their denominator is
zero, which fails promotion. Localization coverage is M/W; abstention/fallback rate is (W-M)/W.
Also report semantic coverage S/W, token abstention (S-M)/S, U/W, and clause/reason counts. Do not
remove difficult completions from these denominators.

| Gate | Frozen minimum / maximum before the first update |
|---|---|
| Complete provenance | Exactly 8 eligible train prompts, 4 original-ID rollouts each; 32 aligned rewards and masks; zero validation/heldout payloads; all evidence persisted |
| Useful terminal signal | W >= 4 and at least 2 mixed terminal-reward groups |
| End-to-end coverage | M >= 2, M/W >= 20%; equivalently incorrect-case abstention <= 80% |
| Token mapping | M/S >= 80%; every unmapped candidate uses exact E2 fallback |
| Update-relevant localization | U >= 2, drawn from at least 2 distinct mixed groups |
| Error attribution review | Review every proposed span/mask against source SQL, gold, AST difference and strict execution repair: zero confirmed wrong-clause/incorrect-boundary/unsafe acceptances; unresolved candidate count = 0 |
| Terminal/fallback invariants | Binary strict reward unchanged; correct rows retain global positive credit; constant groups remain zero; nested/alias/unsafe/uncertain rows abstain; zero semantic-abstention overrides and zero fallback mismatches |
| Numerical/interface invariants | Loss, advantages, old/current log probabilities, importance ratios and logit gradients finite in preflight; padding/EOS untouched by local negative credit; saved-row/mask alignment exact |
| Scale control | Use local-unscaled only; target coefficients <= abs(E2 advantage), no N/K multiplier, original DAPO denominator; report N, K, N/K and credit-mass reduction for each localized row |

Review is a human/local-control review of private evidence before the update phase, bounded by the
session deadline; no silent auto-approval. It establishes reference-relative repair consistency,
not unique semantic causality on all databases. Review all S candidates, including mapping
abstentions, to avoid judging only convenient successes. A detected localization error or unresolved
attribution stops this smoke even if that row could be removed after looking at its outcome.
Execution failures and blocked unsafe SQL are ordinary abstentions; any unsafe SQL actually
accepted/executed, gate bypass, or evidence leakage is a hard stop.

The 20% coverage and two-group thresholds ensure a minimally exercised local path on this fixed
sample; they do not provide a precise population estimate. Report the fixed denominator and
uncertainty, never an accuracy or efficacy claim. Failed thresholds keep Revise; a reliable but
low-coverage sample does not alone prove global No-Go. Any revised threshold requires a new
prospective protocol, not relabeling this result.

## Credit and update contract

Keep E2's binary strict execution reward and `scale_rewards=none` centering:

    A_i = r_i - mean(r_j in group_i)
    a_it = A_i*m_it  if reliable localization and A_i < 0; otherwise A_i
    L = -sum_it v_it*min(rho_it*a_it, clip(rho_it, .8, 1.28)*a_it)/D

D remains the original valid-token denominator for the accumulation window, with no extra
gradient-accumulation division. Reuse E2's beta 0, learning rate 5e-6, batch 1, accumulation 4,
constant scheduler, single iteration and truncation masking. Only the existing `(B,T)` advantage
seam changes. The future runner must test buffer order/device/dtype and actual CPU autograd before
it is reviewable for launch; no live E5 trainer is enabled by this document.

Unscaled localization reduces negative credit mass by K/N. This is intentional for the engineering
smoke and confounds location with reduced total negative pressure; it cannot demonstrate that
localization itself improves learning. Compute terminal-only and N/K mass-matched loss/credit
diagnostics on the saved batch without additional optimizer updates. N/K remains diagnostic only
because it can amplify a very short suspected span. Credit mass is not model-parameter gradient
norm. No validation-based normalization choice is permitted.

## Time, memory, cost and stops

Planning only, assuming the already-retained compatible environment and inputs:

| Item | Estimate / boundary |
|---|---|
| Hardware | One already-approved topology class, RTX 4090 D 48 GB; no multi-GPU |
| Expected active session | 20-40 minutes including capture, bounded review, updates and reload |
| Expected peak memory | Approximately 12-20 GiB; historical E2 smoke measured 11,736 MiB, formal 19,804 MiB; E5 itself is unmeasured |
| Reference rate | Historical CNY 2.59/hour, must be reconfirmed before any launch |
| Expected GPU charge | CNY 0.86-1.73 at that reference rate, excluding disk/storage charges |
| Cumulative hard deadline | 60 minutes from paid start, including all preflight/review/failure time; auto-shutdown at or before that deadline |
| Budget ceiling | CNY 2.59 GPU container charge at the reference rate; reapprove estimate if price differs; actual invoice only enters spend.csv |

Gate failure, incomplete capture, missing assets, parent/selection mismatch, validation/heldout
access, evidence-write failure, wrong attribution, non-finite state, row drift, optimizer-before-
audit, OOM, or deadline/budget exhaustion stops immediately. No OOM retry with changed settings.
After updates, no finite nonzero adapter delta or failed independent reload means smoke failure.
Always stop resources after completion/failure; launch and provider shutdown configuration still
require a later explicit authorization. This design creates no cloud job or automatic continuation.

## Minimum durable evidence

Private mode-0600 files, outside Git/public artifacts:

1. Training selection: eight ordered example/group identities, `split=train`, eligibility, prepared
   prompt text and exact prompt IDs, source manifest identity, selected DB linkage, E1/base identity,
   generation config and seed. Reuse existing identity receipts; no per-stage duplicate hashing.
2. Replay: 32 completion texts and ORIGINAL generated IDs, group/rollout order, masks, EOS/truncation
   state, terminal reward/status, and E1 old-policy log probabilities. Flush after each group.
3. Pre-update audit: clause/character/token spans, mask or explicit fallback reason, repair/evaluator
   evidence, W/S/M/U denominators, per-clause errors, N/K and all gates; reviewer decision and
   `optimizer_steps_before_gate=0`. Stop receipts are required even with no updates.
4. If promoted: up to eight update records with finite loss/gradient/ratio telemetry, actual update
   count, adapter delta, final adapter identity and separate-process reload result. Retain the
   replay even when the update/reload fails.
5. Session: approved source commit, environment versions, phase timestamps, peak memory, cumulative
   wall time, stop reason, zero validation/heldout counters and eventual provider invoice.

Only SQL-free aggregates/gate decisions and references to existing artifact identities enter Git.
No question, SQL, prompt, example ID, database path, or generated token sequence is published.
The unique next action is local-control publication/review of this branch packet and plan; a
separately scoped CPU implementation and explicit launch approval are still needed before any GPU
session. This executor's push lacks HTTPS credentials, and the GitHub connector rejects tree
writes with 403 `Resource not accessible by integration`. A Git bundle preserves the original
local commits for publication by the authenticated local client; no remote branch is claimed.

## Design-packet verification

On 2026-10-21, exact CPython 3.11.15 `make verify` passed 431 tests with four optional-runtime
skips. The focused `tests/test_e5_credit.py tests/test_e5_interface.py` run passed 111 tests with
the retained real tokenizer asset. These regressions verify the existing CPU prototype; they do
not certify an implemented smoke runner. No diagnostic receipt was regenerated, no benchmark
payload was opened, and the spend ledger stayed unchanged.
