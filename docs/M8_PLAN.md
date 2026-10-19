# M8 Research Comparison Plan

Status: completed; E1/E2/E3/E4 training evidence and common validation are recorded.

## Question and order

M8 first asks whether formal outcome-only RL (E2) changed model quality relative to its immutable
SFT parent (E1). Only after that common evaluation is recorded do E3 and E4 run from the same E1
parent. This keeps three different questions separate:

1. `E1 -> E2`: did terminal execution reward help?
2. `E2 -> E3`: does reliability-gated dense structural feedback add useful learning signal?
3. `E3 -> E4`: does normalizing reward dimensions separately improve optimization?

## Phase A — one-exam E1/E2 comparison

E1 is not regenerated. Its accepted full1209 result is reused. E2 is evaluated once on the exact
M4 full1209 validation selection (`3dc2fcbd...f0b1`), using the same deterministic prompts, greedy
decoding, 384-token limit, and frozen M2 evaluator. Mini-Dev remains untouched.

The primary metric is strict execution accuracy. Secondary metrics are behavior-compatible
execution accuracy, valid-SQL rate, SQL-only rate, and failure categories. Training reward is not
an evaluation metric and is never compared numerically with validation accuracy.

If the already generated raw E1 evaluation can be recovered without regenerating E1, M8 also
reports paired correctness transitions and an exact McNemar test. Missing raw E1 evidence does not
justify another expensive E1 generation; the accepted aggregate result and frozen common protocol
remain the primary comparison.

Measured M4 SFT evaluation took 6,437.5 seconds. Phase A therefore plans two hours and caps one
session at four hours / CNY 10.36 at the historical CNY 2.59/hour. Current price, available disk,
auto-shutdown, source commit, adapter manifest, and prepared-bundle identity must be checked before
launch. These numbers are ceilings, not authorization.

## Phase B — dense-reward arms

E3 and E4 each start independently from the immutable E1 adapter and reuse E2's 256 prompts, four
generations, 1,024-rollout budget, generation settings, and optimizer settings. E3 uses one raw
utility followed by group normalization. E4 normalizes execution, gated validity, and gated
structure per dimension before applying the frozen weights.

The E2-only common-evaluation runner is now generalized to accept only explicitly registered E2 or
E3 identities. It rejects condition, adapter, selection, prompt, decoding, evaluator, and heldout
drift. E3 reuses rather than regenerates the accepted E1/E2 results and generates only its own
full1209 predictions. The execution order remains E3 then E4 so that the dense-feedback effect is
measured before the normalization intervention.

## E3 common-validation packet

`configs/m8_e3_common_eval.toml` preregisters one E3 pass over the identical full1209 selection,
with the formal adapter manifest `890d84cd...d5f1`. It reuses E1/E2 aggregates and, when existing
raw rows are available at no inference cost, reports paired E1↔E3 and E2↔E3 transitions with exact
McNemar tests. Missing old raw rows never authorizes regeneration. Expected runtime is two hours;
the four-hour/CNY 10.36 ceiling, current-price check, auto-shutdown, and explicit launch approval
remain mandatory. Mini-Dev stays sealed.

The local packet passes 289 complete tests and 64 M8-focused tests. A read-only cloud preflight
loaded the actual 1,209-example bundle, matched selection `3dc2fcbd...f0b1`, recomputed the retained
adapter as `890d84cd...d5f1`, and observed heldout counters 0/0. No model inference occurred during
that preflight.

## E3 common-validation result

E3 completed the same full1209 exam at execution commit `9a5f8de`. Strict accuracy was 41.6873%,
behavior-compatible accuracy 43.4243%, valid-SQL rate 89.8263%, and SQL-only rate 100%. Relative to
E1, strict transitions were 15 regressions versus 27 improvements (exact McNemar p=0.08843).
Relative to E2, they were 25 regressions versus 17 improvements (p=0.27996). Thus E3 is directionally
better than E1 but not significant, and it does not beat E2. This separates “more reward variation”
from “better downstream policy.” Heldout use remained 0/0. The public result is
`results/m8/m8-e3-common-eval-summary.json`.

## E4 smoke packet

`configs/m8_e4_smoke.toml` preregisters the minimal eight-prompt E4 engineering smoke. It reuses
E3's parent, ordered prompt selection, four-completion groups, optimizer, decoding, component
scorer, fixed weights, evidence requirements, and 30-minute/CNY 1.30 ceiling. The only scientific
change is `sum_then_normalize -> normalize_then_sum`; config-contract tests fail on any other drift.
The retained E2/E3 smoke bundles have the same canonical identity
`82cfce66...c3265`, which must be reproduced before E4 model loading. Smoke success establishes
execution viability only and cannot trigger formal E4 automatically. The preregistration packet
passes 300 complete tests and 75 M8-focused tests in the pinned Python 3.11.15 / TRL 1.9.0 cloud
environment without loading a model. See `docs/M8_E4_PLAN.md`.

The subsequent E4 smoke completed 8/8 optimizer updates and 32 rollouts in 314.72 session seconds.
It produced seven mixed groups, a nonzero adapter delta, an independently reloadable adapter, and
heldout/validation counters of zero. This validates the E4 execution path; it is not a model-quality
comparison and does not authorize formal E4 automatically.

## Formal E4 packet

`configs/m8_e4_formal.toml` preregistered formal E4 as the exact aggregation-only counterpart to
formal E3. The run completed all 256 prompts, 1,024 rollouts, and 256 updates from immutable E1,
with 235 mixed groups, a nonzero adapter update, successful independent reload, and zero
validation/heldout use. Only `sum_then_normalize` changed to `normalize_then_sum`. This is training
evidence; the same full1209 validation is still required before any quality claim.

## E4 common-validation packet

`configs/m8_e4_common_eval.toml` freezes one E4 generation pass on selection
`3dc2fcbd...f0b1`. It accepts only the formal adapter manifest `719bed30...3522`, reuses E1/E2/E3
results without regeneration, and produces an SQL-free four-arm comparison. If retained raw rows
are available, paired McNemar analyses are added at no inference cost. The expected two-hour run
retains the four-hour/CNY 10.36 ceiling; Mini-Dev remains sealed.

## E4 common-validation result

E4 completed the frozen full1209 exam at execution commit `eb1e132`. It reached 41.4392% strict
execution accuracy, 43.3416% behavior-compatible accuracy, 89.4955% valid SQL, and 100% SQL-only
output. Strict accuracy was +0.7444 points versus E1, -0.9098 versus E2, and -0.2481 versus E3.
Paired exact McNemar p-values were 0.2110, 0.1081, and 0.7283 respectively, so none of those E4
contrasts crossed p=0.05. E4 generated 56,662 tokens in 6,432.09 seconds at 8.809 tokens/s; heldout
use remained 0/0. The SQL-free result is `results/m8/m8-e4-common-eval-summary.json`.

The completed ladder therefore supports a narrow conclusion: outcome-only E2 is the best measured
validation arm, while E3/E4 created much more mixed reward signal without improving generalization.
Separate component normalization did not rescue the dense reward in this budget. This controlled
negative result closes M8 without tuning weights after observing validation results.

## Phase A result

Formal E2 completed the frozen full1209 validation exam at execution commit `95da887`. Relative to
the reused E1 result, strict execution accuracy increased from 40.6948% to 42.3490% (+1.6543
percentage points), behavior-compatible accuracy increased from 42.3490% to 44.2514% (+1.9024
points), and valid-SQL rate increased from 87.5931% to 90.7361% (+3.1431 points). SQL-only rate
remained 100%.

The existing raw E1 evidence enabled paired analysis: strict correctness had 19 regressions and 39
improvements (exact two-sided McNemar p=0.01193); behavior-compatible correctness had 18 regressions
and 41 improvements (p=0.00379). These are positive validation results, not heldout claims. The run
generated 57,638 tokens in 6,786.30 seconds at 8.493 tokens/s; actual provider billing remains
pending. The public result is `results/m8/m8-common-eval-summary.json`.

## Acceptance

Phase A passes when E2 has one complete full1209 result, source identities match, heldout use is
zero, and an SQL-free E1/E2 report is saved. M8 as a whole passes only when E2/E3/E4 have comparable
common-evaluation results and reward/sample-efficiency diagnostics, or when a predeclared failure
or reduced-form rule is reported without changing the hypothesis after seeing results.
