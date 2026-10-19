# M8 E4 Normalize-Then-Sum Plan

Status: completed; engineering smoke, formal E4 training, and common validation passed their gates.

## Controlled question

E4 asks whether separately normalizing each reward dimension produces a better optimization signal
than E3's normalize-after-summing rule. It starts independently from the immutable E1 adapter and
uses the same reward components and frozen weights:

```text
R_E4,i = z_G(r_exec_i)
       + 0.05 * z_G(g_conf_i * r_valid_i)
       + 0.20 * z_G(g_conf_i * r_struct_i)
```

E3 and E4 differ scientifically only in aggregation:

- E3: add raw components into one utility, then normalize the utility within the four-completion
  group;
- E4: normalize execution, gated validity, and gated structure separately within the group, then
  apply the same `1.0 / 0.05 / 0.20` weights and add them.

No reward weight, prompt, parent adapter, rollout budget, optimizer, decoding setting, evaluator, or
data split changes with E4.

## Engineering smoke

`e4_m8_gdpo_smoke_001` uses the same eight prompts, four completions per prompt, 32 rollouts, eight
optimizer updates, and 30-minute/CNY 1.30 ceiling as the accepted E3 smoke. The config loader
mechanically compares E4 with `configs/m8_e3_smoke.toml` and rejects any scientific drift beyond the
registered aggregation intervention.

Before model loading, cloud control must regenerate the prepared bundle and match both its ordered
sample IDs and canonical bundle SHA-256 to the retained E3 smoke bundle:

```text
bundle_sha256 = 82cfce66c7728820f22ba4f759f8cbb18ca49abe0c55e96c10f2be1d275c3265
sample_ids = [1345, 765, 353, 2943, 3388, 4017, 2665, 1564]
```

The sample IDs are private execution evidence and must not be copied into a public result. The
public packet records only the aggregate bundle identity.

## Acceptance and stops

The smoke passes only if all eight prompts and 32 completions reach the reward/component telemetry,
at least one reward group is mixed, loss and gradient norms remain finite, the adapter delta is
nonzero, a separate process reloads the same adapter manifest, and heldout counters remain 0/0.
E4's `mean_group_relative_reward` must be approximately zero and must not be mislabeled as terminal
reward.

Stop before or during training on any commit/input/bundle mismatch, unsupported pinned-TRL
aggregation value, `xgrammar` or environment failure, metadata misalignment, NaN/Inf, no reward
variation, zero adapter delta, reload mismatch, OOM, heldout access, or wall/cost breach. A smoke
failure does not authorize hyperparameter tuning or formal E4.

## Cloud entry point after approval

The existing bounded post-training orchestrator is reused:

```sh
export M7_CONFIG=configs/m8_e4_smoke.toml
export M7_APPROVED_COMMIT=<accepted-clean-full-sha>
export M7_DATASET_ROOT=<retained-m2-database-root>
export M7_TRAINING_MANIFEST=<training-manifest.json>
export M7_M4_OUTPUT_DIR=<retained-e1-output-directory>
export M7_OUTPUT_DIR=<fresh-e4-smoke-output-outside-git>
sh scripts/run_m7_cloud.sh
```

The legacy `M7_*` variable names identify the shared runner, not the E2 arm. Formal E4 remains a
separate future preregistration and must not start automatically after smoke success.

## Pre-execution verification

The exact source packet passed 300 complete tests and 75 M8-focused tests in Python 3.11.15 with
TRL 1.9.0. The focused test constructs `GRPOConfig` using the E4 arguments and confirms that the
pinned wheel accepts `normalize_then_sum`. A training-only cloud preflight then regenerated all
eight ordered sample IDs and reproduced canonical bundle SHA-256 `82cfce66...c3265` plus file
SHA-256 `7dc21256...ccf2`, exactly matching retained E2/E3 smoke evidence with heldout 0/0. No model
was loaded and no GPU training occurred.

## Smoke outcome

`e4_m8_gdpo_smoke_001` executed from commit `8376c57` and completed eight prompts, 32 rollouts, and
eight optimizer updates in 314.72 session seconds. Seven groups were mixed and one had zero
variance; the adapter changed by L2 0.10458 and independently reloaded with manifest
`e16ce695...ba5d5`. Peak memory was 11,736 MiB, heldout and validation use were zero, and the public
SQL-free evidence is `results/m8/m8-e4-smoke-summary.json`.

Attempt 001 stopped before model load because the retained `xgrammar 0.2.3` installation lacked
`libxgrammar_bindings.so`. Reinstalling only that exact package restored `xgrammar` and
`GRPOTrainer` imports plus `pip check`; attempt 002 then passed. This environment incident is not a
failed reward experiment. The tiny stochastic smoke establishes execution viability only and does
not show that E4 improves validation accuracy.

## Formal E4 preregistration

`e4_m8_gdpo_formal_001` is the aggregation-only counterpart to accepted formal E3. It starts again
from immutable E1 and preserves formal E3's ordered 256-prompt selection, four generations per
prompt, 1,024 rollouts, 256 optimizer updates, 64-step checkpoints, optimizer, decoding, component
scorer, confidence gate, reward weights, evidence requirements, and stop conditions. The only
scientific intervention is `sum_then_normalize -> normalize_then_sum`.

Before model loading, the cloud run must regenerate the formal prepared bundle and match its
canonical SHA-256 and ordered private sample identities to retained formal E3 evidence. This is a
single input-identity check, not a new data audit. The plan expects about 90 minutes and retains the
three-hour/CNY 7.77 ceiling at the historical rate; price and automatic shutdown must be confirmed
at launch.

Training passes only after all 256 updates and 1,024 rollouts are recorded, loss and gradient norms
are finite, the adapter delta is nonzero, an independent process reloads the saved adapter, the
reward/component telemetry is complete, heldout and validation counters remain 0/0, and the frozen
signal gate reaches at least 32 mixed groups and a 0.125 mixed-group fraction. A successful run is
training evidence, not model-quality evidence; E4 must later take the same full1209 validation exam.

## Formal E4 outcome

`e4_m8_gdpo_formal_001` completed from commit `359a673` with all 256 updates and 1,024 rollouts.
It produced 235 mixed groups (91.80%), 21 zero-variance groups, finite loss and gradients, and a
nonzero adapter delta of 0.35576. A separate process reloaded adapter manifest
`719bed30...3522`; peak memory was 19,804 MiB, session time was 3,436.65 seconds, and both heldout
and validation counters remained zero.

Attempt 001 stopped during preparation because the supplied root duplicated the manifest's
`databases/` prefix. Attempt 002 stopped during import because a provider-managed symlink to the
exact `xgrammar 0.2.3` native library had no target. Neither attempt loaded a model. The successful
run used the same wheel's extracted native library from an external runtime directory, after two
independent import checks, without changing package versions or the scientific contract.

E4's 235 mixed groups are slightly fewer than E3's 239. That is useful optimization evidence but
not a model-quality result. The next and only valid comparison is one same-protocol full1209 E4
validation run; Mini-Dev remains sealed.

## Common-validation packet

`configs/m8_e4_common_eval.toml` preregisters a single E4 pass over the same 1,209-example selection
used by E1, E2, and E3. The generic evaluator accepts only condition `E4`, formal adapter experiment
`e4_m8_gdpo_formal_001`, and adapter manifest `719bed30...3522`; it also checks sample order, prompt
hashes, greedy decoding, evaluator identity, and heldout 0/0 before publishing evidence.

E1/E2/E3 aggregates are reused rather than regenerated. Existing raw rows may be read only to
compute paired E1↔E4, E2↔E4, and E3↔E4 McNemar statistics. Missing old raw evidence merely omits the
corresponding paired analysis; it never authorizes new inference. The SQL-free summarizer reports all
four arms and E4 deltas. Expected runtime is about two hours with a four-hour/CNY 10.36 ceiling at
the historical rate. This packet is local preparation, not launch authorization.

## Common-validation outcome

`e4_m8_common_validation_001` completed once on the frozen 1,209-example selection. E4 scored
41.4392% strict execution accuracy, 43.3416% behavior-compatible accuracy, 89.4955% valid SQL, and
100% SQL-only output. It trailed E2 by 0.9098 strict-accuracy points and E3 by 0.2481 points; paired
exact McNemar p-values were 0.1081 and 0.7283. The E1 comparison was +0.7444 points with p=0.2110.
All source identities passed, heldout remained 0/0, and no earlier condition was regenerated.

The E3-to-E4 aggregation intervention therefore produced no measured quality improvement. E4 is
accepted as a controlled negative result rather than retuned after evaluation.
