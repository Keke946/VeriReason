# M8 E3 Dense-Reward Execution Plan

Status: engineering smoke, formal training, and common full1209 validation completed.

## Controlled question

E3 asks whether adding reliability-gated validity and structural feedback improves learning over E2
terminal correctness alone. E3 starts independently from immutable E1. It does not continue from
E2. The formal E3 run reuses E2's exact 256 training prompts, four completions per prompt, 1,024
rollouts, optimizer, learning rate, decoding, seed, and checkpoint schedule.

The only intended intervention is the reward arm:

- E2: group-relative strict execution correctness;
- E3: form `U = r_exec + g_conf * (0.05*r_valid + 0.20*r_struct)`, then normalize `U` within each
  four-completion group.

`g_conf` closes dense feedback when parsing, schema resolution, supported-construct analysis, or
safe execution is uncertain. Terminal correctness remains available independently.

## Two-stage execution

1. Run `e3_m8_dense_grpo_smoke_001` on the same eight prompts as the E2 engineering smoke. It proves
   arm dispatch, dense component telemetry, optimizer updates, adapter change, and separate reload.
   Expected time is about 10 minutes; hard limit is 30 minutes / CNY 1.30 at the historical rate.
2. Only if the smoke passes, run `e3_m8_dense_grpo_formal_001` on the same 256 prompts as formal E2.
   The formal ceiling remains three hours / CNY 7.77 at the historical rate.

Neither stage accesses validation or heldout. E3 model quality is measured later using the same
full1209 validation protocol already used for E1/E2. E4 remains unstarted until E3 training and
common evaluation complete.

## Formal outcome

`e3_m8_dense_grpo_formal_001` executed from commit `1732ea19e08460422c2d75bb19b4f356638c1993`
and completed all 256 prompts, 1,024 rollouts, and 256 optimizer updates. The signal gate passed:
239/256 groups contained mixed reward and 17 had zero variance, compared with 102/256 and 154/256
for formal E2. The adapter changed by L2 0.30966 and passed a separate-process reload with manifest
SHA-256 `890d84cd...d5f1`. Peak memory was 19,804 MiB; training and complete-session wall times were
3,387.67 and 3,506.27 seconds. Validation and heldout use were both zero.

The 363/1,024 strict-correct and 786/1,024 valid counts describe sampled on-policy training
completions, not full1209 accuracy. Mean entropy was 0.19975, with a low final logged value of
0.03636; there was no completion clipping or length explosion, but common validation must determine
whether the stronger dense signal improved or over-concentrated the policy. Public SQL-free evidence
is `results/m8/m8-e3-formal-summary.json`.

## Smoke outcome

`e3_m8_dense_grpo_smoke_001` executed from commit `5579446` and passed its evidence closure. It
completed eight prompts, 32 rollouts, and eight optimizer updates; seven reward groups were mixed
and one had zero variance. The adapter delta was nonzero, its independent reload identity matched,
peak GPU memory was 11,736 MiB, end-to-end wall time was 304.91 seconds, and heldout use was zero.
The public SQL-free aggregate is `results/m8/m8-e3-smoke-summary.json`.

The first attempt stopped before model loading because the retained `xgrammar 0.2.3` installation
was missing its RECORD-listed native library. Reinstalling only the exact package restored the
import, after which the same frozen smoke contract passed. This is an environment incident, not a
failed reward experiment. The successful tiny run establishes execution and signal viability; it
does not establish E3 accuracy or superiority over E2.

The orchestrator closes the run only after comparing the training and separate-process reload
adapter identities, checking all prompt groups and completions are represented in reward/component
telemetry, confirming a nonzero adapter update and mixed reward signal, and verifying zero heldout
use. It writes `posttraining-summary.json`; process exit codes alone are not treated as acceptance.

## Cloud entry point

The smoke deliberately reuses the tested M7 orchestration surface; the E3 config changes its arm
dispatch. After control has approved a clean commit and checked the retained paths, run with:

```sh
export M7_CONFIG=configs/m8_e3_smoke.toml
export M7_APPROVED_COMMIT=<approved-full-sha>
export M7_DATASET_ROOT=<retained-m2-database-root>
export M7_TRAINING_MANIFEST=<training-manifest.json>
export M7_M4_OUTPUT_DIR=<retained-e1-output-directory>
export M7_OUTPUT_DIR=<fresh-persistent-e3-smoke-output>
sh scripts/run_m7_cloud.sh
```

The legacy `M7_*` variable names are retained for compatibility; the loaded experiment identity
and `arm="E3"` are validated before model loading. The output directory must be fresh, and the
formal E3 config must not be substituted during the smoke session.

## Common-validation entry point

Formal training does not establish model quality. After the local E3 evaluation packet passes, the
retained adapter is evaluated exactly once with the generic M8 runner:

```sh
export HF_HOME=/root/cloud/huggingface
PYTHONPATH=src python scripts/run_m8_eval.py \
  --config configs/m8_e3_common_eval.toml \
  --prepared /root/verireason-m8-input/prepared-full-validation.json \
  --adapter-dir /root/verireason-m8/e3_m8_dense_grpo_formal_001/adapter \
  --output /root/verireason-m8/e3_m8_common_validation_001/e3-full.json
```

The source commit must be the separately accepted execution commit, the output directory must be
fresh, and the adapter manifest must match before model load. Existing E1 and E2 raw evaluations
may be used only for paired statistics; they are never regenerated. The summarizer publishes no SQL.

## Common-validation outcome

The E3 pass completed at execution commit `9a5f8de` with heldout 0/0. Strict accuracy was 504/1,209
(41.6873%): 12 net paired improvements over E1 but eight net paired regressions relative to E2.
Exact McNemar p-values were 0.08843 for E1↔E3 and 0.27996 for E2↔E3. E3 therefore establishes a
controlled negative result: dense feedback greatly reduced zero-variance groups during training but
did not outperform outcome-only E2 on common validation. E4 tests aggregation, not new weights.
