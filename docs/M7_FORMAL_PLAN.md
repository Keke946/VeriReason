# M7 Formal E2 Outcome-only GRPO Plan

Status: local preregistration pending control review; no paid execution is authorized by this file.

## Goal and claim boundary

Run the formal outcome-only RL baseline from the immutable E1 adapter. This run measures whether a
bounded terminal-reward budget produces a stable, nontrivial E2 checkpoint. It does not use heldout
data and does not claim an accuracy gain until the common M8 validation comparison is complete.

## Frozen run

- Parent: `e1_m4_formal_sft_001` final update 671.
- Data: 256 unique, outcome-blind, hash-selected M2 training prompts with frozen critical-feature
  coverage; validation and heldout are excluded.
- Sampling: four completions per prompt, 1,024 rollouts total.
- Optimization: 256 updates, one complete four-completion group per update, learning rate `5e-6`,
  DAPO loss, truncation masking, no rollout reuse, and `beta=0`.
- Reward: strict SQLite execution correctness only. Project group normalization is applied once;
  TRL `scale_rewards` remains `none`.
- Runtime: direct Transformers generation, checkpoints every 64 updates, one 48 GB GPU.

The smoke completed eight updates in 113.8 training seconds and 313.6 end-to-end seconds with
11,736 MiB peak memory. Linear training-only scaling projects about 60.7 minutes for 256 updates.
The operational estimate is 1.5 hours; the process hard stop is three hours / CNY 7.77 at the
historical CNY 2.59/hour rate. Price and provider auto-shutdown must be reconfirmed before launch.

## Signal and promotion gate

The smoke observed 2/8 mixed-reward groups. That estimate is noisy, so it does not justify adaptive
prompt selection or outcome-based filtering. The formal selector remains outcome-blind. The run is
allowed to finish its fixed budget, then requires both:

- at least 32 mixed-reward groups; and
- a mixed-reward fraction of at least 12.5%.

This is a promotion gate, not an early-stop rule. Falling below it is a valid sparse-reward result:
the checkpoint and diagnostics are retained, but E2 is not presented as having received adequate
learning signal and M8 must reconsider group size or informative-group sampling as a declared
follow-up rather than silently changing the experiment.

## Bounded reward-hacking review

The scorer publishes SQL-free aggregates only: strict-correct count, valid count, confidence-gate
count, mean structural score, and evaluator status counts. Trainer evidence also retains reward
variance, completion length, entropy, clipping, and truncation metrics. The review checks for unsafe
or resource-limit growth, collapsing validity, extreme length/truncation, entropy collapse, and a
reward increase unsupported by the later common validation evaluation. M6 adversarial tests remain
the verifier-level defense. No raw SQL is exported publicly and no separate audit bureaucracy is
added to this baseline.

## Stops and outputs

Stop on parent/manifest drift, heldout access, NaN/Inf, reward-row misalignment, missing adapter
delta, checkpoint/reload failure, minimum-batch OOM, or the wall/cost ceiling. Preserve the latest
64-step checkpoint on interruption. Required outputs are the final adapter, separate-process reload,
aggregate reward/component telemetry, peak memory, wall time, and zero-heldout evidence.

The next model-quality comparison belongs to M8, where E1/E2/E3/E4 use the same evaluation protocol.

## Measured result

The fixed run completed 256 updates and 1,024 rollouts in 3,413.74 end-to-end seconds. It produced
102 mixed groups (39.84%), passed both preregistered signal gates, changed the adapter by L2 0.5224,
peaked at 19,804 MiB, and passed a separate-process reload. Training-sample terminal reward was
334/1,024 (32.62%); this is on-policy reward, not validation accuracy. Valid completion rate was
77.64%, no completion was truncated, and heldout/validation use was 0/0.

The bounded reward-hacking review found one unsafe completion and one resource-limit completion;
both were blocked by the evaluator. Quarterly entropy remained between 0.1817 and 0.2194 on
average, mean length moved from 51.70 to 47.24 tokens, and reward was not monotonically increasing.
There is no aggregate evidence of gross reward exploitation, but M8 common-protocol validation is
required before making any quality claim. Public evidence is in
`results/m7/m7-e2-formal-summary.json`; the 75 MiB final adapter and raw SQL-free logs are retained
under ignored `artifacts/m7-e2-formal-001/`.
