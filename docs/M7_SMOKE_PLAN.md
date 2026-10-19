# M7 E2 Outcome-only GRPO Smoke Plan

Status: executed successfully at commit `91a95bad2ca6c2c0cf2c083ee2d57c72cf5f35d0`; formal E2 is
not yet authorized.

## What this run proves

This is an engineering smoke, not an accuracy experiment. It must prove that the immutable E1 SFT
adapter can generate multiple SQL candidates, receive real SQLite execution rewards, update its
LoRA weights with GRPO, save a checkpoint, and reload that checkpoint. It makes no claim that eight
updates improve validation accuracy.

The data path is:

```text
8 deterministic training prompts
  -> 4 sampled SQL completions per prompt (32 rollouts)
  -> frozen read-only SQLite evaluator
  -> strict execution reward in {0, 1}
  -> group-relative E2 advantage
  -> 8 LoRA optimizer updates
  -> save and reload adapter
```

No validation or heldout example is selected, loaded, or evaluated in this smoke.

## Exact choices and rationale

- Parent: immutable `e1_m4_formal_sft_001`, not the base model and not a continued SFT run.
- Selection: eight operationally eligible training rows, ranked by a fixed hash, with join,
  aggregate, and subquery coverage. Model outcomes and rewards are not selection inputs.
- Group size: four completions. With E1's development strict accuracy near 0.41 this is likely to
  produce useful mixed groups without the rollout cost of eight; the smoke records what actually
  happens and does not assume it.
- Direct Transformers generation: `use_vllm=false`. This removes a second inference engine from the
  first training test. vLLM is reconsidered only if formal-run throughput requires it.
- `scale_rewards="none"`: `TRLGroupRewardAdapter` already implements the project-frozen group
  centering/normalization. Letting TRL scale again would silently alter the experiment.
- DAPO loss and truncation masking: active tokens, rather than completion length, normalize the
  policy loss; completions cut off at the token ceiling do not create misleading gradients.
- `beta=0`: the smoke avoids a reference model and tests the minimum viable RL path. A nonzero KL
  coefficient remains a formal-run decision informed by memory and drift diagnostics.
- Eight optimizer updates: batch 1 x accumulation 4 consumes one complete four-completion group per
  update, so every selected prompt is used once.

## Pass evidence

The run passes only if it records finite loss and gradient norm, at least one changed adapter tensor,
a reloadable saved adapter, complete reward-group diagnostics, peak GPU memory, wall time, and zero
heldout use. A zero mixed-group fraction is not a software failure, but it blocks scaling directly to
the formal E2 run because GRPO then received no relative learning signal.

Stop on parent/manifest mismatch, heldout access, reward-row misalignment, NaN/Inf, OOM at the
already-minimal batch, absent adapter delta, reload failure, or the two-hour process deadline. Before
launch, the provider rate and retained E1/data paths must be rechecked. The provider auto-shutdown
ceiling is three paid hours / CNY 7.77, leaving one hour for startup, upload, and shutdown overhead;
the expected working session remains roughly 1–1.5 hours.

The cloud entry point is `scripts/run_m7_cloud.sh`. It requires an approved clean Git commit, the
retained M2 dataset/manifest, the retained M4 output containing `prepared-train.json` and `adapter`,
and a fresh output directory outside Git. One cumulative two-hour process deadline covers local
verification, selection, training, and the separate-process adapter reload.

## What happens after the smoke

The control task reviews correctness, memory, throughput, completion truncation, mixed/constant
reward groups, reward mean, and adapter reload. Only then is the formal E2 sample/step budget frozen.
E3/E4 do not run merely because E2 smoke succeeds.

## Measured result

The run completed all eight optimizer steps on 32 rollouts in 313.60 seconds end to end. Maximum
gradient norm was 0.6484, adapter delta L2 was 0.1334, and peak GPU memory was 11,736 MiB. The
saved adapter passed a separate-process reload and heldout use was 0/0. Mean terminal reward was
0.21875. Two of eight completion groups were mixed-reward; six were constant-reward and therefore
contributed no group-relative policy gradient. This is a successful systems smoke and direct
evidence that terminal-only E2 feedback is sparse on this sample. It is not evidence of improved
validation accuracy. The SQL-free aggregate and source-evidence hashes are recorded in
`results/m7/m7-e2-smoke-summary.json`.
