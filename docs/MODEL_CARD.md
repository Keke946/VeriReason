# Model Card: VeriReason E2 Text-to-SQL Adapter

## Model description

VeriReason E2 is a rank-16 LoRA adapter for the exact base revision
`Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`. It first received one epoch of
answer-only QLoRA supervised fine-tuning and then 256 updates of outcome-only GRPO using strict
SQLite execution correctness as the verifiable reward.

The adapter is designed to translate an English question plus deterministic SQLite schema text into
exactly one read-only `SELECT` or `WITH ... SELECT` query.

## Intended use

- Research and educational Text-to-SQL experiments.
- Reproducing the VeriReason E1/E2 comparison.
- Local inference on databases whose users are authorized to query.

It is not a database access-control mechanism and must not be allowed to execute unrestricted SQL.
Use the included read-only evaluator or an equivalently isolated database role.

## Training and evaluation

| Item | Value |
|---|---|
| Base | Qwen3-4B-Base, fixed revision above |
| SFT | 5,364 examples, one epoch, 671 optimizer updates |
| RL | 256 prompts, 4 completions/prompt, 1,024 rollouts, 256 updates |
| Adapter | LoRA rank 16, alpha 32, all attention/MLP projections |
| Primary validation | 1,209 examples, 10 databases, greedy decoding |
| Strict execution accuracy | 42.3490% |
| Behavior-compatible accuracy | 44.2514% |
| Valid SQL rate | 90.7361% |
| SQL-only rate | 100% |

The E2-vs-E1 strict improvement is 1.6543 percentage points with exact paired McNemar
`p=0.01193`. Mini-Dev was not loaded or evaluated.

The `v1.0.0` release archive `verireason-e2-adapter.tar.gz` has SHA-256
`9e17b2a5acb5ecbe78c778465dd8920ea3fd643340847243ba3892fbd366bcf8`. After extraction, the adapter
directory manifest is `3492a3ec870f7c6e8e672459f025fc96a9a8cba543b27b7df4fc5f6c6bfbd3b7`.

## Limitations and risks

- The model can produce syntactically valid but semantically incorrect SQL.
- The benchmark contains noisy labels; operational filters do not guarantee semantic perfection.
- The model has not been evaluated for sensitive schemas, adversarial instructions, access-control
  bypass, or non-SQLite dialects.
- Execution equivalence on one database instance is not proof of universal semantic equivalence.
- Reported accuracy is validation accuracy, not a private test or leaderboard score.

## Licenses and attribution

The VeriReason code is Apache-2.0. The
[Qwen3-4B-Base model card](https://huggingface.co/Qwen/Qwen3-4B-Base) declares Apache-2.0.
Training data comes from BIRD and declares CC BY-SA 4.0. Users are responsible for complying with
the upstream model and dataset terms. The adapter archive is distributed separately from the Git
source history.
