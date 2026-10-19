# VeriReason: Verifiable Post-Training for Text-to-SQL LLMs

## Abstract

VeriReason is a consumer-GPU study of supervised fine-tuning and reinforcement learning with
verifiable rewards (RLVR) for cross-domain Text-to-SQL. Starting from a fixed
`Qwen/Qwen3-4B-Base` revision, the project builds a database-disjoint BIRD protocol, a sandboxed
SQLite evaluator, a one-epoch QLoRA baseline, and three controlled GRPO reward arms. Outcome-only
GRPO improved strict execution accuracy from 40.69% after SFT to 42.35% on the same 1,209-example
validation exam. Two denser structural-reward variants produced substantially more non-constant
training groups but did not outperform outcome-only GRPO. The result is both a positive small-model
RLVR result and a controlled negative result about naive dense reward shaping. The selected model
was then served through vLLM and verified in a Git-only clean-room release gate.

## Research question

Can reliability-gated structural feedback improve the learning efficiency or execution accuracy of
small-model Text-to-SQL RLVR over terminal execution reward alone, without increasing reward
hacking?

The causal ladder is:

| Arm | Intervention | Strict execution accuracy |
|---|---|---:|
| E0 | Frozen base model | 0.00% |
| E1 | One-epoch QLoRA SFT | 40.69% |
| E2 | E1 + terminal execution reward GRPO | **42.35%** |
| E3 | E1 + gated dense utility, sum then normalize | 41.69% |
| E4 | E1 + same utility, normalize each component then sum | 41.44% |

E2 improved over E1 by 1.65 percentage points. Its paired strict transitions were 39 improvements
and 19 regressions (exact two-sided McNemar `p=0.01193`). E3 and E4 improved over E1 but trailed E2;
their differences from E2 were not statistically significant at 0.05.

## System design

```text
fixed BIRD sources
      |
      v
safe materialization -> database-disjoint train/validation manifests
      |                              |
      v                              v
answer-only QLoRA SFT          sandboxed SQLite verifier
      |                              |
      +----------> E2/E3/E4 GRPO <---+
                         |
                         v
              frozen full1209 evaluation
                         |
                         v
                vLLM serving benchmark
```

### Data and evaluation

- Source: BIRD filtered train at a fixed Hugging Face revision.
- Operational set: 5,364 train and 1,209 validation examples over disjoint 59/10 databases.
- Heldout: BIRD Mini-Dev 500 remained sealed and was never loaded by SFT, RL, evaluation, or
  serving experiments.
- Evaluator: immutable read-only SQLite URIs, SQLite authorizer, child-process isolation, wall/CPU/
  VM/result-size limits, and deterministic error-to-zero semantics.
- Metrics: strict bag/order-aware execution accuracy and a BIRD-compatible behavior metric.

### SFT

E1 uses 4-bit NF4 QLoRA with rank 16 over all attention and MLP projection layers. Only assistant
SQL tokens and EOS contribute to the loss. One complete pass over 5,364 eligible examples produced
671 optimizer updates. Base and E1 were decoded greedily on the identical validation rows and
evaluator.

### Verifiable-reward training

Every RL arm independently starts from immutable E1 and uses the same 256 training prompts, four
rollouts per prompt, optimizer settings, seed, and 256 updates.

- **E2** uses terminal strict execution correctness.
- **E3** combines terminal correctness with confidence-gated valid-SQL and eleven-dimensional SQL
  structure similarity, then normalizes the combined utility within each prompt group.
- **E4** uses identical components and weights but normalizes each component within the group before
  summation.

E2 produced 102/256 mixed-reward groups. E3 and E4 produced 239/256 and 235/256, respectively.
Therefore the dense reward successfully reduced reward sparsity, but the common validation exam
showed that more reward variation was not the same as better generalization.

## Serving result

The best arm, E2, was loaded as a static rank-16 adapter into vLLM's OpenAI-compatible endpoint.
Across 192 measured requests on a fixed 64-example validation slice, every request succeeded.

| Concurrency | Request throughput | TTFT p50 |
|---:|---:|---:|
| 1 | 1.73 req/s | 94 ms |
| 4 | 4.51 req/s | 93 ms |
| 8 | 6.51 req/s | 116 ms |

Peak GPU memory was 29,246 MiB. The slice's 46.88% strict accuracy is an integration check, not a
replacement for the 1,209-example model-quality result.

## Reproducibility and evidence

Every headline metric maps to an experiment ID, fixed config, Git commit, and SQL-free result JSON
in `experiments/registry.yaml`. `make m10-verify` checks the full test suite and the exact hashes of
19 public evidence files without loading a model, database, or heldout row. A Git-only archive in a
fresh Python 3.11.15 environment passed 314 tests with four optional-runtime skips.

## Limitations

- The 1,209-example validation set is the model-selection exam; Mini-Dev has not been consumed, so
  this release does not claim a final heldout score or BIRD leaderboard result.
- The RL study uses 256 prompts and a 4B base model. It establishes a controlled comparison, not a
  scaling-law claim.
- E3/E4 provide sequence-level dense utilities. They do not localize advantages to individual SQL
  clauses or tokens; this is the strongest explanation to test next.
- Actual provider invoices after M3 were not captured in the spend ledger and must not be inferred
  from program wall time.
- The CPU clean-room gate validates code and public evidence, not a fresh rerun of expensive GPU
  experiments.

## Artifact map

- Full experiment identities: `experiments/registry.yaml`
- Reproducibility: `docs/REPRODUCIBILITY.md`
- Model card: `docs/MODEL_CARD.md`
- Data card: `docs/DATA_CARD.md`
- Follow-up research: `docs/RESEARCH_FOLLOWUP.md`
- Interview explanations: `docs/INTERVIEW.md`
