# Research Design: Reliable Dense Feedback for Consumer-GPU Text-to-SQL RLVR

Last updated: 2026-08-13

## Recommendation

Build the project in two layers. The protected fifteen-day core delivers a reproducible base -> QLoRA SFT -> outcome-only RLVR -> focused multi-reward comparison -> evaluation -> serving chain. Only after that chain is green do we attempt clause-token credit, informative-group resampling, a 7B confirmation, or multi-turn agent RL.

The core working hypothesis is:

> Under a fixed consumer-GPU and rollout budget, reliability-gated dense SQL feedback with per-reward normalization improves learning efficiency or held-out execution accuracy over terminal execution reward alone, without increasing verifier exploitation.

This is a testable hypothesis, not a promised positive result.

## Why this is not a naive Text-to-SQL project

A naive implementation calls a model, extracts a SQL string, runs it once, and assigns 0/1. This project instead treats the verifier and learning signal as research objects:

1. **Verifier reliability:** executable is not equivalent to correct; a candidate can match by accident on one database instance or exploit weak comparison semantics.
2. **Sparse credit:** final 0/1 feedback does not say which part of a complex query was useful.
3. **Multi-reward optimization:** combining validity, structure, and execution before group normalization can collapse distinct reward information.
4. **Uninformative groups:** when every completion for a prompt has the same reward, a group-relative update carries little or no useful signal.
5. **Compute efficiency:** a method that gains one point with ten times the rollouts is not automatically better for the stated constraint.

## Evidence from recent primary sources

| Source | Relevant result | Use in this project | Why not copy the whole system |
|---|---|---|---|
| [Arctic-Text2SQL-R1, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1345/) | Strong curated data and SFT make a simple execution reward surprisingly competitive | Outcome-only baseline and a warning not to over-engineer rewards | The released result is a scale/data recipe, not our controlled small-GPU reward study |
| [SQL-R1, NeurIPS 2025 official code](https://github.com/DataArcTech/SQL-R1) | Cold start, SQL-specific rewards, and small synthetic data can support RL reasoning | Output/data conventions and a reproducible upstream baseline | Its script assumes 8 GPUs and its reward/evaluator code is hard-coded and too permissive for the final verifier |
| [ReViSQL, 2026 official code](https://github.com/uiuc-kang-lab/ReViSQL) | Verified data alone produces large gains; its trainer exposes constant-group filtering and formal-equivalence-aware soft rewards | Data-quality audit, verifier design, optional informative-group sampling | Its reported models are far larger and its training entry point is tied to Tinker |
| [EXPO-SQL, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1107/) | Clause-level execution feedback beats uniform query-level credit, especially on challenging SQL | U1 clause/token credit design | Paper-scale training uses 8xH100 80 GB and the claimed public code was unavailable during this audit |
| [SQL-ASTRA, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.920/) | Column-set partial reward and trajectory aggregation reduce sparse feedback in agentic SQL | Dense structural reward inspiration and U3 trajectory study | Multi-turn training expands environment and rollout cost beyond the protected core |
| [Progress-SQL, 2026](https://arxiv.org/abs/2606.06825) | Progressive rewards measure improvement across SQL correction turns | U3 progressive-reward reference | No mature public implementation was available during this audit |
| [SERL-SQL, 2026-08-01](https://arxiv.org/abs/2608.00485) | Teacher-student likelihood gaps localize credit to SQL/tool-action tokens | Highest-risk U3 research direction | Very new preprint, code only promised, and a training-time teacher adds compute and engineering risk |
| [DAPO, 2025](https://arxiv.org/abs/2503.14476) | Dynamic sampling, token-level loss, asymmetric clipping, and overlength handling stabilize RL | DAPO loss defaults, zero-variance diagnostics, optional resampling | The original system is 32B-scale; only isolated, measurable ideas transfer |
| [GDPO, ICML 2026 official code](https://github.com/NVlabs/GDPO) | Normalize reward dimensions separately before aggregation to prevent advantage collapse | Core E4 optimizer comparison through the maintained TRL integration | Do not vendor the repository: its README identifies a non-commercial source-code license despite conflicting GitHub metadata |
| [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) | Maintained support for async rewards, DAPO/Dr-GRPO loss forms, GDPO-style aggregation, vLLM, tools, and diagnostics | Core training substrate | We still own SQL execution safety, data, rewards, tests, and experiment design |
| [LoRA Without Regret, Hugging Face](https://huggingface.co/docs/trl/lora_without_regret) | All-linear adapters, adequate rank, and bounded effective batch size can make LoRA competitive | PEFT configuration prior and ablation | Exact ranks remain empirical for our model/data |

## Core method

### Reward dimensions

Freeze exact equations at M2. The initial design is:

- `r_exec`: terminal execution equivalence under the official evaluator and a process-isolated read-only SQLite runner.
- `r_valid`: bounded syntactic/execution validity signal; it must never outweigh correctness.
- `r_struct`: bounded partial structural agreement computed from parsed SQL components such as referenced columns, joins, predicates, grouping, ordering, and set operations.
- `g_conf`: a reliability gate that suppresses partial credit for parse failures, unsafe statements, unsupported constructs, ambiguous evaluator cases, or adversarially detected shortcuts.

The dense intervention is conceptually `r_exec + g_conf * (r_valid, r_struct)`. We do not grant unrestricted edit-distance or format points, because these are easy to game. Formal equivalence (for supported queries) and multi-database/test-suite checks are evaluation or gating tools, not universal proof claims.

### Optimization conditions

- Outcome baseline: terminal reward only.
- Dense-standard: sum the reward dimensions, then normalize the combined score within each completion group.
- Dense-decoupled: normalize each reward dimension within the group, then aggregate them; use the maintained TRL implementation rather than copying license-ambiguous GDPO source.
- All conditions share prompts, initialization, generation count, token budget, and evaluation protocol.

Use current TRL's DAPO-style token-loss normalization as the implementation default if the M1 compatibility smoke passes. This eliminates a known length-normalization problem but is not counted as our experimental contribution. Record completion length, clipping, entropy, reward-component variance, zero-standard-deviation group fraction, and GPU hours.

## Data and evaluation route

- Final development benchmark: [BIRD Mini-Dev](https://github.com/bird-bench/mini_dev), initially the 500 high-quality SELECT-only subset. Treat it as held out: do not use it for reward design, early stopping, prompt repair, or manual training-data edits.
- SFT/RL candidates: the official BIRD filtered training subset and a controlled subset of [SynSQL-2.5M](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M). SynSQL is Apache-2.0 and supplies complexity labels and reasoning traces.
- Training-time validation: create a frozen, database-disjoint validation slice only from approved training sources; never split Mini-Dev into a tuning set.
- Optional clean-data study: ReViSQL's expert-verified BIRD subset, only after its redistribution terms and exact split provenance are recorded.
- Primary metric: full-500 conservative heldout execution accuracy under both behavior-compatible
  and strict evaluators; the two operationally ineligible gold cases are fixed zero for every model.
- Secondary metric: eligible-498 heldout execution accuracy under both evaluator semantics.
- Formal candidate execution, validation, and RL terminal reward share the amended 10-second wall,
  12/13-second CPU, 100k-row, 64-MiB, and 2-GiB resource boundary.
- Robustness: SQL difficulty slices, unseen databases/schemas, valid-query rate, and test-suite/multi-database equivalence where the benchmark supports it.

No final test examples enter prompts, reward tuning, early stopping, or manual data repair.

## Model route

`Qwen/Qwen3-4B-Base` is the preferred M1 candidate because it is Apache-2.0, large enough for a meaningful SQL study, and still plausible for QLoRA plus LoRA RL on 48 GB. Keep a 1.7B fallback so the end-to-end chain is never blocked. A code-specialized checkpoint may be included in the M1 baseline gate, but model comparison must not consume the research budget.

The core claim is at the selected small scale. A 7B run confirms only the single most informative comparison and never replaces missing 4B ablations.

## Fifteen-day feasibility contract

By Day 4 the repository must run data -> inference/tiny SFT -> evaluation. By Day 10 it must contain a stable SFT checkpoint and outcome-only RL checkpoint. By Day 12 it must contain at least one controlled dense/decoupled comparison. Days 13–15 finish serving, clean-room reproduction, the report, cards, demo, and interview evidence.

If time or compute degrades:

1. reduce training examples/steps while keeping identical comparison budgets;
2. fall back from 4B to 1.7B;
3. reduce formal cells from the full 2x2 to outcome-only versus dense-decoupled, preserving short diagnostic runs for the missing cells;
4. postpone clause-token credit, dynamic resampling, 7B, and all multi-turn work;
5. never drop the verifier tests, frozen evaluation, base/SFT/RL comparison, or reproducibility evidence.

The deliverable is guaranteed as a tested experimental chain, not as a guaranteed accuracy improvement.

## Code reuse policy

Reuse is encouraged when the code is high quality and license-compatible. Every imported or substantially adapted component records repository URL, exact commit, license, original file, and local modifications. We expect to reuse training infrastructure and official evaluation code heavily. The project-owned work is the consumer-GPU integration, safe verifier, reward gates, experiment matrix, diagnostics, tests, measured results, and technical explanation.

Do not present copied code as original. Do present a well-attributed reproduction plus a controlled new intervention as engineering and research work.
