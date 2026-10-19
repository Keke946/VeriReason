# Project Charter

## Working title

VeriSQL-RL: trustworthy and sample-efficient Text-to-SQL post-training on consumer GPUs.

## Problem

Many student LLM projects stop at a fine-tuning notebook or a chat UI. VeriReason must demonstrate a defensible end-to-end engineering and experimental chain: governed data, a frozen evaluation protocol, supervised fine-tuning, verifiable-reward post-training, reward-hacking analysis, ablations, efficient serving, and reproducible evidence.

## Intended audience

- LLM algorithm and model-engineering interviewers
- Applied ML teams that value training, evaluation, and deployment together
- Reviewers who need to reproduce the central claims on modest cloud hardware

## In scope

- Cross-domain Text-to-SQL with automatically verifiable outcomes
- A pretrained open-weight model in the approximately 1.7B–4B range for the core study, with one optional 7B confirmation run
- Dataset construction, cleaning, deduplication, versioning, and frozen splits
- Base-model evaluation
- Parameter-efficient SFT
- Outcome-only GRPO as the post-training baseline
- A reliability-gated dense SQL reward and a justified modern GRPO-family training configuration
- Reward unit tests and reward-hacking analysis
- Base versus SFT versus post-trained evaluation
- A small, hypothesis-driven ablation suite
- Quantized or memory-conscious serving and latency/throughput measurement
- Public-quality README, model/data cards, technical report, and interview notes

## Out of scope

- Foundation-model pretraining
- Multi-node or cluster-scale training as a requirement
- Multiple unrelated task domains
- Unbounded hyperparameter sweeps
- A UI-only chatbot presented as the main contribution
- Claims without saved, reproducible evidence

## Accepted technical direction

- Foundation: Hugging Face TRL/PEFT with only the smallest necessary custom trainer surface
- Task: cross-domain Text-to-SQL with sandboxed execution as the central verifier
- Adaptation: QLoRA SFT followed by LoRA-based verifiable-reward post-training
- Baseline reward: terminal execution correctness
- Research intervention: reliable dense structural feedback, multi-reward normalization, and optional informative-group sampling
- Evaluation: official task evaluator plus a reusable verifier, robustness slices, reward diagnostics, and GPU-hour/sample-efficiency accounting
- Serving: vLLM or another justified local serving engine

The exact 1.7B–4B checkpoint is selected by the M1 quality/throughput/license gate. The 15-day core does not depend on a 7B model, multi-turn agent training, a teacher model, or an unavailable research repository.

## Central research question

Under a consumer-GPU budget, can reliability-gated dense SQL feedback improve the sample efficiency and robust execution accuracy of small-model RLVR over terminal execution reward alone, without increasing reward hacking? The core study separates two effects:

1. reward information: terminal-only versus terminal plus reliable partial structural feedback;
2. multi-reward optimization: sum-then-normalize GRPO versus per-reward normalize-then-sum aggregation.

Clause-token credit assignment, dynamic resampling, and multi-turn execution correction are gated upgrades rather than dependencies of the core result.

## Definition of done

The project is resume-ready when all of the following are true:

1. A clean environment can run a documented smoke path from data to evaluation.
2. Dataset versions and frozen splits are reproducible and checked for leakage.
3. Base, SFT, and post-trained checkpoints are evaluated with the same protocol.
4. At least four hypothesis-driven ablations or controlled comparisons are reported.
5. Reward functions have tests and documented attack/failure cases.
6. Every headline number maps to an experiment ID, Git commit, config, and result file.
7. A serving endpoint is benchmarked for correctness, memory, latency, and throughput.
8. README, technical report, model/data cards, architecture diagram, and demo are complete.
9. The user can explain the central training, reward, evaluation, and serving trade-offs in a mock interview.

The result does not have to show a positive gain to be complete. A well-controlled negative result is acceptable when the implementation, diagnostics, attack analysis, and conclusion are reproducible.

## Day 0 decisions

- The 1.7B–4B model gate and E0–E4 core experiment ladder are accepted.
- The 48 GB-first compute route and CNY 250 core cap are accepted.
- Exact model, environment, and optional two-GPU topology remain measurements made at M1, not scope decisions left open at Day 0.
