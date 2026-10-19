# VeriReason Working Agreement

## Mission

Build a resume-grade, reproducible small-model post-training project covering data, supervised fine-tuning, verifiable-reward training, evaluation, serving, and technical communication. Pretraining a foundation model is out of scope.

## Required context at the start of every task

Before editing or running expensive work, read:

1. `docs/PROJECT.md`
2. `docs/PLAN.md`
3. `docs/STATE.md`
4. `docs/DECISIONS.md`
5. `experiments/registry.yaml`
6. `docs/COMPUTE.md`
7. `experiments/spend.csv`

Then inspect `git status` and the five most recent commits. Summarize the current phase, the next acceptance criterion, and any dirty files before acting.

## Source of truth

- Git-tracked files are authoritative. Chat transcripts and external notes are not.
- `docs/PROJECT.md` owns stable scope and the final definition of done.
- `docs/PLAN.md` owns milestone status and acceptance criteria.
- `docs/STATE.md` owns the current snapshot and exactly one next action.
- `docs/DECISIONS.md` owns durable architectural and experiment decisions.
- `experiments/registry.yaml` owns experiment identity, configuration, hardware, status, and measured metrics.
- `docs/COMPUTE.md` owns the approved hardware and cost policy; `experiments/spend.csv` owns paid-session accounting.
- Machine-generated raw outputs live under `results/` or an ignored artifact directory. Never type invented metrics into documentation.

## Work discipline

- Work on one milestone and one explicit outcome at a time.
- Do not silently change the frozen task, test split, primary metric, base model, or compute budget.
- Start with a cheap smoke run before any full training run.
- Do not download large models or datasets, start paid cloud resources, or publish artifacts without user confirmation.
- Do not start a run that violates the spend gates in `docs/COMPUTE.md`.
- Never commit secrets, access tokens, raw private data, model checkpoints, caches, or large generated artifacts.
- Preserve user changes and avoid destructive Git commands.
- Prefer configuration files over hard-coded experiment parameters.
- Every reward function and evaluator requires positive, negative, and adversarial tests.
- Any result reported in the README or report must be traceable to an experiment ID and a saved result file.

## Parallel work

- Use subagents only when the user explicitly requests delegation or parallel agent work for the phase.
- Prefer subagents for read-heavy exploration, dataset/license review, test design, log analysis, and independent code review.
- Avoid parallel edits to core schemas, training configuration, `docs/STATE.md`, `docs/PLAN.md`, or `experiments/registry.yaml`.
- Use separate branches/worktrees for independent write-heavy tasks. The main branch has one integrator.

## New task and agent handoff protocol

- Never silently open, request, or start another Codex task or subagent.
- When a new task or agent would materially help, first tell the user explicitly:
  1. whether to open a new Codex task, use internal subagents, or continue in the current task;
  2. the exact number of new tasks or agents;
  3. why each one is needed and what it owns;
  4. the exact copy-paste prompt for each new Codex task, or the exact assignment for each proposed subagent;
  5. which source-of-truth files it must read before acting;
  6. what return packet it must bring back.
- Wait for explicit user authorization before creating or using a subagent. User-created Codex tasks are opened by the user unless the user explicitly asks Codex to create them.
- The required return packet is: outcome summary, files changed, commands and tests run, experiment IDs and measured evidence, unresolved risks or decisions, Git commit/hash when applicable, and the proposed single next action.
- The control task integrates returned work and updates `docs/STATE.md`; execution tasks and agents must not independently advance the global milestone unless their prompt grants that ownership.

## Completion protocol

A task is not complete until it:

1. runs `make verify` plus the milestone's verification commands;
2. records experiments and real metrics in `experiments/registry.yaml`;
3. updates `docs/STATE.md` and, when applicable, `docs/PLAN.md` and `docs/DECISIONS.md`;
4. reports changed files, commands run, evidence, unresolved issues, and the next action;
5. leaves the working tree in a reviewable state.

The committed pre-commit hook and CI workflow enforce the durable-state guard. Do not bypass the hook to force an inconsistent commit.

## Learning and interview readiness

At the SFT, GRPO, evaluation, and serving gates, pause for a user teach-back. Record confirmed explanations and unresolved questions in `docs/INTERVIEW.md`. A phase is technically complete only when its code and metrics pass; it is interview-ready only after the user can explain the main design choices and failure modes.
