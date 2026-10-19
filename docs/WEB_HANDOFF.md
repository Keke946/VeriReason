# Web and CLI Handoff

This page is the shortest safe entry point for continuing VeriReason from a new Codex task that
does not share the original chat history.

## Repository and current result

- Public repository: `https://github.com/Keke946/VeriReason`
- Stable release: `v1.0.0`
- Selected model: Qwen3-4B-Base plus the released E2 LoRA adapter
- Best frozen validation result: 42.3490% strict execution accuracy on 1,209 examples
- SFT baseline: 40.6948%; paired exact McNemar p-value: 0.01193
- Dense E3/E4 rewards increased training signal but did not outperform E2
- Mini-Dev heldout use remains zero

## Required reading order

Before changing code or running experiments, read:

1. `AGENTS.md`
2. `docs/PROJECT.md`
3. `docs/PLAN.md`
4. `docs/STATE.md`
5. `docs/DECISIONS.md`
6. `experiments/registry.yaml`
7. `docs/COMPUTE.md`
8. `experiments/spend.csv`
9. `docs/RESEARCH_FOLLOWUP.md`

Git-tracked files are the source of truth. Do not infer experimental facts from chat history.

## CPU-only continuation

The normal development path does not require a GPU:

```bash
git clone https://github.com/Keke946/VeriReason.git
cd VeriReason
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make verify
make m11-verify
```

Codex Cloud can clone the GitHub repository, edit it, run these tests, and return a branch or pull
request. A local Codex CLI can perform the same work directly on the Mac. Ordinary ChatGPT web
chats can review uploaded files but cannot directly access the Mac filesystem.

## Next research gate

The only recommended E5 direction is terminal-anchored clause/token-localized execution credit.
Start with a CPU-only feasibility implementation and adversarial unit tests. Do not start GPU work
until the local gate can localize a failure conservatively and fall back to E2 terminal reward when
the responsible clause is ambiguous. DAPO-style informative-group filtering is a secondary
efficiency ablation, not the main contribution.

## Boundaries

- Do not access Mini-Dev or change the frozen 1,209-example validation exam.
- Do not start paid GPU resources without a named experiment, estimate, and user approval.
- Do not publish private manifests, raw SQL evidence, credentials, or provider data.
- The public release contains the selected E2 adapter. E1/E3/E4 adapters are optional archival
  artifacts; reproducing those arms from the pinned configs requires retraining if they are waived.

## Suggested first prompt for a new web task

> Clone `Keke946/VeriReason`, read the required files listed in `docs/WEB_HANDOFF.md`, and report the
> current milestone, accepted headline result, unresolved M11 storage decision, and the CPU-only E5
> feasibility gate. Do not edit files, start GPU resources, or access heldout data until after that
> report.
