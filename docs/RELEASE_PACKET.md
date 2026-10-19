# Release and Interview Packet

## One-sentence project description

VeriReason is a reproducible single-GPU Text-to-SQL post-training study that compares QLoRA SFT
with three execution-verifiable GRPO reward designs and carries the selected 4B adapter through
sandboxed evaluation, vLLM serving, and clean-room release verification.

## Resume-ready bullets

- Built an end-to-end post-training pipeline for Qwen3-4B on BIRD, including database-disjoint data
  governance, 4-bit QLoRA SFT, execution-verifiable GRPO, isolated SQLite evaluation, and vLLM
  serving on one 48-GB GPU.
- Improved strict execution accuracy from 40.69% after SFT to 42.35% with outcome-only GRPO on an
  identical 1,209-example validation set (`+1.65` points; exact paired McNemar `p=0.01193`), while
  preserving 100% SQL-only output and zero heldout access.
- Designed and audited dense structural-reward ablations that increased mixed-reward groups from
  102/256 to 239/256 but did not beat terminal reward, establishing a controlled negative result
  and motivating clause/token-localized credit assignment.
- Served the selected LoRA through a vLLM OpenAI-compatible endpoint, completing 192/192 requests
  and scaling throughput from 1.73 to 6.51 requests/s at concurrency 1 to 8.

## Thirty-second explanation

The project asks whether more detailed, reliability-gated SQL rewards help a 4B model learn better
than a simple execution-correctness reward. I first built a leakage-controlled BIRD split and a
sandboxed SQLite verifier, then trained one QLoRA SFT baseline and three parity-locked GRPO arms.
Outcome-only GRPO was best: 42.35% strict accuracy versus 40.69% after SFT. The denser rewards made
training signal much less sparse but generalized slightly worse, so the main insight is that richer
sequence-level scores are not enough; credit likely needs to be localized to the clause or tokens
that caused the error.

## Ninety-second explanation

The input is an English question plus a deterministic serialization of an unseen SQLite schema; the
output must be exactly one read-only query. The dataset pipeline pins BIRD revisions, safely extracts
archives, partitions by database rather than row, checks leakage and database integrity, and keeps
Mini-Dev sealed. The evaluator executes SQL in a child process with immutable read-only access,
statement authorization, and wall/CPU/memory/result-size limits.

E1 is one epoch of 4-bit NF4 QLoRA with loss only on assistant SQL tokens. E2 independently starts
from E1 and uses strict execution correctness as the GRPO reward. E3 adds confidence-gated valid-SQL
and structural F1 components before group normalization; E4 normalizes each component before
summing. All RL arms use the same 256 prompts, four rollouts, optimizer, seed, and 1,209-example
exam. E2 significantly improved E1, while E3/E4 produced more nonzero advantages but did not improve
accuracy. This isolates a useful negative result rather than hiding it through post-hoc tuning.

The chosen E2 adapter was finally loaded through a production-style vLLM endpoint. The Git release
contains configs, code, tests, SQL-free evidence, cards, and a one-command CPU verification gate;
large upstream data and base weights are reconstructed from pinned revisions.

## Questions to be ready for

1. Why is execution accuracy stronger than exact SQL-string match, and where can it still be wrong?
2. Why must splits be database-disjoint rather than random rows?
3. What changes in QLoRA, and why are prompt tokens excluded from loss?
4. How does GRPO form group-relative advantages, and why do all-equal groups give no signal?
5. Why did E3/E4 produce more signal but lower validation accuracy?
6. How do the SQLite sandbox and operational-eligibility policy avoid verifier-induced label edits?
7. What does McNemar's paired test measure here?
8. What do TTFT and request throughput reveal that model accuracy does not?
9. What does `make m11-verify` reproduce, and what still requires a GPU/data download?
10. Why is clause/token-localized execution credit the next justified experiment?

These are study prompts, not completed teach-backs. Mastery remains recorded only in
`docs/INTERVIEW.md` after the user explains the design and one failure mode independently.

## Publication and storage checklist

- [ ] Run `make m11-verify` in the canonical Python 3.11.15 environment.
- [ ] Authenticate GitHub CLI and create/push `Keke946/VeriReason`.
- [ ] Confirm public CI passes the CPU job and Docker job.
- [ ] Upload the selected E2 adapter as the `v1.0.0` release asset and publish its archive SHA-256.
- [ ] Copy E1, E3, E4 adapters and private manifest/receipt evidence to a second durable location if
      no-retraining replay of every arm is desired.
- [ ] Capture missing provider invoices or leave the documented accounting limitation unchanged.
- [ ] Only after the migration audit, delete the retained provider disk; never treat Git push alone
      as proof that ignored large/private artifacts were copied.
