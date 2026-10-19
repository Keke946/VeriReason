# M3 Learning Notes

Last updated: 2026-08-24

This note preserves the concepts the user temporarily delegated during Day 4A. It is a later
teach-back aid, not evidence that the interview gate has been completed.

## Data identity

A sample ID says which question is selected. A SHA-256 says whether the complete ordered selection
and its contract changed. `select_tiny_slice()` in `src/verireason/m3_contract.py` ranks eligible
rows with a fixed seed, preserves the M2 split, covers required SQL features, and hashes the ordered
sample IDs. It never uses model results to choose easier examples.

## Train, validation, and heldout

Train supplies gold SQL for parameter learning. Validation is the frozen internal exam used for the
Base/SFT comparison. Heldout is the future final exam: merely reading its failures and changing a
prompt would consume information even if no heldout row were trained. M3 accepts only
`operational_eligible=true` train/validation rows and records zero heldout loaded/evaluated counts.

## Prompt, tokenizer, labels, and loss mask

The prompt contains a SQL-only system instruction, deterministic schema, and question. The target
is the full gold SQL. The Qwen tokenizer turns both into integer token IDs. Labels are `-100` over
system/user/assistant-prefix/padding tokens and equal the target token IDs over SQL plus EOS. Thus
the model sees the prompt as context but cross-entropy measures only next-token prediction of the
answer. See `build_messages()`, `tokenize_answer_only()`, and `AnswerOnlyCollator`.

## Loss, gradients, and a real update

At each answer position, cross-entropy is small when the model assigns high probability to the gold
next token and large when it assigns low probability. Backpropagation turns loss into gradients—the
direction each trainable LoRA parameter should move. Gradient accumulation collects several small
micro-batches before `optimizer.step()`. A log containing loss alone does not prove training.

`train_adapter()` records positive supervised-token and trainable-parameter counts, finite loss,
nonzero finite gradient norm, at least one optimizer step, and a nonzero before/after LoRA parameter
delta. `verify_update_evidence()` rejects missing evidence.

## Why Base/SFT comparison is fair

`comparison_contract()` hashes the ordered validation slice, every prompt, decoding configuration,
model identity, and evaluator limits. The fresh reload process must reproduce that exact contract.
The two systems therefore answer the same exam under the same rules; a negative SFT result is still
valid.

## Adapter reload

Saving files is insufficient proof. `run_m3_base_train.py` saves the adapter and records its file
hashes. `run_m3_reload_eval.py` starts as a separate operating-system process, checks those hashes,
loads the fixed base plus PEFT adapter, then evaluates. This detects missing, corrupt, or
process-memory-only state.

## Why the fixed slice has eight training examples

One micro-batch contains one example. Four micro-batches accumulate gradients before one
`optimizer.step()`, and the smoke permits exactly two optimizer steps. Therefore `1 × 4 × 2 = 8`
examples fills both accumulation windows exactly; no partial gradient bucket is silently discarded.
Validation also uses eight fixed questions because this is an engineering same-exam smoke, not an
accuracy estimate. Real token inspection may veto this preregistration, but it cannot replace a
failed fixed row or choose a more favorable seed.

## Paid-session safety gates

`scripts/orchestrate_m3.py` checks an externally approved full Git SHA, a clean tree, a brand-new
resolved output directory outside Git, and at least 20 GiB free on the actual output filesystem.
Every subprocess shares one 14,400-second countdown. On expiry, the orchestrator records timeout
evidence, terminates the active process group, and escalates to kill after a short grace period.
The fourth hour is safety headroom: it does not add samples, steps, retries, or tuning. Important
evidence is atomically replaced or incrementally appended so a provider shutdown does not erase
already completed stages.

## Later teach-back questions

1. Why can validation be used in M3 while heldout cannot?
2. Why are prompt tokens visible to the model but assigned label `-100`?
3. Why do loss and gradient norm alone not prove an optimizer update?
4. What does the validation slice hash protect against?
5. Why must the adapter be evaluated in a fresh process?

## What the real slice demonstrated

Attempt 003 turned the earlier contracts into measured evidence. Eight one-example micro-batches
filled two four-batch accumulation windows, so two `optimizer.step()` calls occurred. Loss changed
from 0.6484 to 0.4097, gradient norm was 1.3314, and the LoRA parameters moved by L2 1.4325. The
saved adapter was then loaded by PID 4775 after the training process PID 1666 ended.

On the same eight validation prompts, Base produced no extractable SQL and scored 0/8. Tiny SFT
produced SQL-only output on 8/8 and executed correctly on 3/8; four queries returned the wrong result
and one timed out. This tiny set cannot estimate general accuracy, but it proves that data, loss,
optimizer, serialization, reload, generation, and safe execution work end to end. The user teach-back
is still pending; these notes do not mark the concepts mastered.
