# E8 Repair-SFT Smoke Protocol

## Purpose

E7 demonstrated that the fixed 256-source training-only bundle yields 688 verified repair examples
and 256 KEEP examples. E8 asks whether a short supervised warm-up teaches the E1 model to use this
feedback without increasing its tendency to rewrite correct SQL. This is a capability smoke, not a
benchmark claim and not RL.

## Leakage-safe split

The 52 E7 databases are ranked once by a fixed hash seed. Ten databases form the audit partition;
the remaining databases form the training partition. All variants of a source remain with its
database, so database IDs and source IDs have zero overlap. No project validation or Mini-Dev row is
loaded.

Training is exactly one epoch. Each training source contributes one deterministic repair example
and its one KEEP example, producing a 1:1 action balance. Other repair variants are not silently
used. The audit pool retains all variants; the measured smoke deterministically selects four
repairs and two KEEP cases from each of the ten audit databases (60 cases total).

## Comparison

The same audit identities are run under a 2x2 matrix:

| Condition | Feedback arm | Control arm |
|---|---|---|
| Immutable E1 adapter | bounded database feedback | feedback withheld |
| Repair-SFT adapter | bounded database feedback | feedback withheld |

Generation is greedy with the same token budget. `KEEP` resolves to the unchanged initial SQL;
otherwise the response must be strict SQL-only. Final correctness uses the unchanged strict
execution evaluator. Executability alone receives no credit.

## Preregistered continuation gate

The smoke passes only if all conditions hold:

- repair-SFT + feedback fixes at least five audit repairs across at least three databases;
- repair rate improves by at least 10 percentage points over E1 + feedback;
- repair-SFT feedback exceeds its no-feedback control by at least 5 percentage points;
- KEEP correctness is at least 90% and falls no more than 5 points below E1;
- all four cells produce zero unsafe actions.

A failed gate stops the direction with no repair-action RL. Thresholds are not changed after seeing
outputs.

## Runner contract

`scripts/orchestrate_e8_repair_sft.py` owns one bounded session and requires a clean, explicitly
approved commit plus external paths for the private split, retained training databases and E1
adapter. It runs the focused verification gate before model work, then:

1. `run_e8_repair_sft_train.py` verifies the E1 identity, tokenizes all 354 examples with prompt
   labels masked, and performs one remainder-correct epoch (45 optimizer updates);
2. it atomically publishes the changed adapter and SQL-free finite-loss/gradient evidence;
3. `run_e8_repair_sft_eval.py` starts as a distinct process, reloads the adapter, evaluates the
   immutable E1 and E8 adapters on the identical 60 audit cases under both arms, and writes private
   rows mode `0600` plus a SQL-free aggregate.

The real tokenizer length/mask receipt is computed before base-model loading. Any overlength row,
identity drift, non-finite training value, missing adapter delta, reload failure, unsafe action, or
session wall limit stops the run. No automatic retry is implemented.

## Compute plan

Reuse one RTX 4090 D 48 GB, the immutable E1 adapter and the existing 4-bit QLoRA stack. The fixed
60-case four-cell audit requires 240 completions. Planning estimate is 60–90 minutes; hard stop is
120 minutes or CNY 5.18 at the historical rate. This document does not authorize launch. A
runner/reload contract and the real split receipt must pass before a GPU start.
