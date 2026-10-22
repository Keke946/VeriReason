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
used. The audit partition retains all repair variants and its KEEP examples.

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

## Compute plan

Reuse one RTX 4090 D 48 GB, the immutable E1 adapter and the existing 4-bit QLoRA stack. Planning
estimate is 30 minutes; hard stop is 60 minutes or CNY 2.59 at the historical rate. This document
does not authorize launch. A runner/reload contract and the real split receipt must pass before a
GPU start.
