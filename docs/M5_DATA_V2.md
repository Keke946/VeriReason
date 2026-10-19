# M5 Data-v2 Candidate Pool

## Outcome

The first data-v2 artifact is a versioned candidate pool, not a new training run. It selects 1,678
of the 5,364 operationally eligible training examples (31.28%) and covers all 59 training databases.
No validation or heldout row enters the pool.

The private mode-0600 manifest remains outside Git. The SQL-free public receipt is
`results/m5/m5-data-v2-candidate-receipt.json`; its selection SHA-256 is
`5bca7a995aa2f4d56cc2403164ed0719b51bda7393cfcd8a83fbb3a43f5116ef`.

## Selection rule

An example enters when its Gold SQL contains at least one of:

- two or more joins;
- a subquery;
- `GROUP BY`;
- `HAVING`.

The rule is deterministic and outcome-blind. It reads only the training split and Gold structural
features. It does not read validation IDs, validation SQL, model predictions, execution rewards, or
heldout data.

Reason counts overlap because one example can exercise several capabilities:

| Reason | Candidate count |
|---|---:|
| At least two joins | 1,048 |
| Group-by | 495 |
| Subquery | 393 |
| Having | 62 |

The pool contains 493 examples that the old proxy calls `simple`. This is not a contradiction: the
old proxy gives one point for the presence of joins regardless of whether there are one, two, or
five. The new multi-join signal captures schema-path difficulty that the coarse label misses.

## What this artifact does not decide

Selection is not permission to duplicate all 1,678 rows in SFT. Three possible consumers remain
separate experiments:

1. Targeted replay SFT: mix a controlled fraction of candidates with the original distribution.
2. Correction SFT: run the current model on training candidates, retain verified failures, and train
   on correction pairs. This is more targeted but requires GPU generation and safe execution.
3. Verifiable RL: use the candidate pool for informative groups, while preserving a comparable
   evaluation protocol.

The project should choose among these only after estimating compute and checking whether the
candidate pool is sufficiently diverse by reason and database. The original M4 result remains the
fixed E1 baseline and is never overwritten.

## Core implementation

`build_data_v2_candidates()` in `src/verireason/m5_data_v2.py` hard-fails on non-training rows or
duplicate IDs, calculates structural reasons, and returns two different artifacts:

- a private manifest containing identities and selection reasons;
- a public receipt containing only counts, policy, hashes, and privacy assertions.

This split prevents benchmark leakage while keeping the experiment reproducible.
