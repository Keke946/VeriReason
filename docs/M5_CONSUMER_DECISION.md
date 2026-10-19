# M5 Consumer Decision

## Decision

Use the 1,678-row data-v2 pool in two roles:

1. `E1T`, a bounded targeted-replay SFT control;
2. a candidate source for the E2–E4 RL sampling policy.

Correction SFT remains a conditionally triggered backup. This decision freezes experiment roles and
causal relationships, not unmeasured optimizer or rollout hyperparameters.

## Causal graph

All experimental branches start from the same immutable M4 `E1` adapter:

```text
E1 ──> E1T  targeted replay SFT control
 |
 ├──> E2   terminal execution reward
 ├──> E3   terminal + reliable dense reward, sum then normalize
 └──> E4   same components, normalize each reward then sum
```

E1T can never replace E1 as the RL parent. Otherwise the E2–E4 effect would mix RL with additional
supervised training, destroying the intended causal comparison.

## Why E1T is bounded

E1T answers one narrow question: can simply replaying difficult Gold examples explain an apparent
RL gain? It gets at most one pass over the frozen candidate pool and must have separately
preregistered optimizer, schedule, evaluation promotion gate, time, and cost before execution. It
must not delay the outcome-only E2 baseline.

## Why Correction SFT is conditional

Correction SFT needs model generation over training rows, safe execution, equivalence/noise review,
and a new correction interface. It is activated only if a frozen trigger fires—for example, the RL
smoke produces mostly constant-zero groups or stable RL fails specifically on schema grounding.
Without a trigger, it adds cost and confounds the main research question.

## Evaluation boundary

The full 1,209 validation set has already informed M5 error analysis. It is therefore development
evidence, not an untouched final test. Existing dev256 may gate promotion and final953 may compare
promoted experiments. The private BIRD Mini-Dev 500 remains untouched and is used only once after
the experiment matrix and model-selection rules are frozen.

This is adaptive development, not leakage: no validation row enters training, and the final heldout
exam is still isolated.

## Self-audit

- Direction matches the charter's central RLVR research question.
- E1T is a control, not a new headline or replacement baseline.
- E2, E3, and E4 retain a common parent and evaluator.
- Candidate selection contains training rows only and uses no model outcomes.
- Validation-guided capability selection is disclosed.
- Heldout remains unused and unauthorized.
- No optimizer, rollout count, learning rate, or cloud budget is frozen before measurement.
- A negative result remains publishable; no branch needs to improve accuracy to count as complete.
