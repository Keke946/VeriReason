# E6 execution-feedback repair probe

## Decision under test

The accepted E5 result showed that the conservative single-clause localizer could not assign
credit to any of 19 real terminal-wrong outputs. E6 poses a different hypothesis: database
execution feedback may help the unchanged E1 policy produce a better **next repair action**, even
when the original SQL's responsible tokens cannot be localized.

This first probe is inference-only. It does not create an optimizer, update an adapter, access
validation/Mini-Dev, replace the fixed examples, or reinterpret the failed E5 gate.

## Frozen comparison

The source is all 32 immutable-E1 records captured by `e5_capture_audit_smoke_001`. Every state is
used, including 13 initially correct and 19 initially wrong outputs. For each state the same E1
model generates two candidates in each arm:

- `feedback`: original question/schema, original SQL, and bounded executor feedback;
- `control`: the same repair instruction and generation budget, with executor feedback withheld.

The same per-state random seed is reset for both arms. The model may emit exactly `KEEP` or one
read-only SQL query. Gold SQL/results are never placed in either prompt; they are used only by the
unchanged strict evaluator after generation. Execution success itself receives no reward.

Feedback is bounded to executor status, normalized errors, row/column counts and at most three
result rows / 1,200 characters. Resource failures are reported as resource failures, not as clause
diagnoses. Raw SQL, database feedback and completions stay in the private result; the public summary
contains aggregates only.

## Decision rule

This is a feasibility gate, not a statistical quality claim. Promotion to a separately reviewed
repair-action RL design requires all of:

1. at least two initially wrong states repaired by the feedback arm but not by the control arm;
2. those states come from at least two distinct training examples;
3. at least two feedback state-pairs contain mixed terminal rewards;
4. feedback produces more repaired completions than control;
5. feedback damages no more initially correct completions than control;
6. neither arm produces an unsafe action.

Failure means stop or redesign with zero optimizer updates. Passing means only that a repair-action
training design is worth reviewing; it does not authorize RL training or claim validation gain.

## Planned execution

One retained 48 GB GPU, fixed Qwen3-4B-Base + E1 adapter, 128 new repair completions. Expected wall
time is 20--40 minutes; the cumulative program ceiling is 60 minutes / CNY 2.59 at the historical
rate. Model/data/cache inputs must already exist, and the output directory must be fresh and private.
