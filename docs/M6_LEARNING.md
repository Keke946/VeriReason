# M6 Reward Learning Note

## What GRPO receives

The verifier first produces per-completion components:

- `r_exec`: strict execution correctness, 0 or 1;
- `r_valid`: safe, supported execution validity, 0 or 1;
- `r_struct`: structural similarity in `[0, 1]`;
- `g_conf`: whether dense feedback is reliable, 0 or 1.

The confidence gate multiplies the non-terminal signals. A malformed, unsafe, schema-invalid,
timed-out, or unsupported prediction cannot earn partial credit merely by containing SQL-looking
tokens.

## Why group normalization matters

GRPO compares completions sampled for the same prompt. For values `x` in one group:

```text
z(x_i) = (x_i - group_mean) / group_population_std
```

If population standard deviation is below `1e-6`, every normalized value is zero. A group where all
completions receive the same reward provides no relative learning signal.

E2 returns terminal correctness only. E3 builds one raw utility and normalizes it. E4 separately
normalizes execution, gated validity, and gated structure before applying the fixed weights. E3 and
E4 therefore answer a real experimental question: whether a small dense component is suppressed by
the scale/variance of the combined reward.

## Minimal example

If two completions are both execution-wrong but have reliable structural scores 0 and 1:

- E2 returns `(0, 0)` and cannot distinguish them;
- E3 returns approximately `(-1, +1)` and favors the structurally closer answer;
- if `g_conf=0` for both, E3/E4 return zero dense signal.

## Core code

- `RewardComponents` carries the four frozen per-completion values.
- `population_zscores()` implements population, not sample, standard deviation and handles constant
  groups conservatively.
- `group_rewards()` implements E2, E3, and E4 without coupling the math to TRL. Framework integration
  comes only after this unit is fully tested.

## Implemented local extractor gate

The extractor now uses pinned SQLGlot 30.17.0 in SQLite mode. It reuses the independent SQL safety
scanner, parses exactly one query, rejects unsupported or nondeterministic constructs, and then
schema-qualifies identifiers. Table aliases are canonicalized and literal values become typed
placeholders: changing `age > 3` to `age > 4` does not manufacture a different structural target,
while changing a number to text does.

The eleven outputs are sets. Duplicate projections or predicates therefore cannot accumulate
reward, while extra genuine structure lowers F1 precision. If parsing, schema resolution, construct
support, or safe execution is uncertain, `g_conf=0` and `r_struct=0`; terminal execution reward is
still computed independently.

The local attack suite covers malformed SQL, unknown and ambiguous identifiers, recursive and
unused CTEs, NATURAL JOIN, parameters, nondeterministic functions, alias renaming, reordered
conjunctions, and gratuitous projections/joins. A real-schema audit over the frozen 6,496 eligible
training SQL supports 6,488 (99.8768%). Seven queries intentionally fail closed because they use
nondeterministic time functions. One executable Gold query reuses alias `T2` for both `location`
and `user`; SQLGlot cannot assign `locationid` to one physical table without ambiguity. These eight
queries keep terminal execution reward but receive `g_conf=0` and no structural reward. Heldout
remained untouched.

`scripts/audit_m6_training_schema.py` performs that next gate. It verifies the frozen training
manifest identity, selects only operationally eligible training rows, deduplicates by the frozen M2
identity of exact `(db_id, SQL)`, reads all 69 SQLite schemas in immutable read-only mode, and emits only aggregate
coverage counts. It hard-fails on heldout content, path escape, identity drift, or count drift.

An exhaustive discrete-state audit for completion groups of size 2, 4, and 8 found no E4 reward
inversion: every execution-correct completion remains above every execution-wrong completion in the
same group. This is a regression test and must be rerun if group size or reward weights change.

## TRL boundary

`TRLGroupRewardAdapter` implements only the framework seam. Under the pinned TRL 1.9.0 callback
contract, it receives standard string prompts/completions, token IDs, and repeated dataset metadata.
It requires each contiguous block of `num_generations` rows to share one explicit
`reward_group_id`, calls a supplied component scorer once per completion, and delegates every E2/E3/E4
number to `group_rewards()`. It rejects incomplete groups, reordered identities, misaligned columns,
non-integer token IDs, conversational-format inputs, and invalid scorer outputs. Tests compare two
complete groups against the framework-independent equations for all three arms. Model loading and
RL training are not part of this gate.

## Production component scorer

`SQLiteRewardComponentScorer` completes the training-time path. It resolves only relative database
paths under the configured training root, executes Gold once per contiguous completion group,
caches that Gold result and schema for the group, and executes every generated completion through
the frozen read-only evaluator. Strict result equality becomes `r_exec`; successful safe execution
becomes `r_valid`; the existing SQLGlot comparison supplies `r_struct` and `g_conf`. If Gold is
executable but structurally unsupported, terminal reward survives while dense feedback closes.
Synthetic SQLite tests cover correct, execution-equivalent, wrong-result, unsafe, path-escape, Gold
cache, and unsupported-Gold behavior without loading a model.
