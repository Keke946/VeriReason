# Post-v1 E5: CPU-only feasibility gate

Owner: BiBo. Experiment: `e5_cpu_feasibility_001`. Date: 2026-10-20.
Scope: synthetic training-only SQLite, no model, GPU, cloud launch, validation rows, or Mini-Dev.
Accepted E1–E4 evidence, reward equations, evaluator, configs, and v1 remain unchanged.

## Hypothesis and implementation audit

E3/E4 supplied sequence-level structural utilities. E5 instead asks whether reliably identified
error regions should receive the negative terminal-derived policy gradient. E2 remains the anchor:
strict execution correctness is the only reward and final objective. This is an EXPO-SQL-inspired
research variant, not a replication or a claim to implement the paper's full algorithm.

Reviewed `reward_contract.py`, `m6_component_scorer.py`, `sql_evaluator.py`, `sql_features.py`,
`trl_reward_adapter.py`, `m7_runtime.py`, and `scripts/run_m7_smoke_train.py`:

- E2's callback returns binary `r_exec`; the trainer performs group centering. Its accepted config
  uses `scale_rewards=none`, DAPO loss, and beta=0. No second normalization is appropriate.
- M6 already provides isolated read-only execution, strict comparison, schema loading, and a pinned
  SQLGlot parser/qualifier. E5 reuses these; no evaluator or reward-weight fork is needed.
- The existing callback returns one scalar per completion. It cannot communicate token advantages.
  A future narrow, pinned-TRL loss integration is required; E5 is deliberately not a registered
  trainable arm. The original E2/E3/E4 runner cannot launch it by changing a config.
- The M6 structural representation masks literals. E5 must instead retain literal values in AST
  comparisons, or predicate/limit mistakes would disappear.

## Conservative localization algorithm

1. Reject any split other than `train` before I/O. The diagnostic entry point accepts no input data
   path: it creates a disposable synthetic database. A split string alone is NOT a provenance gate
   for future real-data use; that requires the existing frozen train-manifest loader.
2. Run gold and prediction through the unchanged evaluator. Gold failure aborts scoring; never
   fabricate terminal truth. Return ordinary E2 credit immediately for strict-correct predictions,
   including equivalent SQL. Execution failures and comparator resource failures abstain.
3. Admit only a single flat SELECT with known clause arguments and resolvable schema. Reject
   nested queries/CTEs, set operations, windows, self-joins, aliases (including safe aliases for now),
   uncertain functions, unsupported modifiers, and bare aggregate columns. These restrictions
   apply only to localization, never to terminal eligibility.
4. Compare seven literal-preserving AST clause signatures: SELECT, FROM/JOIN, WHERE, GROUP BY,
   HAVING, ORDER BY, LIMIT. Ignore comments, but do not canonicalize away SQL values. Require
   exactly one changed clause and an existing source span on BOTH sides. Missing clauses have no
   generated token target and abstain. LIMIT currently abstains because total ordering is unproven;
   presence of ORDER BY alone would not prove it.
5. Use SQLGlot token positions to locate that clause in the ORIGINAL completion. FROM includes all
   JOIN/ON text. Quoted keywords and parentheses cannot introduce top-level clause boundaries.
6. Splice only that clause from the training reference, reparse the actual splice, check its AST
   against the reference, and require strict execution correctness after repair. Otherwise abstain.
   At most three sandbox executions per sample; no prefix search or unbounded repair search.

This establishes **reference-relative single-clause repair evidence on the observed database**,
not unique semantic causality. A first divergent prefix is deliberately not accepted as attribution:
later aggregation, aliases, and equivalent rewrites can invalidate that inference. Multiple
interacting errors, engine error-message localization, and general incremental execution are deferred.

## Credit contract and exact fallback

`ExecutionCredit` contains binary terminal reward, reason, optional character span, and the exact
source completion for alignment (no added hashes). `TokenCredit` contains per-token advantages,
a mask, and a fallback reason. SQL-containing objects remain transient; aggregate evidence is SQL-free.

For E2's existing centered terminal advantage A and mask m:

- Correct completions and nonnegative advantages: preserve A at every completion token.
- Reliably localized incorrect completion with A < 0: use A * m, with m=1 only in its clause.
- Abstention or uncertain tokenizer offsets: use A everywhere, exactly E2 terminal-only behavior.
- All-wrong and all-correct groups retain zero advantage; this prototype does NOT solve E2's
  zero-variance problem by inventing local positive reward or filtering prompts.

Offsets are Python character offsets into the exact completion. Reject incomplete coverage,
overlap, stale completion text, out-of-range offsets, and tokens crossing a clause boundary with
non-whitespace content on both sides. Special (0,0) tokens receive no local credit. The caller must
still apply E2's ordinary EOS/padding/truncation mask. Preserve the DAPO denominator; do not normalize
by selected-token count. This reduces total negative gradient mass on localized samples and must be
measured as part of the intervention, not hidden as a purely cosmetic change.

## Acceptance and decision rules

Before interpreting results: zero unexpected localization/terminal outcomes on the authored suite;
nonzero useful coverage across several clauses; exact fallback and no positive local reward;
read-only safety; all eight requested test categories; `make verify` and focused tests pass.
Report coverage over all cases AND incorrect cases, and abstention over incorrect cases separately.
No coverage threshold is tuned against validation, and this authored set is not a population sample.

- **Go:** local invariants plus representative training-only audit and actual tokenizer/loss CPU
  alignment evidence support a separately reviewable smoke packet. Never automatic GPU permission.
- **Revise:** bounded prototype works but coverage, ambiguity, or integration evidence is incomplete.
- **No-Go:** unresolved unsafe execution, systematic false attribution, or failure to preserve E2.

## Risks and bounded next gate

1. Reference bias: one-clause repair can reward benchmark-specific syntax; one database cannot
   distinguish all semantically wrong queries from coincidental execution matches.
2. Low/biased coverage: no invalid-SQL localization, aliases, nesting, missing clauses, or LIMIT.
   The negative-result path may dominate real rollouts; authored cases cannot estimate its rate.
3. Reward hacking: empty-result predicates, constants, duplicate rows, nondeterministic functions,
   comment keywords, unsafe statements, and alias tricks are exercised. No partial scalar reward
   is exposed. Clause padding, equivalent rewrites, and choosing locally penalized versus fallback
   forms can still change gradient mass; absence of an observed exploit is not proof of safety.
4. Token alignment: character-offset tests are not a real Qwen tokenizer proof. BPE merges,
   Unicode normalization/byte offsets, EOS, truncation, distributed row ordering, and loss reduction
   require an exact pinned-runtime CPU integration gate before any GPU smoke.
5. Additional evaluator latency is bounded per sample but not measured on real training databases.

Decision for this stage: **Revise; no 8-prompt GPU smoke**. See the generated
`results/e5/e5-cpu-feasibility.json` and registry for measured counts. The single next action is a
CPU-only tokenizer/loss integration and training-provenance audit packet; if local training artifacts
are unavailable, record that blocker without starting cloud or substituting validation data.

## Verification receipt

Exact Python 3.11.15 with `requirements/m10-cpu-lock.txt` passed `make verify` (389 passed,
four optional-runtime skips) and `make e5-verify` (69 focused tests plus the synthetic audit).
Ruff, compileall, and `uv pip check --python .venv-e5/bin/python` check the CPU environment.
Commands use `PYTHON=.venv-e5/bin/python`. The audit JSON records its measured wall time and
runtime versions; no torch, TRL, Transformers, or vLLM module was imported by that runner.

The subsequent CPU interface gate is documented in `docs/E5_INTERFACE.md`. It verifies real
Qwen tokenizer geometry and analytical DAPO loss semantics, but retains **Revise** because real
training replay payloads and original generated IDs are unavailable in this checkout.
