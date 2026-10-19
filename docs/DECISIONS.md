# Decision Log

Decisions are append-only. If a decision changes, add a superseding entry rather than rewriting history.

## ADR-000 — Repository is the project memory

- Date: 2026-08-13
- Status: accepted
- Decision: Git-tracked files, experiment records, and commits are authoritative. Chats and optional external documents are not sources of truth.
- Rationale: The project must survive context compaction, phase-specific chats, and parallel work without losing decisions or evidence.
- Consequence: Every execution task must update project state before completion.

## ADR-001 — Exclude foundation-model pretraining

- Date: 2026-08-13
- Status: accepted
- Decision: Use an existing open-weight pretrained model; invest project time in SFT, verifiable-reward post-training, evaluation, and deployment.
- Rationale: This matches the learning objective, compute constraints, and compressed delivery schedule.
- Consequence: Any pretraining exercise is optional background learning and cannot block the project.

## ADR-002 — Fifteen-day staged delivery

- Date: 2026-08-13
- Status: accepted
- Decision: Target a resume-ready result in fifteen concentrated working days, with a vertical slice by Day 4 and the central SFT/GRPO result by Day 10.
- Rationale: Preserve the full technical chain while compressing wall-clock time through scope control, small models, smoke runs, and parallel non-conflicting work.
- Consequence: Only one domain and a small hypothesis-driven experiment matrix are allowed.

## ADR-003 — Task domain

- Date: 2026-08-13
- Status: proposed
- Decision: Use cross-domain Text-to-SQL as the verifiable task unless the user rejects it after reviewing a concrete example.
- Rationale: Both provide executable rewards; Text-to-SQL offers a strong combination of model training, structured generation, database execution, and product relevance.
- Consequence: The model will map a natural-language question plus database schema to SQL; execution results provide the central correctness signal. Dataset, verifier details, and split remain unfrozen until M2.

## ADR-004 — Mechanically enforce project-state invariants

- Date: 2026-08-13
- Status: accepted
- Decision: Run `scripts/check_project_state.py` through `make verify`, the repository pre-commit hook, and CI.
- Rationale: Collaboration rules must be testable and must fail loudly when plan, state, or required evidence becomes inconsistent.
- Consequence: A task cannot be considered complete or committed when the guard fails; milestone-specific tests will be added to this baseline check as implementation grows.

## ADR-005 — Cloud compute envelope

- Date: 2026-08-13
- Status: proposed
- Decision: Use one on-demand 4090 D 48 GB instance at the displayed CNY 2.59/hour for M1–M9 GPU work. Target CNY 150–200 total, impose a CNY 250 hard cap, and require explicit confirmation for a single run estimated above CNY 30.
- Rationale: The 48 GB option leaves useful headroom for 1.7B QLoRA/GRPO rollouts and reduces failed-run risk. Renting only during active work is more important than minimizing the hourly price.
- Consequence: No cloud deployment occurs until M1 reviews the image and persistence settings; every paid session is entered in `experiments/spend.csv`.

## ADR-006 — Delivery deadline

- Date: 2026-08-13
- Status: accepted
- Decision: Complete the fifteen working-day plan by 2026-08-31, with daily workload allowed to vary.
- Rationale: The user can flex daily hours but wants a two-to-three-week delivery before September.
- Consequence: Wall-clock compression comes from staged smoke runs, bounded experiments, and parallel non-conflicting work, not from deleting the SFT/GRPO/evaluation chain.

## ADR-007 — Public release posture

- Date: 2026-08-13
- Status: accepted
- Decision: Prepare a public GitHub repository, report, and demo. Publish processed data and model adapters only after upstream license and redistribution checks.
- Rationale: A public, reproducible artifact materially improves resume credibility and is acceptable to the user.
- Consequence: License provenance, secrets scanning, data cards, and model cards are release gates.

## ADR-008 — Text-to-SQL domain is accepted

- Date: 2026-08-13
- Status: accepted
- Supersedes: ADR-003
- Decision: Use cross-domain Text-to-SQL as the single project domain, with database execution as the central verifiable outcome.
- Rationale: The user accepted Text-to-SQL after comparing alternatives, while requiring that the project contain a defensible research question rather than only a library call or a copy of one reward function.
- Consequence: All implementation, data, evaluation, post-training, serving, and interview work must support this domain. Changing domains requires a new superseding ADR.

## ADR-009 — Fifteen-day core is protected from research upgrades

- Date: 2026-08-13
- Status: accepted
- Decision: Preserve a complete fifteen-day release containing data, evaluation, SFT, outcome-only RLVR, one focused reward/optimization comparison, serving, and documentation. Clause-token credit, 7B scaling, teacher-guided learning, and multi-turn agent RL are post-core upgrades.
- Rationale: Cutting-edge additions improve the ceiling only if they do not create a single point of failure for the minimum credible project.
- Consequence: Upgrade work cannot begin until the core path it depends on passes its smoke gate. A negative research result is publishable; a missing reproducible core is not.

## ADR-010 — Reliability-gated dense reward study

- Date: 2026-08-13
- Status: proposed
- Decision: Study whether terminal execution reward plus reliable partial SQL feedback improves sample efficiency and held-out accuracy over terminal reward alone. Compare standard sum-then-normalize aggregation with per-reward normalize-then-sum aggregation, and measure reward hacking and zero-variance groups.
- Rationale: Recent Text-to-SQL work identifies coarse or sparse credit as a bottleneck, while recent general LLM-RL work identifies multi-reward normalization and uninformative groups as optimization bottlenecks. Current TRL exposes the required multi-reward, DAPO-loss, vLLM, and GDPO-style aggregation primitives, making the core comparison implementable without recreating a distributed RL stack.
- Consequence: M2 must freeze the reward equations and confidence gates before formal training. Clause-token weighting and dynamic resampling remain U1 until the core trainer is stable.

## ADR-011 — Revised model envelope and compute topology gate

- Date: 2026-08-13
- Status: proposed
- Supersedes on acceptance: ADR-005's 1.7B-only model assumption; the existing cost cap remains unchanged
- Decision: Prefer an Apache-2.0 4B base model for the central study, retain a 1.7B fallback, and decide between one 48 GB GPU and two 24 GB GPUs from a measured M1 smoke benchmark. Use four 24 GB GPUs only for an approved 7B confirmation; do not plan around seven GPUs.
- Rationale: A 4B model offers a stronger SQL reasoning substrate while remaining compatible with LoRA/QLoRA on 48 GB. One 48 GB card is cheaper and simpler at the displayed prices; two 24 GB cards may be faster when rollout serving and training fit separately.
- Consequence: No model download, cloud deployment, or paid benchmark begins before user confirmation and M1 image/storage review.

## ADR-012 — Day 0 research and compute route is accepted

- Date: 2026-08-13
- Status: accepted
- Accepts: ADR-010 and ADR-011
- Supersedes: ADR-005 where its model-size assumption or target budget differs
- Decision: Use the E0–E4 reliability-gated dense-reward study as the fifteen-day core; prefer an Apache-compatible 4B model with a 1.7B fallback; begin with one 4090 D 48 GB; retain a CNY 250 hard cap and the per-run approval gate.
- Rationale: After a detailed walkthrough, the user confirmed the protected core, research ladder, fallback route, and staged upgrades.
- Consequence: M0 is complete. M1 may measure environment, model, storage, and topology choices but may not silently change the research question or spend policy.

## ADR-013 — Explicit task and agent handoffs

- Date: 2026-08-13
- Status: accepted
- Decision: Before requesting a new Codex task or any subagent, explicitly tell the user the exact count, purpose, copy-paste prompt or assignment, file ownership, and required return packet. Wait for authorization before using subagents; do not silently create user-facing tasks.
- Rationale: Long projects need multiple bounded contexts without losing user visibility, ownership, or integration discipline.
- Consequence: Each handoff returns an outcome summary, changed files, commands/tests, experiment evidence, unresolved issues, commit/hash when applicable, and exactly one proposed next action. The control task remains responsible for global state integration.

## ADR-014 — M1 environment, model, and topology selection

- Date: 2026-08-14
- Status: accepted
- Decision: Lock the M1 environment in `configs/environment.toml`, select `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`, retain the fixed Qwen3-1.7B revision as fallback, and use one RTX 4090 D 48 GB. Do not run a paid 2x24 GB comparison without a later quantified bottleneck.
- Rationale: Both candidates produced execution-correct SQL on all four engineering-smoke cases, but 4B followed the SQL-only instruction on 4/4 cases versus 1/4 for 1.7B. The 4B run used 8158 MiB peak device memory and generated 122 tokens at 28.33 completion tok/s, leaving ample single-device headroom. Two 24 GB GPUs cost more and add distribution complexity without solving an observed constraint.
- Consequence: M2 and the core study use the fixed 4B revision. The smoke results establish viability and relative instruction following only; they are not reported as benchmark accuracy. Reconsider topology only after a registered workload demonstrates a concrete memory or rollout/training concurrency bottleneck.

## ADR-015 — Accept M1 closure with an explicit provenance limitation

- Date: 2026-08-14
- Status: accepted
- Decision: Accept M1 after independent control-task verification and proceed to M2 without restarting the paid GPU instance solely to recover the missing transitive `pip freeze`.
- Rationale: The direct environment versions, model revisions, raw result files, source-tree reconstruction, deterministic SQL-only recount, and stable closure manifest are all recorded and independently verified. A paid restart would recover one missing secondary artifact but would not materially change the model/topology gate.
- Consequence: The missing GPU `pip freeze` must remain disclosed. All formal experiments from M2 onward must start from a clean committed source tree and capture the complete environment and source identity before execution.

## ADR-016 — Freeze the M2 BIRD data, evaluator, and reward contract

- Date: 2026-08-14
- Status: accepted for Day 3A; materialization pending control approval
- Decision: Use fixed-revision filtered BIRD labels for training/development with a deterministic 59/10 database split and original BIRD Mini-Dev SQLite as isolated heldout evaluation. Report independently authored `bird_ex_behavior_compatible` and strict execution accuracy, with strict correctness as terminal reward. Freeze the reliability-gated E2–E4 equations and weights before training. Exclude ReViSQL for unclear licensing and keep SynSQL outside M2.
- Rationale: One governed data/evaluation contract prevents Base, SFT, and RL comparisons from changing task identity or evaluator semantics. Database-level isolation, content hashes, hard-versus-review leakage categories, and two metric semantics expose contamination and equivalence tradeoffs without claiming unlicensed official code reproduction.
- Consequence: Mini-Dev cannot influence prompts, early stopping, rewards, model selection, or manual repair. Header metadata is provenance while observed SHA-256 is artifact identity. Day 3B must run on separately approved storage, complete database hashes/integrity/leakage, and freeze manifests before M2 can complete; M6 implements but cannot redesign dense rewards.

## ADR-017 — Repair the M2 evaluator and materialization closure

- Date: 2026-08-15
- Status: accepted for Day 3A closure; Day 3B remains unapproved
- Supersedes: ADR-016 only where its initial empty-dimension structural scoring or incomplete materialization gate conflicts
- Decision: Classify SQL by its top-level statement so CASE `END` and `REPLACE()` remain legal expressions while write/environment statements remain forbidden and authorizer-enforced. Use complete bipartite matching for unordered tolerant float rows. Normalize `r_struct` over active dimensions only, with all-empty score zero. Require safe archive extraction, a hashed source receipt referenced by all manifests, relative database paths, restricted heldout files, full unique-gold execution preflight, and atomic final publication.
- Rationale: Full-label probing exposed 602 legitimate gold queries rejected by token-position-insensitive filtering, while greedy tolerant matching and empty-dimension rewards were order/exploit risks. Reproducible evaluation also requires archive identity and gold executability to be proven before manifests freeze.
- Consequence: Day 3B cannot publish if any gold query is unsafe, syntactically invalid, fails execution, times out, or exceeds resources. It cannot edit gold SQL or relax limits. First-observed archive hashes live in the source receipt, so configuration is not changed after manifest hashing. M2 remains in progress until renewed control approval and successful materialization.

## ADR-018 — Bound evaluator IPC and comparison before cloud materialization

- Date: 2026-08-15
- Status: accepted for final Day 3A pre-cloud closure; Day 3B remains unapproved
- Decision: Drain a one-way worker Pipe before process join; retain terminate/kill/reap timeout cleanup. Gold preflight fully traverses results without returning rows. Cap unordered float candidate pairs at 1,000,000 before adjacency allocation and expose comparator resource exhaustion separately from inequality and SQL failure. Validate resumable HTTP 206 transfers from Content-Range total rather than treating the remaining fragment length as the full artifact length.
- Rationale: A 20,000-row control probe demonstrated Queue backpressure deadlock under join-before-drain, and the previously unbounded float adjacency could consume quadratic memory. Resume provenance also needs representation length rather than response-fragment length.
- Consequence: Large valid results no longer become false timeouts solely because of IPC. Gold preflight transfers bounded metadata, strict accuracy is zero with a diagnostic `comparator_resource_limit` when its frozen comparison budget is exceeded, and a resumed artifact is accepted only when range arithmetic and final size agree. M2 remains in progress until renewed control approval and Day 3B gates pass.

## ADR-019 — Safely materialize the observed nested BIRD training archive

- Date: 2026-08-15
- Status: rejected and superseded by ADR-020
- Decision: Treat `train/train_databases.zip` as an explicitly configured nested archive. Safely validate and extract the frozen outer archive, observe the nested central directory only under 10,000-member and 10-GiB hard caps, then run the same safe extractor with the exact observed values. Record nested SHA-256, size, count, and uncompressed bytes in the hashed source receipt. Discover the training database root uniquely from all 69 expected database IDs and raise the target-filesystem preflight floor to 40 GiB.
- Rationale: Cloud inspection showed the outer training ZIP contains a deflated inner ZIP, so one-layer extraction cannot locate SQLite files and HTTP Range cannot directly inspect the inner central directory. Guessing its layout or directly invoking `unzip` would violate the frozen archive-safety and provenance contract.
- Consequence: The stopped session's already-downloaded archives can be reused after review. A missing/ambiguous root, cap overrun, unsafe inner member, nested corruption, or receipt failure leaves only `.partial` state and blocks publication. No M3 work begins.

## ADR-020 — Gate nested expansion by observed capacity and cumulative spend

- Date: 2026-08-15
- Status: proposed pre-cloud capacity closure
- Supersedes: ADR-019's 10 GiB inner ceiling and 40 GiB initial free-space floor; its nested-path, safe-extraction, provenance, and root-discovery decisions remain
- Decision: Use 50 GiB only as the nested archive's hard uncompressed safety ceiling and require 70 GiB free initially on the actual target filesystem. After safely expanding the outer ZIP and reading the inner central directory, but before inner extraction, re-read that filesystem and require `observed inner uncompressed bytes + heldout expected uncompressed bytes + 10 GiB reserve`. Persist the observation and gate evidence before deciding. Treat CNY 12.95 as the cumulative Day 3B hard stop; a restart requires the verified first-session charge and uses only `12.95 - first charge`.
- Rationale: The outer archive's 9.35 GB uncompressed total mostly measures the compressed inner ZIP and cannot bound final SQLite expansion. A second paid attempt without both a realistic capacity ceiling and the first invoice could predictably stop again or reset the approved budget incorrectly.
- Consequence: More than 50 GiB observed inner expansion, less than 70 GiB initial free space, a failed dynamic capacity equation, or an unavailable first-session bill blocks restart/materialization. The 50 GiB ceiling is not reported as dataset size. No cloud resource or M3 work starts from this closure.

## ADR-021 — Preserve the failed gold preflight without changing protocol

- Date: 2026-08-15
- Status: awaiting control decision
- Decision: Treat `m2_day3b_materialization_001` as stopped and forbid final manifests after the real-database gold preflight produced non-`ok` results. Preserve `.partial` data and mode-0600 private evidence; publish only aggregate failure categories and hashes. Do not edit gold SQL, increase CPU/row limits, reclassify outcomes, restart cloud, or enter M3 without a new control decision.
- Rationale: Capacity and extraction passed, but 167 workers ended under Linux signal 24, 55 IPC EOF records observed an unstable exit code, and 22 gold queries crossed the frozen row cap. These combine an evaluator classification issue with a substantive protocol-limit question; silently changing either after seeing results would violate preregistration.
- Consequence: M2 remains `in_progress`. Control must decide how to classify CPU-limit exits, stabilize IPC exit reporting, and handle known-valid gold queries above the current result cap. Leakage and database identities must be persisted again before any eventual manifest publication.

## ADR-022 — Separate diagnostic reruns from the frozen evaluator

- Date: 2026-08-15
- Status: proposed local diagnostic closure; cloud execution unapproved
- Decision: Stabilize IPC EOF handling by bounded join followed by controlled terminate/kill/reap when needed, then classify a negative exit through `signal.Signals`. `SIGXCPU` is `resource_limit/cpu`; parent wall expiry is `timeout/wall`; row and byte caps are `resource_limit/rows` and `resource_limit/bytes`; other signals, caught worker exceptions, and unexplained exits remain `worker_error`. Preregister two separate `diagnostic_only` profiles and rerun only the 244 unique queries recorded non-`ok` by the accepted 96bcb4c evidence, with an atomic private checkpoint and redacted aggregate report.
- Rationale: The old parent read `exitcode` before joining after IPC EOF, producing 55 unstable `None` records, while preflight folded unknown worker states into execution errors. Larger limits are scientifically useful to distinguish legitimate 3–10-second SQL and measure 100k–1M result sizes, but using them as formal evaluator/RL limits without evidence and control review would violate preregistration.
- Consequence: The accepted failed-run evidence is never overwritten. Diagnostic Profile A uses 10-second wall, 12/13-second CPU, and formal row/byte caps; Profile B uses status-only 30-second wall, 32/33-second CPU, 1,000,000-row and 512-MiB safety ceilings. Neither profile publishes manifests or changes `configs/m2_data.toml`. M2, cloud, and M3 remain stopped pending control review.

## ADR-023 — Bind diagnostics to private evidence and bounded output

- Date: 2026-08-15
- Status: proposed final local diagnostic closure; cloud execution unapproved
- Decision: Before parsing failure records or labels, hash and match the accepted private preflight, training labels, heldout labels, formal M2 config, and source receipt. Bind those identities, the complete diagnostic config, selection, profiles, session limit, and query fingerprints into `plan_sha256`. Resolve output paths and reject Git-worktree or materialization-`.partial` destinations, including symlink traversal. Require checkpoint mode exactly `0600` on creation and resume. Enforce a 5,400-second program-level session wall gate before each query and report incomplete state without claiming success when reached.
- Rationale: A 244-query count alone cannot prove that the diagnostic uses the accepted failed run or frozen labels. A resumable private artifact also creates leakage risk if its path or permissions are weak, while per-query timeouts do not bound total paid-session duration.
- Consequence: Public reports expose only the five hashes, plan hash, completion state/counts, and split aggregates—never SQL, example IDs, database IDs, paths, or checkpoint content. Evidence/config drift, unsafe resolved output, non-0600 resume, or the session gate stops safely. Formal evaluator/RL limits, 96bcb4c evidence, spend ledger, cloud state, and M3 remain unchanged.

## ADR-024 — Preserve completed failure-only diagnostic without changing formal limits

- Date: 2026-08-15
- Status: evidence recorded; protocol decision pending control
- Decision: Accept the 244/244 diagnostic run as measurement only. Preserve its public aggregate and private checkpoint identity, but do not promote Profile A/B limits into `configs/m2_data.toml`, edit gold SQL, publish manifests, rerun the full preflight, or enter M3. Treat five training plus two heldout wall timeouts, one training worker error, and three Profile-B results above one million rows as unresolved control decisions.
- Rationale: The diagnostic eliminated unstable exit codes and recovered 214 earlier training failures, proving most prior errors were evaluator-limit/IPC artifacts. It also produced genuine residual boundary cases that the frozen formal evaluator cannot silently redefine after observation.
- Consequence: M2 remains `in_progress`. Cloud is off, private identities remain undisclosed, and the next action is a control decision on protocol treatment before any further execution or manifest publication.

## ADR-025 — Amend formal resources and freeze operational eligibility after diagnosis

- Date: 2026-08-16
- Status: approved by control; local finalizer closure pending review
- Decision: Preserve the original 3-second wall/2-second CPU config as immutable historical evidence and create a new formal version using 10-second wall, 12-second CPU soft, 13-second CPU hard, 100,000 rows, 64 MiB result bytes, and 2 GiB VM. The same amended boundary governs RL terminal reward, validation, and benchmark candidate execution. Mark exactly 28 unique training SQL operationally ineligible without deleting or editing labels: five wall timeouts, one worker error, and 22 row-cap outcomes. Keep all 500 heldout examples private; use full-500 conservative execution accuracy as primary and eligible-498 as secondary for both behavior-compatible and strict metrics.
- Rationale: The accepted diagnostic recovered 214 legal training SQL under 10/12 seconds, proving the historical 3/2 budget caused false rejections. Profile B remained measurement-only and showed why the 100k/64-MiB formal caps must not be silently expanded. Eligibility is an execution-resource contract, not a judgment that excluded gold SQL is wrong.
- Consequence: Finalization must prove `6282 + 214 = 6496 = 6524 - 28`, heldout 498/2, post-exclusion split balance and feature coverage, full evidence hashes, private permissions, integrity and leakage, and atomic publication without rerunning gold SQL or modifying the source `.partial`. M2 remains `in_progress`, cloud stays off, `spend.csv` stays unchanged, and M3 remains forbidden until control accepts final manifests.

## ADR-026 — Publish M2 manifests from frozen evidence and close the milestone

- Date: 2026-08-16
- Status: accepted and executed
- Decision: Accept the evidence-only finalization at source commit `75557aa5fc2cfd6f108210da0f5dfda69b481394`. Align raw Profile-A fallback outcomes with the already accepted public `worker_error` aggregate while retaining CPU-limit hard failure. Never follow or copy the two observed unreferenced, non-SQLite Mini-Dev `.sql` symlinks; record only their aggregate count and continue rejecting every database or other symlink. Publish only after all evidence, eligibility, 59/10 coverage, integrity, leakage, privacy, relative-path, and atomic-publication gates pass.
- Rationale: The first publication attempt exposed a mismatch between raw worker status and the accepted aggregate; the second exposed two irrelevant archive convenience links. Neither changed gold SQL, database content, split, evaluator limits, or eligibility. Explicit classification and non-following behavior preserve the frozen protocol while preventing unsafe external-path traversal.
- Consequence: Manifest index `0e1182eb16bf5bed4b6d5505b11fcf6860689abe9f0435e3d2b3fc33bf7d1cda` is the M2 publication identity. M2 is complete with 6,496/28 training eligibility, 498/2 heldout eligibility, 59/10 database coverage, 80/80 integrity, passed hard leakage gates, mode-0600 heldout evidence, and zero gold re-execution. Large/private artifacts remain off Git; M3 stays pending until a new scoped task begins.

## ADR-027 — Freeze the M3 vertical-slice contract before paid execution

- Date: 2026-08-24
- Status: accepted for Day 4A; paid execution unapproved
- Decision: Train only assistant SQL plus EOS, with all prompt and padding labels masked to `-100`.
  Serialize deterministic SQLite table/column/type/PK/NOT NULL/FK metadata without sample values;
  remove only complete trailing table blocks on overflow and never truncate gold SQL. Select only
  operationally eligible M2 train/validation rows using one fixed seed, stable IDs/hashes, and
  join/aggregate/subquery coverage. Freeze tiny counts only after real tokenizer-length evidence,
  never from model outcomes. Compare Base and fresh-reloaded QLoRA using identical ordered
  validation prompts, greedy decoding, and the unchanged M2 evaluator. Heldout consumption is zero.
- Rationale: M3 measures end-to-end reproducibility, not accuracy optimization. Answer-only masking
  aligns loss with SQL generation; deterministic identity and a shared comparison contract prevent
  sample/prompt drift; a fresh process proves the saved adapter is independently usable. Real data
  is absent locally, so synthetic contract tests must precede one bounded cloud session.
- Consequence: Day 4A may complete locally without claiming GPU measurements. The paid run stops
  after token-length inspection until control freezes counts, then permits at most two optimizer
  steps. M3 completes only after real update evidence, adapter reload, both execution metrics,
  VRAM/time/throughput/cost/failure evidence, `make verify`, registry/state updates, and teach-back
  follow-up. M4 and RL remain forbidden.

## ADR-028 — Harden M3 as one fail-closed paid session

- Date: 2026-08-24
- Status: accepted locally; paid execution unapproved
- Supersedes: ADR-027's runtime count-freeze pause; all M2, prompt, masking, comparison, and
  evaluator contracts remain unchanged
- Decision: Preregister exactly 8 train and 8 validation rows. With batch size 1, gradient
  accumulation 4, and maximum two optimizer steps, training fills two complete accumulation
  windows. Inspect all real token lengths first, then fail without replacement if the fixed rows
  cannot fit or cover join/aggregate/subquery. Require an externally supplied exact approved commit,
  clean tree, brand-new resolved output outside Git, and 20 GiB actual filesystem headroom before
  GPU stages. Run all subprocesses under one cumulative 14,400-second monotonic deadline with
  process-group terminate/kill and persisted timeout evidence.
- Rationale: M3 is an engineering vertical slice, not an accuracy estimate, so eight validation
  examples are enough to prove the same-exam path and eight training examples exactly exercise two
  optimizer updates. A mid-session control pause wastes paid GPU idle time; prose-only capacity and
  wall limits do not enforce safety.
- Consequence: One `run_m3_cloud.sh` invocation proceeds from preflight through final validation or
  exits nonzero with evidence. It cannot accept count environment variables, resume into an old
  output directory, reset the wall budget per stage, switch seed, or replace an excluded row.
  Commit `3243a273e1751e8d6b49fa95adb24614fb41ff99` is superseded for paid execution by the separately
  pinned hardened closure `1cb5431ecbada21968dff2659ad66d3f9e02af98`. No cloud, model download,
  heldout read, M4, or RL is authorized here.
  At CNY 2.59/hour the script planning ceiling is CNY 10.36, while the honest expectation remains
  1.5–3 hours; the extra hour cannot be used to expand scope. Durable atomic JSON, append-only stage
  logs, Base/training progress checkpoints, and atomic adapter-directory publication preserve as
  much completed evidence as practical before timeout or provider shutdown.

## ADR-029 — Accept the audited M3 vertical slice and close the milestone

- Date: 2026-08-24
- Status: accepted and executed
- Decision: Accept attempt 003 at execution commit
  `530a0f22f491882def5e13b8344d1310c1a7960d`, separately pinned by
  `f1a50c463c011153fbe6ea098e5c8a72e53af150`. Close M3 from the independently recomputed
  `results/m3/m3-vertical-slice-summary.json`. Preserve the fixed 8/8 slice, zero heldout use,
  answer-only masking, two optimizer updates, adapter manifest/fresh-process evidence, and identical
  Base/SFT comparison contract. Treat completion throughput as generation-plus-evaluator end-to-end
  throughput. Record CNY 3.88284165 only from the two user-verified provider invoices; never promote
  the program's elapsed-time estimate to an actual bill.
- Rationale: The returned evidence passes tokenizer-shape and integer-token checks, slice/hash/token
  reconstruction, row-level metric recount, finite update diagnostics, saved-adapter hashing, fresh
  PID reload, evaluator-error/gold-preflight gates, and private artifact permission/hash audit. Base
  scored 0/8 and tiny SFT 3/8 on both required metrics; positive improvement is not an M3 acceptance
  condition, but the measured result is retained without seed or sample shopping.
- Consequence: M3 is technically complete and cumulative M1–M3 cloud spend is CNY 13.52768593.
  The 8-example result is an engineering vertical slice, not a statistical accuracy claim or a
  formal SFT checkpoint. M4 remains pending and requires a new scoped authorization. The user
  teach-back remains open in `docs/INTERVIEW.md`, so technical completion does not imply interview
  readiness.

## ADR-030 — Pre-register M4 formal QLoRA before cloud execution

- Date: 2026-08-27
- Status: superseded before paid execution by ADR-031; control did not approve this default
- Decision: Freeze the default at 2,048 token-eligible train rows spanning all 59 training DBs,
  selected by one outcome-blind hash seed with six-feature coverage. Freeze 1,024 as the only
  pre-result fallback. Train two epochs with batch 1, accumulation 8, 512 default updates, 4-bit
  NF4/bf16 rank-16 all-linear LoRA, paged AdamW 8-bit at 1e-4, cosine schedule and 3% warmup. Save
  every 64 updates and retain three complete resumable checkpoints. The final update is selected
  without early stopping. Freeze dev256 and full1209 validation IDs before inference; E0/E1 share
  IDs, prompts, decoding and the M2 evaluator. Heldout use remains zero.
- Rationale: M3 measured ample 48GB headroom and proved the exact prompt/masking/runtime path, but
  its 8 rows cannot justify either full 5,364 training or a tiny-only formal claim. The default is a
  256x scale-up that covers every training DB while keeping training and the expensive Base
  generation inside one resumable session. Outcome-independent checkpoint choice prevents
  validation leakage.
- Consequence: Day 5A closes locally without a model result. A current price and exact commit need
  control approval before cloud. Derived planning is 10 hours/CNY 25.90 expected and 14 hours/CNY
  36.26 hard stop at the historical CNY 2.59/hour, with at least 30 GiB persistent free space.
  M4 remains pending; M5/M6/M7/RL and heldout remain forbidden.

## ADR-031 — Replace the unapproved M4 sample default with full eligible data

- Date: 2026-09-01
- Status: superseded in protocol details by ADR-032; paid execution pending control approval
- Decision: Preserve ADR-030 and commit `1852fc4` as the unapproved first draft, but replace its
  default prospectively with all 5,364 operationally eligible train rows for one epoch. Keep
  2,048×2 only as a control-approved pre-run capacity/time fallback and 1,024×2 as emergency-only.
  Use batch 1, accumulation 8, exactly 671/512/256 updates and explicit warmups 21/16/8. For the
  full route, average the final four real losses over four via four 2×-scaled real microbatches plus
  four zero-gradient padding forwards; never drop or relabel a real row. Save complete resumable
  state every 64 updates, select only the final update, and require dev256 plus final full1209 E0/E1
  same-exam evaluation with zero heldout use.
- Rationale: Full×1 has 5,364 example-passes versus 4,096 for 2,048×2, only 30.96% more training
  compute. It covers every eligible row and removes selector sampling bias, while the dominant
  full1209 evaluation cost is identical for both routes. M3 measured ample per-step VRAM headroom;
  no evidence shows dataset cardinality makes full×1 infeasible on 48GB.
- Consequence: The M3-rate projection separates A training (1.14 h) from dev/full E0/E1 evaluation
  (12.09 h), giving 13.22 h before unknown overhead. Plan 14 h/CNY 36.26 and hard-stop at 18 h/CNY
  46.62 at the historical rate, both requiring control approval above the CNY 30 gate. M4 remains
  pre-cloud pending; no GPU, heldout, M5, M6, M7, or RL is authorized.

## ADR-032 — Freeze M4 as one primary epoch; make the dev curve advisory evidence

- Date: 2026-09-01
- Status: accepted and pinned-environment verified; paid execution pending control approval
- Decision: The only M4 primary experiment is all 5,364 eligible rows for one epoch, exactly 671
  updates, with the existing one-epoch 671-step cosine schedule and 21-step warmup. At updates 512
  and 671 retain checkpoints and measure fixed dev256 teacher-forced answer-only token-weighted
  loss. Publish raw losses, training loss window, gradient norm, LR, finite state, time, checkpoint
  hashes and `(loss512-loss671)/loss512` in an epoch-extension decision packet. The 1% value is an
  advisory reference, not automatic continuation, early stopping or checkpoint choice. The program
  stops and publishes E1 at update 671. A later epoch2, if approved, is a distinct extension with a
  new LR/scheduler/budget preregistration and never overwrites E1. Freeze mutually exclusive dev256
  and final953 IDs; produce each model's full1209 generation exactly once and derive final953 from it.
- Rationale: A primary result must be independently interpretable and must not convert a post-hoc
  curve reading into an invisible second training protocol. Full1209 is the primary fair report,
  while teacher forcing is a low-cost diagnostic that does not duplicate SQL generation.
- Consequence: Four checkpoints are retained so 512/671 survive. The M3-rate projection is 1.14 h
  training plus 9.97 h unique full1209 generation and at most about 0.11 h teacher diagnostics;
  reserve 12 h and hard-stop 14 h (CNY 31.08/36.26 at the historical rate). M4 remains pending;
  heldout and later milestones remain forbidden.

## ADR-033 — Repair the M4 Transformers loss/gradient-accumulation boundary

- Date: 2026-09-01
- Status: accepted locally; exact-wheel cloud preflight remains required
- Decision: Pin the custom M4 Trainer contract to `transformers==5.14.1`. Qwen3's `forward` has
  `**kwargs`, so Trainer initially infers `model_accepts_loss_kwargs=True`. M4 custom
  `compute_loss` does not use `num_items_in_batch`; after Trainer construction it must assert that
  inferred value, set it to false, and assert the override. This forces Trainer's ordinary
  `current_gradient_accumulation_steps=8` division. Require a real torch/Trainer parameter-update
  regression with a `**kwargs` forward: eight ordinary records must update as their mean; four
  2×-scaled real records plus four zero-scale padding forwards must update as their four-record
  mean. Read required 512/671 metrics from final Trainer log history and fail if loss, grad norm,
  LR, elapsed time or checkpoint hash is missing.
- Rationale: The prior mock algebra did not exercise the version-specific `training_step` branch,
  so it could miss an eight-fold global gradient scale error. Zero-scale padding must remain a real
  forward to preserve the eight-microbatch window and resume semantics.
- Consequence: Local environments lacking the exact wheel report `unavailable_not_verified`; they
  cannot silently certify this regression. The cloud orchestrator performs the required exact
  wheel/source gate before token work or training. M4 remains pre-cloud pending and no model result,
  cloud session, heldout use or later milestone is authorized.

## ADR-034 — Accept the M4 primary E1 result; keep epoch 2 as a separate human decision

- Date: 2026-09-02
- Status: accepted technically; provider invoice and user teach-back pending
- Decision: Accept the completed `e1_m4_formal_sft_001` primary experiment at execution commit
  `f187167a882bf6b9fd2b152220fabbf770e92960`. It trained all 5,364 eligible examples for one epoch
  and 671 updates, passed the exact Trainer source/parameter-update gate, and evaluated Base and SFT
  once each on the same full1209 validation selection. Base strict accuracy was 0.0000 and SFT strict
  accuracy 0.40695; SFT behavior-compatible accuracy was 0.42349 and SQL-only rate 1.0000. Heldout
  use remained zero. Retain the final adapter and checkpoint-671 outside Git and publish the SQL-free
  summary under `results/m4/m4-public-summary.json`.
- Rationale: The locally reconstructed summary byte-matches the downloaded cloud summary and the raw
  Base/SFT files pass the same-exam contract. The dev256 teacher-forced loss fell only 0.1797% from
  update 512 to 671, below the 1% advisory reference, but that reference is not an automatic rule.
- Consequence: No second epoch has occurred or is implied. Any epoch-2 run must be a new experiment
  with a fresh LR/scheduler/cost contract after user review. Actual provider billing must be entered
  before formal M4 closure; M5 and later training remain unapproved.

## ADR-035 — Defer epoch 2 and authorize reversible local M5 diagnosis

- Date: 2026-09-02
- Status: accepted by user for local analysis only
- Decision: Keep M4 accounting open until the provider invoice is visible, but do not idle the
  project. Run deterministic, no-cost M5 analysis on the already downloaded full1209 SFT evidence.
  Publish only aggregate diagnostics; keep questions, SQL, sample IDs, database IDs, and paths out
  of Git. Do not treat the SQL-text complexity proxy as semantic truth. Epoch 2 remains deferred,
  not permanently rejected, and no cloud, heldout, data-v2 training, or RL is authorized here.
- Rationale: The primary SFT result and raw evidence are already technically audited. Local failure
  analysis is reversible and can identify whether more SFT, targeted data, or execution rewards are
  justified without spending GPU budget or contaminating heldout evaluation.
- Consequence: M5 enters `in_progress`. Its first aggregate pass must be followed by a deterministic
  private taxonomy audit before data-v2 is frozen.

## ADR-036 — Use targeted SFT as a bounded control and keep RL as the mainline

- Date: 2026-09-05
- Status: accepted by user; execution parameters pending measurement
- Decision: Complete M5 with the versioned 1,678-row training-only candidate pool. Plan `E1T` as a
  bounded targeted-replay SFT control, while E2, E3, and E4 independently start from the immutable
  M4 E1 adapter. E1T cannot become the RL parent. Keep Correction SFT as a trigger-only backup. The
  full1209 validation evidence is disclosed as development-used after M5 analysis; the private
  Mini-Dev 500 remains the untouched, single-use final heldout gate.
- Rationale: E1T controls for the possibility that extra supervised exposure, rather than reward
  learning, explains improvement. A common E1 parent preserves the E2–E4 causal comparisons.
  Correction SFT would add generation, verifier, equivalence, and interface complexity before a
  measured failure justifies it.
- Consequence: Experiment roles and leakage boundaries are frozen in
  `configs/m5_consumer_policy.toml`; optimizer, rollout, budget, and promotion parameters remain
  pending cheap preflight. M5 is technically complete. Three ambiguity cases remain quarantined
  until a future already-approved data session; none informs training.

## ADR-037 — Use a pinned, fail-closed SQLGlot extractor for dense rewards

- Date: 2026-09-05
- Status: accepted for local implementation; real-schema audit pending
- Decision: Pin SQLGlot 30.17.0 (MIT) and use its SQLite parser, scope graph, and schema qualifier as
  the front end for the frozen eleven-dimensional M6 structural reward. The project-owned layer
  preserves self-join roles, resolves derived-column lineage, replaces literal values with typed
  placeholders, uses occurrence tokens for count-sensitive grammar, and closes `g_conf` on any
  parse, schema, lineage, unsupported-construct, or safe-execution uncertainty.
- Rationale: A maintained parser is safer and more auditable than handwritten SQL parsing, while
  raw AST similarity is insufficient for aliases, CTEs, self-joins, duplicate projections, and
  reward-hacking attempts. Terminal execution correctness remains independent of extractor support.
- Consequence: The local unit/attack gate may be committed, but M6 remains in progress until all
  6,496 eligible training SQL are audited with their real schemas and TRL integration is separately
  verified. Heldout and cloud training remain unauthorized.

## ADR-038 — Accept measured M6 extractor coverage with eight fail-closed Gold queries

- Date: 2026-09-12
- Status: accepted after real training-schema audit
- Decision: Accept 6,488/6,496 supported unique training SQL (99.8768%). Keep structural feedback
  disabled for seven nondeterministic time-function queries and one executable Gold query that
  assigns alias `T2` to two physical tables. Do not add a dataset-specific alias heuristic. These
  eight queries remain eligible for terminal execution reward with `g_conf=0`.
- Rationale: Duplicate aliases make physical lineage ambiguous even when SQLite resolves individual
  columns permissively. Guessing lineage would violate the reliability claim for a gain of at most
  one structural target. Fail-closed behavior is the intended intervention, not data deletion.
- Consequence: Real-schema coverage is closed and M6 may proceed to framework integration. The
  frozen M2 data, evaluator, reward weights, and heldout isolation remain unchanged.

## ADR-039 — Accept the E2 systems smoke without claiming model improvement

- Date: 2026-09-19
- Status: accepted technically; actual provider invoice pending
- Decision: Accept `e2_m7_outcome_grpo_smoke_001` as proof that immutable E1 can generate four
  candidates per training prompt, receive strict SQLite terminal rewards, complete eight GRPO LoRA
  updates, save a changed adapter, and reload it in a separate process. Keep M7 in progress and do
  not report this smoke as an accuracy result.
- Rationale: The run had finite optimization evidence, nonzero adapter delta, 11,736 MiB peak GPU
  memory, successful reload, and zero heldout use. However, only 2/8 groups were mixed-reward and
  6/8 were constant, so the tiny sample mainly demonstrates reward sparsity and systems viability.
- Consequence: Formal E2 must be planned from the measured signal rate and must include a bounded
  reward-hacking review. E3/E4 and heldout remain gated; a new paid session requires separate user
  approval and current price confirmation.

## ADR-040 — Preregister a 256-group formal E2 baseline with a post-run signal gate

- Date: 2026-09-19
- Status: proposed pending control approval and current provider price
- Decision: Run formal E2 from immutable E1 on 256 outcome-blind training prompts, four completions
  each, 256 optimizer updates, and checkpoints every 64 updates. Retain the measured direct-
  generation, `beta=0`, DAPO, and single-normalization path. Apply a post-run promotion gate of at
  least 32 mixed groups and a 12.5% mixed-group fraction; do not outcome-filter prompts or stop early
  based on favorable reward observations.
- Rationale: Smoke memory was only 11,736 MiB and linear training-time scaling predicts about one
  hour. The 25% smoke mixed-group estimate is too noisy to redesign sampling, while 256 groups give
  a bounded baseline and preserve fair E2/E3/E4 budgets. A post-run gate reports sparse feedback
  honestly without biasing the fixed run.
- Consequence: The expected session is 1.5 hours with a three-hour/CNY 7.77 ceiling at the historical
  rate. SQL-free component and trainer telemetry support the M7 reward-hacking review. No cloud
  launch, validation comparison, E3/E4, or heldout access is authorized by this proposal.

## ADR-041 — Accept formal E2 and close M7 without an accuracy claim

- Date: 2026-09-19
- Status: accepted technically; actual provider invoice pending
- Decision: Accept `e2_m7_outcome_grpo_formal_001` as the completed outcome-only RL baseline. The
  run completed its fixed 256-group budget, passed the 32-group/12.5% signal gate with 102 mixed
  groups, produced a changed and independently reloadable adapter, and kept validation/heldout use
  at zero. Close M7 technically and defer all model-quality claims to M8 common evaluation.
- Rationale: Training and reload evidence are complete, peak memory was 19,804 MiB, and aggregate
  reward-hacking diagnostics show no truncation, entropy collapse, or length explosion. One unsafe
  and one resource-limit completion were correctly blocked. On-policy reward 334/1,024 cannot be
  compared directly with E1 validation accuracy because prompts and sampling differ.
- Consequence: M8 must evaluate E1 and E2 under the same frozen validation protocol before deciding
  whether outcome-only RL helped. E3/E4 remain unstarted, Mini-Dev remains untouched, and actual
  provider cost must be recorded from the invoice rather than inferred from program wall time.

## ADR-042 — Reuse E1 and evaluate E2 once on the frozen full1209 exam before dense-reward training

- Date: 2026-09-19
- Status: accepted locally; paid execution pending control approval
- Decision: Make the first M8 outcome a common-protocol E1/E2 comparison. Reuse the accepted E1
  full1209 aggregate and generate only E2 on the identical selection, prompt contract, greedy
  decoding, and M2 evaluator. Do not regenerate E1. Use existing raw E1 evidence for paired
  transitions and an exact McNemar test if it can be recovered without inference; otherwise report
  the aggregate comparison and the provenance limitation. Run E3 before E4 only after Phase A and
  after replacing the current E2-only trainer contract with an explicitly tested multi-arm entry.
- Rationale: Formal E2 training reward is on sampled training prompts and is not comparable to E1
  validation accuracy. Reusing accepted E1 evidence saves approximately 6,437 seconds of duplicate
  generation while preserving the causal exam. Separating Phase A from runner generalization makes
  it harder to accidentally label an E2-only implementation as dense-reward training.
- Consequence: Phase A plans two hours with a four-hour/CNY 10.36 ceiling at the historical rate,
  subject to current price, invoice, auto-shutdown, and explicit approval. Mini-Dev remains sealed.
  E3/E4 remain unstarted and must independently initialize from immutable E1.

## ADR-043 — Accept the positive E1/E2 common-validation result and proceed to E3

- Date: 2026-09-23
- Status: accepted technically; actual provider invoice pending
- Decision: Accept `e2_m8_common_validation_001` as the common-protocol E1/E2 comparison. Treat the
  +1.6543-point strict-accuracy change as a positive validation result supported by paired evidence,
  not as a heldout or final benchmark claim. Proceed next to an independently configured E3 run
  from immutable E1; do not continue training from E2.
- Rationale: E2 answered 39 previously wrong examples correctly while regressing on 19 previously
  correct examples, producing exact two-sided McNemar p=0.01193. Behavior-compatible accuracy and
  valid-SQL rate also improved, while SQL-only stayed fixed at 100%. The same sample IDs, prompts,
  greedy decoding, evaluator, and selection hash rule out the most important comparison drift.
- Consequence: E2 is now a measured outcome-only baseline. M8 remains in progress until E3/E4 or
  the predeclared reduced form is complete. Mini-Dev remains sealed, and the provider invoice must
  be recorded rather than inferred from the 6,786-second generation time.

## ADR-044 — Preregister E3 as a one-variable dense-reward intervention

- Date: 2026-09-23
- Status: accepted locally; paid smoke pending control approval
- Decision: Run E3 independently from immutable E1, first as the same eight-prompt engineering
  smoke used by E2 and then, only after review, with E2's exact 256-prompt/1,024-rollout formal
  budget. Preserve prompt identities, decoding, optimizer, learning rate, seed, update count, and
  checkpoint cadence. Change only the reward arm from E2 terminal correctness to E3's reliability-
  gated utility followed by within-group normalization.
- Rationale: This makes E2-to-E3 a controlled test of whether valid-SQL and structural partial
  credit improve learning signal or downstream accuracy. Starting E3 from E2 would confound the
  reward intervention with additional training, while changing prompts or budget would weaken the
  comparison. A small smoke catches integration errors before the formal GPU spend.
- Consequence: The shared runner now accepts explicitly validated E2 or E3 contracts and labels E3
  reward telemetry as group-relative reward rather than terminal accuracy. The smoke ceiling is 30
  minutes/CNY 1.30 at the historical rate; the formal ceiling remains three hours/CNY 7.77. Neither
  ceiling authorizes launch, validation, E4, or heldout use.

## ADR-045 — Accept E3 engineering smoke and retain the controlled formal contract

- Date: 2026-09-25
- Status: accepted technically; actual provider invoice pending
- Decision: Accept `e3_m8_dense_grpo_smoke_001` as systems and learning-signal evidence. Keep the
  preregistered 256-prompt/1,024-rollout formal E3 contract unchanged and require a separate launch
  approval. Do not treat the smoke's sampled correctness counts as validation accuracy.
- Rationale: The fixed eight-prompt run completed all eight updates, produced mixed dense reward in
  seven groups, changed the adapter, independently reloaded the exact same adapter identity, stayed
  within 11,736 MiB, and used no heldout data. One constant group correctly produced zero relative
  signal. The earlier E2 smoke had only two mixed groups, but this stochastic eight-group comparison
  is too small to claim that E3 is better; the controlled formal run and common validation are still
  required.
- Consequence: The next GPU action is formal E3 from immutable E1 after fresh price and shutdown
  confirmation. E3 validation, E4, and Mini-Dev remain gated. The missing `xgrammar` shared library
  found on the first attempt is recorded as an environment defect, not a model or reward failure.

## ADR-046 — Accept formal E3 training and require common validation before E4

- Date: 2026-10-01
- Status: accepted technically; actual provider invoice pending
- Decision: Accept `e3_m8_dense_grpo_formal_001` as the completed reliability-gated dense-reward
  training arm. It completed the same 256-prompt/1,024-rollout budget as E2 from the same immutable
  E1 parent and changed only the reward aggregation. Do not interpret its on-policy completion
  counts as validation accuracy; evaluate the saved adapter next on the frozen full1209 protocol.
- Rationale: E3 passed its signal and reload gates and increased mixed-reward groups from E2's
  102/256 to 239/256 while holding the causal controls fixed. The last logged entropy was low even
  though mean entropy, completion length, and truncation did not show gross failure, so the common
  exam—not training reward—must decide whether denser feedback helped or over-concentrated behavior.
- Consequence: The next experiment is E3 common validation after local preregistration and separate
  launch approval. E4, Mini-Dev, and final claims remain gated; actual provider cost must come from
  the invoice rather than inferred wall time.

## ADR-047 — Reuse the common evaluator for one frozen E3 validation pass

- Date: 2026-10-01
- Status: locally preregistered; paid execution pending control approval
- Decision: Generalize the accepted M8 evaluation contract from a hard-coded E2 condition to a
  closed registry of E2 and E3 identities. Generate only E3 on the frozen full1209 exam, reuse E1/E2
  aggregates, and use existing raw E1/E2 rows for paired tests only when recoverable without
  inference. Preserve greedy decoding, sample order, prompt hashes, and the M2 evaluator.
- Rationale: One generic, identity-checked path avoids duplicating evaluator logic while maintaining
  the causal comparison. Re-running E1 or E2 would spend GPU without adding information and could
  introduce temporal or implementation drift.
- Consequence: The E3 pass expects two hours and has the same four-hour/CNY 10.36 ceiling as E2.
  Launch still requires a clean accepted commit, current price, auto-shutdown, retained input
  identities, and explicit approval. E4 and Mini-Dev remain gated.

## ADR-048 — Accept E3 as a controlled negative result and proceed to E4

- Date: 2026-10-02
- Status: accepted technically; provider invoice pending
- Decision: Accept `e3_m8_common_validation_001` as the frozen E3 model-quality result. E3 scored
  41.6873% strict accuracy, improving 0.9926 points over E1 but trailing E2 by 0.6617 points. Neither
  paired contrast crossed p=0.05. Retain terminal-only E2 as the current best validation condition
  and proceed to E4 without changing reward components, weights, prompts, budget, or E1 parent.
- Rationale: E3 increased mixed training groups from 102/256 to 239/256, proving denser optimization
  signal, but that signal did not translate into superior common-validation accuracy. E4 is the
  preregistered causal test of whether separately normalizing reward dimensions corrects scale
  interference; post-hoc weight tuning would weaken the experiment.
- Consequence: E4 requires a smoke before formal training. Mini-Dev remains sealed. The exact
  SQL-free E3 summary and hash-verified 1.2 MiB raw file are local; retrieval used the next approved
  start and did not rerun E3 inference. Actual E3 session cost remains invoice-only.

## ADR-049 — Preregister a parity-locked E4 engineering smoke

- Date: 2026-10-02
- Status: accepted locally; paid execution pending control approval
- Decision: Register `e4_m8_gdpo_smoke_001` as an eight-prompt, 32-rollout, eight-update smoke from
  immutable E1. Preserve the E3 smoke's prompts, generation, optimizer, component scorer, weights,
  evidence gates, and limits; change only experiment identity, arm, and aggregation from
  `sum_then_normalize` to `normalize_then_sum`. Require the regenerated bundle to match retained E3
  bundle identity `82cfce66...c3265` before model loading.
- Rationale: E4 is causal only when aggregation is the sole scientific intervention. The tiny smoke
  cheaply tests pinned-TRL compatibility, reward alignment, optimizer update, and reload before any
  formal budget is considered.
- Consequence: Smoke success makes the execution path eligible for review but does not authorize a
  formal E4 run, common validation, weight tuning, or Mini-Dev access.

## ADR-050 — Accept E4 smoke and keep formal E4 separately gated

- Date: 2026-10-02
- Status: accepted technically; provider invoice and teach-back pending
- Decision: Accept `e4_m8_gdpo_smoke_001` as engineering evidence for the normalize-then-sum path.
  It completed 8/8 updates with seven mixed groups, nonzero adapter change, independent reload, and
  zero validation/heldout access from the exact E2/E3 smoke prompts.
- Rationale: The successful attempt demonstrates framework compatibility and a real optimization
  signal while preserving the causal control. Its 32 stochastic rollouts are far too small to
  support accuracy or superiority claims.
- Consequence: Formal E4 requires a new parity-locked preregistration. The pre-model `xgrammar`
  import failure remains recorded as an environment incident; it does not alter reward evidence.

## ADR-051 — Preregister formal E4 as an aggregation-only comparison

- Date: 2026-10-02
- Status: accepted locally; paid execution pending control approval
- Decision: Register `e4_m8_gdpo_formal_001` from immutable E1 with exactly formal E3's 256 prompts,
  1,024 rollouts, 256 updates, optimizer, decoding, reward components, confidence gate, weights,
  evidence, and stops. Change only aggregation from `sum_then_normalize` to
  `normalize_then_sum`.
- Rationale: E3 delivered much denser learning signal but did not beat E2 on common validation.
  Holding every other factor fixed makes E3-to-E4 a causal test of per-component normalization
  rather than a post-hoc hyperparameter search.
- Consequence: Contract validation fails on any additional E3/E4 drift. A successful E4 training
  run proves optimization completion only; the same frozen full1209 exam is required before a
  model-quality conclusion, and Mini-Dev remains sealed.

## ADR-052 — Accept formal E4 training and require common validation

- Date: 2026-10-04
- Status: accepted technically; provider invoice and exact remote/local hash comparison pending
- Decision: Accept `e4_m8_gdpo_formal_001` as the completed parity-locked normalize-then-sum
  training run. It completed 256 updates and 1,024 rollouts from immutable E1, produced 235/256
  mixed groups, changed and independently reloaded its adapter, and used zero validation/heldout
  examples. Treat both earlier pre-model stops as invocation/environment incidents, not failed
  reward experiments.
- Rationale: The successful evidence satisfies every preregistered training and signal gate without
  changing data, model, optimizer, reward components, weights, or aggregation contract. E4's 235
  mixed groups do not exceed E3's 239 and none of the sampled training statistics measure
  generalization.
- Consequence: Preregister one unchanged full1209 E4 common-validation run before comparing E3 and
  E4 quality. Do not tune weights, rerun training, or access Mini-Dev. Provider cost remains
  invoice-only, and the local evidence packet records its still-pending exact remote hash check.

## ADR-053 — Preregister one E4 full1209 common validation

- Date: 2026-10-04
- Status: accepted locally; paid execution pending control approval
- Decision: Evaluate the accepted formal E4 adapter exactly once on the frozen 1,209-example
  validation selection, using the same prompt, decoding, evaluator, and reporting protocol as
  E1/E2/E3. Generate only E4 predictions; reuse all prior aggregates and use existing raw rows only
  for paired statistics.
- Rationale: Training diagnostics cannot answer whether normalize-then-sum improves generalization.
  Reusing the identical exam and old evidence isolates aggregation without paying to regenerate
  prior arms or introducing evaluation drift.
- Consequence: Missing E1/E2/E3 raw rows may reduce paired analysis but never authorize new
  inference. E4 adapter or protocol drift, nonzero heldout use, and any attempt to access Mini-Dev
  fail closed. The four-hour/CNY 10.36 ceiling is a bound, not launch authorization.

## ADR-054 — Accept the four-arm M8 result and retain E2 as the best measured arm

- Date: 2026-10-06
- Status: accepted technically; provider invoice and user teach-back pending
- Decision: Accept `e4_m8_common_validation_001` as the final M8 common-validation result. E4 scored
  41.4392% strict accuracy, trailing E2 by 0.9098 points and E3 by 0.2481 points. None of E4's paired
  strict contrasts crossed p=0.05. Retain outcome-only E2 at 42.3490% as the best measured validation
  arm and close E3/E4 as controlled negative dense-feedback comparisons.
- Rationale: All four arms used the same 1,209 examples, prompts, greedy decoding, evaluator, and
  E1 parent where applicable. E3/E4 greatly increased mixed training groups but did not improve
  downstream accuracy, and changing weights after seeing validation would invalidate the causal
  comparison. A reproducible negative result satisfies the project charter.
- Consequence: M8 is technically complete with Mini-Dev still sealed. No dense-reward retuning or
  additional GPU run is authorized. The next gate is the user's M8 evaluation teach-back, followed
  by a separately preregistered M9 serving benchmark. Provider billing remains invoice-only.

## ADR-055 — Serve E2 through a real endpoint and benchmark task-correctness with systems metrics

- Date: 2026-10-06
- Status: accepted locally; paid execution pending control approval
- Decision: Select E2, the best common-validation arm, for M9. Statically load its rank-16 adapter
  into the pinned vLLM OpenAI-compatible server and benchmark 64 deterministic validation prompts
  at concurrency 1/4/8. Measure streaming TTFT, end-to-end latency, request/token throughput, and
  ready/peak GPU memory; evaluate concurrency-1 SQL with the frozen M2 evaluator.
- Rationale: A generic random-token benchmark would measure engine speed but not whether the served
  adapter and prompt path still solve the project task. A custom streaming client retains real
  Text-to-SQL prompts and correctness while using the standard production endpoint. Static adapter
  loading avoids the security and reproducibility risks of runtime LoRA mutation.
- Consequence: M9 needs one separately approved GPU session. It cannot access Mini-Dev, tune model
  selection, or claim that the 64-example serving slice replaces full1209 accuracy. Raw SQL remains
  ignored; only a SQL-free aggregate may be committed.

## ADR-056 — Accept M9 and keep the 64-example serving slice separate from model selection

- Date: 2026-10-06
- Status: accepted technically; provider invoice and serving teach-back pending
- Decision: Accept attempt 003 as the M9 serving result. Report its task correctness together with
  TTFT, end-to-end latency, throughput, startup time, and GPU memory, while retaining E2's full1209
  evaluation as the model-quality result.
- Rationale: The serving slice exercises the real API, adapter, prompt template, streaming client,
  and SQLite evaluator end to end. Its deterministic 64 examples are sufficient for an integration
  benchmark but too small and selected for serving systems measurement rather than model comparison.
- Consequence: M9 is technically complete with 192/192 successful requests and heldout 0/0. M10
  proceeds without more GPU work. The two pre-readiness failures are environment evidence, not
  failed model-quality trials.

## ADR-057 — Accept archive-only clean-room verification as M10 completion

- Date: 2026-10-06
- Status: accepted
- Decision: Freeze a CPU-only reproduction surface at implementation commit `392b8a3`: exact
  Python 3.11.15, locked verification dependencies, Make and README entry points, a non-root Docker
  image, CI jobs, and a deterministic allowlist/manifest over 19 SQL-free public artifacts.
- Rationale: Re-running SFT/RL is neither necessary nor affordable for installation reproducibility.
  Exporting the commit with `git archive` and passing the entire 314-test and evidence gate in a new
  environment proves that ignored raw files, old editable installs, and private data are not hidden
  dependencies.
- Consequence: M10 is complete with CNY 0 GPU cost. Docker was unavailable locally, so only the
  fresh virtual-environment path is locally measured; the checked-in Linux container must run in
  first public CI. M11 may package the release but cannot claim CPU verification reproduced the
  expensive model-training results.

## ADR-058 — Separate the Git release, model asset, and reconstructible upstream data

- Date: 2026-10-19
- Status: accepted locally; public GitHub publication and CI pending
- Decision: Publish source, configs, tests, cards, runbooks, and SQL-free aggregate evidence in the
  Git repository. Publish the selected E2 adapter as a versioned release asset whose extracted
  directory must match manifest `3492a3ec...bd3b7`. Do not commit base weights, BIRD databases,
  benchmark rows, raw SQL generations, or private heldout manifests; reconstruct upstream content
  from pinned revisions and keep private evidence in separate durable storage.
- Rationale: This makes the useful release cloneable and independently verifiable without turning
  Git history into an unsafe multi-gigabyte data store. The adapter is small enough for a release
  asset, while the base and BIRD sources already have immutable upstream identities.
- Consequence: A Git push alone does not authorize deleting the provider disk. E1/E3/E4 adapters
  and private manifest/receipt evidence must first have two verified durable copies, or the user
  must explicitly accept that those arms require retraining for full replay. Exact Python 3.11.15
  CI and the release-asset checksum remain M11 gates.

## ADR-059 — Prefer localized execution credit over further scalar reward tuning

- Date: 2026-10-19
- Status: research recommendation; implementation and paid execution not authorized
- Decision: If a post-v1 E5 experiment is approved, keep E2's terminal correctness anchor and
  parity-locked budget, but localize advantages to conservatively identified SQL clause/token spans
  using execution errors and incremental clause execution. Fail closed to E2 terminal-only credit
  whenever localization is ambiguous. Treat DAPO-style filtering of constant-reward groups as a
  secondary efficiency ablation rather than the headline intervention.
- Rationale: E3/E4 increased mixed groups from 102/256 to 239/256 and 235/256 yet both trailed E2.
  The evidence argues against another sequence-level structural scalar or post-hoc weight search.
  EXPO-SQL directly targets the remaining credit-assignment problem and is more scientifically
  distinct from the completed causal ladder.
- Consequence: E5 begins only after local parser/localization tests, an offline training-only audit,
  and an eight-prompt smoke are separately reviewed. It uses the same E1 parent, formal prompt
  budget, evaluator, and full1209 exam and never opens Mini-Dev. Failure of the conservative local
  gate ends the upgrade without weakening the complete v1 result.
