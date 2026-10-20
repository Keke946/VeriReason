# Current Project State

Last updated: 2026-10-20

## Current phase

- Milestone: M11 — Release and interview gate
- Status: completed technically; GitHub v1.0.0, public CI, minimal cloud-artifact migration, and
  reproducibility gates are complete. Personal teach-backs are tracked as a separate learning task
  and are not falsely marked mastered.
- Final M2 cloud source commit: `75557aa5fc2cfd6f108210da0f5dfda69b481394`
- Post-v1: E5 CPU-only feasibility implementation and synthetic audit completed; decision
  **Revise**, not ready for GPU smoke. The subsequent real-tokenizer/analytical-loss CPU gate
  passes its bounded interface checks; real training replay evidence is missing locally.
  Preferred project-owner name: **BiBo**.

## Accepted facts

- The project focuses on fine-tuning and post-training, not foundation-model pretraining.
- The intended chain includes data, baseline evaluation, SFT, verifiable-reward training, ablations, serving, and interview preparation.
- The target schedule is fifteen concentrated working days.
- Git-tracked project files are the source of truth; chat transcripts are temporary execution context.
- Work will use one control task, milestone-scoped execution tasks, and selective parallel read/review work.
- There is no local GPU; cloud GPU rental is acceptable when planned and measured.
- The target is a resume-ready result in two to three weeks and before 2026-09-01.
- Public release is acceptable for code, license-compatible processed data, model adapters, reports, and a demo.
- The task domain is cross-domain Text-to-SQL.
- The project must go beyond merely calling an existing library: it will test a focused research hypothesis derived from recent Text-to-SQL and general LLM-RL work.
- A complete 15-day result is mandatory. Clause-level token credit, larger-model confirmation, and multi-turn agent training are upgrades that cannot block that release.
- The 4B-preferred/1.7B-fallback model gate, 48 GB-first topology, CNY 250 core cap, and E0–E4 experiment ladder are accepted.
- Any need for a new Codex task or subagent must be announced in advance with the exact count, prompts, ownership, and required return packet; no agent is started silently.
- Cloud Python is fixed at 3.11.15. The prior local `.venv` used Python 3.11.7 and its combined pytest runner exited 139; M1 closure replaced it with a clean Python 3.11.15 environment. All 14 tests and the complete local gate now exit normally.
- The accepted GPU stack is Python 3.11.15, PyTorch 2.11.0+cu130, CUDA runtime 13.0, Transformers 5.14.1, TRL 1.9.0, PEFT 0.19.1, vLLM 0.25.1, Accelerate 1.14.0, bitsandbytes 0.49.2, and Datasets 4.7.0 on Ubuntu 22.04.5 with NVIDIA driver 610.43.03.
- `m1_env_1x48_001` passed on one RTX 4090 D with 47.399 GiB visible memory and compute capability 8.9.
- Both fixed model revisions completed the four-case engineering smoke with 4/4 valid and execution-correct SQL. This is not benchmark accuracy.
- Qwen3-4B-Base is selected: all four responses obeyed the SQL-only instruction, versus 1/4 for 1.7B; it generated 122 versus 309 tokens and used 8158 MiB peak device memory at 28.33 completion tok/s.
- One 48 GB GPU is selected. No 2x24 GB test is justified because 4B used only 16.6% of the 49140 MiB device, while 2x24 GB costs more and adds distributed complexity without a measured bottleneck.
- The paid M1 session used 1.023611 GPU hours and cost CNY 2.65115277. Provider shutdown released the GPU and stopped GPU billing; retained system-disk usage continues at CNY 0.00003/GB/hour.
- The two model runs executed an uncommitted pre-`51decb1` source tree based on `f709f3a`, not either commit verbatim. Its 17-file manifest is `ed518be19adf972c20020b6d9114c2ae7212a846b411c936397bd25ef408561c`; reconstructing `51decb1` with only `configs/environment.toml` status restored from `accepted_m1_gpu_validated` to `proposed_pending_cloud_validation` reproduces that hash exactly.
- `results/m1/gpu-pip-freeze.txt` was not generated or synchronized during the paid session. Recovering it from the stopped system disk would require user-approved paid restart; the exact direct dependency versions and imported runtime versions remain recorded, but the transitive freeze is missing.
- The closure source manifest is `db724749651c81a42befcd7dc33dde54bbc60d28acb04e73d33bb4a2732a2910` over 12 committed package/config/script files, excluding generated egg-info. It will correspond to the M1 closure commit because documentation and registry files are outside that manifest set.
- Final control-task verification independently passed the project guard, 14 tests, local environment check, Ruff lint/format, compileall, `pip check`, deterministic SQL-only recount, and closure-manifest reconstruction. M1 is accepted without a paid GPU restart; the missing transitive GPU `pip freeze` remains an explicit provenance limitation, and every later formal experiment must run from a clean committed source tree.
- M2 uses `birdsql/bird23-train-filtered@4068469807b255fcfc0816bdd520946fe460d256` for training/development and original BIRD Mini-Dev SQLite labels at `birdsql/bird_mini_dev@f65faf4ae3b638c1fa6df1d3370c8d92c8366301` as a strictly isolated 500-example heldout set; both declare CC BY-SA 4.0.
- ReViSQL is excluded for unclear dataset licensing and SynSQL is outside the M2 core. Neither enters the materialized manifests without a new decision.
- The frozen database-hash split gives 59/10 databases and 5,381/1,220 examples; validation is 18.482% and includes every frozen critical SQL feature. The single seed passed without seed search.
- The local SHA-256 identities first observed for filtered labels and Mini-Dev labels are `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8` and `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2`.
- Label-only leakage precheck has zero cross-split `db_id`, joint-sample, normalized-question, normalized-SQL, and Jaccard findings. Database SHA, renamed-identical-database, and schema checks remain pending until Day 3B obtains SQLite files.
- The independent evaluator freezes `bird_ex_behavior_compatible` and `strict_execution_accuracy`; strict correctness is the terminal RL signal. Safety and comparison semantics pass local positive, negative, equivalent, illegal, malicious, empty, `NULL`, float, duplicate, order, and timeout tests.
- E2, E3, and E4 reward equations, confidence gate, feature weights, zero-variance normalization, and a 0.75 raw correct/incorrect margin are frozen in `docs/M2_PROTOCOL.md`; dense extraction and integration remain M6 work.
- Range inspection initially measured the outer archives; the first approved Day 3B attempt subsequently downloaded both archives and discovered the nested training ZIP. That first attempt performed no extraction or manifest publication. The resumed attempt later passed the 70 GiB initial gate and two-level extraction, as recorded below.
- The closure audit reproduced 540/6,601 filtered and 62/500 Mini-Dev false static rejections caused by treating CASE `END` and `REPLACE()` as dangerous at any position. Top-level statement classification now accepts all 7,101 gold labels statically with zero remaining rejection categories while preserving authorizer-backed write and environment protections.
- Strict unordered float comparison now uses complete bipartite matching and is invariant to row permutation under the tolerance relation.
- Structural reward now normalizes only over active gold/prediction feature dimensions; all-empty scores zero. The incorrect raw maximum remains 0.25 and the correct/incorrect margin remains at least 0.75.
- Day 3B publication is gated on safe two-level ZIP extraction, a hashed source receipt referenced by every manifest, relative database paths, mode-0600 heldout evidence, full unique-gold execution preflight, and same-filesystem atomic rename from `.partial`. The actual target filesystem must initially have at least 70 GiB free.
- Large query results now use a one-way Pipe that the parent drains before joining the worker; a 20,000-row collected result completes without the former Queue deadlock, while real unbounded SQL remains wall-time bounded and its process is reaped.
- Gold execution preflight uses `collect_rows=false`: the child fully traverses results and enforces the frozen row/byte/time limits but sends only status and result metadata over IPC.
- Unordered float comparison has a frozen 1,000,000 candidate-pair budget checked before adjacency allocation. Evaluation distinguishes true inequality, SQL execution failure, and `comparator_resource_limit`.
- HTTP resume provenance accepts only a validated 206 Content-Range whose total equals the completed file; 200 responses still require Content-Length to equal the file size.
- The first approved Day 3B session reused the retained RTX 4090 D instance at CNY 2.59/hour and the root ZFS filesystem with about 100 GiB free. Both labels and both archives downloaded successfully and remained on the retained disk; the provider GPU was shut down before extraction. Its charge was subsequently verified before the resumed attempt.
- Observed archive identities are training `66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc` (8,919,543,554 bytes) and Mini-Dev `aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be` (799,944,582 bytes). Their final source receipt is referenced by the published manifest index.
- Outer BIRD `train.zip` contains deflated member `train/train_databases.zip`. The 10 GiB inner ceiling in `b1059a7` was rejected as too low because outer uncompressed size describes the compressed inner ZIP, not final SQLite expansion. The closure uses a 50 GiB hard ceiling without claiming an observed size, requires 70 GiB initially, then rechecks the actual target filesystem before inner extraction against `observed inner + heldout expected + 10 GiB reserve`. Capacity evidence is persisted before the decision.
- The first Day 3B attempt charge was verified before the second start. CNY 12.95 remains the cumulative Day 3B hard stop across all attempts, never a reset budget.
- The resumed Day 3B run safely observed a 9,347,158,408-byte inner ZIP with 1,525 members, SHA-256 `902fed2d1e75d42dd8c9896e006489acdc8518b2c73b21ba3ae1e42d57ea9aaa`, and 32,408,008,953 uncompressed bytes. Initial and dynamic capacity gates passed; their evidence SHA-256 is `275c525dbcd69e459f6f7cc901399b3faa57dfeafeefe4b6a7bcef3c4794c7b3`.
- Gold preflight blocked publication: training had 6,282/6,524 unique SQL `ok`, 220 execution errors, and 22 row-limit failures; heldout had 498/500 `ok` and 2 execution errors. Across both splits, 167 workers ended with Linux signal 24, 55 IPC EOF records captured an unstable `None` exit code, and 22 queries exceeded 100,000 rows. No syntax, unsafe, or wall-timeout statuses occurred.
- The original materialization attempt published no final manifest because preflight stopped safely; its mode-0600 private failure evidence remains preserved outside Git. The later evidence-only finalizer repeated and persisted the complete integrity/leakage audits before publication.
- The first two Day 3B attempts cost CNY 1.00987346 and CNY 1.78134210, totaling CNY 2.79121556 at that checkpoint. All attempts are now consolidated in the final spend record below.
- Commit `96bcb4c`'s observed 220 training and 2 heldout execution errors remain immutable historical evidence; control accepted that all are SIGXCPU or pre-join IPC EOF/unstable-exit records, not demonstrated gold-SQL errors. The 22 row-limit records remain legitimate queries whose actual result sizes are unmeasured.
- The local diagnostic closure joins/reaps workers before final IPC-failure classification, resolves negative exits through `signal.Signals`, and records non-null exit code, signal name, limit kind, and worker state. CPU, wall, row, byte, other-signal, and caught-worker failures no longer collapse into `execution_error`.
- `configs/m2_diagnostic.toml` preregisters failure-only Profile A (10-second wall, 12/13-second CPU, historical-formal row/byte caps) and Profile B (30-second wall, 32/33-second CPU, 1,000,000 rows, 512 MiB). Both remain `diagnostic_only`; their accepted input identity is preserved as historical config `configs/m2_data_v1_3s.toml`, SHA-256 `d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b`.
- The diagnostic runner writes a mode-0600 atomic checkpoint after every query, resumes without duplicate execution, never modifies `.partial`, and emits only split-level aggregates without SQL, example IDs, database IDs, or paths.
- The evidence/privacy closure freezes and verifies SHA-256 for the private preflight, both label files, formal M2 config, and source receipt before parsing any private query identity. All five hashes, the full diagnostic config, profiles, session limit, selection, and query fingerprints enter `plan_sha256`; any change rejects checkpoint resume.
- Diagnostic output is resolved before use and must lie outside both the Git worktree and materialization `.partial`, including through symlinks. New and resumed checkpoints must be exactly mode `0600`; group/world permissions are fatal. Public output exposes only hashes, completion counts, `complete`, `plan_sha256`, and aggregate statistics.
- The runner checks a frozen 5,400-second total session wall limit before each query. Reaching it atomically preserves an incomplete checkpoint and forbids `complete=true`. This does not replace Profile A/B per-query timeouts.
- The approved failure-only cloud diagnostic ran from exact commit `e6c56af36f521b823def81126143a00c9306eb42`, verified all five inputs, and completed 244/244 in 965 seconds with plan SHA-256 `3ded2e3718c9d2d397c07dd6c2792651dbf3992113d500733fbc562dd699cd6b`. The public aggregate is `results/m2/m2-diagnostic-summary.json`; the private checkpoint remained mode `0600` with SHA-256 `2e8748e29ba87ff94ecda079ae01cf7ae8105ce0965f45e01c1b02e66614c8f7` and is not committed.
- Profile A recovered 214/220 training failures: 82 completed within 3 seconds, 82 in 3–5 seconds, and 50 in 5–10 seconds; five training queries timed out and one remained a worker error. Both heldout Profile-A queries timed out. No CPU-limit or unstable-exit result remained. Profile B measured nine results at 100k–250k rows, six at 250k–500k, four at 500k–1M, and three above the one-million-row diagnostic ceiling; no byte limit, timeout, or worker error occurred.
- The diagnostic session container cost was CNY 2.70870832; conservatively including CNY 0.00277268 visible disk charges gives CNY 2.71148100 for that attempt and CNY 5.50269656 at the pre-finalization checkpoint.
- Control approved a versioned formal-resource amendment after reviewing the complete diagnostic. The historical 3-second wall/2-second CPU config is preserved byte-for-byte at `configs/m2_data_v1_3s.toml`, SHA-256 `d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b`. The amended formal config uses 10-second wall, 12/13-second CPU, 100,000 rows, 67,108,864 result bytes, and 2,147,483,648 VM bytes, SHA-256 `1fa045ebf72a29d9209f5b37f1042126a81e66e3f7511b1c1fc8a8b21d553181`.
- Operational eligibility preserves every label and SQL. Exactly 28 unique training SQL are ineligible: five diagnostic wall timeouts, one worker error, and all 22 queries above the unchanged formal row cap. The evidence identity requires `6,282 + 214 = 6,496` eligible and `6,524 - 28 = 6,496`; any mismatch is fatal. Actual affected example-row counts and post-exclusion train/validation distributions are computed only by the private finalizer.
- Heldout remains all 500 private examples: 498 eligible and two ineligible. Primary reporting is `full_heldout_conservative_execution_accuracy` with denominator 500 and the two ineligible gold queries fixed to zero for every model. Secondary reporting is `eligible_heldout_execution_accuracy` with denominator 498. Both behavior-compatible and strict metrics report both scopes.
- The finalizer verifies that both the original private preflight and diagnostic checkpoint are exactly mode `0600`, in addition to their hashes, plan SHA, receipt, labels, historical config, and amended config identities. It never runs gold SQL or modifies the preserved materialization `.partial`; it hard-links databases into a separate `.finalization.partial`, repeats integrity and database/schema/leakage audits, requires eligible samples to retain exactly 59 train and 10 validation databases, writes relative logical paths and a mode-0600 heldout manifest, then atomically publishes on the same filesystem. Tests bind source SQLite SHA-256, byte length, and content across successful read-only integrity checks and simulated publication failure. Failure leaves explicit private finalization evidence and no final directory.
- Evidence-only finalization published successfully in 289 seconds with `gold_sql_reexecuted=0`. The accepted manifests contain 6,601 training and 500 private heldout examples. Operational filtering excludes 28 affected training rows (17 train, 11 validation), leaving 5,364 train and 1,209 validation examples across exactly 59/10 databases; validation is 18.3934% and retains every critical SQL feature.
- All 80 SQLite databases passed immutable read-only integrity checks. The final leakage audit passed with zero cross-split hard failures for database IDs, database SHA-256, joint samples, heldout exposure, and renamed-identical databases; review-only duplicate counts remain reported. The original 1,603-file database tree retained identity summary `c7b65515238fd1262d60e56619b2555892a0e9c10d76000e0b99aa981aed4279` before and after publication.
- Manifest index SHA-256 is `0e1182eb16bf5bed4b6d5505b11fcf6860689abe9f0435e3d2b3fc33bf7d1cda`. Referenced identities are training `d90ffab32e296248376c542b736aef5ce7f547acc702a9eab41073789cf5b72a`, private heldout `2f6257d2ceef020587183b18105ed7e7899c84dd2010161f4dc41a4d01faa3a1`, eligibility `fd9a3873c2f819d4e0d29eafceb7a6586abeb8cecb7215b91c5e619426634b57`, leakage `266c47959d3f250971a64cbcc87650299af44b903f3f10f512ff8a13cdeb3db2`, finalization evidence `d72f933523ef854f19255e1d3aaee65954427c1d9dddb0d4f98f3ebd5cf09888`, and source receipt `da189fd0f69d7713c16acfe434738e1f867074665c2b95d7d78b3ace29741ffe`. Heldout remains mode `0600` and outside Git.
- Two unreferenced, non-SQLite `.sql` convenience symlinks in Mini-Dev pointed outside the data tree. Finalization never followed or copied them and recorded an aggregate count of two; every database or other symlink type remains fatal. The earlier failed staging tree was preserved as private operational evidence.
- Final Day 3B cost is CNY 6.99369151: CNY 6.98580551 of container charges across 2.697222 billed hours plus CNY 0.00788600 of conservatively included visible disk charges. The last session cost CNY 1.49099495, the cumulative CNY 12.95 gate passed, the provider instance is shut down, and the combined record is in `experiments/spend.csv`.

## Accepted Day 0 route

1. Model gate: `Qwen/Qwen3-4B-Base` is the preferred core candidate; retain a 1.7B fallback. Freeze the exact model only after M1 measures baseline quality, memory, throughput, and license suitability.
2. Research question: compare terminal execution reward with reliability-gated dense SQL feedback, and test standard versus per-reward normalization.
3. Hardware: begin M1 with one 4090 D 48 GB at the displayed CNY 2.59/hour. Benchmark two 24 GB cards only if separate rollout/training processes appear valuable; do not deploy before image review.
4. Budget: retain a CNY 250 hard cap for the 15-day core and require explicit confirmation for any single run above CNY 30. Every upgrade receives a separate budget decision.
5. Schedule: release the complete core within fifteen concentrated working days and target 2026-08-31; start upgrades only after the release path is green.
6. Release: public code and reports; public processed data and adapters only when upstream licenses permit.

## Open items owned by later milestones

- The selected M1 base checkpoint is `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`; 1.7B remains the fallback.
- M3 must consume only `operational_eligible=true` train/validation rows through the frozen M2 loader and use the identical amended evaluator boundary for baseline and later training comparisons.
- M6 implements the already-frozen dense structural rewards and attack suite; it may not choose equations or weights after observing experiments.

## M3 Day 4A accepted local facts

- M3 uses answer-only supervision: system, user, assistant-prefix, and padding labels are `-100`;
  full gold SQL plus EOS participates in loss.
- Schema v1 includes deterministic tables, columns, declared types, primary keys, NOT NULL, and
  foreign keys. It reads no sample values and removes only complete trailing table blocks on
  overflow; gold SQL is never truncated.
- Tiny selection uses a fixed seed, eligible-only M2 train/validation boundaries, required
  join/aggregate/subquery coverage, ordered sample IDs, and SHA-256 identities. Counts are
  preregistered at 8 train and 8 validation; the real token report may stop the run but cannot
  trigger replacement, seed shopping, or a count change.
- Base and SFT are bound to the same ordered validation IDs, prompt hashes, greedy decoding, fixed
  model revision, and unchanged M2 evaluator limits.
- A real update requires supervised tokens, trainable LoRA parameters, finite loss, nonzero finite
  gradient norm, at least one optimizer step, and a detected adapter parameter delta.
- Adapter reload must occur in a separate process and preserve the saved adapter file manifest.
- Local evidence uses synthetic fixtures because the real M2 manifest/SQLite files remain only on
  the stopped retained cloud disk. No real M3 metric, optimizer update, VRAM, throughput, time, or
  cost has been measured.
- Heldout examples loaded and evaluated remain exactly zero. No cloud resource, model/data download,
  M4 work, or RL work occurred in Day 4A.
- Commit `3243a273e1751e8d6b49fa95adb24614fb41ff99` was the initial Day 4A closure but did not
  automate every pre-cloud gate. Hardened execution closure
  `1cb5431ecbada21968dff2659ad66d3f9e02af98` supersedes it for paid execution and requires its
  full SHA through `M3_APPROVED_COMMIT` before any GPU stage.
- The hardened orchestrator requires a new resolved output directory outside Git, 20 GiB actual
  filesystem headroom, and one cumulative 14,400-second process-level deadline with timeout
  evidence and terminate/kill escalation. Expected runtime remains 1.5–3 hours; the fourth hour is
  completion headroom, not additional experiment scope. Critical JSON evidence is atomic, stage
  logs append, Base and optimizer-step progress are checkpointed, and adapter publication uses a
  same-filesystem partial-directory rename.
- The full local gate passes 178 tests and the independent M3 gate passes 43 contract/orchestrator
  tests. A
  compatibility guard treats CPython builds without SQLite extension-loading support as already
  unable to load extensions; builds exposing the API still disable it explicitly. Evaluator safety,
  comparison semantics, and frozen limits are unchanged.
- Paid attempt 001 at approved commit `1cb5431ecbada21968dff2659ad66d3f9e02af98` passed
  source/environment/disk gates but stopped before model loading during real token inspection:
  SQLite returned `to=NULL` for a legal `REFERENCES parent_table` foreign key and schema v1 assumed
  an explicit referenced column. Heldout use, Base evaluation, optimizer steps, and adapter output
  all remained zero. The local compatibility closure preserves explicit-column rendering and emits
  deterministic column-free `REFERENCES parent_table` text for the implicit-parent-key form.
- Compatibility execution commit `5475d185a547d5ef3f2faab2e211cff43fdb248c` is the only approved
  source for attempt 002; the 8/8 slice, seed, prompt, model, evaluator, and offline-cache contract
  are unchanged.
- Attempt 002 passed real token inspection and fixed 8/8 slice publication, then stopped after model
  loading but before Base generation because Transformers 5.14.1 returned a BatchEncoding mapping
  and `_chat_ids()` iterated its key names into `input_ids`. The tokenizer compatibility closure
  normalizes only one integer token sequence and adds fail-closed publication/load validation.
- Tokenizer compatibility execution commit `530a0f22f491882def5e13b8344d1310c1a7960d` is the only
  approved source for attempt 003.
- Attempt 003 completed from execution commit `530a0f22f491882def5e13b8344d1310c1a7960d`; the later
  pin commit is `f1a50c463c011153fbe6ea098e5c8a72e53af150`. The identities are intentionally distinct.
- The real tokenizer preflight observed Transformers 5.14.1 `BatchEncoding` output and normalized
  71 prompt tokens to exact integers. Its synthetic prompt had 9 answer/supervised tokens and
  proved answer-only masking before real slice publication.
- The frozen real slice contains exactly 8 train and 8 validation rows, heldout use is 0/0, and its
  bundle SHA-256 is `3b18e7d1e59e42cbb12d7d61c08e3031602c9bcf7e1d8e8c3161c6002b6d58e5`.
  Train coverage is join 7, aggregate 2, subquery 1; validation coverage is join 4, aggregate 3,
  subquery 2. Every selected target fit without truncation or replacement.
- Tiny QLoRA trained 33,030,144 parameters over 461 supervised tokens. It completed 8 microsteps
  and 2 optimizer updates; observed loss moved from 0.64838308 to 0.40970221, gradient norm was
  1.33139443, and adapter delta L2 was 1.43249453. These are update/smoke diagnostics, not a formal
  convergence claim.
- Adapter manifest `f1f9f41e53e1e9a4d44d4b64d7db296c2f0ddcafa5f62bb4d9aea97ab047f883`
  matched the saved files, Base/train artifact, and final result. Reload PID 4775 differed from the
  Base/train PID 1666.
- Base and SFT evaluated the same ordered eight validation IDs and prompt hashes with identical
  greedy decoding and frozen M2 evaluator. Independent row recount gives Base behavior/strict/
  SQL-only 0/8, 0/8, 0/8 and SFT 3/8, 3/8, 8/8. SFT failures were four wrong results and one timeout;
  no evaluator error or gold-preflight failure occurred.
- Peak Base/train memory was 12,278 MiB by `nvidia-smi` (9.592 GiB torch allocated, 11.557 GiB
  reserved). Reload peaked at 3,734 MiB. Reported completion throughput is generation-plus-evaluator
  end-to-end throughput: Base 15.809 and SFT 6.080 token/s.
- Complete attempt-003 session time comes from `session-completion.json`: 622.617 seconds. Two
  user-verified provider container invoices, CNY 1.43241388 and CNY 2.45042777, total CNY
  3.88284165. The program's wall-derived estimate is not used as actual cost. Cumulative M1–M3
  spend is CNY 13.52768593.
- `results/m3/m3-vertical-slice-summary.json` is a curated, SQL-free closure artifact generated by
  an independent audit of all 30 private attempt-003 files. File/directory permissions are
  0600/0700 and the inventory SHA-256 is
  `829447ecef9c481f6b6c4f4eee4ffc2f0ab378319aaa0f3f32714b90e2cf7cfb`.
- M3 is technically complete. The delegated learning note is preserved, but the user teach-back
  remains unchecked; this affects interview readiness, not the reproducible engineering gate.

## M4 Day 5B accepted execution facts

- Exact execution commit `f187167a882bf6b9fd2b152220fabbf770e92960` passed the pinned
  Transformers 5.14.1 source contract and all real Trainer regression cases before training.
- The primary experiment completed all 5,364 operationally eligible train examples once: 5,368
  planned microbatches, 671 optimizer updates, four real final microbatches plus four zero-gradient
  padding forwards, and no resume.
- Required checkpoints at updates 512 and 671 have finite loss/gradient/LR/time/hash evidence. The
  final adapter, checkpoint-671 optimizer state, raw Base/SFT evaluation evidence, logs, and a
  dereferenced archive are retained locally outside Git; the public summary is
  `results/m4/m4-public-summary.json` with SHA-256
  `78348da5f336dcbb7972a42b984eb03e3d703668877eaaba6283622142ed0136`.
- Unique full1209 E0/E1 generation used the same selection, prompt hashes, greedy decoding, and M2
  evaluator. Base behavior/strict accuracy was 0.0000/0.0000; SFT behavior/strict accuracy was
  0.42349/0.40695. SFT SQL-only and valid-SQL rates were 1.0000 and 0.87593. Heldout remained 0/0.
- The archived session completed in 41,804.54 seconds. Dev256 teacher-forced loss changed from
  0.22349429 at update 512 to 0.22309262 at 671, a 0.1797% reduction. The 1% figure is advisory;
  no epoch-2 extension has been approved or run.

## M5 local error-analysis status

- The user explicitly authorized continued local work while the M4 provider invoice is temporarily
  unavailable. This does not waive the invoice requirement for accounting closure.
- A deterministic SQL-free aggregate of all 1,209 SFT validation results is recorded at
  `results/m5/m5-error-analysis.json`; heldout use remains 0/0.
- A deterministic 60-row private audit worksheet has been created with mode 0600 outside Git: 32
  wrong-result, 20 execution-error, and all eight rare-tail cases. The SQL-free selection receipt is
  `results/m5/m5-audit-selection.json`.
- The M5 taxonomy and validator are implemented. Private annotation covers all 60 rows. An
  independent blind review examined the six ambiguity-sensitive cases: three are closed as benchmark
  ambiguity/Gold defects and three require targeted read-only database evidence. The current private
  state has 58 high- and two medium-confidence labels, with three cases still open. All 20 sampled execution errors currently
  map to schema grounding, while wrong-result cases span schema, predicate, aggregation, projection,
  calculation, cardinality, composition, and four ambiguity/equivalence candidates.
- A training-only, outcome-blind data-v2 candidate pool is versioned: 1,678/5,364 examples across
  all 59 training databases, selected by multi-join, subquery, group-by, or having structure. Its
  public SQL-free receipt is `results/m5/m5-data-v2-candidate-receipt.json`. This is a candidate
  pool, not authorization to retrain or duplicate every selected row.
- M5 consumer roles are frozen: E1T is a bounded supervised control; E2–E4 all start independently
  from immutable E1; Correction SFT is trigger-only. Validation is disclosed as development-used,
  while Mini-Dev 500 remains untouched for the final single-use gate.
- M6 group reward math is implemented independently of TRL. E2 terminal rewards, E3 sum-then-
  normalize, E4 normalize-then-sum, confidence gating, population standard deviation, constant-group
  behavior, non-finite inputs, and unknown-arm failure paths pass local tests. The pinned SQLGlot
  SQLite extractor emits the frozen eleven dimensions, schema-resolves identifiers, canonicalizes
  aliases and typed literals, and fails closed under the local attack suite. E4 terminal dominance is
  exhaustively checked for registered group sizes 2/4/8. Real training-schema coverage and framework
  integration remain pending. A training-only aggregate audit runner now verifies the frozen
  manifest identity, 6,573 eligible rows, 6,496 unique SQL, 69 real SQLite schemas, path containment,
  and zero heldout input before measuring extractor coverage. No RL run has started and heldout
  remains untouched.
- Strict-correct/wrong-result/execution-error counts are 492/567/142. Only eight examples occupy the
  combined syntax/timeout/resource/unsafe tail.
- Complex gold queries score 22.22% strict accuracy; subquery, group-by, and having are the weakest
  sufficiently visible structural signals. These are proxy diagnostics, not new reward labels.
- No epoch-2 extension, data-v2 training, RL, or heldout evaluation has started. One short paid
  session accessed the retained training databases for the M6 schema audit; its provider invoice is
  pending and no model computation occurred.
- The M6 real-schema audit ran from execution commit `fcab4f1` against the frozen training manifest
  `d90ffab...5b72a`: 6,488/6,496 exact `(db_id, SQL)` identities are supported (99.8768%) across
  all 69 training databases. Seven nondeterministic time-function queries and one Gold query with a
  duplicate `T2` alias fail closed. All eight retain terminal execution reward and receive
  `g_conf=0`; heldout use remained 0/0. The aggregate result is
  `results/m6/m6-training-schema-audit.json`.
- The pinned TRL 1.9.0 wheel source confirms that custom reward callbacks receive ordered prompts,
  completions, completion token IDs, and repeated dataset columns. `TRLGroupRewardAdapter` now
  validates explicit contiguous group identities and metadata alignment, then delegates E2/E3/E4
  values to the framework-independent math. Synthetic two-group parity and failure tests pass; no
  model, optimizer, or RL loop has run.
- `SQLiteRewardComponentScorer` now connects generated SQL to the frozen read-only evaluator,
  strict result comparison, schema-aware structure score, confidence gate, and TRL adapter. It
  reuses one Gold execution/schema per contiguous completion group and rejects database paths
  outside the configured training root. End-to-end synthetic SQLite tests cover correct,
  execution-equivalent, wrong-result, unsafe, cache, unsupported-Gold, and path-escape behavior.
  M6 is technically complete; interview mastery is not claimed.
- M7 E2 smoke executed from commit `91a95bad2ca6c2c0cf2c083ee2d57c72cf5f35d0` on one RTX 4090 D
  48 GB. Eight outcome-blind training prompts produced 32 rollouts and eight optimizer updates.
  Finite loss and gradients were recorded, maximum gradient norm was 0.6484375, adapter delta L2
  was 0.13338757, peak GPU memory was 11,736 MiB, and a separate process reloaded the saved adapter.
  Heldout use was 0/0. Two of eight groups were mixed-reward and six had zero reward variance;
  therefore the path is valid but the smoke makes no accuracy claim. The public aggregate is
  `results/m7/m7-e2-smoke-summary.json`. Its three source JSON hashes were verified after download,
  but the temporary raw local copies expired before closure; recovering the retained cloud copies
  would require a separately approved paid restart. Actual provider billing is still pending.
- Formal E2 executed from commit `98ea3c0c795e7b5300a09348573b2795ba17ef29` on 256 outcome-blind
  training prompts and 1,024 rollouts. All 256 updates completed; 102 groups were mixed-reward
  (39.84375%), the preregistered signal gate passed, maximum gradient norm was 1.578125, adapter
  delta L2 was 0.52239771, peak memory was 19,804 MiB, and the adapter independently reloaded with
  manifest `3492a3ec...bd3b7`. Training-sample terminal reward was 334/1,024 and valid rate was
  795/1,024; neither is validation accuracy. One unsafe and one resource-limit completion were
  blocked, truncation remained zero, and quarterly entropy/length showed no aggregate collapse or
  explosion. Validation and heldout use were 0/0. The public evidence is
  `results/m7/m7-e2-formal-summary.json`; the final adapter and raw SQL-free logs are retained under
  ignored `artifacts/m7-e2-formal-001/`. Actual provider billing remains pending.
- M8 Phase A is frozen in `configs/m8_common_eval.toml` and `docs/M8_PLAN.md`. It reuses the accepted
  E1 aggregate and evaluates formal E2 exactly once on the same full1209 validation selection
  `3dc2fcbd...f0b1`, deterministic prompt contract, greedy decoding, and unchanged M2 evaluator.
  Regenerating E1 is forbidden. If existing raw E1 evidence is recoverable at no generation cost,
  the SQL-free report also includes paired transitions and an exact McNemar test; its absence does
  not justify another E1 run. Mini-Dev remains untouched. The current M7 trainer is correctly
  recognized as E2-specific and cannot run E3/E4 by a TOML-only change.
- M8 Phase A executed from commit `95da8873c7631e028d149bcde0e1d8ee9ce2c993` on the frozen
  full1209 validation selection. E2 strict execution accuracy was 512/1,209 (42.3490%), versus E1
  492/1,209 (40.6948%), a +1.6543-point change. Behavior-compatible accuracy improved by 1.9024
  points and valid-SQL rate by 3.1431 points; SQL-only remained 100%. Paired strict transitions were
  19 regressions and 39 improvements, with exact two-sided McNemar p=0.01193. Behavior-compatible
  transitions were 18/41 with p=0.00379. The result is positive on validation but is not a heldout
  claim; heldout use was 0/0. The public aggregate is `results/m8/m8-common-eval-summary.json`, and
  the ignored raw evidence hash is `735c2925...b165`. Actual provider billing remains pending.
- E3 is preregistered as a controlled dense-reward intervention. Its smoke and formal contract both
  start from immutable E1 and preserve E2's selected prompts, generation settings, optimizer, seed,
  update count, and checkpoint schedule. The shared runner explicitly dispatches E2/E3 and names E3
  telemetry `mean_group_relative_reward`.
- The E3 smoke orchestrator now closes aggregate evidence rather than trusting subprocess exit codes
  alone. It requires complete reward/component coverage, a nonzero adapter update and mixed signal,
  centered E3 group rewards, matching train/reload adapter identity, a successful independent
  reload, bounded runtime, and zero heldout use before publishing `posttraining-summary.json`.
- `e3_m8_dense_grpo_smoke_001` passed from execution commit `5579446`. It completed eight prompts,
  32 rollouts, and eight optimizer updates in 304.91 session seconds, with 11,736 MiB peak GPU
  memory. Seven of eight groups had mixed reward and one had zero variance; the adapter changed by
  L2 0.10633 and independently reloaded with matching manifest SHA-256. Across the sampled
  completions, seven were strict-correct, 21 were valid/confidence-gated, and 11 ended in execution
  error. Heldout use remained 0/0. These are engineering and on-policy signal diagnostics, not
  validation-accuracy evidence. Public evidence is `results/m8/m8-e3-smoke-summary.json`.
- The first smoke attempt failed before model load because the installed `xgrammar 0.2.3` package
  lacked its RECORD-listed `libxgrammar_bindings.so`. Reinstalling exactly `xgrammar==0.2.3` with no
  dependency changes restored the library; `GRPOTrainer` import and `pip check` passed before the
  successful rerun. The failed attempt remains separately retained on the cloud disk.
- Formal E3 executed from commit `1732ea19e08460422c2d75bb19b4f356638c1993` and completed the
  controlled 256-prompt/1,024-rollout/256-update budget from immutable E1. The adapter changed by
  L2 0.30966, independently reloaded with manifest `890d84cd...d5f1`, used 19,804 MiB peak memory,
  and accessed zero validation or heldout examples. Training took 3,387.67 seconds and the complete
  session took 3,506.27 seconds.
- E3 produced mixed dense reward in 239/256 groups (93.36%), versus 102/256 (39.84%) under E2;
  only 17 E3 groups had zero variance. Its 363 strict-correct and 786 valid sampled completions are
  on-policy training diagnostics, not validation accuracy. Mean logged entropy was 0.19975 and the
  final value was 0.03636; there was no completion clipping or length explosion, but the frozen
  full1209 exam is required to determine downstream quality. The four downloaded source JSONs
  match their cloud hashes, and the public aggregate is `results/m8/m8-e3-formal-summary.json`.
- The E3 common-validation packet is locally verified. The actual retained prepared bundle contains
  exactly 1,209 examples with selection SHA-256 `3dc2fcbd...f0b1`, the retained adapter recomputes
  to manifest `890d84cd...d5f1`, and heldout counters are 0/0. The generic runner accepts only the
  registered E2/E3 identities and rejects adapter, sample, prompt, decoding, evaluator, and heldout
  drift. Full verification passes 289 tests; the M8 focused gate passes 64 tests.
- `e3_m8_common_validation_001` completed all 1,209 frozen validation examples from execution commit
  `9a5f8de`. E3 strict accuracy is 504/1,209 (41.6873%), behavior-compatible accuracy is 43.4243%,
  valid-SQL rate is 89.8263%, and SQL-only remains 100%. Heldout use is 0/0.
- Relative to E1, E3 gains 0.9926 strict-accuracy points with 15 regressions and 27 improvements
  (exact McNemar p=0.08843). Relative to E2, E3 loses 0.6617 points with 25 regressions and 17
  improvements (p=0.27996). Dense E3 therefore delivered much more training signal but did not
  beat terminal-only E2 under common validation; this is an accepted controlled negative result.
- The SQL-free summary at `results/m8/m8-e3-common-eval-summary.json` exactly matches cloud SHA-256
  `fe5b86e8...827f`. The 1.2 MiB raw E3 evaluation was retrieved without rerunning inference to the
  ignored path `results/m8/raw/e3_m8_common_validation_001-e3-full.json`; its local SHA-256 exactly
  matches the recorded cloud identity `db638a33...f2e`. Actual provider cost is pending.
- `e4_m8_gdpo_smoke_001` is preregistered as an eight-prompt, 32-rollout, eight-update smoke from
  immutable E1. Its contract mechanically differs from E3 smoke only in experiment identity, arm,
  and `normalize_then_sum` aggregation; retained E2/E3 prompt-bundle identity is
  `82cfce66...c3265`. The exact packet passed 300 complete and 75 M8-focused tests under Python
  3.11.15 / TRL 1.9.0, including real SQLite-to-E4 reward flow and `GRPOConfig` acceptance. A
  training-only cloud preflight regenerated the identical ordered eight-prompt bundle with canonical
  SHA-256 `82cfce66...c3265`, file SHA-256 `7dc21256...ccf2`, and heldout 0/0. No model load, GPU
  training, validation, or heldout access occurred.
- `e4_m8_gdpo_smoke_001` then completed from execution commit `8376c57`: eight prompts, 32
  rollouts, eight optimizer updates, seven mixed groups, one zero-variance group, adapter delta
  0.10458, 11,736 MiB peak memory, and 314.72 seconds end to end. A separate process reloaded adapter
  manifest `e16ce695...ba5d5`; heldout and validation use remained zero. Attempt 001 was an
  environment-only failure before model load due to a missing `xgrammar 0.2.3` native library;
  reinstalling only the same package fixed the import, and attempt 002 passed. Downloaded SQL-free
  evidence matches all recorded hashes. The public aggregate is
  `results/m8/m8-e4-smoke-summary.json`; provider cost remains pending.
- `e4_m8_gdpo_formal_001` completed from execution commit `359a673` as the exact aggregation-only
  counterpart to formal E3: immutable E1 parent, 256 prompts, 1,024 rollouts, and 256 updates.
  It produced 235/256 mixed groups and 21 zero-variance groups, changed its adapter by L2 0.35576,
  independently reloaded manifest `719bed30...3522`, used 19,804 MiB peak memory, and accessed zero
  validation or heldout examples. The complete session took 3,436.65 seconds.
- Two preceding E4 attempts stopped before model load: one corrected a duplicated database-root
  prefix, and one exposed a missing provider-managed `xgrammar 0.2.3` native-library symlink target.
  The successful attempt used the exact wheel's extracted library through `LD_LIBRARY_PATH` after
  two independent import checks, without changing source, data, config, or package versions.
- E4 had 235 mixed groups versus E3's 239, so per-component normalization did not increase the
  observed frequency of mixed group signal. Its 366 strict-correct and 796 valid sampled training
  completions remain on-policy diagnostics, not validation accuracy. Public SQL-free evidence is
  `results/m8/m8-e4-formal-summary.json`; exact remote/local hash comparison and provider billing
  remain pending.
- `e4_m8_common_validation_001` completed from execution commit `eb1e132` on the unchanged
  full1209 exam. E4 scored 41.4392% strict, 43.3416% behavior-compatible, 89.4955% valid SQL, and
  100% SQL-only. Strict deltas were +0.7444 points versus E1, -0.9098 versus E2, and -0.2481 versus
  E3; paired exact McNemar p-values were 0.2110, 0.1081, and 0.7283. The adapter, selection, prompt,
  decoding, evaluator, and heldout-zero identities passed; only E4 predictions were generated.
- E2 remains the best measured validation arm at 42.3490% strict accuracy. E3's 239/256 and E4's
  235/256 mixed training groups show denser reward variation, but neither dense-reward arm improved
  common-validation accuracy over E2. M8 closes with a reproducible controlled negative result.
  The SQL-free four-arm summary is `results/m8/m8-e4-common-eval-summary.json`; the raw E4 evidence
  is retained locally under the ignored `results/m8/raw/` directory. Provider billing is pending.
- `m9_e2_vllm_serving_001` was locally preregistered to serve the measured-best E2 rank-16 adapter
  through a static vLLM OpenAI-compatible endpoint. Its deterministic validation-only hash64 slice
  contains 64 of the frozen 1,209 examples with selection SHA-256 `10d2280b...ff13a`; the same slice
  is measured at concurrency 1/4/8 after two warmups, for 192 measured requests total. Streaming
  client timestamps produce TTFT and end-to-end latency, usage events produce token throughput,
  `nvidia-smi` samples ready/peak memory, and concurrency-1 outputs use the unchanged M2 evaluator.
  Raw SQL evidence stays ignored and only a whitelisted SQL-free aggregate may enter Git. The
  formal benchmark later completed in attempt 003; heldout access did not occur.
- M9 cloud attempt 001 stopped during model loading and before endpoint readiness or benchmark
  requests because vLLM 0.25.1 defaults prefix caching to enabled. The runner now passes the
  supported explicit `--no-enable-prefix-caching` flag and regression-tests that command/config
  agreement; attempt 001 produced no model-quality or serving metrics.
- M9 cloud attempt 002 also stopped before endpoint readiness or benchmark requests. The pinned
  vLLM selected FlashInfer's sampler, whose first-use JIT could not execute the provider-restored
  0644 CUDA `cicc` binary. The runner now sets the supported
  `VLLM_USE_FLASHINFER_SAMPLER=0` switch, preserving greedy decoding through the Torch sampler
  without mutating the provider CUDA installation.
- M9 cloud attempt 003 completed from execution commit `b47fa46`: the static E2 endpoint became
  ready in 155 seconds and all 192 measured requests succeeded. Request throughput was 1.7314,
  4.5062, and 6.5132 requests/s at concurrency 1/4/8; TTFT p50 was 0.0940/0.0932/0.1157 seconds,
  and end-to-end latency p95 was 0.9311/1.4682/1.7720 seconds. Ready/peak GPU memory was
  28,972/29,246 MiB.
- The deterministic 64-example serving slice scored 30/64 strict execution correct (46.875%),
  32/64 behavior-compatible (50%), 95.3125% valid SQL, and 100% SQL-only. This small slice checks
  that deployment preserved task behavior; it does not supersede E2's full1209 accuracy. The
  public summary SHA-256 is `85b20f14...07b8`, its private raw identity is `9fc7343c...6ce1`, and
  heldout use remained 0/0. Provider billing is pending.
- M10 implementation commit `392b8a3` provides an exact Python 3.11.15 CPU lock, Dockerfile,
  Make/README entry points, two CI jobs, and a deterministic manifest over 19 public SQL-free
  artifacts. The release manifest SHA-256 is `67f10d78...5f7f2`.
- A fresh environment created from a Git-only archive of `392b8a3` passed 314 tests with four
  expected optional-runtime skips, Ruff, compileall, local environment validation, `pip check`, and
  public-evidence reconstruction. It loaded no GPU stack, model, database, validation raw row, or
  heldout data. Local Docker was unavailable; the identical container build is configured as a CI
  gate and must not be described as locally executed.
- The M11 local release candidate adds Apache-2.0 licensing, citation metadata, a concise public
  README, technical report, model/data cards, three-level reproducibility guide, research follow-up,
  release/interview packet, and an identity-checked `verireason-demo` entry point for the selected
  E2 adapter. The M11 gate binds those files to the accepted 1,209-example E1/E2 evidence, paired
  p-value, 192 serving requests, 19 SQL-free artifacts, and heldout 0/0.
- Incremental M11 verification passed 320 tests with four expected optional-runtime skips under a
  clean Python 3.11.13 environment because the installed macOS Python 3.11.7 still exits 139 when
  importing pytest. This is not a replacement for the prior exact-3.11.15 M10 receipt; the changed
  release candidate still requires exact-3.11.15 public CI before the release is marked complete.
- The public repository is `https://github.com/Keke946/VeriReason`. A sanitized current-tree
  snapshot was used rather than exposing all 40 development-history commits. Public commit
  `dc133fc` passes both exact-Python-3.11.15 and Docker CI jobs; the first snapshot failed only
  because five ignored SQL-free summaries were absent, and the follow-up commit added them.
- The selected E2 adapter is already backed up locally with manifest
  `3492a3ec...bd3b7`. GitHub Release `v1.0.0` contains the 54,304,343-byte
  `verireason-e2-adapter.tar.gz`; GitHub's remote digest exactly matches SHA-256
  `9e17b2a5...bcf8`. E1, E3, E4 adapters and private M2 manifest/receipt evidence have not yet been
  proven to have two durable copies. The retained provider disk must not be deleted until that
  migration audit passes or the user explicitly accepts retraining instead of no-retraining replay.
- `docs/WEB_HANDOFF.md` now gives a new web or CLI task the minimum reading order, CPU-only setup,
  accepted headline result, E5 feasibility gate, and privacy/compute boundaries. Codex CLI 0.163.0
  was also verified on this Intel macOS 12.7 host with a real read-only GPT-6.1-Sol response.
- The minimal cloud archive is now local under ignored `artifacts/cloud-backup/`: final E1, E3,
  and E4 adapters plus the final M2 manifest/receipt/audit JSONs. The dereferenced adapter weights
  are 132,187,888, 66,127,776, and 66,127,776 bytes; the private heldout manifest retained mode
  `0600`. Databases, base weights, caches, environments, intermediate checkpoints, and optimizer
  state were intentionally not copied. The provider disk remains available for possible E5 work.
- Current literature review selects an EXPO-SQL-inspired terminal-anchored clause/token-localized
  credit experiment as the only strong E5 candidate. DAPO-style informative-group filtering is a
  secondary efficiency ablation. Further E3/E4 scalar-weight tuning and Mini-Dev access remain
  rejected; E5 is a recommendation, not GPU authorization.
- M11 technical closure separates artifact readiness from personal interview readiness. All release
  gates are complete; `docs/INTERVIEW.md` preserves the outstanding teach-backs without blocking or
  reopening the v1 milestone.

## Post-v1 E5 CPU-only evidence

- `e5_cpu_feasibility_001` reuses the frozen SQLite evaluator, SQLGlot, schema gate, and E2 terminal
  reward. It provides conservative single-clause reference-relative repair spans and a pure token
  credit adapter, without adding E5 to the GPU trainer dispatch or changing E1–E4.
- The built-in synthetic training suite has 37 cases: four terminal-correct, 33 incorrect. Eleven
  incorrect cases localize across six clause classes; 22 abstain. Coverage is 11/37 overall and
  11/33 among incorrect cases (33.33%); incorrect-case abstention is 66.67%. Expected terminal and
  clause outcomes have zero mismatches. These are authored challenge cases, not BIRD coverage.
- LIMIT, nested/alias ambiguity, missing clauses, execution failures, and unsupported semantics
  remain terminal-only. Exact fallback, source-text binding, Unicode offsets, and E2 scorer parity
  are tested. No local bonus or signal for all-zero terminal groups is invented.
- Evidence: `results/e5/e5-cpu-feasibility.json`; design and remaining risks: `docs/E5_DESIGN.md`.
  No validation/heldout input, GPU, model, or paid resource was used. Spend ledger is unchanged.
- Decision: **Revise**. Real tokenizer/loss integration, representative training-only coverage,
  finite-database attribution limits, and gradient-mass effects remain unresolved. No smoke starts.

## E5 CPU tokenizer/loss follow-up

- `e5_cpu_interface_001` uses the actual fixed-revision Qwen3 tokenizer backend, no model weights.
  Eight geometry cases map successfully; five uncertain-ID/text/EOS/boundary cases use exact E2
  fallback. Across the 37 synthetic execution cases, all 11 localizable candidates map to tokens,
  with zero overrides of semantic abstention. Nested/alias geometry is not semantic permission.
- The inspected TRL 1.9.0 source already supports `(B,T)` advantages. A CPU seam changes only
  advantages, retaining E2 terminal group centering, masks and the original DAPO denominator.
  Analytical synthetic-logit loss gradients pass 27 finite-difference coordinates with maximum
  absolute error about 1.41e-11. No real trainer/autograd/optimizer/model run is claimed.
- Local unscaled masks reduce negative update mass. In the six-token/two-target-token fixture,
  negative logit-gradient L1 shrinks to one third; an explicit N/K control restores credit mass
  while tripling selected-token scale. This control is not selected for GPU training.
- Source audit reads only the three SQL-free E2/E3/E4 formal-training summaries. This checkout
  lacks training manifests, prepared rows, databases, generated SQL, and original generated IDs.
  Real training/token coverage remains null. The current runner does not export SQL/IDs or enable
  completion logging; independent private capture has not been demonstrated.
- Decision stays **Revise**; no eight-prompt smoke, cloud start, validation/heldout payload read,
  or change to E1–E4. Evidence: `results/e5/e5-cpu-interface.json`; `docs/E5_INTERFACE.md`.

## Exactly one next action

Obtain and audit a minimal existing local training-only replay bundle from the original retained
artifacts, explicitly recording whether original generated IDs were ever retained; no cloud/GPU,
new inference, validation, or heldout access to fill missing evidence.
