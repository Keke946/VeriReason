# M2 Day 3A Evidence and Day 3B Plan

Last updated: 2026-08-15

## Day 3A outcome

Day 3A froze and locally implemented the M2 data/evaluation contract. It did not download either
database archive, start cloud resources, or enter M3. The complete protocol is in
`docs/M2_PROTOCOL.md`; machine constants are in `configs/m2_data.toml`.

The 2026-08-15 closure audit found that the original static analyzer treated expression tokens as
statement tokens: it rejected 502 filtered and 62 Mini-Dev gold queries containing CASE `END`, plus
38 filtered queries using `REPLACE()`. The repaired analyzer classifies the top-level statement and
retains the SQLite authorizer/read-only process as the security boundary. A full rerun accepts all
6,601 filtered and all 500 Mini-Dev gold SQL strings; remaining static rejection categories are
empty. This is static evidence only, not a substitute for Day 3B execution preflight.

The closure also replaces order-dependent greedy float matching with complete bipartite matching,
changes structural reward to active-dimension normalization with all-empty score zero, and requires
source receipts, safe ZIP extraction, relative manifest paths, private heldout files, all-gold
execution preflight, and atomic publication. These measured closure results are registered as
`m2_day3a_closure_001` and saved under `results/m2/day3a-gold-static-analysis.json`.

The final pre-cloud audit then reproduced a second defect: joining the worker before draining a
Queue caused a valid 20,000-row result to block on IPC backpressure and be mislabeled timeout. The
executor now drains a one-way Pipe before join, reliably reaps timed-out workers, and returns only
status metadata for gold preflight while still traversing every row. Regression tests prove the
20,000-row collected and status-only paths complete, an unbounded recursive query still terminates,
and row/byte caps remain active. Float-bag adjacency is checked against a frozen 1,000,000-pair
budget before allocation, with `comparator_resource_limit` recorded separately. HTTP 206 resume
receipts now validate Content-Range length and completed representation size.

Only two small label files were downloaded to `/private/tmp` from immutable Hugging Face revisions.
They are not Git artifacts:

| Artifact | Revision | Content-Length | ETag | Last-Modified | Observed SHA-256 |
|---|---|---:|---|---|---|
| filtered training labels | `4068469807b255fcfc0816bdd520946fe460d256` | 2,654,294 | `"b3ae3560c14aede19d7faea28f6ad3bb86d85714"` | not supplied | `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8` |
| Mini-Dev SQLite labels | `f65faf4ae3b638c1fa6df1d3370c8d92c8366301` | 278,513 | `"f1ee766b3e438395cd3f8282adc10d483f6d91de"` | not supplied | `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2` |

These are first-observed local identities, not claimed upstream checksums. A later SHA mismatch is a
hard failure. A header-only change with the same SHA produces a warning.

At the disk snapshot, the local filesystem had 17,249,624 KiB (about 16.45 GiB) free. HTTP Range
inspection, without archive download, established:

| Archive | Content-Length | ZIP member uncompressed bytes | Entries | Metadata fetched |
|---|---:|---:|---:|---:|
| BIRD `train.zip` | 8,919,543,554 | 9,353,941,605 | 10 | 66,731 |
| Mini-Dev `minidev_0703.zip` | 799,944,582 | 3,539,193,972 | 246 | 102,491 |

The Mini-Dev download-plus-extract theoretical peak is 4,339,138,554 bytes (about 4.04 GiB), which
would leave about 12.41 GiB at the snapshot before filesystem/tool overhead. Despite exceeding the
10 GiB floor, no archive was downloaded because Day 3A is metadata-only. The complete training
archive alone has a theoretical download-plus-extract peak of 18,273,485,159 bytes (about 17.02
GiB), exceeding local free space, so it is categorically forbidden on this workstation.

## Split and leakage precheck

The fixed SHA-256 database-rank split and single approved seed selected 59 training and 10
validation databases. It produced 5,381 training examples (81.518%) and 1,220 validation examples
(18.482%), inside the frozen 10%–25% gate. No seed search occurred.

| Proxy feature | Train | Validation |
|---|---:|---:|
| join | 4,164 | 855 |
| aggregate | 2,375 | 621 |
| subquery | 395 | 108 |
| group by | 495 | 150 |
| set operation | 16 | 1 |
| order by | 917 | 205 |

Validation therefore contains every pre-registered critical feature. Complexity labels are:

| Split | Simple | Moderate | Complex | Highly complex |
|---|---:|---:|---:|---:|
| train | 2,858 | 2,059 | 398 | 66 |
| validation | 650 | 442 | 91 | 37 |

The report also records every database's sample count. At label level there are no cross-split
`db_id`, full joint sample fingerprint, or heldout-interface exposures. Review-only findings are 3
within-split normalized-question pairs, 91 within-split normalized-SQL pairs, and 12 within-split
question Jaccard pairs at least 0.90; all corresponding cross-split counts are zero. Two identical
joint samples occur within a split and are reported without converting a cross-split rule into a
new failure rule. Database SHA, renamed-identical-database, and schema-similarity checks remain
explicitly pending until Day 3B materializes SQLite files.

## Evaluator evidence

The independent evaluator rejects non-query, multi-statement, write, pragma, attach, extension, and
transaction behavior before execution and through a SQLite authorizer. Execution uses URI
`mode=ro&immutable=1`, `query_only`, disabled extensions, a disposable spawned process, wall and CPU
limits, progress interruption, and result caps.

Golden tests establish that `bird_ex_behavior_compatible` ignores row order and duplicate
multiplicity, preserves value column position, and scores illegal/exceptional predictions zero.
Strict tests cover bag versus ordered results, duplicate multiplicity, column arity including empty
results, `NULL`, tolerant finite floats, equivalent SQL, read-only file identity, malicious SQL,
syntax errors, gold preflight failure, and a bounded recursive-query timeout. This is independently
written behavior compatibility, not official evaluator source reproduction.

The IPC regression suite additionally covers a 20,000-row collected result, a 20,000-row
status-only gold preflight, status-only row and byte caps, and an actually unbounded recursive CTE.
Comparator tests verify both permutation-invariant complete matching below budget and diagnostic
rejection before any pair check above budget.

## First Day 3B attempt and safe stop

Control approved Day 3B from exact clean commit
`cba44c4a8a0229723da6e7e9015dd5abf007153d`. The retained Xiangongyun instance reported one RTX
4090 D 48 GB at CNY 2.59/hour, Python 3.11.15, and 107,368,022,016 bytes free on the actual root ZFS
target filesystem. The preflight `make m2-verify` passed 83 tests. Both labels and both archives were
downloaded; a transient TLS EOF on the first training-label request was logged and the bounded retry
succeeded.

| Artifact | Bytes | Observed SHA-256 |
|---|---:|---|
| filtered training labels | 2,654,294 | `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8` |
| Mini-Dev labels | 278,513 | `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2` |
| BIRD training archive | 8,919,543,554 | `66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc` |
| BIRD Mini-Dev archive | 799,944,582 | `aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be` |

Read-only central-directory inspection showed that the training database payload is the deflated
nested member `train/train_databases.zip`, not a directly usable database directory. The one-layer
materializer could not satisfy the database-root contract, so the session stopped before extraction,
SQLite access, gold preflight, leakage audit, or manifest publication. The raw downloads remain on
the retained disk. Its actual cost was not inferred from timestamps; the provider record was later
verified before the resumed attempt and is included in the resumed-result section below.

The capacity closure validates and safely extracts the frozen outer archive, then observes the nested
central directory under a 10,000-member and 50-GiB hard ceiling. The ceiling is not an observed data
size. Before inner extraction, it re-reads the actual target filesystem and requires the observed
inner uncompressed bytes plus 3,539,193,972 heldout bytes plus a 10-GiB reserve. Structured evidence
is saved before the pass/fail decision. The initial target gate is 70 GiB. Synthetic tests use only
metadata and mocked capacity; they do not allocate large files. All 7,101 gold labels remain
statically accepted.
The closure passes 99 local tests, including the 70-GiB boundary, approximately 40-GiB synthetic
inner metadata under the 50-GiB ceiling, over-ceiling rejection, dynamic pass/fail, and proof that
the extractor is not called when the dynamic gate fails.

## Day 3B resumed execution result

The reviewed `9dc25ceaf76072a7e8a0a1ccdb12bc1bfe3f2eb8` bundle passed its SHA, clean
HEAD, environment, 99-test, source-identity, target-path, and 70-GiB gates. The materializer observed
the nested training ZIP as 9,347,158,408 bytes, 1,525 members, SHA-256
`902fed2d1e75d42dd8c9896e006489acdc8518b2c73b21ba3ae1e42d57ea9aaa`, and
32,408,008,953 uncompressed bytes. These are actual observations; the separate 50-GiB value remains
only a safety ceiling.

The initial capacity gate passed with 107,369,463,808 free bytes against 75,161,927,680 required.
After outer extraction, the dynamic gate passed with 98,791,849,984 free bytes against
46,684,621,165 required. The latter equals 32,408,008,953 observed inner bytes plus
3,539,193,972 heldout bytes plus 10,737,418,240 reserve bytes. Capacity evidence SHA-256 is
`275c525dbcd69e459f6f7cc901399b3faa57dfeafeefe4b6a7bcef3c4794c7b3`.

Gold execution preflight then blocked publication:

| Split | Total | Unique | `ok` | Execution error | Resource limit |
|---|---:|---:|---:|---:|---:|
| Training | 6,601 | 6,524 | 6,282 | 220 | 22 |
| Heldout | 500 | 500 | 498 | 2 | 0 |

There were no unsafe, syntax-error, or wall-timeout statuses. Sanitized failure classification found
167 worker IPC EOFs ending with Linux signal 24 (`SIGXCPU`), 55 IPC EOFs recorded before the child
exit code stabilized, and 22 results crossing the frozen 100,000-row cap. The first two reveal CPU
limit/IPC classification behavior; the last is a protocol-limit question. This report does not
expose heldout SQL or example IDs. Private evidence remains mode `0600` outside Git with SHA-256
`2408b34df2e4e55544c77c49efb5a8c42364d86ac9e018168ea4b5dfeff98e02`.

The hard leakage gate did not raise before preflight, but its in-memory report was scheduled to be
written only after preflight. Because publication stopped first, no persistent audit manifest exists
and no final leakage-clean claim is made. No training, heldout, audit, or index manifest was
published. Source receipt SHA-256 is
`da189fd0f69d7713c16acfe434738e1f867074665c2b95d7d78b3ace29741ffe`; gold summary SHA-256 is
`8e1b47ff105f4db2bc0c256f1fab8296da1930c4e30e57add469db22f62c128a`.

The materializer ran 2,005 seconds and exited 1 by design. Verified visible costs were CNY
1.00987346 for the first attempt and CNY 1.78134210 for the second, totaling CNY 2.79121556 and
leaving CNY 10.15878444 under the cumulative CNY 12.95 hard stop. The instance is confirmed shut
down. `experiments/spend.csv` remains deferred until Day 2–3 closure as instructed.

## Local evaluator diagnostic closure

Control accepted the 96bcb4c stopped closure without treating the 222 IPC/CPU records as demonstrated
gold-SQL errors. The evaluator now joins after Pipe EOF before inspecting the exit code, reaps a
worker on every path, and emits structured `exit_code`, `signal_name`, `limit_kind`, and
`worker_state` fields. `SIGXCPU` maps to `resource_limit/cpu`; parent wall expiry maps to
`timeout/wall`; row and byte caps map independently; other signals, caught exceptions, and
unexplained exits remain `worker_error`. Gold preflight no longer folds worker failures into
`execution_error`, and its public serialization omits per-database failure details.

The separate `configs/m2_diagnostic.toml` preregisters two `diagnostic_only` profiles:

| Profile | Prior selection | Wall | CPU soft/hard | Row ceiling | Byte ceiling |
|---|---|---:|---:|---:|---:|
| A | SIGXCPU or unstable IPC | 10 s | 12/13 s | 100,000 | 64 MiB |
| B | row-limit only | 30 s | 32/33 s | 1,000,000 | 512 MiB |

Profile B remains status-only but traverses every row. The runner resolves only prior non-`ok`
identities against frozen label files, writes a mode-0600 atomic checkpoint after every query, and
skips completed fingerprints on resume. It refuses to write below a `.partial` directory and never
publishes manifests. The public aggregate separates training and heldout and reports only the
preregistered time/size buckets and failure counts; SQL, example IDs, database IDs, and paths are
excluded.

The formal evaluator configuration was not edited and remains SHA-256
`d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b`.
The diagnostic profile SHA-256 is
`0dbf750c487a2fd2dcbb4f3c7cefabd16b121f9b9024ef5be452ccafeae07d37`.
The later cloud diagnostic did not overwrite the 96bcb4c result or change this formal configuration.

The final local evidence/privacy gate freezes the accepted private-preflight, training-label,
heldout-label, formal-config, and source-receipt SHA-256 values in the diagnostic config. The CLI
verifies all five files before parsing private JSON or selecting SQL. The full source identity,
configuration, profiles, selection, 5,400-second session limit, and selected query fingerprints are
hashed into `plan_sha256`; resume rejects any drift. Resolved output must be outside Git and the
materialization `.partial`, and checkpoint creation/resume requires exact mode `0600`.

The public report now contains `plan_sha256`, the five input hashes, `complete`, `completed_count`,
`expected_count`, and split-level aggregates only. It cannot contain SQL, example IDs, database
identities, or filesystem paths. The program checks its 5,400-second session wall gate before every
new query; reaching it atomically saves the checkpoint and keeps `complete=false`.

### Completed failure-only cloud diagnostic

Control approved one run from exact commit `e6c56af36f521b823def81126143a00c9306eb42`.
All five evidence hashes and the formal/diagnostic config hashes matched. The runner completed
244/244 selected queries in 965 seconds with plan SHA-256
`3ded2e3718c9d2d397c07dd6c2792651dbf3992113d500733fbc562dd699cd6b`.
The public aggregate is preserved at `results/m2/m2-diagnostic-summary.json`. The private checkpoint
was mode `0600`, SHA-256 `2e8748e29ba87ff94ecda079ae01cf7ae8105ce0965f45e01c1b02e66614c8f7`,
and remains outside Git.

Training Profile A recovered 214/220: 82 completed within 3 seconds, 82 in 3–5 seconds, and 50 in
5–10 seconds. Five timed out and one remained a worker error. Both heldout Profile-A queries timed
out. No CPU-limit or unstable-exit result remained. Training Profile B measured nine queries at
100k–250k rows, six at 250k–500k, four at 500k–1M, and three above one million; it had no byte-limit,
timeout, or worker-error result.

The run therefore completed diagnostically but triggered the preregistered stop conditions. The
instance was shut down. Formal evaluator/RL limits remain unchanged, no manifest was published,
and the unresolved timeout, worker, and over-one-million-row cases return to control.

### Historical execution command

The approved run used the retained disk and exact reviewed source, resolved private database roots
read-only from the preserved `.partial` tree, and ran:

```bash
PYTHONPATH=src python scripts/diagnose_m2_failures.py \
  --private-preflight "$PRIVATE_PREFLIGHT" \
  --training-labels "$TRAINING_LABELS" \
  --heldout-labels "$HELDOUT_LABELS" \
  --source-receipt "$SOURCE_RECEIPT" \
  --formal-config configs/m2_data_v1_3s.toml \
  --config configs/m2_diagnostic.toml \
  --training-database-root "$PRIVATE_TRAINING_DB_ROOT" \
  --heldout-database-root "$PRIVATE_HELDOUT_DB_ROOT" \
  --output-dir "$M2_DIAGNOSTIC_OUTPUT"
```

The database-root variables remained private and were not copied into the public report. The run
used 965 seconds. Provider-visible container cost was CNY 2.70870832; conservatively including
CNY 0.00277268 visible disk charges gives CNY 2.71148100 for this attempt and CNY 5.50269656
cumulative Day 3B cost. Wall timeouts, one worker error, and three over-one-million-row observations
triggered the preregistered stop; the instance was confirmed shut down without another execution.

## Operational-eligibility amendment and evidence-only finalizer

Control accepted the complete diagnostic and approved a versioned formal amendment. The original
3-second wall/2-second CPU config is preserved at `configs/m2_data_v1_3s.toml`, SHA-256
`d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b`. The amended formal
`configs/m2_data.toml` uses 10-second wall, 12/13-second CPU, 100,000 rows, 64 MiB results, and
2 GiB VM, SHA-256 `1fa045ebf72a29d9209f5b37f1042126a81e66e3f7511b1c1fc8a8b21d553181`.
The 1,000,000-row/512-MiB Profile B remains diagnostic-only.

The finalizer config SHA-256 is
`67d7990a989e16824bea087a36984d294d75e960776b035fa08519907ff3a3ad`. It verifies the accepted
private preflight, mode-0600 diagnostic checkpoint, source receipt, both labels, historical and
amended configs, diagnostic plan, and 244/244 completion before resolving any private identity. It
must prove 6,496 eligible/28 ineligible training unique SQL and 498/2 heldout eligibility. It keeps
all labels, records exclusion reasons and actual affected example-row counts, and reruns no gold
SQL.

Publication hard-links the preserved database tree into a new sibling
`m2-v1.finalization.partial`, repeats SQLite integrity, database SHA/schema and leakage checks,
validates post-exclusion split ratio and feature coverage, writes relative paths and a mode-0600
heldout manifest, then atomically renames to `m2-v1`. The original `m2-v1.partial` is never edited,
renamed, or deleted. Any failure leaves private `finalization-failure.json` in the finalization
staging directory and no final directory.

After separate control approval, the exact cloud command is:

```bash
export M2_PARTIAL=/root/verireason-m2-data/processed/m2-v1.partial
export M2_DIAGNOSTIC_CHECKPOINT=/root/verireason-m2-diagnostics-private/diagnostic-checkpoint-private.json
export M2_TRAINING_LABELS=/root/verireason-m2-data/raw/train-labels.jsonl
export M2_HELDOUT_LABELS=/root/verireason-m2-data/raw/minidev-labels.json
export M2_FINAL_DIR=/root/verireason-m2-data/processed/m2-v1

make m2-finalize
```

Before execution, the source `.partial` and checkpoint must exist while both `M2_FINAL_DIR` and
`$M2_FINAL_DIR.finalization.partial` must be absent. Estimated wall time is 0.25–0.75 hours and
estimated incremental cost is CNY 0.65–1.94 at a reverified CNY 2.59/hour. Stop without publication
on any price, Git, evidence hash/mode, plan/completion, 6,496/28, 498/2, label/database count,
integrity, leakage, split ratio/coverage, path, filesystem, staging/final collision, permission,
atomic rename, 90-minute, session CNY 3.89, or cumulative CNY 12.95 failure. No archive download,
extraction, gold execution, database mutation, raw deletion, or M3 action is permitted.

## Superseded pre-resume Day 3B plan

The following plan documents the gates used for the now-completed resumed attempt. It is retained
for provenance, not as authorization to restart after the gold-preflight stop. Any further cloud
execution requires a new control decision.

### Reuse and capacity gate

The first attempt confirmed that the retained root filesystem survived restart and contains the
downloaded sources. A resumed Day 3B must recheck that exact filesystem and have at least 70 GiB
free. Both `findmnt -T` and `df -B1` must target the actual data parent, and the
materializer repeats the filesystem check. If fewer than 70 GiB are free, stop before extraction and ask
control to approve a larger persistent disk or a CPU-only materialization route.

The materializer may temporarily hold the raw archives, the 9,353,941,605-byte outer training tree,
and the inner expanded content. After reading the inner central directory but before extracting it,
the dynamic requirement is `observed inner uncompressed bytes + 3,539,193,972 heldout bytes +
10,737,418,240 reserve bytes`. The current free bytes and result are written to
`capacity-gates.json` and the successful source receipt. Operator-owned raw archives are never
deleted.

### Duration and cost

At the time this plan was written, the downloads were complete but a resumed duration could not be
converted into an authorized cost until the first session's provider bill became visible. CNY 12.95
was and remains the cumulative Day 3B hard stop, not a fresh allowance per restart:

```text
remaining_hard_stop_cny = 12.95 - verified_first_session_actual_charge_cny
```

The first charge was later verified and control authorized the resumed attempt. That attempt is now
stopped; the remaining allowance alone does not authorize another start. Provider price, billing,
protocol disposition, and a fresh control approval would all be required.

### Code synchronization and commands

The resumed attempt transferred a Git bundle made from the reviewed clean repair commit and fetched
it into the existing cloud clone. The commands below are historical execution provenance and must
not be reused as authorization for another paid start:

```bash
git bundle create /tmp/verireason-m2-nested-fix.bundle HEAD
# Upload the bundle through the reviewed provider transfer path.

cd /root/verireason-m2
git fetch /root/verireason-m2-nested-fix.bundle HEAD
git checkout --detach FETCH_HEAD
git status --short --branch
git rev-parse HEAD
conda activate verireason-m1
python -m pip install -e '.[dev]'
make m2-verify

export M2_DATA_ROOT=/root/verireason-m2-data
findmnt -T "$M2_DATA_ROOT"
df -B1 "$M2_DATA_ROOT"

stat -c '%n %s' "$M2_DATA_ROOT/raw/train.zip" "$M2_DATA_ROOT/raw/minidev.zip" \
  "$M2_DATA_ROOT/raw/train-labels.jsonl" "$M2_DATA_ROOT/raw/minidev-labels.json"
sha256sum "$M2_DATA_ROOT/raw/train.zip" "$M2_DATA_ROOT/raw/minidev.zip" \
  "$M2_DATA_ROOT/raw/train-labels.jsonl" "$M2_DATA_ROOT/raw/minidev-labels.json"

PYTHONPATH=src python scripts/materialize_m2_data.py \
  --training-labels "$M2_DATA_ROOT/raw/train-labels.jsonl" \
  --training-label-headers "$M2_DATA_ROOT/raw/train-labels.headers" \
  --training-archive "$M2_DATA_ROOT/raw/train.zip" \
  --training-archive-headers "$M2_DATA_ROOT/raw/train.headers" \
  --training-database-root auto \
  --heldout-labels "$M2_DATA_ROOT/raw/minidev-labels.json" \
  --heldout-label-headers "$M2_DATA_ROOT/raw/minidev-labels.headers" \
  --heldout-archive "$M2_DATA_ROOT/raw/minidev.zip" \
  --heldout-archive-headers "$M2_DATA_ROOT/raw/minidev.headers" \
  --heldout-database-root minidev/MINIDEV/dev_databases \
  --target-dir "$M2_DATA_ROOT/processed/m2-v1"
```

The `curl --dump-header` files preserve the final response's observed status, `Content-Length`,
`Content-Range`, ETag, and Last-Modified. A 200 length must equal the completed file; a resumed 206
must have a valid byte range whose total equals the completed file size. Archive member paths were
observed read-only; the training root is derived from all expected database IDs and the observed
Mini-Dev root is fixed above. The materializer validates and extracts ZIP members itself—raw `unzip`
extraction is forbidden. Label SHA-256 must match the frozen values. First-observed archive hashes
are written to `source-receipt.json`; its hash and the final configuration hash flow into all
manifests without a post-generation configuration edit.

### Return artifacts and stop conditions

Return only the small `source-receipt.json`, `manifest-index.json`, training manifest, mode-0600
heldout manifest and private preflight details through an access-controlled path, audit manifest,
public gold-preflight summary, command log, hashes, `findmnt`/`df` snapshots, environment freeze,
and test output. Do not return SQLite archives to Git. The heldout-gold manifest must never enter a
train/validation loader or public artifact.

Stop before or during work if the first session's actual provider charge is unavailable, the reviewed
repair commit differs, target-filesystem free space is below 70 GiB,
`findmnt` and `df` identify an unexpected filesystem, an HTTP length is
unexpected, a known label SHA changes, an archive transfer is incomplete, ZIP integrity fails, any
SQLite integrity check fails, validation leaves the 10%–25% range or loses a critical SQL type, a
hard leakage rule fires, any gold preflight status is not `ok`, provider price changes materially,
the cumulative Day 3B cost reaches CNY 12.95, or cumulative wall time reaches 5 hours. Do not change
the seed, edit gold SQL, or raise
limits to make a gate pass. Any failure leaves only the named `.partial` directory for review; no
final directory is published.
