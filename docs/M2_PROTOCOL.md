# M2 Frozen Data and Evaluation Protocol

Last updated: 2026-08-16

Status: completed after control-approved operational-eligibility amendment and evidence-only Day 3B
manifest publication. No gold
SQL is edited or deleted, and the diagnostic Profile-B limits remain non-formal.

## Scope and selected sources

The training source is `birdsql/bird23-train-filtered` at immutable revision
`4068469807b255fcfc0816bdd520946fe460d256`: 6,601 label rows across 69 databases. The labels and
the associated original BIRD database archive are CC BY-SA 4.0. The project-held evaluation source
is the original 500-example, 11-database BIRD Mini-Dev SQLite release, with its label mirror fixed at
revision `f65faf4ae3b638c1fa6df1d3370c8d92c8366301`. Mini-Dev is held out from prompt repair, early
stopping, reward tuning, model selection, and manual data revision.

Source URLs, revisions, expected counts, observed HTTP metadata, and execution limits are frozen in
`configs/m2_data.toml`. The authoritative upstream references are:

- [BIRD project](https://bird-bench.github.io/)
- [BIRD paper](https://arxiv.org/abs/2305.03111)
- [filtered BIRD training labels](https://huggingface.co/datasets/birdsql/bird23-train-filtered)
- [BIRD Mini-Dev repository](https://github.com/bird-bench/mini_dev)
- [BIRD Mini-Dev labels](https://huggingface.co/datasets/birdsql/bird_mini_dev)
- [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

ReViSQL is excluded because no sufficiently clear dataset license was found. SynSQL is excluded from
the M2 core to avoid adding a large synthetic distribution before the BIRD-only vertical slice is
measured. Neither exclusion may be reversed without a new control decision and provenance review.

## Artifact identity and disk gate

For every remote artifact the manifest records URL, immutable revision when available,
`Content-Length`, `ETag`, and `Last-Modified`. These headers are provenance, not content identity.
The first authorized download records the observed local SHA-256; no unpublished upstream checksum
is invented. Every later materialization must match that SHA-256. A header change with an unchanged
SHA-256 produces a warning; a changed SHA-256 is a hard failure.

The first approved Day 3B download observed and froze archive SHA-256 identities
`66e9e3115b59559554013aa3b124156249f30437a6b4e4f96de3d2dfb5ae8cbc` for training and
`aeb211c0e39010bbdae3838bb5e8bd27dc446ed77495b1709f85ccc9bf67f2be` for Mini-Dev. They are now
strong pre-extraction checks in configuration and will also flow into the hashed source receipt.
They are observed project identities, not claims of upstream-published checksums.

Successful materialization writes the observed archive SHA-256 values to an immutable
`source-receipt.json` together with URL, revision where applicable, byte length, ETag,
Last-Modified, label identities, and the final configuration SHA-256. The receipt's own SHA-256 is
referenced by the training, private heldout, and audit manifests. Archive identity is therefore
frozen without modifying configuration after manifests are generated.

Day 3B captures each download's response headers with `curl --dump-header`. For HTTP 200 the final
Content-Length must equal the completed file size. For a resumed HTTP 206 response, the materializer
requires a valid `Content-Range: bytes start-end/total`, checks response Content-Length against the
range length, and checks the completed file against `total`. It records status, response length,
Content-Range, representation length, ETag, and Last-Modified before computing SHA-256 locally.
Metadata drift is reported; only a stable SHA-256 establishes content continuity after first
observation.

The complete 8,919,543,554-byte BIRD training archive is forbidden on the current workstation.
Before any Mini-Dev archive download, the ZIP central directory must be inspected with HTTP Range
requests to establish compressed and total uncompressed bytes. The operation is forbidden unless
the calculated download/unpack peak leaves at least 10 GiB free. Day 3A downloads labels or HTTP
metadata only; databases are materialized in Day 3B after explicit approval.

Day 3B requires at least 70 GiB free on the filesystem that will actually contain the data. The
operator runs `findmnt -T TARGET_PARENT` and `df -B1 TARGET_PARENT`; the materializer independently
checks the nearest existing target parent. Checking `/workspace` does not establish capacity for a
different persistent mount.

The resumed run observed the nested training ZIP as 9,347,158,408 compressed bytes, 1,525 members,
and 32,408,008,953 uncompressed bytes, with SHA-256
`902fed2d1e75d42dd8c9896e006489acdc8518b2c73b21ba3ae1e42d57ea9aaa`. Initial and dynamic
capacity gates passed. Gold preflight nevertheless blocked publication: training recorded
6,282/6,524 unique SQL as `ok`, with 220 execution errors and 22 resource-limit outcomes; heldout
recorded 498/500 `ok`, with 2 execution errors. These historical observations remain bound to the
historical config and do not authorize gold edits. Private failure details remain outside Git.

## Control-approved formal resource amendment

The historical formal configuration is retained verbatim at `configs/m2_data_v1_3s.toml`, SHA-256
`d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b`; it uses 3-second wall and
2-second CPU. The amended `configs/m2_data.toml`, SHA-256
`1fa045ebf72a29d9209f5b37f1042126a81e66e3f7511b1c1fc8a8b21d553181`, freezes:

- wall timeout 10.0 seconds;
- CPU soft/hard 12/13 seconds, with hard required to equal soft plus one;
- 100,000 result rows, 67,108,864 result bytes, and 2,147,483,648 VM bytes;
- 1,000,000 float-pair checks.

RL terminal reward, validation evaluation, and benchmark candidate execution must instantiate this
same version. Profile B's 1,000,000 rows and 512 MiB remain diagnostic-only and cannot enter formal
evaluation.

## Worker diagnostics and failure-only profiles

Every returned `QueryResult` carries `exit_code`, `signal_name`, `limit_kind`, and `worker_state`.
After Pipe EOF or OS error, the parent performs a bounded join before reading the exit code; a live
worker is terminated, killed if necessary, and reaped. Pipe endpoints close on every path. A final
diagnostic record never contains `exit_code = null`. The classification contract is:

| Condition | Status | `limit_kind` |
|---|---|---|
| child exits under `SIGXCPU` | `resource_limit` | `cpu` |
| parent wall deadline expires | `timeout` | `wall` |
| result exceeds formal/diagnostic row ceiling | `resource_limit` | `rows` |
| result exceeds formal/diagnostic byte ceiling | `resource_limit` | `bytes` |
| other signal, caught worker exception, or unexplained exit | `worker_error` | unset |

This corrects classification mechanics but does not reinterpret the accepted 96bcb4c stopped-run
record. Its 167 `SIGXCPU`, 55 unstable IPC EOF, and 22 row-limit outcomes remain the historical
observations.

`configs/m2_diagnostic.toml` is a separate `diagnostic_only` preregistration. Profile A selects only
the prior SIGXCPU/unstable-IPC unique queries and uses a 10-second wall limit, 12/13-second CPU
soft/hard limits, and the formal 100,000-row/64-MiB caps. Profile B selects only the prior row-limit
queries, traverses results status-only, and uses a 30-second wall limit, 32/33-second CPU limits,
1,000,000 rows, and 512 MiB. These values are diagnostic safety ceilings, never automatic changes
to the formal evaluator or RL protocol.

The runner resolves selected SQL from frozen labels plus the private failure identities, never edits
the materialization `.partial` tree, and atomically checkpoints each completed query in a mode-0600
file. Resume skips completed fingerprints. Public output separates training and heldout aggregates;
it contains duration/size buckets and failure categories but no SQL, example IDs, database IDs, or
database paths. Diagnostic success cannot publish a manifest; control must review its evidence.

Before any private JSON or label parsing, the CLI computes and requires these frozen identities:

| Input | SHA-256 |
|---|---|
| private gold preflight | `2408b34df2e4e55544c77c49efb5a8c42364d86ac9e018168ea4b5dfeff98e02` |
| filtered training labels | `2ee64aa593e1adc59be7d5b9ce2395372a8ee27fdd94a3fb4d2fe09f3cd34bc8` |
| heldout labels | `88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2` |
| historical formal `m2_data_v1_3s` config | `d468ca938964597832ce5fb4c23b452fa362e8b6770c9d73a0de80d6c669598b` |
| source receipt | `da189fd0f69d7713c16acfe434738e1f867074665c2b95d7d78b3ace29741ffe` |

All five identities plus the full diagnostic config, profiles, selection, session limit, and selected
query fingerprints enter `plan_sha256`. Resume requires that exact plan. Output paths are resolved
through symlinks and must be outside both the Git worktree and the source materialization
`.partial`. The private checkpoint must be exactly mode `0600` when created and whenever resumed;
any group/world bit is a hard failure.

The runner has a separate `maximum_session_wall_seconds = 5400` gate checked before each new query.
If reached, it atomically saves current private progress, marks the run incomplete, emits no
`complete=true` claim, and stops. Public output includes `plan_sha256`, `complete`, global completed
and expected counts, the five input hashes, and split aggregates only. This session gate does not
replace either profile's per-query wall timeout.

The approved diagnostic completed 244/244 with plan SHA-256
`3ded2e3718c9d2d397c07dd6c2792651dbf3992113d500733fbc562dd699cd6b`. It recovered 214
training Profile-A queries but observed five training and two heldout wall timeouts plus one
training worker error. Profile B measured 9/6/4 queries in the 100k–250k/250k–500k/500k–1M
buckets and three above one million rows. These are diagnostic observations, not amendments to
the formal limits or permission to publish a manifest.

## Operational eligibility and heldout metric scopes

The original 6,601 training rows and all gold SQL remain in the manifest. A query is
`operational_eligible=false` only when frozen evidence identifies one of 28 unique training SQL:
five 10-second wall timeouts, one worker error, or 22 results above the unchanged 100,000-row cap.
Each row retains an `exclusion_reason`; SFT, RL, and train/validation loaders emit only eligible rows.
The finalizer must prove `6,282 + 214 = 6,496` and `6,524 - 28 = 6,496`, then report actual affected
example rows and post-exclusion database/example/complexity/feature distributions by train and
validation. The validation ratio must remain 10%–25%, every critical SQL feature must remain, and
eligible rows must still cover exactly 59 train databases and 10 validation databases. Those counts
are derived from the frozen 69-database corpus and 10-database validation allocation; an exclusion
that empties any database is a publication-blocking failure.

All 500 heldout examples remain in a mode-0600 private manifest: 498 eligible and two ineligible.
For both `bird_ex_behavior_compatible` and `strict_execution_accuracy`, the primary
`full_heldout_conservative_execution_accuracy` fixes the denominator at 500 and assigns both
ineligible gold cases zero for every model. Secondary `eligible_heldout_execution_accuracy` fixes
the denominator at 498. Neither scope exposes heldout identities or permits heldout use in prompts,
model selection, early stopping, reward tuning, or manual revision.

The evidence-only finalizer verifies both private evidence hashes and requires both the original
private preflight and diagnostic checkpoint to have exact mode `0600`, then verifies diagnostic
plan/completion, receipt, labels, and both formal config identities. It does not execute gold SQL. It hard-links the
already materialized databases into a new same-filesystem `.finalization.partial`, repeats SQLite
integrity, database SHA/schema and full leakage audits, writes relative logical paths plus training,
private heldout, eligibility, leakage, evidence, and index manifests, and atomically renames only
after every gate passes. Read-only integrity checks and failed publication attempts must leave each
source database byte length, content, and SHA-256 unchanged. The original materialization `.partial`
is never modified or removed.

Final publication satisfied this contract with 6,496/28 eligible/ineligible training unique SQL,
498/2 heldout eligibility, 5,364/1,209 eligible train/validation rows across 59/10 databases, 80/80
SQLite integrity, passed hard leakage gates, and `gold_sql_reexecuted=0`. Manifest index SHA-256 is
`0e1182eb16bf5bed4b6d5505b11fcf6860689abe9f0435e3d2b3fc33bf7d1cda`; the private heldout
manifest remains mode `0600` and outside Git. The 1,603-file source database identity summary
remained `c7b65515238fd1262d60e56619b2555892a0e9c10d76000e0b99aa981aed4279` before and after
publication.

## Canonical examples and database files

The canonical example interface contains source ID and revision, upstream row ID, `db_id`, question,
gold SQL, optional evidence, schema reference, database SHA-256, and split. UTF-8 strings retain
their original text; normalized forms are derived audit fields, never replacements. SQLite files are
opened read-only. A manifest is not frozen until every referenced database exists, passes
`PRAGMA integrity_check`, and has a recorded byte length and SHA-256.

Database paths in every manifest are relative POSIX logical paths below the published dataset root;
machine paths such as `/root`, `/persistent`, and `/workspace` are forbidden. The heldout manifest
and failure-detail report are mode `0600`, live only under ignored/access-controlled storage, and
must not enter Git.

ZIP extraction rejects absolute names, `..`, backslashes, symbolic links, special files,
case-conflicting or duplicate names, and file/directory prefix conflicts. Member count and summed
uncompressed bytes must equal the Range-observed values, and reading every member verifies CRC and
declared member length. Work is written below an explicit `.partial` directory. Only after source,
database, leakage, split, and gold-execution gates all pass is that directory atomically renamed to
the final path on the same filesystem. Failures never create the final directory.

The observed BIRD training layout is two-level: outer member `train/train_databases.zip` is itself
a deflated ZIP, so its central directory cannot be inspected independently with HTTP Range. The
outer archive is first validated against its frozen 10-member/9,353,941,605-byte contract and safely
extracted. The exact configured nested path is then observed under immutable hard ceilings of 10,000
members and 50 GiB uncompressed. The 50 GiB value is a safety ceiling, not an observed data size.
Exceeding either ceiling is a hard stop.

After central-directory observation and before any inner member is extracted, the materializer
re-reads capacity on the filesystem containing the actual target. The dynamic requirement is:

```text
required_bytes = observed_inner_uncompressed_bytes
               + heldout_expected_uncompressed_bytes
               + 10 GiB reserve
```

The observed inner size, heldout expectation, reserve, required bytes, current filesystem free
bytes, and pass/fail result are written to `.partial/capacity-gates.json` and, on success, the hashed
source receipt. Failure occurs before the inner extractor is called. On success the same full safe
extractor uses the exact observation. The nested SHA-256, byte size, member count, and uncompressed
bytes enter the receipt before the transient outer extraction is removed. The training database root
is not guessed: exactly one extracted root must resolve every one of the 69 expected `db_id` files.

## Deterministic split

Only the approved filtered training source is divided into training and validation. Database IDs
are ranked by SHA-256 of `seed + NUL + db_id`; the first 10 ranked IDs form validation and all others
form training. The fixed seed is `verireason-m2-bird-filtered-v1`. Mini-Dev is a separate heldout
split and is never passed through a train/validation data interface.

Before locking the materialized manifest, the precheck reports database and example counts and
ratios, examples per database, and a deterministic SQL-complexity proxy. The proxy records the
presence of join, aggregate, subquery, group-by, set-operation, order-by, having, and window
features. Its severity score is `join + aggregate + group_by + having + order_by + 2*subquery +
2*set_operation + 2*window`; labels are simple (0–1), moderate (2–3), complex (4–5), and highly
complex (6+). If validation contains outside 10%–25% of examples or has zero occurrences of any
frozen critical feature, materialization stops for control review. The seed must not be searched or
changed to improve the result.

## Leakage, duplication, and contamination

The audit reports within-split and cross-split counts separately. These findings are hard failures:

- a `db_id` appears in more than one split;
- a SQLite SHA-256 appears in more than one split;
- a complete joint fingerprint `(database identity, normalized question, normalized gold SQL)`
  appears in more than one split;
- a train/validation interface exposes a Mini-Dev gold SQL;
- byte-identical real databases cross splits under different names.

Normalized-question duplicates, normalized SQL/AST duplicates, identical or similar schemas, and
question-token Jaccard similarity at least 0.90 are review findings only. They never fail by
themselves. The report includes both within- and cross-split totals. Database hashes and schema
similarity require Day 3B files; Day 3A must label them pending rather than claim a clean result.

## Safe execution

Predictions must contain exactly one `SELECT` or `WITH ... SELECT` statement. The static analyzer
classifies the top-level statement rather than rejecting keywords at arbitrary expression
positions: `CASE ... END`, nested CASE, and `REPLACE()` are legal expressions, while
`WITH ... DELETE/UPDATE/INSERT/REPLACE` is rejected. One terminating semicolon may be followed only
by whitespace/comments. Comments and quoting do not permit hidden extra statements. DDL, DML,
transaction, attach/detach, pragma, vacuum, reindex, analyze, extension loading, and other write or
environment-changing operations are rejected during classification and by a SQLite authorizer.
The static classifier is not the security boundary: the connection also uses SQLite URI
`mode=ro&immutable=1`, `PRAGMA query_only=ON`, disabled extensions, and an allow-list authorizer.

Each query runs in a disposable child process with a wall timeout, SQLite progress interruption,
CPU/address-space bounds where supported, and result row/byte caps. A one-way multiprocessing Pipe
is drained by the parent before it joins the child; this ordering prevents a large serialized result
from blocking child exit. On wall timeout the parent terminates, then kills if necessary, joins, and
closes both Pipe endpoints on every path. Timeout, unsafe SQL, malformed
SQL, multiple statements, unsupported behavior, authorizer denial, and resource exhaustion are
scored zero. They are stable outcome categories, not evaluator crashes. Gold-query failure is a
dataset preflight hard failure and invalidates the affected evaluation rather than silently scoring
the model.

Before publication, Day 3B executes every unique exact `(db_id, gold_sql)` pair using these frozen
limits and the real SQLite file. The private preflight report records total and unique SQL counts,
`ok/unsafe/syntax_error/execution_error/timeout/resource_limit`, duration count/total/mean/minimum/
p50/p95/p99/maximum, per-database failed unique SQL and affected-example counts, and failure
example IDs. Public summaries omit failure details and never contain heldout gold SQL. Any non-`ok`
gold result blocks all final manifests; gold SQL is not edited and limits are not raised to hide a
failure.

Gold preflight calls the executor with `collect_rows=false`. The child still fetches every row and
enforces the same timeout, 100,000-row cap, and 64 MiB serialized-result estimate, but returns only
status, column count, row count, result bytes, execution duration, and error category over IPC. Model
prediction evaluation uses `collect_rows=true` because both result sets are required for comparison.

## Frozen execution metrics

`bird_ex_behavior_compatible` is independently implemented behavior compatibility, not a copy or
line-by-line reproduction of an upstream evaluator. It compares successful row tuples as sets:
row order and duplicate multiplicity are ignored, but values retain column position. An illegal or
exceptional prediction scores zero.

`strict_execution_accuracy` compares successful results with column arity preserved. If the gold
query has a top-level `ORDER BY`, row sequence and duplicate multiplicity must match; otherwise row
multisets must match. `NULL` equals only `NULL`; finite numeric values use the fixed absolute and
relative tolerances; non-finite floats never compare equal. Empty results compare equal only when
column arity matches. Unordered float-containing bags use complete bipartite matching under the row
tolerance relation, so the result is independent of either input row order. The strict metric is the
primary terminal correctness signal and the terminal RL reward. Base, SFT, and RL must call the same
evaluator version and frozen heldout manifest.

Before allocating the bipartite adjacency structure, strict comparison checks the frozen
`maximum_float_pair_checks = 1,000,000` budget against `prediction_rows * gold_rows`. Exceeding it
returns `comparator_resource_limit`, never an unqualified inequality or crash. Evaluation records
distinguish `equal`, `not_equal`, `sql_execution_error`, and `comparator_resource_limit`; only
`equal` earns strict execution correctness.

## Pre-registered reward equations

M2 freezes these definitions and parameters. M6, not M2, implements the dense feature extractor,
attack suite, and training integration.

`r_exec in {0,1}` is strict execution correctness. `r_valid in {0,1}` is one only when the
prediction is a single, safe, execution-supported SQL query that parses and executes within all limits.
`r_struct in [0,1]` is the weighted mean of canonical set/multiset F1 scores over this fixed feature
set: select expression/operator signatures 0.20; referenced tables 0.10; referenced columns 0.10;
join edge/type signatures 0.15; WHERE predicate column/operator/connective signatures 0.15;
GROUP BY 0.08; HAVING 0.05; ORDER BY expression/direction 0.07; LIMIT presence/value class 0.03;
set operations 0.04; and subquery nesting profile 0.03. The weights sum to 1.00.

For gold features `G_k` and prediction features `P_k`, define active dimensions
`A = {k | G_k is non-empty or P_k is non-empty}` and:

```text
r_struct = sum_{k in A}(w_k * F1(G_k, P_k)) / sum_{k in A}(w_k)
```

Empty/empty dimensions enter neither numerator nor denominator. If `A` is empty, `r_struct=0`, a
conservative value that an empty output cannot exploit. Standard set/multiset F1 applies only to
active dimensions. Extra structure lowers precision, and identical set-valued features are
deduplicated before scoring.

`g_conf in {0,1}` is one only when gold and prediction parse, schema-resolve, are supported by the
fixed extractor, and the prediction executes within limits. A gold parse/schema failure is a
preflight hard failure. Prediction parse failure, unsafe SQL, timeout, runtime failure, or a
construct unsupported by safe execution sets `g_conf=r_valid=r_struct=r_exec=0`. If safe execution
does support a strictly correct query but only the dense structural extractor does not support its
construct, it earns `r_exec=1`, retains `r_valid=1`, and has `g_conf=r_struct=0`; the validity term
remains gated off in utility.

With fixed weights `w_exec=1.0`, `w_valid=0.05`, and `w_struct=0.20`, raw utility is:

```text
U_i = r_exec_i + g_conf_i * (0.05 * r_valid_i + 0.20 * r_struct_i)
```

Thus every correct prediction has `U >= 1.0`, while every incorrect prediction has `U <= 0.25`, a
strict raw margin of at least 0.75. E2 uses terminal reward only: `R_E2,i = r_exec_i`.

For a completion group `G`, define `z_G(x_i) = 0` when population standard deviation is below
`1e-6`; otherwise `z_G(x_i) = (x_i - mean_G(x)) / std_G(x)`. E3 is sum then normalize:

```text
R_E3,i = z_G(U_i)
```

E4 uses the same fixed weights but normalizes each component before summing:

```text
R_E4,i = 1.0 * z_G(r_exec_i)
       + 0.05 * z_G(g_conf_i * r_valid_i)
       + 0.20 * z_G(g_conf_i * r_struct_i)
```

Formatting, SQL length, textual edit distance, explanations, or reasoning traces receive no reward.
Literal text values are reduced to typed placeholders for structural comparison. Repeated identical
features are deduplicated where the grammar is set-valued; gratuitous extra features lower precision,
so repeated or meaningless structure cannot accumulate reward. Safety and execution gates dominate
all partial signals.

## Fallbacks and release limits

If the source is too large locally, Day 3B uses an approved persistent cloud disk and returns only
manifests, reports, and small test fixtures. If a license remains unclear, the source is excluded.
If an official test label is unavailable, it is never reconstructed or fabricated. If Mini-Dev
materialization cannot satisfy the disk gate, evaluation waits for approved remote materialization;
the training split must not be repurposed as heldout test data.
