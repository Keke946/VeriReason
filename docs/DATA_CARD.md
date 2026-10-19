# Data Card: VeriReason BIRD Protocol

## Sources

| Role | Source and immutable revision | License |
|---|---|---|
| Train/development | [`birdsql/bird23-train-filtered`](https://huggingface.co/datasets/birdsql/bird23-train-filtered) at `4068469807b255fcfc0816bdd520946fe460d256` | CC BY-SA 4.0 |
| Sealed heldout | [`birdsql/bird_mini_dev`](https://huggingface.co/datasets/birdsql/bird_mini_dev) at `f65faf4ae3b638c1fa6df1d3370c8d92c8366301` | CC BY-SA 4.0 |

The repository does not redistribute the multi-gigabyte databases or row-level benchmark content.
It records fixed URLs, revisions, observed archive hashes, and the code needed to materialize them.

## Split and filtering

The filtered training source contains 6,601 rows over 69 databases. A deterministic database-level
split prevents any database from occurring in both train and validation. Resource-bounded gold SQL
preflight marked 28 unique queries operationally ineligible without editing the SQL. The resulting
model-development protocol contains:

| Split | Examples | Databases |
|---|---:|---:|
| Train | 5,364 | 59 |
| Validation | 1,209 | 10 |
| Mini-Dev heldout | 500 total; 498 operationally eligible | 11 |

The heldout set stayed private, mode `0600`, and unused throughout M3–M10. If it is ever opened, the
primary metric is the conservative full-500 score, with its two operationally ineligible items fixed
to zero for every model; eligible-498 is secondary.

## Governance checks

- Fixed source revision and SHA-256 receipts.
- Safe two-level ZIP extraction with traversal, symlink, collision, count, and capacity checks.
- SQLite `integrity_check` on all 80 development/heldout databases.
- Database-ID, database-content, normalized example, and schema leakage checks.
- Relative logical paths in manifests; private heldout paths and SQL stay outside Git.
- Immutable read-only execution and fixed CPU/wall/memory/result limits.

## Known issues

BIRD labels are not assumed perfect. Static and real execution audits found long-running, large-
result, and ambiguous examples. Operational ineligibility prevents evaluator infrastructure limits
from silently becoming label edits. This protocol improves reproducibility but does not turn every
gold SQL into human-verified semantic truth.

## Reproduction

Follow `docs/REPRODUCIBILITY.md`. Approximately 40–70 GiB of temporary/free disk is required during
safe materialization. Do not commit the resulting databases, label rows, private manifests, or raw
model generations.
