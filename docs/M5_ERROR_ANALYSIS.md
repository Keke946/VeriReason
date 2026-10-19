# M5 Error Analysis: First Pass

## What this stage asks

M4 proved that one epoch of QLoRA SFT can turn the 4B base model from 0% to 40.69% strict
execution accuracy. M5 now asks why the remaining 717/1,209 predictions fail, so that data-v2 and
post-training target measured weaknesses rather than adding generic training.

This report is generated from the fixed full1209 SFT evaluation and the corresponding validation
metadata. It does not use heldout data and does not publish questions, SQL, sample IDs, database
IDs, or database paths. The machine-readable aggregate is
`results/m5/m5-error-analysis.json`.

The follow-up private audit selection is frozen at 60 examples: 32 wrong-result cases (eight per
gold-complexity stratum), 20 execution-error cases (five per stratum), and all eight rare-tail
cases. The private worksheet is mode 0600 outside Git. Its SQL-free receipt is
`results/m5/m5-audit-selection.json`, with selection SHA-256
`438287c056bfd9725c74ce84f450da9659144214c865f970636ae27aef3e888a`.

The frozen taxonomy is documented in `docs/M5_TAXONOMY.md`. All 60 private rows have draft labels.
An independent blind review examined the six ambiguity/equivalence-sensitive cases: three were
closed as benchmark ambiguity or Gold defects, while three require real read-only database checks
for ties, NULL/uniqueness, or grouping equivalence. The local machine has no copy of those databases,
so the final aggregate remains a draft. The SQL-free review receipt is
`results/m5/m5-independent-review-status.json`.

Validation examples never enter data-v2. They identify weak capabilities only; targeted examples
must be selected from the training split under an outcome-blind rule. This preserves the frozen
validation exam.

## Main result

The dominant problem is semantic correctness, not output formatting:

- 492/1,209 (40.69%) are strictly execution-correct.
- 567/1,209 (46.90%) execute but return the wrong result.
- 142/1,209 (11.75%) fail during execution.
- Only 8/1,209 (0.66%) fall into syntax, timeout, resource-limit, or unsafe tails combined.

Because SQL-only output is already 100%, more format-only SFT is unlikely to be the best next
investment. The largest addressable pool is the 567 executable-but-semantically-wrong queries.

## Where accuracy falls

The deterministic gold-SQL complexity proxy shows:

| Gold complexity | Examples | Strict accuracy |
|---|---:|---:|
| simple | 642 | 44.70% |
| moderate | 441 | 38.78% |
| complex | 90 | 22.22% |
| highly complex | 36 | 38.89% |

The non-monotonic highly-complex result has only 36 examples and should not be read as proof that
those queries are easier. The clearest supported gap is complex versus simple: 22.22% versus
44.70%.

Accuracy by gold feature is 41.68% for aggregate, 37.46% for join, 36.45% for order-by, 32.08% for
subquery, 26.85% for group-by, and 21.43% for having. Set operations and windows have only one and
two examples, so they are diagnostic anecdotes rather than stable estimates.

Across the 10 validation databases, strict accuracy ranges from 17.95% to 59.48%; the median is
35.87%. This wide spread is evidence of schema/domain sensitivity, while the public artifact
intentionally omits database identities.

## Structural diagnostic

The SQL-text feature proxy compares whether a gold feature also appears in the extracted
prediction. Recall is high for joins (90.22%), aggregates (89.18%), and order-by (91.13%), but much
lower for group-by (59.73%), having (35.71%), and especially subqueries (13.21%).

This is not a semantic equivalence metric: an equivalent query can implement the same operation
without the same surface feature. It therefore identifies review candidates and data needs, not a
new correctness label or reward by itself.

## Decision for the next M5 slice

Prioritize three buckets:

1. Executable wrong-result cases: annotate the frozen, SQL-private sample to separate schema
   grounding, wrong column/table, missing condition, join, aggregation, and compositional errors.
2. Execution errors: classify the 142 cases using private error evidence and determine whether
   schema linking or invalid references dominate.
3. Structurally difficult cases: build data-v2 candidates around subquery, group-by/having, and
   complex queries, but only after the private audit confirms the surface proxy corresponds to
   real semantic failures.

The rare syntax/safety/resource tail remains covered by evaluator tests but is not the primary
training target. Epoch 2 stays deferred: the current analysis is reversible and does not decide
whether a separately preregistered extension is ever worthwhile.

## Reproduction

Run from the repository with the downloaded private M4 closure paths substituted as needed:

```bash
PYTHONPATH=src python scripts/analyze_m5_errors.py \
  --evaluation /private/path/sft-full.json \
  --prepared /private/path/prepared-dev.json \
  --prepared /private/path/prepared-final.json \
  --output results/m5/m5-error-analysis.json
```

The checked-in first-pass aggregate SHA-256 is
`47f71b02e808fc00f8dbbd031be1e5851b0a0502270b768fb537b002fabc0cb0`.
