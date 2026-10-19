# M5 Private Error Taxonomy

## Why labels are frozen before counting

If categories are invented after seeing the totals, it is easy to split or merge errors until the
story looks attractive. M5 therefore defines the label meanings first, validates every private
annotation against that vocabulary, and publishes counts only after ambiguous cases receive a
second review.

Each sampled error has one primary cause and zero or more secondary contributors:

| Label | Meaning |
|---|---|
| `schema_grounding` | Wrong or missing table, column, relationship, or schema-qualified reference |
| `projection` | The query returns the wrong fields or omits required output fields |
| `predicate` | A filter, literal, comparison, or requested condition is wrong or missing |
| `aggregation` | COUNT/SUM/AVG, GROUP BY, HAVING, or grouping level is wrong |
| `composition` | Multi-hop logic, subquery, set operation, or decomposition is missing or wrong |
| `cardinality` | DISTINCT, ties, duplicate handling, or all-versus-one answer semantics are wrong |
| `calculation` | Percentage, date/time, rate, or other derived arithmetic is wrong |
| `execution_syntax` | SQL is rejected because of syntax such as an unquoted reserved identifier |
| `efficiency_resource` | A wrong plan causes timeout or excessive result/resource use |
| `safety_generation` | Degenerate output triggers the safety gate |
| `ambiguous_or_equivalent` | Prediction may be equivalent, or the question/gold pair is materially ambiguous |

The categories answer different engineering questions. `schema_grounding` suggests better schema
representations or targeted examples; `aggregation` and `composition` suggest reasoning-focused
data or dense credit; `ambiguous_or_equivalent` suggests evaluator/data review rather than training
the model to imitate a questionable surface form.

## Review contract

- The frozen 60-row worksheet and annotations remain mode 0600 outside Git.
- The first reviewer records primary/secondary labels, confidence, and `review_required`.
- Low-confidence, possibly equivalent, and benchmark-ambiguous cases require a second review.
- The public summary contains counts only—never sample IDs, questions, SQL, database identities, or
  paths.
- Data-v2 is not designed until the reviewed aggregate exists.

## Core code to understand

`validate_annotations()` in `src/verireason/m5_taxonomy.py` enforces identity completeness and the
frozen vocabulary. `summarize_annotations()` discards row identity and text, then counts primary,
secondary, confidence, and error-category cross-tabs. This separation is intentional: private
evidence supports auditability while the repository remains safe to publish.
