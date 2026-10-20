# E5 CPU tokenizer/loss interface and training-source audit

Owner: BiBo. Experiment: `e5_cpu_interface_001`. Parent implementation: `631b4a2`.
Decision: **Revise**. No GPU smoke, cloud launch, model weights, validation payload, or Mini-Dev.

## What this gate establishes

The fixed Qwen3 tokenizer can map conservative character spans to token masks on the authored
SQL suite. The existing TRL 1.9.0 loss accepts token-wise advantages. A CPU analytical mock of
that loss has finite, numerically checked gradients, exact E2 fallback, and the expected change
in negative gradient mass. This is not a real GRPOTrainer run, autograd run, model update, real
rollout-coverage estimate, or permission to train.

## Token mapping algorithm

Use the tokenizer backend from `Qwen/Qwen3-4B-Base` at the accepted revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, with `tokenizers==0.22.2`. This version satisfies
the tokenizer range in the pinned Transformers 5.14.1 wheel. The 7,031,645-byte tokenizer JSON
and 9,678-byte configuration came from that revision's Hugging Face `resolve` URLs; only these
small tokenizer assets were fetched, with no weights or benchmark data.

1. Keep original completion text, original generated token IDs, original completion mask, and
   truncation flag together. Require a contiguous valid prefix ending in one base-model EOS
   (`<|endoftext|>`); padding is excluded from decoding.
2. Decode original content IDs without stripping or normalizing. Require exact original text.
3. Encode that text without added special tokens and require exact equality to the ORIGINAL IDs.
   Equal decoded strings can have different token segmentations. If IDs differ, re-encoded offsets
   cannot be attributed to the generated tokens: fall back to E2.
4. Trim only pure separator whitespace at the clause edges; otherwise a following ` FROM` BPE
   token could appear to overlap a SELECT span solely through its leading space. Then use
   original-string character offsets, not substring search. Reuse E5's coverage, overlap,
   source-text, and boundary checks. A token crossing a clause boundary with meaningful content
   on both sides makes attribution uncertain. Missing/unreliable offsets also fall back.
5. Keep EOS/padding outside localized negative credit and retain the trainer's original completion
   and truncation masks. A fully truncated E2 mask remains zero regardless of fallback credit.

Repeated text is addressed by absolute token positions. White space carried by a byte-level BPE
token may cross a boundary when it adds no meaningful text outside the target. Unicode is not
assumed to be byte-indexed. Tests include ordinary SQL, whitespace/newline/case changes, split
identifiers/numbers, repeated literal text, nesting, alias, Unicode, unavailable/overlapping offsets,
stale text, alternative generation segmentation, missing EOS, and truncation.

Nested/alias geometry tests ONLY establish that offsets can describe a span. They do not supply
semantic credit: the unchanged execution localizer still abstains on those structures.

## Minimal TRL seam and formulas

`TRLGroupRewardAdapter` continues returning E2 binary strict correctness r. TRL's accepted
`scale_rewards=none` yields, for completion i in generation group g:

    A_i = r_i - mean(r_j for j in g)

There is no second normalization or new structural bonus. Correct completions retain the positive
terminal-derived advantage globally. Constant all-zero/all-one groups remain zero.

Source inspection of the exact `trl-1.9.0` wheel shows `_compute_loss` only unsqueezes advantages
when they are 1-D and accepts existing `(B,T)` tensors. The CPU seam `attach_token_advantages`
replaces ONLY the already-centered local-row `advantages` after generation/scoring and local row
slicing, before buffering/minibatch splitting. It preserves all other input objects, including
completion IDs, masks, old log probabilities, and `num_items_in_batch`.

A future narrow subclass must create the new tensor using the original advantage tensor's device
and dtype and retain row order through buffering. It must not patch `_compute_loss` globally or
change reward aggregation. No such live trainer subclass is enabled here.

With original valid-token mask v, localized mask m, and per-completion N=sum(v), K=sum(v*m):

    E2 control:             a_it = A_i
    local unscaled:         a_it = A_i*m_it              (localized A_i < 0 only)
    mass-matched control:   a_it = A_i*m_it*N_i/K_i       (diagnostic only)
    fallback/correct rows:  a_it = A_i

Under the audited beta=0, token-importance, DAPO branch:

    rho_it = exp(log pi_theta(y_it) - log pi_old(y_it))
    L = -sum_it v_it * min(rho_it*a_it, clip(rho_it, .8, 1.28)*a_it) / D

D is the ORIGINAL valid-token count over the accumulation window (single CPU/process here).
Never replace D with the localized-token count or divide the DAPO loss by accumulation steps again.
Tests compare a full batch with accumulated microbatch sums, including padding/truncation masks.

The mock uses stable log-softmax and analytical logit derivatives, checked against central finite
differences. It tests clipping-active and inactive positive/negative/zero advantages. Non-finite
inputs or overflowing importance ratios abort rather than fabricate credit.

Direct masking changes negative update mass. In the six-token/two-target-token fixture, unscaled
negative credit and negative logit-gradient L1 both shrink to one third. N/K restores pre-clipping
credit L1 and restores logit-gradient L1 in this uniform fixture, while tripling each selected
negative-token gradient. It does NOT generally preserve actual model-parameter gradient norms:
token Jacobians, probability distributions, clipping, and shared parameters differ. Keep N/K as an
explicit diagnostic control; it is not a selected training normalization. Before a smoke, examine
real N/K and gradient-scale distributions and preregister a bounded choice without validation tuning.

## Training-only provenance and blocker

The audit reads exactly three existing, legal-to-version, SQL-free training aggregates:

- `results/m7/m7-e2-formal-summary.json`
- `results/m8/m8-e3-formal-summary.json`
- `results/m8/m8-e4-formal-summary.json`

Each records 256 historical training prompts and 1,024 rollouts, and no heldout use. These arms
reuse a controlled prompt budget; do not treat their counts as 768 independent training prompts.
The summaries contain no row-level SQL, generated IDs, or databases. They support provenance of
the earlier runs, not real E5 attribution/coverage. No validation summary is read by this audit.

This checkout has no `data/processed`, `artifacts/cloud-backup`, or
`artifacts/m7-e2-formal-001` directory. The prior STATE archive refers to the original Mac/cloud
environment, not files present in this executor. We did not open a private heldout manifest,
recursively search private archives, start cloud, or reconstruct/download BIRD.

Missing: frozen training manifest with split membership; prepared training row/prompt linkage;
selected training-only SQLite files; and existing generated training SQL plus original IDs,
EOS/truncation metadata. Real training rows and rollout SQL examined are both zero; real coverage
is **unknown/null**, not zero and not the synthetic mapping pass rate.

There is also an evidence-retention limitation: the current `run_m7_smoke_train.py` exports
adapters and aggregate diagnostics, not per-completion SQL/token IDs. Its argument builder does
not enable completion logging; the inspected TRL default is `log_completions=False`. A source-key
audit records this separately. No independent private capture has been demonstrated. Therefore
copying old aggregate logs alone cannot satisfy the generated-ID requirement, and token IDs must
not be invented by re-encoding historical SQL.

## Reproduction and next decision

Install the optional CPU requirements with
`uv pip install --python .venv-e5/bin/python -r requirements/e5-interface.txt`.
Provide only the two small fixed-revision tokenizer files at ignored
`artifacts/e5-interface/qwen3-tokenizer/`; fetch the source wheel with
`python3 -m pip download --no-deps trl==1.9.0 -d artifacts/e5-interface`.
Then run `make e5-interface-verify PYTHON=.venv-e5/bin/python` and `make verify` with the same Python.
The focused target requires actual assets before tests, so missing optional files cannot yield a
misleading passing receipt. All diagnostic data is authored SQL; no model inference takes place.

**Revise remains appropriate:** geometry and the analytical loss seam are feasible, but no real
training attribution/coverage evidence or live pinned-trainer integration has been established.
The unique next action is to obtain a minimal local training-only replay bundle from already
retained artifacts, without cloud start or new inference. If original generated token IDs were not
retained, record that fact explicitly; re-encoded IDs cannot substitute for them. Until this blocker
is resolved, do not promote to an eight-prompt GPU smoke.

## Measured receipt

2026-10-21 follow-up: BiBo's local client inspected the original Mac archive and confirmed that
historical rollout SQL and original completion IDs were not retained. The old replay-recovery
action is closed as an evidence limitation; no cloud search or validation substitute is allowed.
Real historical coverage remains unknown. `docs/E5_SMOKE_PLAN.md` proposes prospective capture
and a mandatory pre-update audit, with no launch authorization. This later external report does
not change the historical CPU receipt below.

`results/e5/e5-cpu-interface.json` records eight successful geometry cases, five exact uncertain
fallbacks, 11/11 mapped synthetic execution-localized candidates, and zero semantic-abstention
overrides. There are 27 numerical-gradient coordinates, maximum absolute error
1.41291561783774e-11, and all measured loss/gradient scales are finite. The explicit negative
logit-gradient L1 values are 0.2484058 (E2), 0.0828019 (unscaled), and 0.2484058 (N/K control).
These are synthetic-logit measurements, not model-parameter gradient norms or accuracy metrics.
The focused interface suite passes 42 tests; `make verify` passes 431 with four optional-runtime
skips. Ruff, compileall, and `uv pip check` pass. No raw training row or generated rollout was read.
