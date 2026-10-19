# VeriReason

**Verifiable post-training for small Text-to-SQL language models.** VeriReason is an end-to-end,
single-GPU study of data governance, QLoRA supervised fine-tuning, execution-reward GRPO,
controlled reward ablations, sandboxed evaluation, vLLM serving, and clean-room reproduction.

```text
BIRD -> database-disjoint manifests -> QLoRA SFT -> E2/E3/E4 GRPO
     -> sandboxed SQLite evaluation -> vLLM serving -> reproducible release
```

The fixed base is
`Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`. All reported model comparisons use
the same 1,209-example, 10-database validation exam, greedy decoding, prompt contract, and
resource-bounded evaluator. BIRD Mini-Dev remained sealed: heldout use was **0/0**.

## Results

| Arm | Training intervention | Strict execution accuracy | Valid SQL |
|---|---|---:|---:|
| E0 | Frozen base | 0.00% | — |
| E1 | One-epoch QLoRA SFT | 40.69% | 87.59% |
| E2 | E1 + terminal execution-reward GRPO | **42.35%** | **90.74%** |
| E3 | E1 + gated dense utility, sum then normalize | 41.69% | 89.83% |
| E4 | E1 + component-wise normalization then sum | 41.44% | 89.50% |

E2 improved strict accuracy over E1 by **1.65 percentage points**. The paired comparison contained
39 improvements and 19 regressions (exact two-sided McNemar `p=0.01193`). E3/E4 created many more
mixed-reward groups (239/256 and 235/256 versus E2's 102/256) but did not beat E2. This is the main
research result: denser sequence-level feedback reduced reward sparsity without improving
generalization, motivating clause/token-localized credit rather than further scalar-weight tuning.

The selected E2 adapter was also served through vLLM. Across 192/192 successful requests,
throughput scaled from 1.73 to 4.51 to 6.51 requests/s at concurrency 1/4/8; TTFT p50 was
94/93/116 ms and peak GPU memory was 29,246 MiB.

## What is included

- deterministic BIRD source receipts, safe materialization, database-level splitting, and leakage
  checks;
- a read-only SQLite evaluator with statement authorization, process isolation, and fixed wall/CPU/
  memory/result limits;
- answer-only 4-bit NF4 QLoRA SFT and parity-locked E2/E3/E4 GRPO pipelines;
- positive, negative, malformed, unsafe, equivalent-query, and reward-hacking tests;
- SQL-free aggregate evidence tied to experiment IDs in `experiments/registry.yaml`;
- vLLM endpoint benchmarking and a local SQLite demo;
- exact Python 3.11.15 CPU verification, Docker, Make targets, and CI.

The repository intentionally excludes benchmark rows, SQLite databases, private heldout manifests,
raw SQL generations, base weights, and model checkpoints. They are either reconstructed from fixed
upstream revisions or distributed separately as release assets.

## Verify the release on CPU

```bash
git clone https://github.com/Keke946/VeriReason.git
cd VeriReason
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/m10-cpu-lock.txt
python -m pip install --no-build-isolation --no-deps .
make m11-verify PYTHON=python PIP_CHECK_PYTHON=python
```

This checks code, tests, lint, dependency health, all 19 allowlisted public evidence files, release
documentation, and the exact headline metrics. It does not download a model or dataset and does not
claim to rerun expensive GPU experiments.

Container equivalent:

```bash
docker build -t verireason:v1 .
docker run --rm verireason:v1
```

## Run the selected adapter

Download and extract `verireason-e2-adapter.tar.gz` from the `v1.0.0` GitHub release. Its directory
manifest must equal
`3492a3ec870f7c6e8e672459f025fc96a9a8cba543b27b7df4fc5f6c6bfbd3b7`; the release archive
SHA-256 is `9e17b2a5acb5ecbe78c778465dd8920ea3fd643340847243ba3892fbd366bcf8`.

```bash
python -m pip install -e '.[gpu]'
verireason-demo \
  --database /path/to/database.sqlite \
  --question 'Which five products have the highest revenue?' \
  --adapter /path/to/verireason-e2-adapter \
  --allow-model-download
```

The demo reconstructs the exact schema/prompt contract and prints one SQL query. Add `--execute`
only for a database you are authorized to query; execution uses the project's read-only sandbox.
Omit `--allow-model-download` to require the fixed base revision to exist in the local cache.

## Reproduce the research pipeline

Three levels are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md):

1. CPU verification of code and published aggregates.
2. Inference with the selected E2 adapter.
3. Full data materialization, SFT, RL, common evaluation, and serving on one 48-GB GPU.

The measured GPU environment was Ubuntu 22.04.5, Python 3.11.15, PyTorch 2.11.0+cu130,
Transformers 5.14.1, TRL 1.9.0, PEFT 0.19.1, and vLLM 0.25.1 on one RTX 4090 D 48 GB.
Configuration files—not chat history—define all experiment parameters.

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Model card](docs/MODEL_CARD.md)
- [Data card](docs/DATA_CARD.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Web/CLI handoff for a new Codex task](docs/WEB_HANDOFF.md)
- [Follow-up research review](docs/RESEARCH_FOLLOWUP.md)
- [Release and interview packet](docs/RELEASE_PACKET.md)
- [Experiment registry](experiments/registry.yaml)
- [Current project state](docs/STATE.md)
- [Interview learning log](docs/INTERVIEW.md)

## Scope and limitations

- The 42.35% result is validation accuracy, not a private-test or BIRD leaderboard claim.
- Mini-Dev was deliberately not used for tuning, early stopping, model selection, or reporting.
- The 256-prompt RL study supports a controlled small-model comparison, not a scaling-law claim.
- Execution equivalence on one database instance is not proof of universal SQL equivalence.
- Actual provider invoices after M3 were not captured and are not inferred from runtime logs.

See the cards and technical report for fuller safety, licensing, and validity limitations.

## License and citation

VeriReason code is released under [Apache-2.0](LICENSE). The Qwen base is Apache-2.0; BIRD data is
CC BY-SA 4.0 and remains subject to its upstream terms. Citation metadata is in
[`CITATION.cff`](CITATION.cff).
