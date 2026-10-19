# M1 Repository and Environment Runbook

Last updated: 2026-08-14

## Status and boundary

M1 passed on 2026-08-14. The GPU environment, both candidate revisions, model choice, topology,
storage, and shutdown behavior were validated on the reviewed Xiangongyun instance. No value in
this document is a measured GPU result unless it names an experiment ID and result file.

This M1 smoke suite is a four-example engineering gate. Its in-memory SQLite score is not the frozen
M2 evaluator and must not be reported as benchmark accuracy.

## Proposed environment lock

`configs/environment.toml` is the accepted machine-readable environment profile.

| Component | Proposed version | Compatibility basis |
|---|---:|---|
| OS | Linux x86_64 | Required by the selected NVIDIA vLLM wheel |
| Python | 3.11.x (`.python-version`: 3.11.15) | Supported by the complete stack |
| PyTorch | 2.11.0 | Exact dependency of vLLM 0.25.1 |
| CUDA runtime | 13.0 | vLLM/PyTorch wheel lane; requires R580+ NVIDIA driver |
| Transformers | 5.14.1 | Satisfies TRL and vLLM constraints |
| TRL | 1.9.0 | Supports vLLM 0.17.0 through 0.25.1 |
| PEFT | 0.19.1 | Current pinned PEFT layer |
| vLLM | 0.25.1 | Latest TRL-supported release inspected for this gate |
| Accelerate | 1.14.0 | Pinned launcher/runtime dependency |
| bitsandbytes | 0.49.2 | Pinned QLoRA dependency; CUDA 13 is supported |
| Datasets | 4.7.0 | Minimum version required by TRL 1.9.0 |

Primary references: [TRL 1.9.0 metadata](https://pypi.org/pypi/trl/1.9.0/json),
[TRL vLLM integration](https://huggingface.co/docs/trl/vllm_integration),
[vLLM 0.25.1 metadata](https://pypi.org/pypi/vllm/0.25.1/json),
[PyTorch versions](https://pytorch.org/get-started/previous-versions/), and
[NVIDIA CUDA 13 driver requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver).

If the cloud host driver is below `580.65.06`, do not install this lane. Re-resolve and test a CUDA
12 lane as a whole; never mix a CUDA 13 vLLM wheel with an older driver and call it reproducible.

## Local clean install

This validates packaging and tests, not CUDA behavior:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make m1-verify
```

## Information required before cloud deployment

Provide screenshots or copied text from these exact provider pages:

1. Instance selection: provider/region, exact GPU name, VRAM, count, availability, hourly price,
   billing increment, and whether stopped instances still incur GPU charges.
2. Image selection/details: image name or ID, Ubuntu version, Python, PyTorch, CUDA toolkit, and any
   displayed NVIDIA driver requirement/version.
3. Storage configuration: system-disk size and price; persistent-volume size, price, mount path,
   survival across stop/release, snapshot behavior, and deletion semantics.
4. Lifecycle/billing help: difference among stop, shut down, release, and delete; auto-shutdown;
   minimum charge; public-IP/network/egress charges; and how the console shows final actual cost.
5. Only if considering 2x24 GB: confirmation both GPUs are on one host and any displayed
   PCIe/NVLink topology. Runtime `nvidia-smi topo -m` is still required.

The 4B and 1.7B repositories are approximately 8.06 GB and 3.45 GB respectively according to their
Hugging Face file listings. Do not use the 100 GB system disk as the only copy of caches or results.
The review should identify a persistent path with enough headroom for both model revisions, the GPU
environment, logs, and later adapters/checkpoints.

## Paid-run gate

Before starting, record and receive approval for:

- objective: validate the environment and compare both fixed model revisions on the same 1x48 GB GPU;
- maximum wall time: 3 hours for environment setup plus both tiny runs;
- estimated maximum GPU cost: CNY 7.77 at the Day 0 displayed rate of CNY 2.59/hour, to be replaced
  by the reviewed current price;
- stop conditions: environment gate fails, either model cannot complete the suite, peak used memory
  reaches 95% of device memory, a command exceeds 30 minutes without progress, or total session time
  reaches 3 hours;
- shutdown: copy results and `pip freeze`, append actual spend, then use the provider's verified
  no-GPU-charge stop/release action.

The estimate excludes unverified storage/network charges. Actual cost comes from the provider bill,
not from the command-duration estimate emitted by the harness.

## GPU install and preflight

Run only after the image passes review:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -e '.[dev,gpu]'
make env-check-gpu
python -m pip check
python -m pip freeze --all > results/m1/gpu-pip-freeze.txt
```

The successful freeze becomes the final transitive lock evidence. A curated lock file is committed
only after this clean install succeeds on the selected Linux image.

## Candidate-model smoke commands

The wrapper refuses downloads unless `ALLOW_MODEL_DOWNLOAD=1` is set. Set `HF_HOME` to the reviewed
persistent mount before running. Execute 1.7B first because it is the cheaper failure detector, then
4B on the same instance and environment.

```bash
export HF_HOME=/reviewed/persistent/path/huggingface
export HOURLY_RATE_CNY=REVIEWED_CURRENT_RATE

ALLOW_MODEL_DOWNLOAD=1 MODEL_KEY=qwen3_1_7b_base \
EXPERIMENT_ID=m1_qwen3_1_7b_transformers_1x48_001 \
sh scripts/run_m1_smoke.sh

ALLOW_MODEL_DOWNLOAD=1 MODEL_KEY=qwen3_4b_base \
EXPERIMENT_ID=m1_qwen3_4b_transformers_1x48_001 \
sh scripts/run_m1_smoke.sh
```

The result JSON records the exact model revision, environment, per-case output, valid-SQL count,
smoke execution count, model load/generation time, completion tokens/s, PyTorch peak allocated and
reserved memory, whole-device sampled memory, and estimated command cost. The paid session's actual
cost is recorded separately in `experiments/spend.csv`.

## Decision rule

Select Qwen3-4B-Base if it completes all four cases without OOM, its measured peak leaves at least
20% of the 48 GB device free, and its throughput/cost is usable for the planned small rollout budget.
The smoke correctness count is a tie-breaker and sanity check, not a research result. Select the
1.7B fallback if 4B fails those resource gates or is materially less reliable on the fixed smoke set.

Do not test 2x24 GB during M1 unless the 1x48 GB evidence shows that colocated vLLM/training cannot
retain safe headroom or a separate-server topology has a quantified throughput hypothesis. If that
reason appears, create a new experiment ID and request a separate paid-run approval first.

## Measured M1 outcome

- Environment: `m1_env_1x48_001` passed with no validation errors.
- Fallback: `m1_qwen3_1_7b_transformers_1x48_001` completed 4/4 execution-correct cases at
  34.85 completion tok/s and 3764 MiB peak device memory, but only 1/4 outputs were SQL-only.
- Preferred: `m1_qwen3_4b_transformers_1x48_001` completed 4/4 execution-correct cases at
  28.33 completion tok/s and 8158 MiB peak device memory; all 4/4 outputs were SQL-only.
- Decision: select the fixed 4B revision and one 48 GB GPU; retain 1.7B as fallback and do not pay
  for a 2x24 GB test without a later quantified multi-process bottleneck.
- Billing: the 1.023611-hour session cost CNY 2.65115277. Provider shutdown released the GPU;
  retained system-disk usage continues at CNY 0.00003/GB/hour.

## Closure provenance and limitations

The model smokes ran from an rsynced working tree without `.git`. Their JSON files therefore record
`git_head: null` and the executed 17-file source manifest
`ed518be19adf972c20020b6d9114c2ae7212a846b411c936397bd25ef408561c`. The registry must not claim
that either `f709f3a` or `51decb1` was the exact executed tree. Deterministic reconstruction shows
that the only difference from `51decb1`'s manifest
`e341d926fb2d5ae0a5c4dcebdd8b7193d5049e3aeb5d5bd6e922c424a94a908e` is
`configs/environment.toml`: execution used `status = "proposed_pending_cloud_validation"`; closure
changed it to `status = "accepted_m1_gpu_validated"`. Restoring that one value reproduces the raw
result manifest exactly. This proves the available executed source content but does not create an
exact historical Git commit after the fact.

The closure implementation excludes generated `*.egg-info` directories from future source
manifests, so a committed source tree has a stable content hash. The historical `ed518...` value is
preserved as generated by the original 17-file algorithm, which included seven egg-info files.
The closure tree's 12-file manifest is
`db724749651c81a42befcd7dc33dde54bbc60d28acb04e73d33bb4a2732a2910`; it covers the package,
configuration, Makefile, and scripts that will be stored in the closure commit.

`strict_sql_only_count` is deterministically derived by `is_strict_sql_only`: after trimming outer
whitespace and allowing one trailing semicolon, the entire response must equal one bare query that
starts with `SELECT` or `WITH`. Markdown fences, explanatory prose, and text before or after the
query fail. Run `PYTHONPATH=src python scripts/summarize_m1_results.py results/m1/m1_qwen3_*.json` against
model-result JSON files to reproduce the counts.

`results/m1/gpu-pip-freeze.txt` was not generated or synchronized before shutdown. Recovering it
requires restarting the paid instance and is not authorized during closure. This missing transitive
freeze is recorded rather than reconstructed or invented.
