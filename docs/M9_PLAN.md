# M9 Serving and Benchmark Plan

Status: completed from execution commit `b47fa46`; provider invoice pending.

## Outcome

Serve the best measured model, E2, through a real OpenAI-compatible vLLM endpoint and report both
model correctness and systems performance. M9 is not a UI exercise: the deliverable is a reusable
API plus traceable TTFT, end-to-end latency, request throughput, completion-token throughput, and
GPU-memory measurements.

## Why E2

E2 is the highest-accuracy common-validation arm at 42.3490% strict execution accuracy. Its formal
rank-16 adapter manifest is `3492a3ec...bd3b7`. Serving E3 or E4 because they are more novel would
contradict the measured model-selection result. The base revision and E2 adapter are statically
loaded at server startup; runtime adapter mutation is disabled.

## Workload

The benchmark deterministically hash-selects 64 examples from the frozen 1,209-example validation
bundle using seed `verireason-m9-serving-v1`. The resulting selection identity is
`10d2280b...ff13a`. Mini-Dev remains sealed and heldout use is 0/0.

After two unmeasured warmups, the same 64 prompts run at concurrency 1, 4, and 8 with greedy,
streaming generation and a 384-token cap. Streaming timestamps produce client-observed time to first
token rather than dividing total latency by token count. The final usage event supplies completion
token counts. The concurrency-1 outputs additionally pass through the unchanged read-only M2 SQLite
evaluator for strict execution, behavior-compatible execution, valid SQL, SQL-only rate, and failure
categories.

## Serving configuration

- Backend: pinned vLLM OpenAI-compatible server.
- Model: fixed Qwen3-4B-Base revision plus static E2 LoRA, served as `verireason-e2`.
- Precision: bfloat16; maximum model length 4,096.
- Maximum active sequences: 8; LoRA rank allocation: exactly 16.
- Prefix caching: disabled so repeated workload phases do not receive a hidden cache advantage.
- GPU memory: sampled every 500 ms with `nvidia-smi`; both ready-state and peak MiB are reported.

The pinned vLLM documentation covers [static LoRA serving](https://docs.vllm.ai/en/v0.25.1/features/lora/),
[online TTFT/throughput benchmarking](https://docs.vllm.ai/en/v0.25.1/cli/bench/serve/), and
[production latency metrics](https://docs.vllm.ai/en/v0.25.1/usage/metrics/). M9 uses a
project-specific client because generic random-text benchmarks cannot also execute generated SQL
against the frozen evaluator.

## Acceptance and stops

The run passes only if the adapter and 64-example selection identities match, `/v1/models` exposes
`verireason-e2`, all 192 measured requests succeed, TTFT/latency/throughput metrics are finite and
positive, SQL-only rate is at least 95%, GPU memory is sampled, evaluator metrics are complete, and
heldout counters remain zero. Accuracy is measured but has no post-hoc promotion threshold.

The session planned 30 minutes with a one-hour/CNY 2.59 ceiling at the historical rate. Before the
completed launch, current price, instance availability, the retained E2 adapter, prepared full1209
bundle, clean execution commit, and shutdown plan were checked. This historical plan does not
authorize any additional paid execution.

## Result

Attempt 003 completed after two environment-only fail-closed attempts. All 192 measured requests
succeeded. Throughput at concurrency 1/4/8 was 1.7314/4.5062/6.5132 requests/s, TTFT p50 was
0.0940/0.0932/0.1157 seconds, and peak GPU memory was 29,246 MiB. The 64-example slice produced
46.875% strict and 50% behavior-compatible execution accuracy, 95.3125% valid SQL, and 100% SQL-only
output. Heldout use was 0/0. The tracked SQL-free evidence is
`results/m9/m9-serving-summary.json`; private request rows remain ignored.
