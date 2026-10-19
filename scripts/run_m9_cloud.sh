#!/bin/sh
set -eu
umask 077

: "${M9_APPROVED_COMMIT:?set the control-approved full SHA}"
: "${M9_PREPARED_FULL1209:?set prepared-full-validation.json}"
: "${M9_E2_ADAPTER_DIR:?set the retained formal E2 adapter directory}"
: "${M9_OUTPUT_DIR:?set a fresh persistent output directory outside Git}"
: "${HF_HOME:?set the retained Hugging Face cache root}"
: "${M9_CONFIG:=configs/m9_serving.toml}"

observed_commit=$(git rev-parse HEAD)
if [ "$observed_commit" != "$M9_APPROVED_COMMIT" ] || [ -n "$(git status --porcelain)" ]; then
  echo "M9 source commit or working tree is not approved" >&2
  exit 1
fi
if [ -e "$M9_OUTPUT_DIR" ]; then
  echo "M9 output directory already exists" >&2
  exit 1
fi
available=$(df -B1 "$(dirname "$M9_OUTPUT_DIR")" | awk 'NR==2 {print $4}')
if [ "$available" -lt 5368709120 ]; then
  echo "M9 output filesystem has less than 5 GiB free" >&2
  exit 1
fi

mkdir -p "$M9_OUTPUT_DIR"
PYTHONPATH=src python scripts/preflight_m9_serving.py \
  --config "$M9_CONFIG" \
  --prepared "$M9_PREPARED_FULL1209" \
  --adapter-dir "$M9_E2_ADAPTER_DIR" \
  --output "$M9_OUTPUT_DIR/preflight.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
server_pid=""
sampler_pid=""
cleanup() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  if [ -n "$sampler_pid" ] && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -lms 500 > "$M9_OUTPUT_DIR/gpu-samples.csv" &
sampler_pid=$!
server_started=$(date +%s)
vllm serve Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 8 \
  --no-enable-prefix-caching \
  --enable-lora \
  --max-loras 1 \
  --max-lora-rank 16 \
  --lora-modules "verireason-e2=$M9_E2_ADAPTER_DIR" \
  > "$M9_OUTPUT_DIR/server.log" 2>&1 &
server_pid=$!

ready=0
for _attempt in $(seq 1 300); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "M9 vLLM server exited before readiness" >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo "M9 vLLM server did not become ready within 10 minutes" >&2
  exit 1
fi
startup_seconds=$(($(date +%s) - server_started))
ready_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

timeout --signal=TERM 2400s env PYTHONPATH=src python scripts/benchmark_m9_serving.py \
  --config "$M9_CONFIG" \
  --prepared "$M9_PREPARED_FULL1209" \
  --server-startup-seconds "$startup_seconds" \
  --ready-gpu-memory-mib "$ready_memory" \
  --output "$M9_OUTPUT_DIR/m9-serving-raw-private.json"

cleanup
server_pid=""
sampler_pid=""
PYTHONPATH=src python scripts/summarize_m9_serving.py \
  --config "$M9_CONFIG" \
  --raw "$M9_OUTPUT_DIR/m9-serving-raw-private.json" \
  --gpu-samples "$M9_OUTPUT_DIR/gpu-samples.csv" \
  --output "$M9_OUTPUT_DIR/m9-serving-summary.json"
