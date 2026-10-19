#!/bin/sh
set -eu

: "${MODEL_KEY:?Set MODEL_KEY to qwen3_4b_base or qwen3_1_7b_base}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to the registered experiment ID}"
: "${HF_HOME:?Set HF_HOME to the reviewed persistent storage path}"
: "${HOURLY_RATE_CNY:?Set HOURLY_RATE_CNY from the reviewed provider page}"

if [ "${ALLOW_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo "Model download is disabled. Set ALLOW_MODEL_DOWNLOAD=1 only after explicit user approval." >&2
  exit 2
fi

results_dir="${RESULTS_DIR:-results/m1}"
output="${results_dir}/${EXPERIMENT_ID}.json"
PYTHONPATH=src python -m verireason.smoke \
  --config configs/m1_smoke.toml \
  --environment-config configs/environment.toml \
  --model-key "$MODEL_KEY" \
  --experiment-id "$EXPERIMENT_ID" \
  --hourly-rate-cny "$HOURLY_RATE_CNY" \
  --output "$output" \
  --allow-download
