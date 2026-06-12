#!/usr/bin/env bash
# Run the 4 VRAM-heavy "off-cluster" cells of the full matrix on the local GPU box.
# Runs the off-cluster (VRAM-heavy) cells of the full matrix. Sequential, resumable, one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

run() {
  local model="$1" quant="$2"
  echo "=== $(date -Is) START $model $quant ==="
  uv run python scripts/run_pilot.py --config configs/full_matrix.yaml \
    --only-model "$model" --only-quant "$quant" --model-dir models \
    --out "results/full_matrix/shards/${model}__${quant}.jsonl"
  echo "=== $(date -Is) DONE  $model $quant ==="
}

run llama-3.1-8b-instruct    Q8_0
run mistral-7b-instruct-v0.3 Q8_0
run qwen3-8b                 Q8_0
run qwen3-4b                 BF16
echo "=== ALL 4 OFF-CLUSTER CELLS DONE $(date -Is) ==="
