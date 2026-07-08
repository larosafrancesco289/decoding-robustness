#!/usr/bin/env bash
# Overnight queue 2026-06-10: (1) MMLU-Pro stratified-subset check, (2) cliff-edge ladder.
# Sequential, one llama-server at a time. Gemma-3-12B needs parallel=4 / n_ctx=12288 (VRAM).
# Each cell is resumable (id-keyed JSONL); rerunning this script only fills gaps.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

LOG=results/overnight_20260610.log
run() {
  echo "=== $(date -Is) $*" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  echo "=== $(date -Is) exit=$? $1..." >>"$LOG"
}

# --- 1. stratified MMLU-Pro check (~25 min) -------------------------------------------
for M in llama-3.1-8b-instruct qwen2.5-7b-instruct; do
  run uv run python scripts/run_pilot.py --config configs/mmlu_strat_check.yaml \
    --only-model "$M" --out results/mmlu_strat_check/strat.jsonl
done
run uv run python scripts/run_pilot.py --config configs/mmlu_strat_check.yaml \
  --only-model gemma-3-12b-it --parallel 4 --n-ctx 12288 \
  --out results/mmlu_strat_check/strat.jsonl

# --- 2. cliff-edge ladder (~5-7 h) -----------------------------------------------------
for M in llama-3.1-8b-instruct qwen2.5-7b-instruct; do
  run uv run python scripts/run_pilot.py --config configs/cliff_ladder.yaml \
    --only-model "$M" --out results/cliff_ladder/ladder.jsonl
done
run uv run python scripts/run_pilot.py --config configs/cliff_ladder.yaml \
  --only-model gemma-3-12b-it --parallel 4 --n-ctx 12288 \
  --out results/cliff_ladder/ladder.jsonl

echo "=== $(date -Is) OVERNIGHT QUEUE DONE" >>"$LOG"
