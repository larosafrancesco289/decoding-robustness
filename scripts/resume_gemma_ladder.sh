#!/usr/bin/env bash
# Resume the Gemma-3-12B cliff-ladder cell (MMLU-Pro remainder, ~1570 records).
# Safe to re-run any number of times: the runner skips ids already on disk.
# parallel=6 (was 4): same total KV budget (n_ctx 12288 -> 2048/slot, proven sufficient
# for MMLU-Pro on the other models), ~1.3-1.5x faster. Run me overnight:
#   nohup scripts/resume_gemma_ladder.sh >> results/overnight_20260610.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
echo "=== $(date -Is) resume gemma ladder (parallel 6)"
uv run python scripts/run_pilot.py --config configs/cliff_ladder.yaml \
  --only-model gemma-3-12b-it --parallel 6 --n-ctx 12288 \
  --out results/cliff_ladder/ladder.jsonl
echo "=== $(date -Is) gemma ladder cell done (exit $?)"
