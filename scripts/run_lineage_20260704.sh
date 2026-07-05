#!/usr/bin/env bash
# Llama lineage cell (2026-07-04): Llama-3-8B + Llama-3.2-3B, Q8, greedy + pure
# temperature, T0.7/1.0/1.3, both frozen N=50 tasks. Reviewer-proofing for the n=1
# fragile-model caveat; see configs/llama_lineage.yaml. Resumable (id-keyed JSONL);
# rerunning only fills gaps. Same launcher pattern as run_overnight_20260610.sh.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

LOG=results/llama_lineage_20260704.log
mkdir -p results/llama_lineage
run() {
  echo "=== $(date -Is) $*" >>"$LOG"
  "$@" >>"$LOG" 2>&1
  echo "=== $(date -Is) exit=$? $1..." >>"$LOG"
}

for M in llama-3-8b-instruct llama-3.2-3b-instruct; do
  run uv run python scripts/run_pilot.py --config configs/llama_lineage.yaml \
    --only-model "$M" --out results/llama_lineage/lineage.jsonl
done
echo "=== $(date -Is) lineage cell complete" >>"$LOG"
