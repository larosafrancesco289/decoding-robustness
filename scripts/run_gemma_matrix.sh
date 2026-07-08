#!/usr/bin/env bash
# Autonomous Gemma-3-12B launcher, PER-QUANT (rev 2026-06-04, slow-link aware). The download
# (scripts/dl_gemma.sh) lands quants one at a time over a ~0.77MB/s link, Q4_K_M first. So we
# run each Gemma cell the moment ITS gguf+sidecar exist AND the Qwen2.5 matrix is done; the
# headline cross-vendor result (Q4_K_M) comes out tonight; Q8/Q6/Q3 follow as they arrive.
# Gemma is 12B -> parallel=4 / n_ctx=12288 (VRAM headroom on the 16GB 5070 Ti). thinking-OFF.
# On first GPU grab we preempt the optional thinking-full cell + retire the old Qwen2.5
# orchestrator (so it can't double-launch thinking-full). After all 4, resume thinking-full.
# pkill patterns never match this script's own cmdline ("bash scripts/run_gemma_matrix.sh").
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

ORDER=(Q4_K_M Q8_0 Q6_K Q3_K_M)
GG_PREFIX="models/google_gemma-3-12b-it"
QWEN25_SHARDS=(
  results/full_matrix/shards/qwen2.5-7b-instruct__Q8_0.jsonl
  results/full_matrix/shards/qwen2.5-7b-instruct__Q6_K.jsonl
  results/full_matrix/shards/qwen2.5-7b-instruct__Q4_K_M.jsonl
  results/full_matrix/shards/qwen2.5-7b-instruct__Q3_K_M.jsonl
)
THINK_CFG="configs/qwen3_thinking_full.yaml"
THINK_OUT="results/thinking_full/qwen3-8b-thinkON__Q8_0.jsonl"
CELL_DONE=6400
PREEMPTED=0
log() { echo "=== [$(date -Is)] $* ==="; }
free_gpu() { while pgrep -x llama-server >/dev/null 2>&1; do sleep 5; done; }

qwen25_done() {
  for s in "${QWEN25_SHARDS[@]}"; do
    n=$( [ -f "$s" ] && wc -l < "$s" || echo 0 )
    [ "$n" -ge "$CELL_DONE" ] || return 1
  done
  return 0
}

preempt_once() {
  [ "$PREEMPTED" -eq 1 ] && return 0
  log "preempting thinking-full + retiring old Qwen2.5 orchestrator (free GPU for Gemma)"
  pkill -f "run_pilot.py --config $THINK_CFG" 2>/dev/null || true
  pkill -f "bash scripts/run_qwen25_matrix.sh" 2>/dev/null || true
  sleep 5
  pkill -x llama-server 2>/dev/null || true
  free_gpu
  PREEMPTED=1
}

log "per-quant launcher up; order: ${ORDER[*]}"
for q in "${ORDER[@]}"; do
  gg="$GG_PREFIX-$q.gguf"
  out="results/full_matrix/shards/gemma-3-12b-it__${q}.jsonl"
  # wait until this quant is downloaded (gguf+sidecar) AND Qwen2.5 matrix is complete
  while ! { [ -f "$gg" ] && [ -f "$gg.meta.json" ] && qwen25_done; }; do sleep 30; done
  preempt_once
  free_gpu
  log "START gemma-3-12b-it $q (parallel=4, n_ctx=12288)"
  uv run python scripts/run_pilot.py --config configs/full_matrix.yaml \
    --only-model gemma-3-12b-it --only-quant "$q" --model-dir models \
    --parallel 4 --n-ctx 12288 --out "$out" \
    || log "WARN run_pilot failed for $q"
  log "DONE gemma-3-12b-it $q ($(wc -l < "$out" 2>/dev/null || echo 0) records)"
done
log "ALL GEMMA CELLS DONE"

# Resume the thinking-ON full cell (sole owner now; optional appendix; keeps GPU busy).
free_gpu
log "resuming thinking-ON full cell (overnight, resumable)"
uv run python scripts/run_pilot.py --config "$THINK_CFG" --enable-thinking --model-dir models \
  --out "$THINK_OUT" >> results/thinking_full/thinking_full_run.log 2>&1 \
  || log "WARN thinking resume failed"
log "THINKING-FULL RESUME COMPLETE; all queued work done"
