#!/usr/bin/env bash
# Autonomous Qwen2.5-7B launcher with PREEMPTION (added 2026-06-04, v2).
# Priority: the headline Qwen2.5 generation-control run must not wait ~15h for the (optional,
# already-spot-checked) thinking-ON full cell. So: run thinking-full during the download window,
# then the moment the download+templates are ready, PREEMPT thinking-full (it's resumable),
# run the 4 Qwen2.5 cells, then RESUME thinking-full overnight (keeps the GPU busy).
# Run in background. Kill patterns target the run_pilot/llama-server, never this script
# (this bash process cmdline is just "bash scripts/run_qwen25_matrix.sh").
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

REPO=bartowski/Qwen2.5-7B-Instruct-GGUF
QUANTS=(Q8_0 Q6_K Q4_K_M Q3_K_M)
THINK_CFG="configs/qwen3_thinking_full.yaml"
THINK_OUT="results/thinking_full/qwen3-8b-thinkON__Q8_0.jsonl"
log() { echo "=== [$(date -Is)] $* ==="; }

free_gpu() {
  while pgrep -x llama-server >/dev/null 2>&1; do sleep 5; done
}

# 1. Wait for all GGUFs to exist and stop growing (download finished).
log "waiting for Qwen2.5 GGUFs to finish downloading"
while :; do
  missing=0
  for q in "${QUANTS[@]}"; do [ -f "models/Qwen2.5-7B-Instruct-$q.gguf" ] || missing=1; done
  if [ "$missing" -eq 0 ] && ! pgrep -f "Qwen2.5-7B-Instruct.*resolve/main" >/dev/null 2>&1; then
    s1=$(stat -c '%s' models/Qwen2.5-7B-Instruct-*.gguf 2>/dev/null | tr '\n' ' '); sleep 25
    s2=$(stat -c '%s' models/Qwen2.5-7B-Instruct-*.gguf 2>/dev/null | tr '\n' ' ')
    [ "$s1" = "$s2" ] && break
  fi
  sleep 25
done
log "GGUFs stable"; ls -la models/Qwen2.5-7B-Instruct-*.gguf | awk '{print $5, $9}'

# 2. Extract chat-template sidecars (CPU; safe to do while thinking-full still runs).
log "extracting chat-template sidecars"
for q in "${QUANTS[@]}"; do
  f="models/Qwen2.5-7B-Instruct-$q.gguf"
  [ -f "$f.meta.json" ] || uv run --extra fetch python scripts/fetch_models.py --local "$f" --repo "$REPO" \
    || log "WARN template extraction failed for $q"
done

# 3. PREEMPT thinking-full to free the GPU for the headline run. Match by config path (this
#    script's own cmdline does NOT contain it, so we won't kill ourselves), then ensure server gone.
log "preempting thinking-full (resumable) to free GPU for Qwen2.5"
pkill -f "run_pilot.py --config $THINK_CFG" 2>/dev/null || true
sleep 5
pkill -x llama-server 2>/dev/null || true
free_gpu
log "GPU free; starting Qwen2.5-7B matrix (headline)"

# 4. Run the 4 Qwen2.5 cells (sequential, resumable; matrix shard convention).
for q in "${QUANTS[@]}"; do
  log "START qwen2.5-7b-instruct $q"
  uv run python scripts/run_pilot.py --config configs/full_matrix.yaml \
    --only-model qwen2.5-7b-instruct --only-quant "$q" --model-dir models \
    --out "results/full_matrix/shards/qwen2.5-7b-instruct__${q}.jsonl" \
    || log "WARN run_pilot failed for $q"
  log "DONE qwen2.5-7b-instruct $q"
done
log "ALL QWEN2.5 CELLS DONE"

# 5. Resume the thinking-ON full cell overnight (optional appendix; keeps GPU busy).
free_gpu
log "resuming thinking-ON full cell (overnight, resumable)"
uv run python scripts/run_pilot.py --config "$THINK_CFG" --enable-thinking --model-dir models \
  --out "$THINK_OUT" >> results/thinking_full/thinking_full_run.log 2>&1 || log "WARN thinking resume failed"
log "THINKING-FULL RESUME COMPLETE; all queued work done"
