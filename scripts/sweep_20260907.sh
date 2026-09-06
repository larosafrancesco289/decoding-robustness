#!/usr/bin/env bash
# 2026 panel temperature sweep + commit bridge, llama.cpp v0.4.0, Sicily. Run after scripts/sweep_smoke_20260906.sh
# passed and the PLACEHOLDER commits in both configs were replaced. Resumable by record id. Order: bridge first
# (cheapest, and it validates the new build against known cells), then Hermes-3 (old-arch model, most comparable),
# then the three 2026 models.
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp-v0.4.0/build/bin/llama-server
LOG=results/sweep_20260907.log
log() { echo "=== $(date -Is) $*" | tee -a "$LOG"; }
grep -q PLACEHOLDER configs/temp_sweep_2026.yaml configs/bridge_v040.yaml && { log "PLACEHOLDER commit still in a config"; exit 2; }
[ -x "$LLAMA_SERVER_BIN" ] || { log "no v0.4.0 llama-server"; exit 2; }
mkdir -p results/bridge_v040 results/temp_sweep_2026
for m in llama-3.2-3b-instruct qwen3-4b; do
  log "START bridge $m"
  uv run python scripts/run_pilot.py --config configs/bridge_v040.yaml --only-model $m --out results/bridge_v040/$m.jsonl >>"$LOG" 2>&1
  log "EXIT=$? bridge $m ($(wc -l < results/bridge_v040/$m.jsonl) records)"
done
for m in hermes-3-llama-3.1-8b qwen3.5-9b gemma-4-e4b-it olmo-3-7b-instruct; do
  log "START sweep $m"
  uv run python scripts/run_pilot.py --config configs/temp_sweep_2026.yaml --only-model $m --out results/temp_sweep_2026/$m.jsonl >>"$LOG" 2>&1
  log "EXIT=$? sweep $m ($(wc -l < results/temp_sweep_2026/$m.jsonl) records)"
done
log "SWEEP COMPLETE"
