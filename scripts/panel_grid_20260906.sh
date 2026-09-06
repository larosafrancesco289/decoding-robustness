#!/usr/bin/env bash
# Full 8-configuration Q8 grid (configs/panel_grid.yaml) for the 2026-panel models that tripped the
# prespecified flag (scripts/flag_panel.py). Usage: scripts/panel_grid_20260906.sh <model> [<model> ...]
# Waits for the temperature sweep to finish (SWEEP COMPLETE in results/sweep_20260907.log) so the two
# never share the GPU, then runs the grid model by model. Resumable by record id.
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp-v0.4.0/build/bin/llama-server
LOG=results/panel_grid_20260906.log
log() { echo "=== $(date -Is) $*" | tee -a "$LOG"; }
[ $# -ge 1 ] || { echo "usage: $0 <model> [...]"; exit 2; }
mkdir -p results/panel_grid
until grep -q "SWEEP COMPLETE" results/sweep_20260907.log 2>/dev/null; do sleep 60; done
for m in "$@"; do
  log "START grid $m"
  uv run python scripts/run_pilot.py --config configs/panel_grid.yaml --only-model "$m" --out results/panel_grid/$m.jsonl >>"$LOG" 2>&1
  log "EXIT=$? grid $m ($(wc -l < results/panel_grid/$m.jsonl) records)"
done
log "GRID COMPLETE"
