#!/usr/bin/env bash
# Direct-HTTPS Gemma-3-12B downloader (added 2026-06-04). The huggingface_hub Xet backend
# kept stalling (832MiB/50MiB hangs) on this box, so we bypass it: aria2c pulls straight from
# resolve/main (resumable -c, auto-retry), then fetch_models.py --local attaches the sidecar
# WITHOUT re-downloading. Priority order Q4_K_M FIRST so the headline cross-vendor cell can run
# tonight; Q8/Q6/Q3 follow overnight for the full quant axis. Link is ~0.77MB/s right now, so
# the .meta.json sidecar (the launcher's "ready" signal) appears only after a quant fully lands.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=bartowski/google_gemma-3-12b-it-GGUF
BASE="https://huggingface.co/$REPO/resolve/main"
ORDER=(Q4_K_M Q8_0 Q6_K Q3_K_M)
log() { echo "=== [$(date -Is)] $* ==="; }

for q in "${ORDER[@]}"; do
  fn="google_gemma-3-12b-it-$q.gguf"
  out="models/$fn"
  if [ -f "$out" ] && [ -f "$out.meta.json" ]; then log "$q already complete, skip"; continue; fi
  log "downloading $q via aria2c"
  aria2c -c -x4 -s4 -j1 --retry-wait=10 --max-tries=0 --summary-interval=30 \
    --console-log-level=warn -d models -o "$fn" "$BASE/$fn" \
    || { log "WARN aria2c failed $q (will be retried next pass)"; continue; }
  log "extracting sidecar for $q (--local, no re-download)"
  uv run --extra fetch python scripts/fetch_models.py --local "$out" --repo "$REPO" \
    || log "WARN sidecar extraction failed $q"
  log "$q DONE"
done
log "GEMMA DOWNLOAD DRIVER FINISHED"
