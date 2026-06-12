#!/usr/bin/env bash
# Wait for the Gemma cliff-ladder resume (run_pilot.py) to exit, then run logit probe v2
# (raw per-position top-20 arrays, all 7 models, Q8). 2026-06-10 overnight chain.
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server

echo "=== $(date -Is) watcher: waiting for run_pilot.py to exit"
while pgrep -f "scripts/run_pilot.py" >/dev/null 2>&1; do
  sleep 120
done
echo "=== $(date -Is) watcher: gemma run gone; waiting 30s for the server to wind down"
sleep 30
if pgrep -x llama-server >/dev/null 2>&1; then
  echo "=== $(date -Is) watcher: llama-server still up, waiting up to 5 min"
  for _ in $(seq 30); do
    pgrep -x llama-server >/dev/null 2>&1 || break
    sleep 10
  done
fi
echo "=== $(date -Is) watcher: starting logit probe v2"
uv run python scripts/logit_probe2.py
echo "=== $(date -Is) watcher: probe v2 done (exit $?)"
