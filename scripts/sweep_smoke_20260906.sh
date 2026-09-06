#!/usr/bin/env bash
# Smoke test for the 2026 temperature sweep (run on Sicily after the GGUFs and the v0.4.0 build exist, GPU free):
# 2 items per task, greedy + T1.3, each of the four models through the NEW llama.cpp build, output to a scratch dir.
# Checks: server starts, template renders, no <think>/<|think|> blocks (Gemma 4, Qwen3.5), greedy parses, CUDA output sane.
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp-v0.4.0/build/bin/llama-server
OUT=results/temp_sweep_2026_smoke; mkdir -p $OUT; LOG=$OUT/smoke.log
[ -x "$LLAMA_SERVER_BIN" ] || { echo "no v0.4.0 llama-server"; exit 2; }
scripts/attach_sidecars.sh 2>&1 | tail -3
for m in hermes-3-llama-3.1-8b qwen3.5-9b gemma-4-e4b-it olmo-3-7b-instruct; do
  echo "=== $(date -Is) smoke $m" | tee -a $LOG
  uv run python scripts/run_pilot.py --config configs/temp_sweep_2026.yaml --only-model $m --limit 2 \
    --out $OUT/$m.jsonl >>$LOG 2>&1; echo "exit=$?" | tee -a $LOG
  uv run python - "$OUT/$m.jsonl" <<'PY'
import json, sys
n = think = 0
for line in open(sys.argv[1]):
    r = json.loads(line); n += 1
    o = r["raw_output"]
    if "<think>" in o or "<|think|>" in o or "<|channel>" in o: think += 1
    print(f"  {r['id'].split('|')[2]:<12} T={r['temperature']:<4} {r['task']:<9} parse={r['parse_method']:<9} correct={r['correct']!s:<5} toks={r['n_completion_tokens']:<5} :: {o[:90]!r}")
print(f"  records={n} think_blocks={think}")
PY
done
