#!/usr/bin/env bash
# Engine/precision check, Sicily, 2026-09-06 morning run. Ordered by how much each arm can
# change the paper: (A) Llama-3.2-3B BF16 on transformers is the arm that could kill the
# framing; (B) the same model as f16 GGUF on llama.cpp isolates engine from precision;
# (C) Llama-3.1-8B int8 on transformers repeats the headline model off llama.cpp;
# (D) Qwen3-4B BF16 on transformers is the robust control (its BF16 llama.cpp cells are
# already in results/full_matrix/shards/qwen3-4b__BF16.jsonl).
# Every runner is resumable by record id, so re-running this script only fills gaps.
set -uo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
export HF_HUB_ENABLE_HF_TRANSFER=0
OUT=results/engine_check; mkdir -p "$OUT"
LOG=$OUT/engine_check_20260906.log
log() { echo "=== $(date -Is) $*" | tee -a "$LOG"; }
run() { log "START $*"; "$@" >>"$LOG" 2>&1; local rc=$?; log "EXIT=$rc $1 $2 $3"; return $rc; }
HF=~/.local/bin/hf

# Official Meta repos if the licence was granted, unsloth's byte-identical mirrors otherwise.
pick_repo() { $HF download "$1" config.json --local-dir /tmp/hfcheck_$$ >/dev/null 2>&1 && echo "$1" || echo "$2"; rm -rf /tmp/hfcheck_$$; }
LLAMA3B=$(pick_repo meta-llama/Llama-3.2-3B-Instruct unsloth/Llama-3.2-3B-Instruct)
LLAMA8B=$(pick_repo meta-llama/Llama-3.1-8B-Instruct unsloth/Llama-3.1-8B-Instruct)
log "HF repos: 3B=$LLAMA3B 8B=$LLAMA8B"

# Preconditions
for m in Llama-3.2-3B-Instruct-Q8_0 Meta-Llama-3.1-8B-Instruct-Q8_0 Qwen_Qwen3-4B-Q8_0; do
  [ -f models/$m.gguf.meta.json ] || { log "MISSING sidecar models/$m.gguf.meta.json (run scripts/attach_sidecars.sh)"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tee -a "$LOG"

# A. Llama-3.2-3B BF16, transformers
run uv run python scripts/run_transformers_axis.py --config configs/llama_lineage.yaml \
  --model-name llama-3.2-3b-instruct --hf-model "$LLAMA3B" \
  --meta models/Llama-3.2-3B-Instruct-Q8_0.gguf.meta.json --out $OUT/llama-3.2-3b-instruct__BF16-hf.jsonl

# B. Llama-3.2-3B f16 GGUF, llama.cpp (same engine as the grid, precision changed)
if [ -f models/Llama-3.2-3B-Instruct-f16.gguf.meta.json ] && [ -x "$LLAMA_SERVER_BIN" ]; then
  run uv run python scripts/run_pilot.py --config configs/engine_check.yaml \
    --only-model llama-3.2-3b-instruct --out $OUT/llama-3.2-3b-instruct__F16-gguf.jsonl
else
  log "SKIP B: f16 GGUF sidecar or llama-server missing"
fi

# C. Llama-3.1-8B int8 (bitsandbytes), transformers
run uv run python scripts/run_transformers_axis.py --config configs/llama_lineage.yaml \
  --model-name llama-3.1-8b-instruct --hf-model "$LLAMA8B" --load-in-8bit --batch-size 4 \
  --meta models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf.meta.json --out $OUT/llama-3.1-8b-instruct__INT8-hf.jsonl

# D. Qwen3-4B BF16, transformers (robust control)
run uv run python scripts/run_transformers_axis.py --config configs/llama_lineage.yaml \
  --model-name qwen3-4b --hf-model Qwen/Qwen3-4B \
  --meta models/Qwen_Qwen3-4B-Q8_0.gguf.meta.json --out $OUT/qwen3-4b__BF16-hf.jsonl

log "ENGINE CHECK COMPLETE"
for f in $OUT/*.jsonl; do echo "$f: $(wc -l <"$f") records" | tee -a "$LOG"; done
