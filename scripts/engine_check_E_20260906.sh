#!/usr/bin/env bash
# Arm E (added 2026-09-06 on Astra's review of the engine check): Llama-3.1-8B-Instruct in NATIVE BF16 through
# transformers, with accelerate CPU offload of the layers that do not fit next to the KV cache in 16 GB
# (weights 16 GB; ~12 GiB resident, the rest streamed per step). Slow but exact weights, unlike arm C's
# bitsandbytes INT8. Greedy + T0.7 + T1.3 only (the cliff endpoints), both tasks, same frozen items.
# Waits for the 4-arm script to finish so the two never share the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=0
OUT=results/engine_check; LOG=$OUT/engine_check_20260906.log
log() { echo "=== $(date -Is) $*" | tee -a "$LOG"; }
until grep -q "ENGINE CHECK COMPLETE" "$LOG" 2>/dev/null; do sleep 60; done
HF=~/.local/bin/hf
pick_repo() { $HF download "$1" config.json --local-dir /tmp/hfcheck_$$ >/dev/null 2>&1 && echo "$1" || echo "$2"; rm -rf /tmp/hfcheck_$$; }
LLAMA8B=$(pick_repo meta-llama/Llama-3.1-8B-Instruct unsloth/Llama-3.1-8B-Instruct)
log "START arm E: $LLAMA8B BF16 + CPU offload"
uv run python scripts/run_transformers_axis.py --config configs/llama_lineage.yaml \
  --model-name llama-3.1-8b-instruct --hf-model "$LLAMA8B" \
  --device-map auto --max-memory "0=12GiB,cpu=26GiB" --batch-size 8 --temperatures 0.7 1.3 \
  --meta models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf.meta.json --out $OUT/llama-3.1-8b-instruct__BF16-hf-offload.jsonl >>"$LOG" 2>&1
log "EXIT=$? arm E"
echo "$OUT/llama-3.1-8b-instruct__BF16-hf-offload.jsonl: $(wc -l <$OUT/llama-3.1-8b-instruct__BF16-hf-offload.jsonl) records" | tee -a "$LOG"
