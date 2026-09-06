#!/usr/bin/env bash
# Temperature-axis sweep models chosen 2026-09-06 (four, at the author's decision): Hermes-3 on the paper's fragile
# Llama-3.1-8B base (post-training contrast), Qwen3.5-9B and Gemma-4-E4B-it (the two most downloaded 2026 small
# models; need a newer llama.cpp), OLMo-3-7B-Instruct (fully open). Q8_0 only, bartowski where he has the file.
set -uo pipefail
cd "$(dirname "$0")/.."
HF=~/.local/bin/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
log() { echo "=== [$(date -Is)] $* ==="; }
PAIRS=(
  "bartowski/Hermes-3-Llama-3.1-8B-GGUF Hermes-3-Llama-3.1-8B-Q8_0.gguf"
  "bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q8_0.gguf"
  "bartowski/google_gemma-4-E4B-it-GGUF google_gemma-4-E4B-it-Q8_0.gguf unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf"
  "bartowski/allenai_Olmo-3-7B-Instruct-GGUF allenai_Olmo-3-7B-Instruct-Q8_0.gguf"
)
for pair in "${PAIRS[@]}"; do
  set -- $pair; repo=$1; fn=$2; alt_repo=${3:-}; alt_fn=${4:-}
  if [ -f "models/$fn" ] || { [ -n "$alt_fn" ] && [ -f "models/$alt_fn" ]; }; then log "$fn present, skip"; continue; fi
  log "downloading $repo :: $fn"
  ok=0
  for attempt in 1 2 3; do
    $HF download "$repo" "$fn" --local-dir models && { ok=1; break; }
    log "attempt $attempt failed"; sleep 20
  done
  if [ $ok = 0 ] && [ -n "$alt_repo" ]; then
    log "falling back to $alt_repo :: $alt_fn"
    for attempt in 1 2 3; do $HF download "$alt_repo" "$alt_fn" --local-dir models && break; sleep 20; done
  fi
  ls -l models/$fn models/$alt_fn 2>/dev/null
done
log "SWEEP DOWNLOADS DONE"
