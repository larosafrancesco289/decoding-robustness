#!/usr/bin/env bash
# Attach the .meta.json sidecar (SHA256 + embedded chat template) to every GGUF in models/
# that lacks one. Idempotent. Needs `uv sync --extra fetch`. The repo id is only recorded,
# so it is resolved from the study configs' hf_repo entries by filename.
set -uo pipefail
cd "$(dirname "$0")/.."
for f in models/*.gguf; do
  [ -e "$f" ] || continue
  [ -f "$f.meta.json" ] && continue
  [ -e "models/.cache/huggingface/download/$(basename "$f").incomplete" ] && { echo "skip (incomplete): $f"; continue; }
  fn=$(basename "$f")
  # nearest hf_repo ABOVE the filename's quant_files entry (a plain -B window can catch the previous checkpoint's repo)
  repo=$(awk -v fn="$fn" '/hf_repo:/{repo=$2} index($0, fn) && repo {print repo; exit}' configs/*.yaml)
  echo "=== $(date -Is) sidecar for $fn (repo ${repo:-unknown})"
  uv run python scripts/fetch_models.py --local "$f" --repo "${repo:-unknown}" --file "$fn" || echo "FAILED: $fn"
done
ls -1 models/*.meta.json 2>/dev/null | wc -l | xargs echo "sidecars present:"
