#!/bin/sh
# One (model,quant) cell of the full matrix per Slurm array task (Slurm cluster). The cell is
# read from a manifest file (passed as $1) at line ($SLURM_ARRAY_TASK_ID + 1); the server
# port is derived from the array index so co-located tasks on a shared 8-GPU node don't
# collide on 8080. Each cell writes its own shard JSONL (the id-keyed store is single-writer,
# so per-cell shards avoid concurrent-append races); shards merge by concatenation later.
#
# Submit (the --gres / --array are set at submit time, see scripts/slurm/submit_matrix.sh):
#   sbatch --array=0-13%4 --gres=gpu:nvidia_geforce_rtx_2080_ti:1 \
#       -o results/full_matrix/logs/cell_%A_%a.log \
#       scripts/slurm/run_cell.sh scripts/slurm/cells_2080.txt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=Teaching
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -J cell
# NB: these ICF GPU nodes are CPU-starved (12 CPUs / 8 GPUs on landonia + the A6000 node),
# so keep --cpus-per-task small (4 here; override to 2 for the contended A6000 node at submit)
# — the runner is GPU-bound, and a big CPU request blocks the node's other 7 GPUs / never
# schedules. damnii 2080 Ti nodes have 40 CPUs and are usually idle.

set -e
cd "${REPO_DIR:-$HOME/decoding-robustness}"
module load cuda/13.1.1
export LLAMA_SERVER_BIN="$PWD/vendor/llama.cpp/build/bin/llama-server"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# ICF compute nodes export http_proxy=wwwcache.inf.ed.ac.uk:3128. httpx honours it, so the
# runner's GET http://127.0.0.1:PORT/health gets routed through the web proxy (which can't see
# the node's localhost) and the server is never seen as healthy -> 300s timeout. We run fully
# offline (local GGUFs + cached datasets), so just drop the proxy and bypass loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ftp_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

MANIFEST="$1"
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "usage: sbatch run_cell.sh <manifest>; got MANIFEST='$MANIFEST'" >&2
    exit 2
fi
# Pick the Nth data line, skipping '#' comments and blank lines.
LINE=$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
MODEL=$(printf '%s\n' "$LINE" | awk '{print $1}')
QUANT=$(printf '%s\n' "$LINE" | awk '{print $2}')
PARALLEL=$(printf '%s\n' "$LINE" | awk '{print $3}')   # optional per-cell override (blank => config default)
if [ -z "$MODEL" ] || [ -z "$QUANT" ]; then
    echo "empty cell at line $((SLURM_ARRAY_TASK_ID + 1)) of $MANIFEST" >&2
    exit 2
fi
PORT=$((8080 + SLURM_ARRAY_TASK_ID))

echo "cell host=$(hostname) job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID model=$MODEL quant=$QUANT port=$PORT parallel=${PARALLEL:-default} $(date -Is)"
EXTRA=""
[ -n "$PARALLEL" ] && EXTRA="--parallel $PARALLEL"
uv run python scripts/run_pilot.py --config configs/full_matrix.yaml \
    --only-model "$MODEL" --only-quant "$QUANT" \
    --model-dir "${MODEL_DIR:-$HOME/models}" \
    --out "results/full_matrix/shards/${MODEL}__${QUANT}.jsonl" \
    --port "$PORT" $EXTRA
