#!/bin/sh
# Submit the cluster's share of the full matrix: the 14 (model,quant) cells that fit an 11GB
# 2080 Ti, one cell per array task, as a single self-throttling array. %4 keeps us inside the
# 4-GPU QOS cap and a good cluster citizen (the rest pend tidily, not a queue flood).
#
# A6000 DROPPED (2026-06-02): the A6000 node is CPU-starved (12 CPU / 8 GPU) and
# contended, so the VRAM-heavy cells run on a local workstation instead. Those cells,
# Llama-3.1-8B Q8_0, Mistral-7B Q8_0, Qwen3-8B Q8_0, and the Qwen3-4B BF16 anchor (see
# cells_a6000.txt, kept for reference), are NOT submitted here.
#
# Run only AFTER the gate has passed. Resumable: re-running skips finished cells for free.
set -e
cd "${REPO_DIR:-$HOME/decoding-robustness}"
LOGDIR=results/full_matrix/logs

# 14 cells that fit an 11GB 2080 Ti (billing weight 0.2, the cheapest card).
sbatch --array=0-13%4 \
    --cpus-per-task=4 \
    --gres=gpu:nvidia_geforce_rtx_2080_ti:1 \
    -o "$LOGDIR/cell2080_%A_%a.log" \
    scripts/slurm/run_cell.sh scripts/slurm/cells_2080.txt

echo "submitted 14 2080 Ti cells (%4). watch: squeue --me"
echo "NOTE: VRAM-heavy cells (cells_a6000.txt + Llama Q8_0) run off-cluster on the workstation."
