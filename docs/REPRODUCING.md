# Reproducing the paper

Map from every artifact in `paper/main.tex` to the script and data that produce it.
All analysis scripts are pure stdlib (or matplotlib for figures) and run on CPU in
seconds; only the generation runs need a GPU.

## Environment

```bash
uv sync
CUDA_ARCH=120 scripts/build_llamacpp.sh      # or 75 for RTX 2080 Ti; pins the commit
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
uv run python scripts/fetch_models.py --config configs/full_matrix.yaml
```

llama.cpp is pinned at commit `5aba5364` (server build 9456). All quantized checkpoints
are Bartowski imatrix GGUFs; SHA256 hashes in `DOWNLOADS.md` and `*.meta.json` sidecars.

## Generation runs (GPU)

| Study arm | Config | Runner | Output |
|---|---|---|---|
| Main matrix (7 models x 4 quants x 8 arms x T0.7/1.0/1.3) | `configs/full_matrix.yaml` | `scripts/run_pilot.py` (Slurm: `scripts/slurm/`) | `results/full_matrix/shards/` |
| Llama lineage cell (Llama-3-8B + Llama-3.2-3B, Q8, temperature-only) | `configs/llama_lineage.yaml` | `scripts/run_lineage_20260704.sh` | `results/llama_lineage/lineage.jsonl` |
| Cliff ladder (3 models, Q8, T1.3-2.0) | `configs/cliff_ladder.yaml` | `scripts/run_pilot.py` | `results/cliff_ladder/ladder.jsonl` |
| Stratified MMLU-Pro replication | `configs/mmlu_strat_check.yaml` | `scripts/run_pilot.py` | `results/mmlu_strat_check/strat.jsonl` |
| High-temperature probe (T2-3) | `configs/hightemp_smoke.yaml` | `scripts/run_pilot.py` | `results/hightemp_smoke/` |
| Thinking-mode spot-check | `configs/qwen3_thinking_full.yaml` | `scripts/run_pilot.py` | `results/thinking_full/` |
| Logit probe v1 (per-prompt top-20 summaries) | -- | `scripts/logit_probe.py` | `results/logit_probe/` |
| Logit probe v2 (raw per-position top-20 arrays) | -- | `scripts/logit_probe2.py` | `results/logit_probe_v2/` |
| Single-token perturbation probe | -- | `scripts/perturb_probe.py` | `results/perturbation/` |

Every run is resumable: records are keyed by
`model|quant|sampler|temperature|task|item|repetition` and existing keys are skipped.
`scripts/audit_shards.py` checks for duplicate ids and reproduces the headline numbers.

## Paper artifacts (CPU)

| Artifact | Producer | Notes |
|---|---|---|
| Appendix tables (`paper/tables/*.tex`) | `scripts/gen_tables.py` | reads the shards directly |
| Main-text Tables 1-2 (inline in `paper/main.tex`) | `scripts/stats_matrix.py` + `scripts/analyze_followups.py` | drop/spread CIs (Table 1), cliff ladder (Table 2) |
| Figures 1-6 (`figures/*.pdf`) | `scripts/make_figures.py` | `uv sync --extra figures` |
| Headline stats + CIs (`STATS.txt`) | `scripts/stats_matrix.py` | item-clustered bootstrap, B=2000 |
| Degeneration decomposition (`DECOMP.txt`, 4.3) | `scripts/decompose_collapse.py` | parse-path split, cap rates |
| Cliff ladder + stratified check (4.4, App. C) | `scripts/analyze_followups.py` | drops vs each model's T0.7 anchor |
| Probe summaries + Spearman (4.6, fig6) | `scripts/analyze_logit_probe.py` | entropy/flat%/margin vs drop |
| Terminated-length censoring (4.6, App. B) | `scripts/analyze_recovery.py` | absorbing-derailment evidence |
| Temperature-scaled escape hazard (4.6) | `scripts/analyze_escape.py` | beyond-top-20 mass at T=1.3 |
| Perturbation analysis (4.6) | `scripts/analyze_perturbation.py` | injected vs control continuations |
| PDF | `cd paper && tectonic main.tex` | template-agnostic preamble |

## Notes for exact reproduction

- Seeds derive deterministically from (global seed 0, model, quant, sampler, T, task,
  item, repetition); greedy is the K=1 anchor. Batched serving (<=8 slots) is
  seed-logged but not bit-exact.
- Records with `n_prompt_tokens == 0` are server-recovered degenerate streams (the
  server response carried no timings); analyses count them as run-to-cap.
- The frozen MMLU-Pro head-50 subset is single-category (business) because the split is
  sorted by category; the stratified replication covers all 14 categories
  (`subset_strategy: stratified_category`).
- Hardware in the paper's runs: RTX 2080 Ti cluster nodes (17/21 main-matrix cells,
  sm_75 build) and one RTX 5070 Ti workstation (4 VRAM-heavy cells + all follow-ups,
  sm_120 build), same llama.cpp commit.
