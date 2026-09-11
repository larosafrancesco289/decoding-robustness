# Reproducing the paper

Map from every exhibit in `paper/main.tex` to the script and records that produce it.
The analysis scripts run on CPU from the tracked records in `results/` in seconds to
minutes; only the generation runs need a GPU.

## Environment

```bash
uv sync --extra figures
CUDA_ARCH=120 scripts/build_llamacpp.sh      # pins the main-grid commit; 75 for an RTX 2080 Ti
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
uv run python scripts/fetch_models.py --config configs/full_matrix.yaml
```

Two llama.cpp builds were used. The main grid, the two further Llama releases, the sweep to
2.0, the stratified check, and the thinking-mode check use commit `5aba5364` (server build
9456). The four 2026 models, their sampler grids, and the bridge run use release v0.4.0
(commit `5266f24d`), which Qwen3.5 and Gemma 4 require. Every quantized model is a Bartowski
imatrix GGUF; SHA-256 hashes are in `DOWNLOADS.md` and in the `*.meta.json` sidecars written
by `scripts/fetch_models.py`.

## Generation runs (GPU)

| Run (paper name) | Config | Launcher | Records |
|---|---|---|---|
| Main grid: 7 models x 4 quantization levels x 8 configurations x T 0.7/1.0/1.3 | `configs/full_matrix.yaml` | `scripts/run_pilot.py` (Slurm: `scripts/slurm/`) | `results/full_matrix/shards/` |
| Two further Llama releases (Llama-3-8B, Llama-3.2-3B), Q8, temperature test | `configs/llama_lineage.yaml` | `scripts/run_lineage_20260704.sh` | `results/llama_lineage/` |
| Temperature test of the four 2026 models (Hermes-3, Qwen3.5-9B, Gemma-4-E4B, OLMo-3), Q8, v0.4.0 | `configs/temp_sweep_2026.yaml` | `scripts/fetch_sweep_20260906.sh`, `scripts/sweep_20260907.sh` | `results/temp_sweep_2026/` |
| Bridge: Llama-3.2-3B and Qwen3-4B temperature test on v0.4.0 | `configs/bridge_v040.yaml` | `scripts/sweep_20260907.sh` | `results/bridge_v040/` |
| Sampler grids of the three flagged 2026 models, Q8 | `configs/panel_grid.yaml` | `scripts/flag_panel.py` (the flag), `scripts/panel_grid_20260906.sh` | `results/panel_grid/` |
| Temperature sweep to 2.0 (Llama-3.1-8B, Qwen2.5-7B, Gemma-3-12B), Q8, every sampler | `configs/cliff_ladder.yaml` | `scripts/run_pilot.py` | `results/cliff_ladder/` |
| Stratified MMLU-Pro check (three models) | `configs/mmlu_strat_check.yaml` | `scripts/run_pilot.py` | `results/mmlu_strat_check/` |
| Thinking-mode check (Qwen3-8B) | `configs/qwen3_thinking_full.yaml` | `scripts/run_pilot.py` | `results/thinking_full/` |
| Engine check: HF Transformers BF16 / INT8 and llama.cpp F16 | `configs/engine_check.yaml` | `scripts/run_transformers_axis.py`, `scripts/engine_check_20260906.sh`, `scripts/engine_check_E_20260906.sh` | `results/engine_check/` |

Every run is resumable: records are keyed by
`model|quant|sampler|temperature|task|item|repetition` and existing keys are skipped.
`scripts/audit_shards.py` checks the main grid for duplicate ids, cell completeness, and
parameter sanity.

## Paper exhibits (CPU)

First build the flat table, then the figures and tables:

```bash
uv run python scripts/paper_data.py      # results/paper_records.parquet
uv run python scripts/paper_figures.py   # figures/*.pdf (+ .png)
uv run python scripts/paper_tables.py    # paper/tables/*.tex, results/paper_numbers.json
```

`results/paper_numbers.json` holds every number the prose cites.

| Exhibit | Producer | Notes |
|---|---|---|
| Table 1 (design) | hand-written `paper/tables/tab_design.tex` | totals from `paper_numbers.json` |
| Fig. 1 (plain-temperature drops, Q8) | `paper_figures.py fig1` | `figures/fig1_drops.pdf` |
| Fig. 2 (outcome composition, MMLU-Pro) | `paper_figures.py fig2` | `figures/fig2_outcomes.pdf`; GSM8K version `figA_outcomes_gsm8k` in Appendix B |
| Table 2 (paired sampler gains and the joint bound) | `scripts/sampler_bound_all.py` then `paper_tables.py` | `results/sampler_bound_all.json`; hand-edited labels in `tab_bound.tex` |
| Fig. 3 (recovery at T=1.3) | `paper_figures.py fig3` | `figures/fig3_recovery.pdf` |
| Fig. 4 (sweep to 2.0) | `paper_figures.py fig4` | `figures/fig4_ladder.pdf` |
| Table 3 (engine check) | `scripts/analyze_engine_check.py`, `paper_tables.py` | `tab_engine_full.tex` |
| Blinded outcome audit (Appendix B) | `scripts/outcome_audit_sample.py` | `results/outcome_audit/` (blind sample, key, labels, summary) |
| Markdown-tolerant regrade (Appendix B) | `scripts/regrade_markdown_check.py` | writes nothing; prints the changes |
| Table 4 (parse outcomes), Table 5 (survivors vs matched greedy) | `paper_tables.py`; `scripts/survivor_baseline.py` for the matched baseline | `tab_decomp.tex`, `tab_survivor.tex` |
| Tables 6-8 (temperature results in full, quantization, stratified check) | `paper_tables.py` | `tab_temp_full.tex`, `tab_quant_drops.tex`, `tab_strat.tex` |
| Table 9 (sweep) | `paper_tables.py` | `tab_ladder.tex` |
| Tables 10-13 (sampler grids and paired contrasts) | `paper_tables.py` | `tab_grid_*.tex`, `tab_contrasts_*.tex` |
| Fig. 8 (gain vs loss, Appendix E) | `paper_figures.py figA` | `figures/figA_gain_loss.pdf` |
| GEE check of the bound (Appendix E) | `scripts/sampler_gee.py` | `results/sampler_gee.json` |

Earlier analysis scripts (`stats_matrix.py`, `decompose_collapse.py`, `analyze_followups.py`,
`gen_tables.py`, `make_figures.py`, the logit-probe and perturbation scripts) belong to a
previous version of the paper and are kept for the record; their outputs from June 2026 are
under `results/legacy_july2026/`.

## Notes for exact reproduction

- Seeds derive deterministically from the global seed 0 and the record identity; greedy is
  the K=1 anchor. The server serves up to eight requests at once, so a run is seed-logged but
  not bit-exact.
- Each request carries exactly one truncation parameter and pins the sampler order through
  the server's `samplers` array. No repetition penalty or other logit processor is sent.
- Records with a completion token count of zero are server-recovered degenerate streams (on
  v0.4.0, an HTTP error without text); every analysis counts them as capped. Appendix A of
  the paper gives their counts per run.
- The client renders each model's official chat template and sends the text, with the
  server's template parser and BOS insertion off; the rendered-prompt SHA-256 is stored in
  every record. On v0.4.0 the Gemma 4 prompt is rendered without a BOS token because the
  server inserts its own.
