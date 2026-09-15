# decoding-robustness

Code, configurations, and every generation record for the paper
**Temperature Fragility and the Conditional Benefits of Truncation Sampling**
(Francesco La Rosa, 2026; [arXiv:2609.15476](https://arxiv.org/abs/2609.15476); source in [`paper/`](paper/)).

## What the study does

Thirteen open-weight instruction-tuned models (1.7B to 12B) answer 50 GSM8K and 50 MMLU-Pro
items at temperatures 0.7, 1.0, and 1.3 in one controlled llama.cpp pipeline, with the same
prompts, parsers, and token budgets for every model. Ten of the models also run eight decoding
configurations: greedy, plain temperature sampling, top-p, top-k, min-p, top-nσ, and two
order variants that apply truncation before temperature. Seven of the models run at four
quantization levels. Three models run a temperature sweep to 2.0 under every sampler.
The study totals 232,700 graded generations, all of which are in [`results/`](results/).

## What it finds

- **Temperature sensitivity differs by model.** Between 0.7 and 1.3, six of the thirteen
  models lose 17 to 38 accuracy points on MMLU-Pro under plain sampling; the other seven lose
  at most 10. The collapse replicates when a fragile model is served by HF Transformers in
  BF16, and a different post-training of the same base model (Hermes-3 vs Llama-3.1-8B)
  halves it.
- **The lost accuracy is degenerate output.** On the fragile models the share of generations
  that run to the token limit or never state an answer rises by 26 to 78 points, while the
  share that states a wrong answer does not rise.
- **No sampler gains where accuracy holds.** On the robust models, no truncation sampler
  improves on plain temperature sampling at 0.7 or 1.0, and a jointly resampled upper bound
  states how large a gain the data leave possible.
- **Truncation recovers accuracy on the collapse.** On the fragile models every truncation
  sampler improves accuracy at 1.3, and in the sweep truncation delays or prevents the collapse
  up to 2.0. The gains reported for truncation samplers are recoveries from a collapse that
  only some models suffer, at temperatures above the ones systems use.

## Reproduce

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) maps every figure and table in the paper to the
script and records that produce it. The analysis runs on CPU from the tracked records:

```bash
uv sync --extra figures
uv run python scripts/paper_data.py       # results/paper_records.parquet (one row per generation)
uv run python scripts/paper_figures.py    # figures/*.pdf
uv run python scripts/paper_tables.py     # paper/tables/*.tex + results/paper_numbers.json
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Regenerating the records needs a GPU with 16 GB, a pinned llama.cpp build
(`scripts/build_llamacpp.sh`; commit `5aba5364` for the main grid, release v0.4.0 for the
2026 models), and the model files listed with SHA-256 hashes in [`DOWNLOADS.md`](DOWNLOADS.md).
Every generation is one resumable JSONL record keyed by
`model|quant|sampler|temperature|task|item|repetition`, with the rendered-prompt hash, seed,
parse outcome, and token counts logged.

## Layout

```
src/decoding_robustness/   the package
  config/                  Pydantic experiment-config schema and YAML loader
  inference/               llama-server client, chat-template rendering, sampler parameters
  tasks/                   dataset loaders, answer parsers, graders
  runner/                  factorial matrix expansion and the resumable generation loop
  analysis/                bootstrap statistics and table helpers
configs/                   YAML experiment configs, one per run
scripts/                   run scripts (GPU) and analysis scripts (CPU); see docs/REPRODUCING.md
results/                   JSONL records per run, the flat parquet table, audit labels
paper/                     LaTeX source, references, generated tables
figures/                   generated figures
tests/                     unit tests for the package
docs/                      reproducing guide
```

## Dev

```bash
uv run pytest -q        # tests
uv run ruff check .     # lint
```

MIT license.

## Citation

```bibtex
@article{larosa2026temperature,
  title   = {Temperature Fragility and the Conditional Benefits of Truncation Sampling},
  author  = {La Rosa, Francesco},
  journal = {arXiv preprint arXiv:2609.15476},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.15476}
}
```
