# decoding-robustness

**Temperature kills models, not answers: decoding robustness is a model property.**

A single-pipeline factorial study of decoding strategies on small open-weight LLMs:
seven instruction-tuned models from five families (1.7B-12B), 4 GGUF quantization levels
plus a BF16 anchor, 8 sampler arms, temperatures 0.7-2.0, GSM8K and MMLU-Pro, plus a
temperature-only cell on two more Llama generations; about 204,000 graded generations on
commodity GPUs. Paper source in [`paper/`](paper/).

## Findings (paper section in parentheses)

- **At deployment temperatures the sampler barely matters; the model is the variable.**
  Under plain temperature at T=1.3 the Llama family collapses: Llama-3.1-8B loses
  34-41pp, Llama-3.2-3B collapses the same way, and Llama-3-8B loses 17pp on MMLU-Pro.
  Every model outside the family stays within 8pp (4.1-4.2).
- **The collapse is degeneration, not wrong answers.** Accuracy conditional on a
  well-formed answer is flat in temperature; what collapses is termination. Standard
  parsers disguise this (GSM8K as wrong answers, MMLU-Pro as parse failure) (4.3).
- **Every model has a cliff; robustness is the cliff's location.** A T=1.3-2.0 ladder
  shows a staircase (Llama by 1.5, Qwen2.5 by 1.7, Gemma-3 by 2.0 on GSM8K); truncation
  samplers shift the cliff right by tail-cutting strength; sampler gains are real but
  live past every model's deployment range (4.4).
- **Quantization does not cause the collapse and does not reorder samplers** (Q8 to Q3;
  the effect grows as models shrink) (4.5).
- **Mechanism:** greedy-path uncertainty ranks with fragility (Spearman +0.96), and a
  forced-token perturbation isolates the difference: one off-distribution token in
  context multiplies the fragile model's chance of drawing the next one (2.3x, against
  at most 1.35x for robust models), so derailment is self-reinforcing (4.6).

## Reproduce

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the full paper-to-script map
(every figure, table, and in-text number). The short version:

```bash
uv sync                                          # env
CUDA_ARCH=120 scripts/build_llamacpp.sh          # pin llama.cpp (sm_120 = RTX 5070 Ti)
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
uv run python scripts/fetch_models.py --config configs/full_matrix.yaml   # GGUFs + SHA256
uv run python scripts/run_pilot.py --config configs/full_matrix.yaml      # the matrix
uv run python scripts/gen_tables.py              # paper/tables/*.tex
uv run python scripts/make_figures.py            # figures/*.pdf  (uv sync --extra figures)
cd paper && tectonic main.tex                    # the PDF
```

Every generation is one resumable JSONL record keyed by
`model|quant|sampler|temperature|task|item|repetition`, with the rendered-prompt SHA256,
seed, parse path, and throughput logged. Quantized checkpoints are Bartowski imatrix
GGUFs with pinned SHA256 hashes ([`DOWNLOADS.md`](DOWNLOADS.md)).

## Layout

```
src/decoding_robustness/   # the package
  config/            # Pydantic experiment-config schema + YAML loader
  inference/         # llama-server client + lifecycle
  tasks/             # dataset loaders, answer parsers, graders
  runner/            # factorial matrix expansion + resumable generation loop
  analysis/          # bootstrap stats + analysis entry points
configs/             # YAML experiment configs (validated against the schema)
scripts/             # run/analysis scripts (see docs/REPRODUCING.md)
results/             # JSONL shards per study arm
paper/               # LaTeX master, references, auto-generated tables
figures/             # generated figures
tests/               # unit tests
docs/                # reproducing guide
```

## Dev

```bash
uv run pytest -q        # tests
uv run ruff check .     # lint
```
