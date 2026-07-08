# decoding-robustness

Code and data for **"Temperature Robustness Is a Model Property, Not a Sampler Choice"**.

A single-pipeline factorial study of decoding strategies on open-weight
instruction-tuned models: seven models from five families (1.7B to 12B), four GGUF
quantization levels plus a BF16 anchor, eight sampler arms, temperatures 0.7 to 2.0,
GSM8K and MMLU-Pro, plus a temperature-only cell on two more Llama generations.
About 204,000 graded generations; every generation is included under `results/` as a
JSONL record, so all numbers in the paper can be recomputed from this repository
without a GPU.

## Findings (paper section in parentheses)

1. **At the temperatures deployments use (T up to 1.3) the sampler barely matters;
   the model is the variable.** Under plain temperature at T=1.3 the Llama family
   collapses: Llama-3.1-8B loses 34 to 41 accuracy points, Llama-3.2-3B collapses
   the same way, and Llama-3-8B loses 17 on MMLU-Pro. Every model outside the
   family stays within 8 points (4.1, 4.2).
2. **The collapse is degeneration, not wrong answers.** Accuracy conditional on a
   well-formed final answer is flat in temperature; what collapses is the
   probability of terminating a coherent answer. Standard parsers disguise this,
   GSM8K as wrong answers and MMLU-Pro as parse failures (4.3).
3. **Every model has a cliff, and robustness is the cliff location.** A ladder up
   to T=2.0 shows Llama collapsing by 1.5, Qwen2.5 by 1.7, and Gemma-3 by 2.0 on
   GSM8K. Truncation samplers shift each cliff right in order of tail-cutting
   strength, so published sampler gains are real but sit past every model's
   deployment range (4.4).
4. **Quantization neither causes the collapse nor reorders the samplers** from Q8
   down to Q3; a real cost appears only on the two smallest models (4.5).
5. **Mechanism.** Greedy-path uncertainty ranks the temperature drops (Spearman
   +0.96), and a forced-token perturbation isolates the difference: one
   off-distribution token in the context multiplies the fragile model's chance of
   drawing the next one by 2.3, against at most 1.35 for robust models, so
   derailment compounds (4.6).

## Reproduce

[`docs/REPRODUCING.md`](docs/REPRODUCING.md) maps every figure, table, and in-text
number to the script and data that produce it. The short version:

```bash
uv sync
CUDA_ARCH=120 scripts/build_llamacpp.sh
export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
uv run python scripts/fetch_models.py --config configs/full_matrix.yaml
uv run python scripts/run_pilot.py --config configs/full_matrix.yaml
uv run python scripts/gen_tables.py
uv run python scripts/make_figures.py
```

The analysis scripts need no GPU and run on the shards in `results/` in seconds.
Model checkpoints are pinned by SHA256 in [`DOWNLOADS.md`](DOWNLOADS.md).

## Where to look in the code

| What | Where |
|---|---|
| How a sampler arm becomes a server request; the pinned chain order | `src/decoding_robustness/inference/sampling.py` |
| Client-side chat-template rendering (the exact prompt is hashed and logged) | `src/decoding_robustness/inference/templating.py` |
| Deterministic per-record seeds (the paired design) | `src/decoding_robustness/seeding.py` |
| Answer parsing with logged parse paths; grading by exact match | `src/decoding_robustness/tasks/parsers.py`, `grading.py` |
| The resumable generation loop and the JSONL record schema | `src/decoding_robustness/runner/loop.py`, `records.py` |
| Item-clustered bootstrap behind Table 1 and the CIs | `scripts/stats_matrix.py` |
| The degeneration decomposition (4.3) | `scripts/decompose_collapse.py` |
| Cliff ladder and stratified replication (4.4, App. C) | `scripts/analyze_followups.py` |
| Mechanism probes (4.6) | `scripts/logit_probe.py`, `logit_probe2.py`, `perturb_probe.py` and their `analyze_*.py` counterparts |

## Layout

```
src/decoding_robustness/   the package
  config/            Pydantic experiment-config schema + YAML loader
  inference/         llama-server client, lifecycle, sampling params, templating
  tasks/             dataset loaders, answer parsers, graders
  runner/            factorial matrix expansion + resumable generation loop
  analysis/          aggregation tables
configs/             YAML experiment configs (validated against the schema)
scripts/             run drivers and analysis scripts (see docs/REPRODUCING.md)
results/             JSONL shards, one record per generation
figures/             generated figures
tests/               unit tests
DOWNLOADS.md         SHA256 manifest of all 31 model files
STATS.txt            headline statistics with bootstrap CIs
DECOMP.txt           the degeneration decomposition
```

## Dev

```bash
uv run pytest -q        # tests
uv run ruff check .     # lint
```
