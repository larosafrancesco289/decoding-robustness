# Full matrix — cross-model results summary

Cluster run complete 2026-06-03. **14 cluster cells** (N=50, K=3, GSM8K + MMLU-Pro, sampler grid
per `configs/full_matrix.yaml`): Llama-3.1-8B {Q6,Q4,Q3}, Mistral-7B-v0.3 {Q6,Q4,Q3}, Qwen3-8B
{Q6,Q4,Q3}, Qwen3-4B {Q8,Q6,Q4,Q3}, Qwen3-1.7B {Q8,Q6,Q4,Q3} — all 6400 records/cell, 0 failures.
**Off-cluster (user's box), pending:** Llama/Mistral/Qwen3-8B Q8_0 + Qwen3-4B BF16.
Tables below are sampler×temperature **pooled over quant** (render via `analysis.SamplerTemperatureTable`).

## Greedy baselines
| Model | greedy GSM8K | greedy MMLU-Pro |
|---|---|---|
| Llama-3.1-8B | 78.7% | 55.3% |
| Mistral-7B-v0.3 | 32.0% | 34.0% |
| Qwen3-8B | 94.0% | 60.0% |
| Qwen3-4B | 95.0% | 61.5% |
| Qwen3-1.7B | 75.5% | 45.5% |

## Headline (PRELIMINARY — discuss before writeup): the temperature-robustness effect is MODEL-DEPENDENT
- **Llama-3.1-8B — classic pattern reproduces (de-risk gate + matrix agree).** Pure `temperature`
  collapses at T=1.3 (GSM8K 78→43%, MMLU-Pro 48→**6%**); `top_p`/`top_k` degrade (top_p MMLU
  51→16%); `min_p` & `top_n_sigma` stay robust (top-nσ GSM8K 77→82%). Temp-LAST arms beat temp-FIRST
  (top_p MMLU @T=1.3: 38% tlast vs 16% tfirst) — the min-p-critique ablation.
- **Qwen3 (8B / 4B / 1.7B) — NO collapse for ANY sampler**, incl. pure temperature, out to T=1.3
  (Qwen3-8B GSM8K ~94% & MMLU ~60% essentially flat across all temps & samplers). Decoding strategy
  barely moves accuracy.
- **Mistral-7B-v0.3 — at the floor** on these 0-shot CoT tasks (~32% GSM8K, ~34% MMLU); no
  temperature or sampler signal to rank.

## Interpretation (hypothesis, for the user to finalize)
The min_p/top-nσ "temperature robustness" advantage is only visible in the **mid-accuracy regime**
(Llama, ~78% GSM8K). It washes out at the **ceiling** (Qwen3 — very peaked *thinking-off*
distributions, and 8B/4B near GSM8K ceiling → high T doesn't induce errors) and at the **floor**
(Mistral). The ranking *direction* (truncation methods ≥ pure temperature at high T) never reverses,
but its *magnitude* is strongly model-dependent and well-powered only for Llama here. This reframes
the cross-model "stability" story: less "stable ranking across models," more "the effect is real
wherever there's headroom to observe it."

**Caveats:** Qwen3 thinking-OFF by design; Qwen3-8B/4B near GSM8K ceiling (compression); Mistral
floor. Per-quant stability + the pooled logistic/GLMM (sampler×quant interaction + item random
effects) are the user's planned analysis — these tables pool over quant. Add the 4 off-cluster
Q8/BF16 shards when they land for the full picture (esp. the Qwen3-4B BF16 anchor).
