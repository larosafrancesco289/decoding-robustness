"""Map a (SamplerSpec, temperature, seed) condition to llama-server /completion params.

This is where the study's confound controls become a concrete request: exactly one
truncation method is active per condition, and the sampler chain order is pinned via
the llama-server ``samplers`` array.

Chain-order decision (default): **temperature first, then the truncation method**, i.e.
``["temperature", "min_p"]``. Rationale:

  * It matches HuggingFace's `generate` ordering (temperature → top_k/top_p), the
    convention the source papers were evaluated under.
  * It makes the probability-space truncators (top_p, min_p) see the *temperature-scaled*
    distribution, which is the temperature-adaptive behaviour the min-p paper relies on.
  * top-nσ is temperature-invariant by construction (it thresholds on logit std, and
    temperature scales max and std together), so its surviving set is unchanged by where
    temperature sits.

Note this departs from llama.cpp's *default* chain, which applies temperature last. The
order is recorded per generation and can be overridden per sampler via ``SamplerSpec.chain``;
the temp-last ablation arms use exactly that override.
"""

from __future__ import annotations

from ..config.schema import DecodingMethod, SamplerSpec

# Decoding method -> the llama.cpp ``samplers``-array token for its truncation stage.
# (Confirmed against common/sampling.cpp: these are the canonical sampler names.)
_CHAIN_TOKEN: dict[DecodingMethod, str] = {
    DecodingMethod.TOP_P: "top_p",
    DecodingMethod.TOP_K: "top_k",
    DecodingMethod.MIN_P: "min_p",
    DecodingMethod.TOP_N_SIGMA: "top_n_sigma",
}

TEMPERATURE_TOKEN = "temperature"


def default_chain(method: DecodingMethod) -> list[str]:
    """The pinned sampler chain for a method (application order; temperature first)."""
    if method in (DecodingMethod.GREEDY, DecodingMethod.TEMPERATURE):
        return [TEMPERATURE_TOKEN]
    return [TEMPERATURE_TOKEN, _CHAIN_TOKEN[method]]


def completion_params(
    sampler: SamplerSpec,
    temperature: float,
    seed: int,
    *,
    n_predict: int,
    stop: list[str] | None = None,
    cache_prompt: bool = True,
    n_probs: int = 0,
    ignore_eos: bool = False,
) -> dict:
    """Build the JSON body for a single POST /completion call (sans ``prompt``).

    Exactly one truncation parameter is included, matching ``sampler.method``; foreign
    truncation params are never sent, so the request is self-describing and the chain is
    unambiguous. ``greedy`` is emitted as temperature 0.0 (llama.cpp decodes argmax then).
    """
    method = sampler.method
    chain = sampler.chain if sampler.chain is not None else default_chain(method)

    if method == DecodingMethod.GREEDY:
        temperature = 0.0  # deterministic argmax anchor, regardless of the swept grid

    params: dict = {
        "samplers": chain,
        "temperature": temperature,
        "seed": seed,
        "n_predict": n_predict,
        "cache_prompt": cache_prompt,
    }

    if method == DecodingMethod.TOP_P:
        params["top_p"] = sampler.top_p
    elif method == DecodingMethod.TOP_K:
        params["top_k"] = sampler.top_k
    elif method == DecodingMethod.MIN_P:
        params["min_p"] = sampler.min_p
    elif method == DecodingMethod.TOP_N_SIGMA:
        params["top_n_sigma"] = sampler.top_n_sigma

    if stop:
        params["stop"] = stop
    if n_probs:
        params["n_probs"] = n_probs
    if ignore_eos:
        params["ignore_eos"] = True

    return params


def chain_label(chain: list[str]) -> str:
    """Human/record-friendly rendering of a sampler chain, e.g. ``temperature->min_p``."""
    return "->".join(chain)
