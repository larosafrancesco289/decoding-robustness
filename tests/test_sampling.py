"""Tests for the SamplerSpec -> /completion params mapping (sampler-chain plumbing)."""

from __future__ import annotations

from decoding_robustness.config.schema import DecodingMethod, SamplerSpec
from decoding_robustness.inference.sampling import (
    chain_label,
    completion_params,
    default_chain,
)


def test_greedy_forces_zero_temperature_and_no_truncation():
    spec = SamplerSpec(method=DecodingMethod.GREEDY)
    params = completion_params(spec, temperature=1.0, seed=7, n_predict=128)
    assert params["temperature"] == 0.0  # greedy decodes argmax regardless of the grid
    assert params["samplers"] == ["temperature"]
    assert not {"top_p", "top_k", "min_p", "top_n_sigma"} & params.keys()


def test_pure_temperature_has_only_temperature_in_chain():
    spec = SamplerSpec(method=DecodingMethod.TEMPERATURE)
    params = completion_params(spec, temperature=0.7, seed=1, n_predict=64)
    assert params["samplers"] == ["temperature"]
    assert params["temperature"] == 0.7


def test_each_truncation_method_sends_only_its_own_param():
    cases = {
        DecodingMethod.TOP_P: ("top_p", dict(top_p=0.95)),
        DecodingMethod.TOP_K: ("top_k", dict(top_k=40)),
        DecodingMethod.MIN_P: ("min_p", dict(min_p=0.05)),
        DecodingMethod.TOP_N_SIGMA: ("top_n_sigma", dict(top_n_sigma=1.0)),
    }
    all_params = {"top_p", "top_k", "min_p", "top_n_sigma"}
    for method, (token, kwargs) in cases.items():
        spec = SamplerSpec(method=method, **kwargs)
        params = completion_params(spec, temperature=1.0, seed=1, n_predict=64)
        # temperature first, then exactly this method's truncation stage
        assert params["samplers"] == ["temperature", token]
        assert token in params
        assert not (all_params - {token}) & params.keys()


def test_default_chain_is_temperature_first():
    assert default_chain(DecodingMethod.MIN_P) == ["temperature", "min_p"]
    assert default_chain(DecodingMethod.GREEDY) == ["temperature"]


def test_explicit_chain_override_is_respected():
    spec = SamplerSpec(method=DecodingMethod.MIN_P, min_p=0.05, chain=["min_p", "temperature"])
    params = completion_params(spec, temperature=1.0, seed=1, n_predict=64)
    assert params["samplers"] == ["min_p", "temperature"]


def test_common_request_fields_present():
    spec = SamplerSpec(method=DecodingMethod.TOP_P, top_p=0.9)
    params = completion_params(spec, temperature=1.0, seed=42, n_predict=256, stop=["<|eot_id|>"])
    assert params["seed"] == 42
    assert params["n_predict"] == 256
    assert params["cache_prompt"] is True
    assert params["stop"] == ["<|eot_id|>"]


def test_chain_label_renders_arrows():
    assert chain_label(["temperature", "min_p"]) == "temperature->min_p"
