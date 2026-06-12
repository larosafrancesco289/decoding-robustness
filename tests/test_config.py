"""Tests for the experiment-config schema, loader, and design-rule validators."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from decoding_robustness.config import (
    DecodingMethod,
    ExperimentConfig,
    SamplerSpec,
    load_experiment_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = REPO_ROOT / "configs" / "pilot_llama.yaml"


def test_pilot_config_loads_and_validates():
    config = load_experiment_config(PILOT_CONFIG)
    assert isinstance(config, ExperimentConfig)
    assert config.phase == "pilot"
    assert len(config.checkpoints) == 1
    assert config.checkpoints[0].family == "llama"
    assert len(config.quant_levels) == 4
    assert len(config.tasks) == 3


def test_decoding_conditions_apply_asymmetric_grid():
    config = load_experiment_config(PILOT_CONFIG)
    conditions = list(config.decoding_conditions())
    # greedy once at T=0 + 7 stochastic methods x 3 temperatures
    # (6 base methods + 2 temperature-last ordering arms for min_p & top_p − greedy = 7)
    assert len(conditions) == 1 + 7 * 3
    greedy = [(s, t) for s, t in conditions if s.method == DecodingMethod.GREEDY]
    assert len(greedy) == 1
    assert greedy[0][1] == 0.0
    # no stochastic condition is emitted at T=0
    assert all(t > 0 for s, t in conditions if s.method != DecodingMethod.GREEDY)


def test_sampler_label_is_stable():
    spec = SamplerSpec(method=DecodingMethod.MIN_P, min_p=0.05)
    assert spec.label == "min_p_min_p0.05"
    assert SamplerSpec(method=DecodingMethod.GREEDY).label == "greedy"


def test_temperature_last_chain_gets_distinct_label():
    # the two ordering arms of one method must not collide on record id
    first = SamplerSpec(method=DecodingMethod.MIN_P, min_p=0.05)  # default = temp-first
    last = SamplerSpec(method=DecodingMethod.MIN_P, min_p=0.05, chain=["min_p", "temperature"])
    assert first.label == "min_p_min_p0.05"
    assert last.label == "min_p_min_p0.05_tlast"
    assert first.label != last.label


def test_method_requires_its_own_parameter():
    with pytest.raises(ValidationError):
        SamplerSpec(method=DecodingMethod.MIN_P)  # missing min_p
    with pytest.raises(ValidationError):
        SamplerSpec(method=DecodingMethod.TOP_P)  # missing top_p


def test_method_rejects_foreign_parameters():
    # orthogonality: a method may only set its own truncation parameter
    with pytest.raises(ValidationError):
        SamplerSpec(method=DecodingMethod.TOP_P, top_p=0.9, min_p=0.05)
    with pytest.raises(ValidationError):
        SamplerSpec(method=DecodingMethod.GREEDY, top_p=0.9)


def test_temperature_grid_must_be_positive_and_unique():
    base = dict(
        name="x",
        phase="pilot",
        checkpoints=[
            dict(name="m", family="f", params_b=8, hf_repo="r"),
        ],
        quant_levels=["Q4_K_M"],
        samplers=[dict(method="greedy")],
        tasks=[
            dict(name="t", hf_dataset="d", revision="main", parser="p"),
        ],
    )
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**base, "temperatures": [0.0, 0.7]})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({**base, "temperatures": [0.7, 0.7]})
    # valid baseline parses
    ExperimentConfig.model_validate({**base, "temperatures": [0.7, 1.0]})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SamplerSpec(method=DecodingMethod.GREEDY, bogus=1)
