"""Tests for deterministic per-sample seed derivation."""

from __future__ import annotations

from decoding_robustness.seeding import derive_seed

BASE = dict(
    global_seed=0,
    model="llama-3.1-8b-instruct",
    quant="Q4_K_M",
    sampler="min_p_min_p0.05",
    temperature=1.0,
    task="gsm8k",
    item_id="0001",
    repetition=1,
)


def test_seed_is_deterministic():
    assert derive_seed(**BASE) == derive_seed(**BASE)


def test_seed_in_32bit_range():
    seed = derive_seed(**BASE)
    assert 0 <= seed < 2**32


def test_seed_varies_with_identity():
    base = derive_seed(**BASE)
    assert derive_seed(**{**BASE, "repetition": 2}) != base
    assert derive_seed(**{**BASE, "quant": "Q8_0"}) != base
    assert derive_seed(**{**BASE, "item_id": "0002"}) != base
    assert derive_seed(**{**BASE, "temperature": 0.7}) != base


def test_same_item_across_conditions_uses_independent_seeds():
    # Paired design: the *same* item under two methods must still get distinct seeds
    # (we pair by reusing the item/prompt, not by reusing the RNG stream).
    a = derive_seed(**{**BASE, "sampler": "top_p_top_p0.9"})
    b = derive_seed(**{**BASE, "sampler": "min_p_min_p0.05"})
    assert a != b
