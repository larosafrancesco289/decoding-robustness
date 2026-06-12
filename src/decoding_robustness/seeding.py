"""Deterministic per-sample seed derivation (SPEC §7.4).

The same item gets the same seed across every decoding condition and quant level,
which enables the paired / common-random-numbers design that cancels between-prompt
variance — the main source of statistical power for the interaction estimate.
"""

from __future__ import annotations

import hashlib


def derive_seed(
    *,
    global_seed: int,
    model: str,
    quant: str,
    sampler: str,
    temperature: float,
    task: str,
    item_id: str,
    repetition: int,
) -> int:
    """Return a deterministic 32-bit seed for one sample.

    The seed is a function of the full sample identity, so it is reproducible and
    stable across runs, yet differs across repetitions of the same condition.
    """
    payload = "|".join(
        str(part)
        for part in (
            global_seed,
            model,
            quant,
            sampler,
            temperature,
            task,
            item_id,
            repetition,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # 32-bit non-negative seed
