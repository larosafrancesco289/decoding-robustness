"""Analysis layer.

M3 adds the first sampler×quant accuracy table (pure aggregation). The GLMM interaction
model, the Bayesian robustness check, and figures arrive at M4.
"""

from __future__ import annotations

from .tables import Cell, SamplerQuantTable, SamplerTemperatureTable

__all__ = ["Cell", "SamplerQuantTable", "SamplerTemperatureTable"]
