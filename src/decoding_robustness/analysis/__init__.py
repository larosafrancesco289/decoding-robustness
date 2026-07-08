"""Analysis layer: pure aggregation into accuracy tables.

Bootstrap statistics and figures live in scripts/ (stats_matrix.py, make_figures.py).
"""

from __future__ import annotations

from .tables import Cell, SamplerQuantTable, SamplerTemperatureTable

__all__ = ["Cell", "SamplerQuantTable", "SamplerTemperatureTable"]
