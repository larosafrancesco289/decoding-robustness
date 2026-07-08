"""Aggregate result JSONL into sampler×quant and sampler×temperature accuracy tables.

Pure aggregation over GenerationRecords; no plotting, no stats model (those live in
scripts/stats_matrix.py and scripts/make_figures.py). The tables answer at a glance
whether the sampler ranking moves as bits drop and whether the pipeline is producing
believable numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import QuantLevel
from ..runner.records import GenerationRecord, RecordStore


@dataclass
class Cell:
    """Aggregate for one (sampler, quant[, task]) group."""

    n: int = 0
    correct: int = 0
    parse_failures: int = 0
    completion_tokens: int = 0

    def add(self, record: GenerationRecord) -> None:
        self.n += 1
        self.correct += int(record.correct)
        self.parse_failures += int(record.parse_method == "failed")
        self.completion_tokens += record.n_completion_tokens

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def mean_tokens(self) -> float | None:
        return self.completion_tokens / self.n if self.n else None


# Canonical quant order (high precision -> low), so columns read left to right as bits drop.
_QUANT_ORDER = [
    q.value for q in (QuantLevel.Q8_0, QuantLevel.Q6_K, QuantLevel.Q4_K_M, QuantLevel.Q3_K_M)
]


@dataclass
class SamplerQuantTable:
    """sampler (rows) × quant (cols) accuracy grid, optionally split by task."""

    cells: dict[tuple[str, str, str], Cell] = field(default_factory=dict)  # (task, sampler, quant)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> SamplerQuantTable:
        table = cls()
        for record in RecordStore(path).read_all():
            key = (record.task, record.sampler, record.quant)
            table.cells.setdefault(key, Cell()).add(record)
        return table

    def tasks(self) -> list[str]:
        return sorted({task for task, _, _ in self.cells})

    def samplers(self) -> list[str]:
        return sorted({sampler for _, sampler, _ in self.cells})

    def quants(self) -> list[str]:
        present = {quant for _, _, quant in self.cells}
        ordered = [q for q in _QUANT_ORDER if q in present]
        return ordered + sorted(present - set(ordered))

    def render(self) -> str:
        """A monospace accuracy table per task (acc% with n), quants as columns."""
        quants = self.quants()
        samplers = self.samplers()
        sampler_w = max((len(s) for s in samplers), default=7) + 1
        lines: list[str] = []
        for task in self.tasks():
            lines.append(f"\n== {task} ==")
            header = "sampler".ljust(sampler_w) + "".join(q.rjust(12) for q in quants)
            lines.append(header)
            lines.append("-" * len(header))
            for sampler in samplers:
                row = sampler.ljust(sampler_w)
                for quant in quants:
                    cell = self.cells.get((task, sampler, quant))
                    if cell and cell.accuracy is not None:
                        row += f"{cell.accuracy:6.1%}({cell.n:>3})".rjust(12)
                    else:
                        row += "-".rjust(12)
                lines.append(row)
        return "\n".join(lines)


@dataclass
class SamplerTemperatureTable:
    """sampler (rows) × temperature (cols) accuracy grid, pooled over quant, split by task.

    This is the headline view for temperature robustness: the sampler×quant table
    pools over temperature and so *hides* the whole story, that pure temperature collapses
    at high T while min_p / top-nσ hold. Here temperature is the column axis and we pool over
    the quant levels, which is the margin the pilot's narrative is stated on.
    """

    cells: dict[tuple[str, str, float], Cell] = field(default_factory=dict)  # (task, sampler, temp)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> SamplerTemperatureTable:
        table = cls()
        for record in RecordStore(path).read_all():
            key = (record.task, record.sampler, record.temperature)
            table.cells.setdefault(key, Cell()).add(record)
        return table

    def tasks(self) -> list[str]:
        return sorted({task for task, _, _ in self.cells})

    def samplers(self) -> list[str]:
        return sorted({sampler for _, sampler, _ in self.cells})

    def temperatures(self) -> list[float]:
        return sorted({temp for _, _, temp in self.cells})

    def render(self) -> str:
        """A monospace accuracy table per task (acc% with n), temperatures as columns."""
        temps = self.temperatures()
        samplers = self.samplers()
        sampler_w = max((len(s) for s in samplers), default=7) + 1
        lines: list[str] = []
        for task in self.tasks():
            lines.append(f"\n== {task} (pooled over quant) ==")
            header = "sampler".ljust(sampler_w) + "".join(f"T={t:g}".rjust(12) for t in temps)
            lines.append(header)
            lines.append("-" * len(header))
            for sampler in samplers:
                row = sampler.ljust(sampler_w)
                for temp in temps:
                    cell = self.cells.get((task, sampler, temp))
                    if cell and cell.accuracy is not None:
                        row += f"{cell.accuracy:6.1%}({cell.n:>3})".rjust(12)
                    else:
                        row += "-".rjust(12)
                lines.append(row)
        return "\n".join(lines)
