"""Generation record schema + append-only JSONL store (SPEC §7.1, §7.2).

Every generation is one self-describing JSONL line. The record carries the full sample
identity (model, quant, sampler, temperature, task, item, repetition) plus the rendered-
prompt hash, the raw output, the graded result, and the length/throughput covariates —
so the JSONL alone reconstructs the run without the config.

Resumability (SPEC §7.2): each record's ``id`` is a deterministic function of its identity
tuple. ``RecordStore`` reads the ids already on disk and the runner skips them, so an
interrupted run resumes from the JSONL and Phase 2 just appends new ids.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def make_record_id(
    *,
    model: str,
    quant: str,
    sampler: str,
    temperature: float,
    task: str,
    item_id: str,
    repetition: int,
) -> str:
    """Deterministic, human-readable id for one sample (the resumability key).

    Readable rather than hashed so a JSONL line is debuggable by eye; the identity tuple
    is unique by construction, so this doubles as the natural key.
    """
    return f"{model}|{quant}|{sampler}|t{temperature}|{task}|{item_id}|r{repetition}"


class GenerationRecord(BaseModel):
    """One graded generation (SPEC §7.1). Append-only; one per JSONL line."""

    model_config = ConfigDict(extra="forbid")

    id: str
    # Identity tuple (also encoded in ``id``, kept explicit so records are self-describing).
    model: str
    quant: str
    sampler: str  # SamplerSpec.label
    params: dict  # the /completion params actually sent (chain, truncation param, ...)
    temperature: float
    task: str
    item_id: str
    repetition: int
    seed: int

    # Prompt / output.
    prompt_hash: str
    raw_output: str

    # Grading.
    parsed_answer: str | None
    parse_method: str  # "strict" | "flexible" | "failed"
    gold: str
    correct: bool

    # Covariates (SPEC §5).
    n_prompt_tokens: int
    n_completion_tokens: int
    latency_s: float | None = None
    tokens_per_second: float | None = None
    stopped: bool = False

    # Provenance.
    server_commit: str | None = None
    timestamp: str = Field(description="UTC ISO-8601 capture time")


class RecordStore:
    """Append-only JSONL store with resumable id-skip (SPEC §7.2)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def existing_ids(self) -> set[str]:
        """Ids already on disk, so the runner can skip completed samples on resume.

        Tolerant of a truncated/interleaved trailing line: a kill mid-write (concurrent
        appends from the batched runner) can leave a partial JSON line. We skip such lines
        with a warning rather than crash the resume — the dropped samples just regenerate.
        """
        if not self.path.is_file():
            return set()
        ids: set[str] = set()
        bad = 0
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    bad += 1
        if bad:
            print(f"[RecordStore] skipped {bad} corrupt line(s) in {self.path}", file=sys.stderr)
        return ids

    def append(self, record: GenerationRecord) -> None:
        """Append one record as a JSON line (fsync-free; OS buffering is fine for resume)."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def read_all(self) -> Iterator[GenerationRecord]:
        """Stream every record back (for accuracy aggregation / analysis).

        Skips corrupt lines (see ``existing_ids``) so analysis never crashes on a partial
        line left by an interrupted run.
        """
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield GenerationRecord.model_validate_json(line)
                except ValueError:
                    print(f"[RecordStore] skipped corrupt line in {self.path}", file=sys.stderr)
