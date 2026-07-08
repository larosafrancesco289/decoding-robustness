"""Resumable generation loop.

Given a *running* server client, a chat template, and a set of (sampler, temperature)
conditions, this loops ``items × conditions × repetitions``, renders each prompt once,
derives the paired seed, runs the completion, grades it, and appends a ``GenerationRecord``,
skipping any id already on disk so the run is interruptible and resumable.

Throughput: with ``concurrency > 1`` the loop keeps that many completions in
flight against an N-slot llama-server (``--parallel N``), which is the batching lever.
Requests run in worker threads (httpx.Client is thread-safe); grading and the JSONL append
stay on the calling thread so the store is written by one writer and ``RunSummary`` is
consistent. Batched serving is seed-logged but not bit-exact; accepted here.

It is intentionally agnostic to *how many* conditions it is given: the smoke drivers pass
a single greedy condition; the matrix passes the full factorial expansion. Server lifecycle
and matrix expansion live elsewhere (one server == one (model, quant) load).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

from ..config.schema import Checkpoint, QuantLevel, SamplerSpec, TaskSpec
from ..inference import (
    ChatTemplate,
    LlamaServerClient,
    chain_label,
    completion_params,
    prompt_sha256,
)
from ..seeding import derive_seed
from ..tasks import TaskItem, grade
from .records import GenerationRecord, RecordStore, make_record_id

Condition = tuple[SamplerSpec, float]


@dataclass
class RunSummary:
    """Aggregate outcome of a loop pass, for the M2 self-check and the M3 table."""

    generated: int = 0  # records newly written this pass
    skipped: int = 0  # ids already on disk (resume)
    correct: int = 0  # over newly generated records
    parse_failures: int = 0  # parse_method == "failed", over newly generated

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.generated if self.generated else None

    @property
    def parse_failure_rate(self) -> float | None:
        return self.parse_failures / self.generated if self.generated else None


@dataclass
class _WorkUnit:
    """One pending generation: its identity, the rendered prompt, and the request params."""

    record_id: str
    item: TaskItem
    rendered: str
    prompt_hash: str
    sampler_label: str
    temperature: float
    repetition: int
    seed: int
    params: dict


def run_conditions(
    *,
    client: LlamaServerClient,
    template: ChatTemplate,
    checkpoint: Checkpoint,
    quant: QuantLevel,
    task: TaskSpec,
    items: Sequence[TaskItem],
    conditions: Iterable[Condition],
    store: RecordStore,
    global_seed: int,
    repetitions: int = 1,
    concurrency: int = 1,
    server_commit: str | None = None,
    stop: list[str] | None = None,
    enable_thinking: bool = False,
    on_record: Callable[[GenerationRecord], None] | None = None,
) -> RunSummary:
    """Run every (item, condition, repetition) not already on disk; append + grade each.

    ``concurrency`` requests are kept in flight at once (match it to the server's
    ``--parallel`` slots for batched throughput); 1 is the deterministic single-stream path.
    """
    seen = store.existing_ids()
    summary = RunSummary()
    conditions = list(conditions)

    # Build the pending work. The prompt depends only on the item, so render (and hash) it
    # once per item and reuse across conditions; this keeps the paired design's prompt identical.
    units: list[_WorkUnit] = []
    for item in items:
        # enable_thinking defaults False, which keeps Qwen3 out of thinking-mode (long outputs that
        # blow the per-slot context budget); harmless for templates that don't reference it
        # (Llama/Mistral/Qwen2.5 ignore the kwarg). Set True only for the thinking-ON spot-check.
        rendered = template.render(
            [{"role": "user", "content": item.prompt}], enable_thinking=enable_thinking
        )
        prompt_hash = prompt_sha256(rendered)
        for sampler, temperature in conditions:
            # Greedy/T=0 is deterministic, so one rep suffices; reps only reduce sampling noise
            # for stochastic methods. Avoids writing 3 identical greedy draws when repetitions>1.
            n_reps = 1 if temperature == 0.0 else repetitions
            for rep in range(1, n_reps + 1):
                record_id = make_record_id(
                    model=checkpoint.name,
                    quant=quant.value,
                    sampler=sampler.label,
                    temperature=temperature,
                    task=task.name,
                    item_id=item.item_id,
                    repetition=rep,
                )
                if record_id in seen:
                    summary.skipped += 1
                    continue
                seed = derive_seed(
                    global_seed=global_seed,
                    model=checkpoint.name,
                    quant=quant.value,
                    sampler=sampler.label,
                    temperature=temperature,
                    task=task.name,
                    item_id=item.item_id,
                    repetition=rep,
                )
                params = completion_params(
                    sampler, temperature, seed, n_predict=task.max_new_tokens, stop=stop
                )
                units.append(
                    _WorkUnit(
                        record_id=record_id,
                        item=item,
                        rendered=rendered,
                        prompt_hash=prompt_hash,
                        sampler_label=sampler.label,
                        temperature=temperature,
                        repetition=rep,
                        seed=seed,
                        params=params,
                    )
                )

    def _run(unit: _WorkUnit) -> tuple[_WorkUnit, object, float]:
        started = time.monotonic()
        result = client.completion(unit.rendered, unit.params)
        return unit, result, time.monotonic() - started

    def _finish(unit: _WorkUnit, result, latency_s: float) -> None:  # noqa: ANN001
        graded = grade(result.content, unit.item.gold, task.parser)
        record = GenerationRecord(
            id=unit.record_id,
            model=checkpoint.name,
            quant=quant.value,
            sampler=unit.sampler_label,
            params={**unit.params, "chain": chain_label(unit.params["samplers"])},
            temperature=unit.temperature,
            task=task.name,
            item_id=unit.item.item_id,
            repetition=unit.repetition,
            seed=unit.seed,
            prompt_hash=unit.prompt_hash,
            raw_output=result.content,
            parsed_answer=graded.parsed_answer,
            parse_method=graded.parse_method,
            gold=unit.item.gold,
            correct=graded.correct,
            n_prompt_tokens=result.tokens_evaluated,
            n_completion_tokens=result.tokens_predicted,
            latency_s=latency_s,
            tokens_per_second=result.predicted_per_second,
            stopped=result.stopped,
            server_commit=server_commit,
            timestamp=datetime.now(UTC).isoformat(),
        )
        store.append(record)
        summary.generated += 1
        summary.correct += int(graded.correct)
        summary.parse_failures += int(graded.parse_method == "failed")
        if on_record is not None:
            on_record(record)

    if concurrency <= 1:
        for unit in units:
            _finish(*_run(unit))
    else:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        futures = [pool.submit(_run, unit) for unit in units]
        try:
            for future in as_completed(futures):
                _finish(*future.result())
        finally:
            # On an unexpected error, cancel queued work instead of draining all of it
            # (the records already written stay on disk for resume).
            pool.shutdown(wait=False, cancel_futures=True)

    return summary
