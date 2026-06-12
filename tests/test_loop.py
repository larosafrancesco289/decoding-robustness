"""Tests for the resumable generation loop, incl. client-side concurrency.

A stub client stands in for llama-server so the loop logic — id-skip resume, grading,
and concurrent execution — is exercised offline.
"""

from __future__ import annotations

import threading
import time

from decoding_robustness.config.schema import (
    Checkpoint,
    DecodingMethod,
    QuantLevel,
    SamplerSpec,
    TaskSpec,
)
from decoding_robustness.inference import ChatTemplate
from decoding_robustness.inference.client import CompletionResult
from decoding_robustness.runner import RecordStore, run_conditions
from decoding_robustness.tasks import TaskItem

_TEMPLATE = ChatTemplate(template="{{ messages[0]['content'] }}", eos_token="</s>")
_CKPT = Checkpoint(
    name="llama-3.1-8b-instruct",
    family="llama",
    params_b=8.0,
    hf_repo="x/y",
    quant_files={QuantLevel.Q8_0: "m.gguf"},
)
_TASK = TaskSpec(name="gsm8k", hf_dataset="openai/gsm8k", revision="main", parser="gsm8k_numeric")
_ITEMS = [TaskItem(item_id=f"i{i}", prompt=f"q{i}", gold="18") for i in range(5)]
_CONDS = [
    (SamplerSpec(method=DecodingMethod.GREEDY), 0.0),
    (SamplerSpec(method=DecodingMethod.MIN_P, min_p=0.05), 1.0),
]


class _StubClient:
    """Returns a fixed completion; counts concurrent in-flight calls."""

    def __init__(self, content="The final answer is 18."):
        self.content = content
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def completion(self, prompt, params):  # noqa: ANN001
        with self._lock:
            self.calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(0.03)  # hold the "request" so concurrent callers actually overlap
            return CompletionResult(
                content=self.content, tokens_predicted=7, tokens_evaluated=3, stopped=True
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def _run(tmp_path, client, concurrency):
    store = RecordStore(tmp_path / "out.jsonl")
    return store, run_conditions(
        client=client,
        template=_TEMPLATE,
        checkpoint=_CKPT,
        quant=QuantLevel.Q8_0,
        task=_TASK,
        items=_ITEMS,
        conditions=_CONDS,
        store=store,
        global_seed=0,
        concurrency=concurrency,
    )


def test_runs_all_units_and_grades(tmp_path):
    store, summary = _run(tmp_path, _StubClient(), concurrency=1)
    assert summary.generated == len(_ITEMS) * len(_CONDS)  # 5 * 2 = 10
    assert summary.correct == 10  # all answer "18" == gold
    assert summary.parse_failures == 0
    assert len(list(store.read_all())) == 10


def test_resume_skips_existing_ids(tmp_path):
    client = _StubClient()
    _run(tmp_path, client, concurrency=1)
    first_calls = client.calls
    # Second pass over the same store: everything already on disk -> all skipped.
    store, summary = _run(tmp_path, client, concurrency=1)
    assert summary.generated == 0
    assert summary.skipped == 10
    assert client.calls == first_calls  # no new completion calls
    assert len(list(store.read_all())) == 10  # no duplicates appended


def test_concurrency_keeps_multiple_in_flight_and_is_consistent(tmp_path):
    client = _StubClient()
    store, summary = _run(tmp_path, client, concurrency=4)
    assert summary.generated == 10
    assert client.max_in_flight > 1  # actually ran concurrently
    assert len({r.id for r in store.read_all()}) == 10  # unique ids, no lost/dup writes


def test_parse_failure_counted(tmp_path):
    store, summary = _run(tmp_path, _StubClient(content="no number here"), concurrency=2)
    assert summary.parse_failures == 10
    assert summary.correct == 0
