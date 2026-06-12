"""Tests for the sampler×quant accuracy aggregation (analysis/tables.py)."""

from __future__ import annotations

from decoding_robustness.analysis import SamplerQuantTable, SamplerTemperatureTable
from decoding_robustness.runner.records import GenerationRecord, RecordStore, make_record_id


def _rec(*, sampler, quant, correct, parse_method="strict", item_id, tokens=100, temperature=0.0):
    identity = dict(
        model="llama-3.1-8b-instruct",
        quant=quant,
        sampler=sampler,
        temperature=temperature,
        task="gsm8k",
        item_id=item_id,
        repetition=1,
    )
    return GenerationRecord(
        id=make_record_id(**identity),
        **identity,
        params={},
        seed=1,
        prompt_hash="x",
        raw_output="...",
        parsed_answer="18" if correct else None,
        parse_method=parse_method,
        gold="18",
        correct=correct,
        n_prompt_tokens=10,
        n_completion_tokens=tokens,
        timestamp="2026-06-01T00:00:00+00:00",
    )


def test_table_accuracy_and_counts(tmp_path):
    store = RecordStore(tmp_path / "p.jsonl")
    # greedy@Q8_0: 2/2 correct; greedy@Q4_K_M: 1/2 correct
    store.append(_rec(sampler="greedy", quant="Q8_0", correct=True, item_id="i0"))
    store.append(_rec(sampler="greedy", quant="Q8_0", correct=True, item_id="i1"))
    store.append(_rec(sampler="greedy", quant="Q4_K_M", correct=True, item_id="i0"))
    store.append(
        _rec(sampler="greedy", quant="Q4_K_M", correct=False, parse_method="failed", item_id="i1")
    )

    table = SamplerQuantTable.from_jsonl(tmp_path / "p.jsonl")
    assert table.cells[("gsm8k", "greedy", "Q8_0")].accuracy == 1.0
    assert table.cells[("gsm8k", "greedy", "Q4_K_M")].accuracy == 0.5
    assert table.cells[("gsm8k", "greedy", "Q4_K_M")].parse_failures == 1


def test_quant_columns_in_precision_order(tmp_path):
    store = RecordStore(tmp_path / "p.jsonl")
    # insert out of order; render order should be Q8 -> Q6 -> Q4 -> Q3
    for quant in ("Q3_K_M", "Q8_0", "Q4_K_M"):
        store.append(_rec(sampler="greedy", quant=quant, correct=True, item_id="i0"))
    table = SamplerQuantTable.from_jsonl(tmp_path / "p.jsonl")
    assert table.quants() == ["Q8_0", "Q4_K_M", "Q3_K_M"]


def test_render_runs_and_includes_task_header(tmp_path):
    store = RecordStore(tmp_path / "p.jsonl")
    store.append(_rec(sampler="min_p_min_p0.05", quant="Q8_0", correct=True, item_id="i0"))
    out = SamplerQuantTable.from_jsonl(tmp_path / "p.jsonl").render()
    assert "== gsm8k ==" in out
    assert "min_p_min_p0.05" in out


def test_temperature_table_pools_over_quant(tmp_path):
    store = RecordStore(tmp_path / "p.jsonl")
    # same sampler+temp across two quants must pool into one (sampler, temp) cell
    store.append(
        _rec(sampler="temperature", quant="Q8_0", correct=True, item_id="i0", temperature=1.3)
    )
    store.append(
        _rec(sampler="temperature", quant="Q4_K_M", correct=False, item_id="i0", temperature=1.3)
    )
    table = SamplerTemperatureTable.from_jsonl(tmp_path / "p.jsonl")
    cell = table.cells[("gsm8k", "temperature", 1.3)]
    assert cell.n == 2
    assert cell.accuracy == 0.5


def test_temperature_columns_sorted(tmp_path):
    store = RecordStore(tmp_path / "p.jsonl")
    for temp in (1.3, 0.7, 1.0):
        store.append(
            _rec(sampler="min_p", quant="Q8_0", correct=True, item_id="i0", temperature=temp)
        )
    table = SamplerTemperatureTable.from_jsonl(tmp_path / "p.jsonl")
    assert table.temperatures() == [0.7, 1.0, 1.3]
    assert "(pooled over quant)" in table.render()
