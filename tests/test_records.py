"""Tests for the JSONL record store: deterministic ids + resumable id-skip (SPEC §7.2)."""

from __future__ import annotations

from decoding_robustness.runner.records import GenerationRecord, RecordStore, make_record_id

_IDENTITY = dict(
    model="llama-3.1-8b-instruct",
    quant="Q4_K_M",
    sampler="greedy",
    temperature=0.0,
    task="gsm8k",
    item_id="gsm8k-test-00000",
    repetition=1,
)


def _record(**overrides) -> GenerationRecord:
    identity = {**_IDENTITY, **overrides}
    return GenerationRecord(
        id=make_record_id(**identity),
        **identity,
        params={"samplers": ["temperature"], "temperature": identity["temperature"]},
        seed=12345,
        prompt_hash="deadbeef",
        raw_output="The final answer is 18.",
        parsed_answer="18",
        parse_method="strict",
        gold="18",
        correct=True,
        n_prompt_tokens=50,
        n_completion_tokens=20,
        timestamp="2026-06-01T00:00:00+00:00",
    )


def test_make_record_id_is_deterministic_and_identity_sensitive():
    base = make_record_id(**_IDENTITY)
    assert base == make_record_id(**_IDENTITY)  # stable
    assert make_record_id(**{**_IDENTITY, "item_id": "gsm8k-test-00001"}) != base
    assert make_record_id(**{**_IDENTITY, "repetition": 2}) != base


def test_append_and_read_roundtrip(tmp_path):
    store = RecordStore(tmp_path / "results.jsonl")
    record = _record()
    store.append(record)
    read_back = list(store.read_all())
    assert len(read_back) == 1
    assert read_back[0].id == record.id
    assert read_back[0].correct is True


def test_existing_ids_enables_resume(tmp_path):
    store = RecordStore(tmp_path / "results.jsonl")
    assert store.existing_ids() == set()  # nothing yet

    first = _record(item_id="gsm8k-test-00000")
    second = _record(item_id="gsm8k-test-00001")
    store.append(first)
    store.append(second)

    ids = store.existing_ids()
    assert ids == {first.id, second.id}


def test_store_creates_parent_dir(tmp_path):
    store = RecordStore(tmp_path / "nested" / "dir" / "out.jsonl")
    store.append(_record())
    assert (tmp_path / "nested" / "dir" / "out.jsonl").is_file()
