#!/usr/bin/env python
"""Sensitivity sweep for the single-token perturbation probe (section 4.6, Tier 3).

perturb_probe.py tested ONE configuration: force the rank-20 candidate at position 48,
continue greedily, measure the T=1.3 escape hazard over the next 32 positions. Reviewers
asked whether the Llama-vs-robust multiplier depends on that choice. This script varies
each design knob around the anchor, one at a time, from the same greedy prefix:

  position   inject the rank-20 candidate at position 16 / 48 / 80
  rank       inject the rank-2 / 5 / 10 / 20 candidate at position 48
  count      inject rank-20 candidates 1 / 2 / 4 times, 16 positions apart from 48

The pos48 / rank20 / count1 arm is the anchor and repeats the original configuration,
which also gives a determinism check against results/perturbation/ (that run used a
single server slot; this one uses `--parallel` slots, so continuous batching may perturb
low-order bits and occasionally flip a greedy path).

Per prompt, all arms share one greedy 128-token prefix decode with n_probs=20. A control
arm (unedited prefix continued greedily) is run per injection position. Multi-injection
arms compare against the position-48 control. Records: one JSON line per prompt holding
every arm; skipped arms (prefix terminated before the injection point) are null.

  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
  export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
  .venv/bin/python scripts/perturb_sweep.py [--only-model M] [--limit N] [--parallel 4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from perturb_probe import CONT_LEN, PREFIX_LEN, TOPN, head_stats  # noqa: E402

from decoding_robustness.config import load_experiment_config  # noqa: E402
from decoding_robustness.config.schema import QuantLevel  # noqa: E402
from decoding_robustness.inference import ChatTemplate, llama_server  # noqa: E402
from decoding_robustness.tasks import load_task  # noqa: E402

POSITIONS = (16, 48, 80)
RANKS = (2, 5, 10, 20)
COUNTS = (1, 2, 4)
ANCHOR_POS = 48
ANCHOR_RANK = 20
SPACING = 16  # positions between successive injections in the count arms
MODEL_ORDER = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
               "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]

GREEDY = {"samplers": ["temperature"], "temperature": 0.0, "seed": 0, "cache_prompt": True,
          "n_probs": TOPN, "post_sampling_probs": False}


def ranked(cands: list[dict], rank: int) -> dict:
    """The rank-th most likely candidate (rank 1 = argmax). Falls back to the last one."""
    ordered = sorted(cands, key=lambda t: t["logprob"], reverse=True)
    return ordered[min(rank, len(ordered)) - 1]


def arm_stats(entries: list[dict]) -> dict:
    return {"len": len(entries), "capped": len(entries) >= CONT_LEN, **head_stats(entries),
            "head_text": "".join(e["token"] for e in entries[:24])}


def continue_from(client, text: str, n_predict: int = CONT_LEN) -> list[dict]:
    r = client.completion(text, {**GREEDY, "n_predict": n_predict})
    return r.raw.get("completion_probabilities", [])


def run_prompt(client, rendered: str, entries: list[dict]) -> dict:
    """Build every arm for one prompt from its greedy prefix; returns the arms dict."""
    n = len(entries)
    jobs: dict[str, tuple[str, str]] = {}  # arm key -> (text, kind)

    def prefix_text(pos: int) -> str:
        return "".join(e["token"] for e in entries[:pos])

    inj_meta: dict[str, dict] = {}
    for pos in POSITIONS:
        if n <= pos:
            continue
        tok = ranked(entries[pos]["top_logprobs"], ANCHOR_RANK)
        inj_meta[f"pos{pos}"] = {"inject_at": pos, "rank": ANCHOR_RANK, "inj_token": tok["token"],
                                 "inj_prob": math.exp(tok["logprob"])}
        jobs[f"pos{pos}|inj"] = (rendered + prefix_text(pos) + tok["token"], "long")
        jobs[f"pos{pos}|ctrl"] = (rendered + prefix_text(pos), "long")
    if n > ANCHOR_POS:
        for rank in RANKS:
            if rank == ANCHOR_RANK:
                continue
            tok = ranked(entries[ANCHOR_POS]["top_logprobs"], rank)
            inj_meta[f"rank{rank}"] = {"inject_at": ANCHOR_POS, "rank": rank,
                                       "inj_token": tok["token"],
                                       "inj_prob": math.exp(tok["logprob"])}
            jobs[f"rank{rank}|inj"] = (rendered + prefix_text(ANCHOR_POS) + tok["token"], "long")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {k: pool.submit(continue_from, client, text) for k, (text, _) in jobs.items()}
        # Multi-injection arms are sequential chains; run them alongside the batch.
        multi_futs = {}
        if n > ANCHOR_POS:
            for count in COUNTS:
                if count == 1:
                    continue
                multi_futs[f"count{count}"] = pool.submit(
                    multi_inject, client, rendered, prefix_text(ANCHOR_POS),
                    entries[ANCHOR_POS]["top_logprobs"], count)
        results = {k: f.result() for k, f in futs.items()}
        multi = {k: f.result() for k, f in multi_futs.items()}

    arms: dict = {}
    for pos in POSITIONS:
        key = f"pos{pos}"
        arms[key] = None if key not in inj_meta else {
            **inj_meta[key], "inj": arm_stats(results[f"{key}|inj"]),
            "ctrl": arm_stats(results[f"{key}|ctrl"])}
    for rank in RANKS:
        key = f"rank{rank}"
        if rank == ANCHOR_RANK:
            arms[key] = arms["pos48"]  # same arm; duplicated for uniform access
            continue
        arms[key] = None if key not in inj_meta else {
            **inj_meta[key], "inj": arm_stats(results[f"{key}|inj"])}
    for count in COUNTS:
        key = f"count{count}"
        if count == 1:
            arms[key] = arms["pos48"]
            continue
        arms[key] = multi.get(key)
    return arms


def multi_inject(client, rendered: str, prefix: str, first_cands: list[dict],
                 count: int) -> dict | None:
    """Inject rank-20 at 48, then again every SPACING positions, `count` times in all."""
    tok = ranked(first_cands, ANCHOR_RANK)
    text = prefix + tok["token"]
    injected = [{"inject_at": ANCHOR_POS, "inj_token": tok["token"],
                 "inj_prob": math.exp(tok["logprob"])}]
    pos = ANCHOR_POS
    for _ in range(count - 1):
        step = continue_from(client, rendered + text, n_predict=SPACING)
        if len(step) < SPACING:
            break  # model terminated inside the gap; record how far we got
        text += "".join(e["token"] for e in step[:SPACING - 1])
        tok = ranked(step[SPACING - 1]["top_logprobs"], ANCHOR_RANK)
        text += tok["token"]
        pos += SPACING
        injected.append({"inject_at": pos, "inj_token": tok["token"],
                         "inj_prob": math.exp(tok["logprob"])})
    final = continue_from(client, rendered + text)
    return {"count_requested": count, "count_done": len(injected), "last_inject_at": pos,
            "injections": injected, "inj": arm_stats(final)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_matrix.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max prompts per task (smoke)")
    ap.add_argument("--out-dir", default="results/perturbation_sweep")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    config = load_experiment_config(args.config)
    config = config.model_copy(update={"server": config.server.model_copy(
        update={"parallel": args.parallel, "n_ctx": args.n_ctx})})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    ckpts = {c.name: c for c in config.checkpoints}
    order = [m for m in MODEL_ORDER if m in ckpts] + [m for m in ckpts if m not in MODEL_ORDER]
    for name in order:
        if args.only_model and name not in args.only_model.split(","):
            continue
        ckpt = ckpts[name]
        fname = ckpt.quant_files.get(QuantLevel.Q8_0)
        path = model_dir / fname if fname else None
        if path is None or not path.is_file():
            print(f"-- {name}: no Q8_0 GGUF, skipped", flush=True)
            continue
        out_path = out_dir / f"{name}.jsonl"
        done = set()
        if out_path.is_file():
            done = {json.loads(line)["key"] for line in open(out_path) if line.strip()}
        template = ChatTemplate.from_meta_file(path.with_suffix(path.suffix + ".meta.json"))
        print(f"== {name} (Q8_0, parallel={args.parallel}) -> {out_path}", flush=True)
        t0 = time.time()
        with llama_server(path, config.server, log_file=out_dir / f"{name}.server.log") as client:
            wrote = skipped = 0
            for task in config.tasks:
                items = list(load_task(task))
                if args.limit:
                    items = items[: args.limit]
                for item in items:
                    key = f"{name}|{task.name}|{item.item_id}"
                    if key in done:
                        continue
                    rendered = template.render([{"role": "user", "content": item.prompt}],
                                               enable_thinking=False)
                    r = client.completion(rendered, {**GREEDY, "n_predict": PREFIX_LEN})
                    entries = r.raw.get("completion_probabilities", [])
                    if len(entries) <= POSITIONS[0]:
                        skipped += 1
                        continue
                    arms = run_prompt(client, rendered, entries)
                    record = {"key": key, "model": name, "task": task.name,
                              "item_id": item.item_id, "prefix_len": len(entries),
                              "prefix_terminated": len(entries) < PREFIX_LEN, "arms": arms}
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    wrote += 1
                    if wrote % 10 == 0:
                        print(f"   ... {wrote} prompts ({time.time() - t0:.0f}s)", flush=True)
            print(f"   done: {wrote} new prompts, {skipped} skipped, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
