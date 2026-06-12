#!/usr/bin/env python3
"""Analyze the 2026-06-10 follow-up runs: stratified MMLU-Pro check + cliff-edge ladder.

[S] Stratified check (results/mmlu_strat_check/strat.jsonl): does the pure-temperature
    collapse ordering reproduce on a category-stratified MMLU-Pro N=50 (vs the frozen
    matrix's all-business head-50)? Reports accuracy by model x T, the T0.7->T1.3 drop
    with an item-clustered bootstrap CI, and the same numbers recomputed from the matrix
    (Q8 only, same conditions) side by side.

[L] Cliff ladder (results/cliff_ladder/ladder.jsonl): accuracy and strict-parse rate
    (the degeneration meter) by model x sampler x T over 1.3/1.5/1.7/2.0, joined to the
    matrix's Q8 T0.7/1.0 rungs as the baseline anchor. Bootstrap CI per rung for the
    drop vs the model's own T0.7 anchor. The "cliff" reading is left to the table: look
    for the first rung where the CI excludes a small drop.

Pure stdlib + numpy; CPU, seconds.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

B = 2000
RNG = np.random.default_rng(0)
SAMPLER_LABEL = {
    "greedy": "greedy", "temperature": "temperature",
    "top_p_top_p0.95": "top_p", "min_p_min_p0.05": "min_p",
    "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
}
MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]


def load(paths_glob, *, q8_only=False):
    """(model, task, sampler, temp) -> item -> [correct]; plus strict-parse counters."""
    by = defaultdict(lambda: defaultdict(list))
    strict = defaultdict(lambda: [0, 0])  # key -> [n_strict, n]
    for path in glob.glob(paths_glob):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if q8_only and r["quant"] != "Q8_0":
                continue
            lab = SAMPLER_LABEL.get(r["sampler"])
            if lab is None:
                continue
            key = (r["model"], r["task"], lab, r["temperature"])
            by[key][r["item_id"]].append(int(r["correct"]))
            s = strict[key]
            s[0] += int(r["parse_method"] == "strict")
            s[1] += 1
    return by, strict


def acc(item_map, items):
    num = den = 0
    for it in items:
        vals = item_map.get(it)
        if vals:
            num += sum(vals)
            den += len(vals)
    return num / den if den else float("nan")


def boot_drop(map_a, map_b):
    items = np.array(sorted(set(map_a) | set(map_b)), dtype=object)
    if not len(items):
        return None
    point = acc(map_a, items) - acc(map_b, items)
    draws = [
        acc(map_a, s) - acc(map_b, s)
        for s in (RNG.choice(items, size=len(items), replace=True) for _ in range(B))
    ]
    return point, np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5)


def fmt_drop(r):
    if r is None:
        return "      --"
    d, lo, hi = r
    return f"{d*100:+5.1f}pp [{lo*100:+5.1f},{hi*100:+5.1f}]"


def main():
    print("=" * 96)
    print("[S] STRATIFIED MMLU-PRO CHECK vs the all-business matrix subset (Q8, pure temperature)")
    print("=" * 96)
    strat, _ = load("results/mmlu_strat_check/*.jsonl")
    matrix, _ = load("results/full_matrix/shards/*.jsonl", q8_only=True)
    print(f"{'model':<26} {'subset':<12} {'acc@0.7':>8} {'acc@1.0':>8} {'acc@1.3':>8}"
          f"  {'drop 0.7->1.3 (95% CI)':>24}")
    for m in MODELS:
        for name, src in (("stratified", strat), ("business", matrix)):
            row = [acc(src[(m, "mmlu_pro", "temperature", t)],
                       list(src[(m, "mmlu_pro", "temperature", t)])) for t in (0.7, 1.0, 1.3)]
            r = boot_drop(src[(m, "mmlu_pro", "temperature", 0.7)],
                          src[(m, "mmlu_pro", "temperature", 1.3)])
            cells = "".join(f" {v*100:7.1f}" for v in row)
            print(f"{m:<26} {name:<12}{cells}  {fmt_drop(r):>24}")
        g = acc(strat[(m, "mmlu_pro", "greedy", 0.0)], list(strat[(m, "mmlu_pro", "greedy", 0.0)]))
        print(f"{'':<26} {'(strat greedy':<12} {g*100:7.1f})")

    print()
    print("=" * 96)
    print("[L] CLIFF LADDER (Q8): accuracy / strict-parse%  by model x sampler x T")
    print("    matrix Q8 rungs T0.7/1.0 shown as anchors; drop CIs are vs the model's T0.7 anchor")
    print("=" * 96)
    ladder, lstrict = load("results/cliff_ladder/*.jsonl")
    mstrict: dict = {}
    matrix2, mstrict = load("results/full_matrix/shards/*.jsonl", q8_only=True)
    temps_anchor = [0.7, 1.0]
    temps_ladder = [1.3, 1.5, 1.7, 2.0]
    for task in ("gsm8k", "mmlu_pro"):
        print(f"\n  TASK {task}")
        head = (f"  {'model':<26} {'sampler':<12}"
                + "".join(f" {f'T{t}':>11}" for t in temps_anchor + temps_ladder))
        print(head + "   (acc% / strict%)")
        for m in MODELS:
            for s in ("temperature", "top_p", "min_p", "top_n_sigma"):
                cells = []
                for t in temps_anchor:
                    key = (m, task, s, t)
                    a = acc(matrix2[key], list(matrix2[key]))
                    sp = mstrict[key]
                    cells.append(f"{a*100:4.0f}/{sp[0]/sp[1]*100:3.0f}" if sp[1] else "    --  ")
                for t in temps_ladder:
                    key = (m, task, s, t)
                    a = acc(ladder[key], list(ladder[key]))
                    sp = lstrict[key]
                    cells.append(f"{a*100:4.0f}/{sp[0]/sp[1]*100:3.0f}" if sp[1] else "    --  ")
                print(f"  {m:<26} {s:<12}" + "".join(f" {c:>11}" for c in cells))
        print(f"\n  drops vs T0.7 anchor (95% CI), {task}, pure temperature:")
        for m in MODELS:
            anchor = matrix2[(m, task, "temperature", 0.7)]
            row = "  ".join(
                f"T{t}: {fmt_drop(boot_drop(anchor, ladder[(m, task, 'temperature', t)]))}"
                for t in temps_ladder
            )
            print(f"    {m:<26} {row}")


if __name__ == "__main__":
    main()
