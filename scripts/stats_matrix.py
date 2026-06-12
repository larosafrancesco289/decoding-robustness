#!/usr/bin/env python3
"""Uncertainty quantification for the full matrix via clustered nonparametric bootstrap.

At N=50/item the right uncertainty story is a bootstrap that resamples ITEMS (the cluster),
not individual generations -- the K reps (and the pooled quants) of one item are correlated.
This is the min-p-critique's ask (proper CIs / power) without a fragile GLMM. Paired where the
quantity compares two temperatures on the same items.

Reports, per model:
  [A] Pure-temperature collapse: accuracy drop T0.7 -> T1.3 (the headline), 95% CI. Llama's CI
      should exclude 0 by a mile; Qwen3's should straddle 0 (a real bounded null).
  [B] Sampler spread at T=1.3: max-min accuracy across the non-greedy samplers, 95% CI -- the
      practitioner number "how much does sampler choice matter at high T".
  [C] Self-consistency: mean distinct parsed answers per item under pure temperature at T=1.3.
  [D] Quant effect: greedy accuracy drop Q8_0 -> Q3_K_M, 95% CI.

Pure stdlib + numpy. Pools over quant (cluster = item_id) unless noted.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

SHARDS = "results/full_matrix/shards/*.jsonl"
B = 2000
RNG = np.random.default_rng(0)
SAMPLER_LABEL = {
    "greedy": "greedy", "temperature": "temperature",
    "top_p_top_p0.95": "top_p", "top_k_top_k40": "top_k",
    "min_p_min_p0.05": "min_p", "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
    "min_p_min_p0.05_tlast": "min_p(tlast)", "top_p_top_p0.95_tlast": "top_p(tlast)",
}
NONGREEDY = ["temperature", "top_k", "top_p", "min_p", "top_n_sigma",
             "top_p(tlast)", "min_p(tlast)"]
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "gemma-3-12b-it",
          "qwen2.5-7b-instruct", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]
TASKS = ["gsm8k", "mmlu_pro"]


def load():
    by = defaultdict(lambda: defaultdict(list))  # (model,task,sampler,temp) -> item -> [correct]
    ans = defaultdict(lambda: defaultdict(list))  # (model,task,temp) -> (item,quant) -> [parsed]
    for f in glob.glob(SHARDS):
        for line in open(f):
            r = json.loads(line)
            lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
            key = (r["model"], r["task"], lab, r["temperature"])
            by[key][r["item_id"]].append(int(r["correct"]))
            if lab == "temperature":
                ans[(r["model"], r["task"], r["temperature"])][(r["item_id"], r["quant"])].append(
                    r["parsed_answer"])
    return by, ans


def acc_from(item_map, items):
    """Pooled accuracy over the given (possibly repeated) item ids."""
    num = den = 0
    for it in items:
        vals = item_map.get(it)
        if vals:
            num += sum(vals)
            den += len(vals)
    return num / den if den else float("nan")


def ci(vals):
    a = np.asarray(vals)
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


def boot_drop(map_a, map_b):
    """Paired bootstrap of acc(a) - acc(b) over shared items (cluster = item)."""
    items = sorted(set(map_a) | set(map_b))
    if not items:
        return None
    items = np.array(items, dtype=object)
    point = acc_from(map_a, items) - acc_from(map_b, items)
    draws = []
    for _ in range(B):
        samp = RNG.choice(items, size=len(items), replace=True)
        draws.append(acc_from(map_a, samp) - acc_from(map_b, samp))
    lo, hi = ci(draws)
    return point, lo, hi


def boot_spread(maps_by_sampler, items):
    """Bootstrap of (max-min accuracy across samplers) at fixed temp; cluster = item."""
    items = np.array(sorted(items), dtype=object)
    def spread(it):
        accs = [acc_from(m, it) for m in maps_by_sampler.values()]
        accs = [a for a in accs if a == a]
        return (max(accs) - min(accs)) if accs else float("nan")
    point = spread(items)
    draws = [spread(RNG.choice(items, size=len(items), replace=True)) for _ in range(B)]
    lo, hi = ci(draws)
    return point, lo, hi


def boot_selfconsistency(group_map):
    """Mean distinct parsed answers/item-group under pure temperature; cluster = item."""
    # group_map: (item,quant) -> [parsed]. cluster by item id.
    by_item = defaultdict(list)
    for (it, _quant), parsed in group_map.items():
        by_item[it].append(len(set(parsed)))
    items = np.array(sorted(by_item), dtype=object)
    if not len(items):
        return None

    def flat(its):
        return np.mean([v for it in its for v in by_item[it]])

    point = flat(items)
    draws = [flat(RNG.choice(items, size=len(items), replace=True)) for _ in range(B)]
    lo, hi = ci(draws)
    return point, lo, hi


def pp(x):
    return f"{x*100:+.1f}"


def main():
    by, ans = load()
    models = [m for m in MODELS if any(k[0] == m for k in by)]
    print(f"Clustered bootstrap (B={B}, resample items). Pooled over quant unless noted.\n")

    for task in TASKS:
        print("=" * 80)
        print(f"TASK: {task}")
        print("=" * 80)
        print("\n[A] Pure-temperature collapse: accuracy drop T0.7 -> T1.3 (95% CI)")
        print("    positive = accuracy fell; CI excluding 0 = real collapse")
        for m in models:
            r = boot_drop(by[(m, task, "temperature", 0.7)], by[(m, task, "temperature", 1.3)])
            if r is None:
                continue
            d, lo, hi = r
            null = "  (null: CI spans 0)" if lo <= 0 <= hi else ""
            flag = "  <-- collapse" if lo > 0.05 else null
            print(f"    {m:<26} {pp(d)}pp  [{pp(lo)}, {pp(hi)}]{flag}")

        print("\n[B] Sampler spread at T=1.3: max-min accuracy across 7 samplers (95% CI)")
        print("    small = sampler choice barely matters")
        for m in models:
            maps = {s: by[(m, task, s, 1.3)] for s in NONGREEDY if by[(m, task, s, 1.3)]}
            if len(maps) < 2:
                continue
            items = set().union(*[set(mp) for mp in maps.values()])
            p, lo, hi = boot_spread(maps, items)
            print(f"    {m:<26} {p*100:4.1f}pp  [{lo*100:.1f}, {hi*100:.1f}]")

        print("\n[C] Self-consistency under pure temperature @T=1.3: mean distinct answers/item")
        print("    1.0 = all reps agree (robust); ->K = reps disagree (temp injecting errors)")
        for m in models:
            r = boot_selfconsistency(ans[(m, task, 1.3)])
            if r is None:
                continue
            p, lo, hi = r
            print(f"    {m:<26} {p:.2f}   [{lo:.2f}, {hi:.2f}]")

    print("\n" + "=" * 80)
    print("[D] Quant effect: greedy accuracy drop Q8_0 -> Q3_K_M (95% CI), pooled tasks")
    print("=" * 80)
    # rebuild greedy maps split by quant
    g = defaultdict(lambda: defaultdict(list))  # (model,quant) -> item -> [correct]
    for f in glob.glob(SHARDS):
        for line in open(f):
            r = json.loads(line)
            if SAMPLER_LABEL.get(r["sampler"]) == "greedy":
                g[(r["model"], r["quant"])][f"{r['task']}|{r['item_id']}"].append(int(r["correct"]))
    for m in models:
        a8, a3 = g[(m, "Q8_0")], g[(m, "Q3_K_M")]
        if not a8 or not a3:
            continue
        r = boot_drop(a8, a3)
        d, lo, hi = r
        print(f"    {m:<26} {pp(d)}pp  [{pp(lo)}, {pp(hi)}]")


if __name__ == "__main__":
    main()
