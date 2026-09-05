#!/usr/bin/env python3
"""Positive bound on the sampler effect inside the deployment band (ARR revision, Tier 2).

For every model x task x T in {0.7, 1.0, 1.3}, and every truncation arm, compute the paired
item-level accuracy difference (arm minus plain temperature). Per item the accuracy is the mean
over its K=3 repetitions and, unless --q8 is given, over the four quantization levels; the
difference is paired because both arms ran the same 50 items. An item-clustered bootstrap
(B=5000) gives the 95% CI of the mean difference. The bound on what a sampler can buy is the
largest upper CI limit across the six truncation arms (temperature-first arms; the two
temperature-last ablation arms are reported separately). A TOST-style equivalence call at margin
delta holds when the whole CI lies inside [-delta, +delta].

Writes results/sampler_bound.json and prints a per-model table. CPU, ~1 min.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

ARMS = ["top_p_top_p0.95", "top_k_top_k40", "min_p_min_p0.05", "top_n_sigma_top_n_sigma1.0"]
ABL = ["top_p_top_p0.95_tlast", "min_p_min_p0.05_tlast"]
SHORT = {"top_p_top_p0.95": "top-p", "top_k_top_k40": "top-k", "min_p_min_p0.05": "min-p",
         "top_n_sigma_top_n_sigma1.0": "top-nσ", "top_p_top_p0.95_tlast": "top-p(tl)",
         "min_p_min_p0.05_tlast": "min-p(tl)"}
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def load(q8only):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (model,task,T,arm) -> item -> [c,n]
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["sampler"] == "greedy" or (q8only and r["quant"] != "Q8_0"):
                continue
            a = acc[(r["model"], r["task"], r["temperature"], r["sampler"])][r["item_id"]]
            a[0] += int(r["correct"])
            a[1] += 1
    return acc


def item_vec(d):
    items = sorted(d)
    return items, np.array([d[i][0] / d[i][1] for i in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--delta", type=float, default=5.0, help="equivalence margin in pp")
    ap.add_argument("--q8", action="store_true", help="Q8 only instead of pooling quant levels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    acc = load(args.q8)
    out = []
    for m in MODELS:
        for task in ("gsm8k", "mmlu_pro"):
            for T in (0.7, 1.0, 1.3):
                items, base = item_vec(acc[(m, task, T, "temperature")])
                for arm in ARMS + ABL:
                    d = acc.get((m, task, T, arm))
                    if not d:
                        continue
                    it2, x = item_vec(d)
                    assert it2 == items
                    diff = (x - base) * 100
                    n = len(diff)
                    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(args.reps)])
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    out.append(dict(model=m, task=task, T=T, arm=arm, mean=float(diff.mean()),
                                    lo=float(lo), hi=float(hi), n=n,
                                    equivalent=bool(lo > -args.delta and hi < args.delta)))
    json.dump(out, open("results/sampler_bound.json", "w"), indent=1)

    print(f"paired arm-minus-temperature accuracy difference (pp), item-clustered 95% CI, "
          f"{'Q8 only' if args.q8 else 'pooled over quant'}; delta={args.delta}")
    print(f"{'model':<26}{'task':<9}{'T':>4}  " + "".join(f"{SHORT[a]:>19}" for a in ARMS + ABL))
    for m in MODELS:
        for task in ("gsm8k", "mmlu_pro"):
            for T in (0.7, 1.0, 1.3):
                row = [o for o in out if o["model"] == m and o["task"] == task and o["T"] == T]
                cells = []
                for a in ARMS + ABL:
                    o = next((r for r in row if r["arm"] == a), None)
                    cells.append(f"{o['mean']:+5.1f} [{o['lo']:+5.1f},{o['hi']:+5.1f}]" if o else "".rjust(19))
                print(f"{m:<26}{task:<9}{T:>4}  " + "".join(c.rjust(19) for c in cells))
    print("\nBOUND: largest upper 95% limit over the four truncation arms, all T<=1.3, per model x task:")
    for m in MODELS:
        for task in ("gsm8k", "mmlu_pro"):
            rows = [o for o in out if o["model"] == m and o["task"] == task and o["arm"] in ARMS]
            hi = max(o["hi"] for o in rows)
            lo = min(o["lo"] for o in rows)
            worst = max(rows, key=lambda o: o["hi"])
            alleq = all(o["equivalent"] for o in rows)
            print(f"  {m:<26}{task:<9} max upper {hi:+5.1f} (at {SHORT[worst['arm']]}, T={worst['T']}), "
                  f"min lower {lo:+5.1f}, all 12 CIs inside ±{args.delta}: {alleq}")


if __name__ == "__main__":
    main()
