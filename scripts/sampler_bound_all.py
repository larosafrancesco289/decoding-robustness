#!/usr/bin/env python3
"""Sampler-effect bound for the rewrite (2026-09-09): every truncation arm vs plain temperature, paired on items.

Main grid: 7 models, pooled over quantization levels (item accuracy = mean over K and levels).
Flagged-model grid: Hermes-3, Qwen3.5-9B, OLMo-3-7B at Q8 (150 records per cell).
Per comparison: paired item-level difference in pp with an item-clustered bootstrap CI (B=5000).
Simultaneous bound: 95th percentile of the bootstrap distribution of the LARGEST mean gain over a set of
comparisons (items resampled jointly per task). Writes results/sampler_bound_all.json and prints a summary.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

B, SEED = 5000, 0
TOL = 1e-9  # numerical zero: an interval endpoint within TOL of 0 counts as touching zero
ARMS = ["top_p", "top_k", "min_p", "top_n_sigma"]
GRID = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", "gemma-3-12b-it",
        "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]
PANEL = ["hermes-3-llama-3.1-8b", "qwen3.5-9b", "olmo-3-7b-instruct"]


def item_acc(df):
    return df.groupby("item_id").correct.mean()


def main():
    df = pd.read_parquet("results/paper_records.parquet")
    df = df[df.chain == "tfirst"]
    out = []
    diffs = {}  # (source, model, task, T, arm) -> np.array of per-item differences (aligned on sorted items)
    for src, models in (("main_grid", GRID), ("main_grid_q8", GRID), ("panel_grid", PANEL)):
        d = df[df.source == src.replace("_q8", "")]
        if src.endswith("_q8"):
            d = d[d.quant == "Q8_0"]
        for m in models:
            for task in ("gsm8k", "mmlu_pro"):
                for T in (0.7, 1.0, 1.3):
                    base = item_acc(d[(d.model == m) & (d.task == task) & (d["T"] == T) & (d.sampler == "temperature")])
                    for arm in ARMS:
                        x = item_acc(d[(d.model == m) & (d.task == task) & (d["T"] == T) & (d.sampler == arm)])
                        assert list(x.index) == list(base.index) and len(x) == 50, (m, task, T, arm, len(x))
                        diffs[(src, m, task, T, arm)] = (x.values - base.values) * 100
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 50, (B, 50))
    for k, v in diffs.items():
        boots = v[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(dict(source=k[0], model=k[1], task=k[2], T=k[3], arm=k[4], mean=float(v.mean()),
                        lo=float(lo), hi=float(hi)))
    json.dump(out, open("results/sampler_bound_all.json", "w"), indent=1)

    def summary(keys, label):
        rows = [o for o in out if (o["source"], o["model"], o["task"], o["T"], o["arm"]) in keys]
        mx = max(rows, key=lambda o: o["mean"])
        mu = max(rows, key=lambda o: o["hi"])
        above = sum(o["lo"] > TOL for o in rows)
        below = sum(o["hi"] < -TOL for o in rows)
        # simultaneous: same item draws for every comparison (items of the two tasks drawn independently but jointly across models/arms)
        mats = np.stack([diffs[k] for k in keys])  # (n_comp, 50)
        est = mats.mean(axis=1)  # observed mean gains
        bmeans = mats[:, idx].mean(axis=2)  # (n_comp, B) bootstrap means, items resampled jointly
        # max-t style simultaneous upper bound (Westfall-Young): max_i g_i + 95th pct of max_i (g_i^b - g_i)
        sim = float(est.max() + np.percentile((bmeans - est[:, None]).max(axis=0), 95))
        pct_max = float(np.percentile(bmeans.max(axis=0), 95))  # older construction, kept for the record
        within3 = sum(abs(o["mean"]) <= 3 for o in rows)
        print(f"{label}: n={len(rows)}  max gain {mx['mean']:+.1f} ({mx['model']} {mx['task']} T{mx['T']} {mx['arm']}); "
              f"max upper {mu['hi']:+.1f} ({mu['model']} {mu['task']} T{mu['T']} {mu['arm']}); simultaneous 95% upper bound on the largest gain (max-t) {sim:+.1f} (95th pct of max: {pct_max:+.1f}); "
              f"CIs above zero {above}, below {below}; |mean|<=3: {within3}")
        return dict(label=label, n=len(rows), max_gain=mx["mean"], max_upper=mu["hi"], simultaneous=sim, pct_max=pct_max, above=above, below=below, within3=within3)

    S = {}
    for src, models, name in (("main_grid", GRID, "main grid pooled"), ("panel_grid", PANEL, "flagged-model grid Q8")):
        for Ts, tl in (((0.7, 1.0), "T<=1.0"), ((1.3,), "T=1.3")):
            keys = [k for k in diffs if k[0] == src and k[3] in Ts]
            S[f"{name} {tl}"] = summary(keys, f"{name} {tl}")
        # robust-only main grid at T<=1.0 (exclude Llama)
        if src == "main_grid":
            keys = [k for k in diffs if k[0] == src and k[3] in (0.7, 1.0) and k[1] != "llama-3.1-8b-instruct"]
            S["main grid robust T<=1.0"] = summary(keys, "main grid, six robust models, T<=1.0")
            keys = [k for k in diffs if k[0] == src and k[3] in (0.7, 1.0) and k[1] == "llama-3.1-8b-instruct"]
            S["main grid llama T<=1.0"] = summary(keys, "main grid, Llama-3.1 only, T<=1.0")
    keys = [k for k in diffs if k[0] == "main_grid_q8" and k[3] in (0.7, 1.0) and k[1] != "llama-3.1-8b-instruct"]
    S["main grid robust Q8 T<=1.0"] = summary(keys, "main grid, six robust models, Q8 only, T<=1.0")
    keys = [k for k in diffs if k[0] in ("main_grid", "panel_grid") and k[3] in (0.7, 1.0)]
    S["all ten T<=1.0"] = summary(keys, "all ten models T<=1.0")
    json.dump(S, open("results/sampler_bound_all_summary.json", "w"), indent=1)
    print("\nper-model max gain / max upper at T<=1.0:")
    for src, models in (("main_grid", GRID), ("panel_grid", PANEL)):
        for m in models:
            rows = [o for o in out if o["source"] == src and o["model"] == m and o["T"] in (0.7, 1.0)]
            print(f"  {m:<26} max gain {max(o['mean'] for o in rows):+5.1f}  max upper {max(o['hi'] for o in rows):+5.1f}  min lower {min(o['lo'] for o in rows):+5.1f}")
    print("\nT=1.3 recovery, fragile models with grids (plain 0.7, plain 1.3, best arm and its acc, gain [CI] vs plain 1.3):")
    for src, models in (("main_grid", ["llama-3.1-8b-instruct"]), ("panel_grid", PANEL)):
        d = df[(df.source == src) & (df.model.isin(models))]
        if src == "main_grid":
            d = d[d.quant == "Q8_0"]
        for m in models:
            for task in ("gsm8k", "mmlu_pro"):
                p07 = d[(d.model == m) & (d.task == task) & (d["T"] == 0.7) & (d.sampler == "temperature")].correct.mean() * 100
                p13 = d[(d.model == m) & (d.task == task) & (d["T"] == 1.3) & (d.sampler == "temperature")].correct.mean() * 100
                base13 = item_acc(d[(d.model == m) & (d.task == task) & (d["T"] == 1.3) & (d.sampler == "temperature")])
                base07 = item_acc(d[(d.model == m) & (d.task == task) & (d["T"] == 0.7) & (d.sampler == "temperature")])
                best = None
                for arm in ARMS:
                    x = item_acc(d[(d.model == m) & (d.task == task) & (d["T"] == 1.3) & (d.sampler == arm)])
                    g = (x.values - base13.values) * 100
                    g07 = (x.values - base07.values) * 100
                    bo = g[idx].mean(axis=1); bo7 = g07[idx].mean(axis=1)
                    rec = dict(arm=arm, acc=x.mean() * 100, gain=g.mean(), lo=np.percentile(bo, 2.5), hi=np.percentile(bo, 97.5),
                               d07=g07.mean(), lo7=np.percentile(bo7, 2.5), hi7=np.percentile(bo7, 97.5))
                    if best is None or rec["gain"] > best["gain"]:
                        best = rec
                print(f"  {m:<24} {task:<9} plain {p07:5.1f} -> {p13:5.1f}; best {best['arm']:<12} {best['acc']:5.1f} "
                      f"gain {best['gain']:+5.1f} [{best['lo']:+5.1f},{best['hi']:+5.1f}]  vs 0.7 {best['d07']:+5.1f} [{best['lo7']:+5.1f},{best['hi7']:+5.1f}]")


if __name__ == "__main__":
    main()
