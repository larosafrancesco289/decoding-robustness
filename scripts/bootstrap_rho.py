#!/usr/bin/env python3
"""Item-clustered bootstrap CIs for the three mechanism Spearman correlations in Sec. 4.6.

Each rho pairs a per-model probe statistic (mean over the 100 frozen prompts) with the
model's MMLU-Pro plain-temperature drop T0.7 -> T1.3 (pooled over quantization levels and
repetitions, 50 items). The point estimate uses n=7 models, so rho cannot be bootstrapped
over models. Instead, each replicate resamples the measurement units with replacement:
the 100 probe prompts within each model for the probe statistic, and the 50 MMLU-Pro items
within each model for the drop (an item's K x quant repetitions travel together). rho is
recomputed on the replicate's seven means. The CI therefore reflects measurement noise
in both coordinates, holding the set of models fixed. Also reports the exact permutation
p-value over the 7! orderings for the point estimate. CPU, seconds.

Usage: python scripts/bootstrap_rho.py [--reps 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from collections import defaultdict

import numpy as np

MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def avg_rank(v):
    """Ranks with ties given their average rank (the standard Spearman convention)."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    s = v[order]
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0, 1])


def escape_cont(probs, temp=1.3):
    inv = 1.0 / temp
    p = np.asarray(probs, dtype=float)
    s_top = np.sum(p[p > 0] ** inv)
    t = max(0.0, 1.0 - p.sum())
    if t <= 0 or s_top <= 0:
        return 0.0
    p20 = p[-1]
    s_cont = (t / p20) * (p20 ** inv) if p20 > 0 else t ** inv
    return s_cont / (s_top + s_cont)


def load_drop_items():
    """model -> array (50,) of per-item [acc@0.7 - acc@1.3] on MMLU-Pro plain temperature."""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (model,T) -> item -> [c,n]
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["task"] == "mmlu_pro" and r["sampler"] == "temperature" \
                    and r["temperature"] in (0.7, 1.3):
                a = acc[(r["model"], r["temperature"])][r["item_id"]]
                a[0] += int(r["correct"])
                a[1] += 1
    out = {}
    for m in MODELS:
        lo, hi = acc[(m, 0.7)], acc[(m, 1.3)]
        items = sorted(lo)
        assert items == sorted(hi), m
        out[m] = np.array([lo[i][0] / lo[i][1] - hi[i][0] / hi[i][1] for i in items]) * 100
    return out


def load_probe():
    """model -> dict of per-prompt arrays: entropy (logit_probe), escape (logit_probe_v2),
    d_esc32 (perturbation, injected minus control)."""
    out = {m: {} for m in MODELS}
    for m in MODELS:
        ent = [json.loads(l)["mean_entropy_lb"]
               for l in open(f"results/logit_probe/{m}.jsonl", encoding="utf-8")]
        out[m]["entropy"] = np.array(ent)
        esc_sum, esc_n = [], []
        for l in open(f"results/logit_probe_v2/{m}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            e = [escape_cont(pos) for pos in r["probs"]]
            esc_sum.append(sum(e))
            esc_n.append(len(e))
        # position-pooled hazard (as in analyze_escape.py); stored as (sum, n) per prompt
        out[m]["escape"] = np.stack([np.array(esc_sum), np.array(esc_n)], axis=1)
        d = []
        for l in open(f"results/perturbation/{m}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if "esc32" in r["inj"] and "esc32" in r["ctrl"]:
                d.append(r["inj"]["esc32"] - r["ctrl"]["esc32"])
        out[m]["d_esc32"] = np.array(d)
    return out


def perm_p(x, y, observed):
    """Exact two-sided permutation p over all 7! orderings of y."""
    n = 0
    tot = 0
    for perm in itertools.permutations(range(len(y))):
        tot += 1
        if abs(spearman(x, [y[i] for i in perm])) >= abs(observed) - 1e-12:
            n += 1
    return n / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    drops = load_drop_items()
    probe = load_probe()
    drop_pt = np.array([drops[m].mean() for m in MODELS])
    print("point drops (pp):", dict(zip(MODELS, np.round(drop_pt, 1))))

    def stat_mean(v):
        return v[:, 0].sum() / v[:, 1].sum() if v.ndim == 2 else v.mean()

    for stat in ("entropy", "escape", "d_esc32"):
        x_pt = np.array([stat_mean(probe[m][stat]) for m in MODELS])
        print(f"  {stat} per-model means:", dict(zip(MODELS, np.round(x_pt, 5))))
        rho_pt = spearman(x_pt, drop_pt)
        boots = np.empty(args.reps)
        for b in range(args.reps):
            xb = np.empty(len(MODELS))
            db = np.empty(len(MODELS))
            for i, m in enumerate(MODELS):
                v = probe[m][stat]
                xb[i] = stat_mean(v[rng.integers(0, len(v), len(v))])
                w = drops[m]
                db[i] = w[rng.integers(0, len(w), len(w))].mean()
            boots[b] = spearman(xb, db)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = perm_p(list(x_pt), list(drop_pt), rho_pt)
        print(f"{stat:<8} rho = {rho_pt:+.2f}  95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]  "
              f"P(rho_b <= 0) = {np.mean(boots <= 0):.4f}  exact perm p = {p:.4f}  "
              f"(reps={args.reps})")


if __name__ == "__main__":
    main()
