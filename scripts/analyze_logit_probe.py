#!/usr/bin/env python3
"""Analyze the logit probe (results/logit_probe/) against temperature fragility.

Per model: mean over the 100 frozen prompts of the per-prompt greedy-path stats
(entropy lower bound, top-1 prob, top-1 minus top-2 margin, fraction of positions with
top-1 < 0.5). Fragility = the matrix's MMLU-Pro pure-temperature drop T0.7->T1.3
(recomputed here from the shards, not hard-coded). Reports the table + Spearman rank
correlations. CPU, seconds.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def mean(v):
    return sum(v) / len(v)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, j in enumerate(order):
            r[j] = i
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx = my = (n - 1) / 2
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def fragility():
    """MMLU-Pro pure-temperature drop T0.7 -> T1.3 (pp), pooled over quants."""
    acc = defaultdict(lambda: [0, 0])
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["task"] == "mmlu_pro" and r["sampler"] == "temperature" \
                    and r["temperature"] in (0.7, 1.3):
                a = acc[(r["model"], r["temperature"])]
                a[0] += int(r["correct"])
                a[1] += 1
    out = {}
    for m in MODELS:
        lo, hi = acc[(m, 0.7)], acc[(m, 1.3)]
        out[m] = (lo[0] / lo[1] - hi[0] / hi[1]) * 100
    return out


def probe_stats():
    stats = defaultdict(lambda: defaultdict(list))
    for f in glob.glob("results/logit_probe/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for k in ("mean_entropy_lb", "mean_top1", "mean_margin", "frac_flat_pos"):
                stats[r["model"]][k].append(r[k])
    return stats


def main():
    drop = fragility()
    stats = probe_stats()
    rows = []
    for m in MODELS:
        d = stats[m]
        if not d:
            continue
        rows.append((m, mean(d["mean_entropy_lb"]), mean(d["mean_top1"]),
                     mean(d["mean_margin"]), mean(d["frac_flat_pos"]), drop[m]))
    rows.sort(key=lambda r: -r[-1])
    print(f"{'model':<26} {'H(lb)':>6} {'top1':>6} {'margin':>7} {'flat%':>6} {'drop pp':>8}")
    for m, h, t1, mg, fl, dr in rows:
        print(f"{m:<26} {h:6.3f} {t1:6.3f} {mg:7.3f} {fl*100:5.1f}% {dr:8.1f}")
    ent = [r[1] for r in rows]
    flat = [r[4] for r in rows]
    mar = [r[3] for r in rows]
    dr = [r[5] for r in rows]
    print(f"\nSpearman(entropy, drop) = {spearman(ent, dr):+.2f}")
    print(f"Spearman(flat%,   drop) = {spearman(flat, dr):+.2f}")
    print(f"Spearman(margin,  drop) = {spearman(mar, dr):+.2f}")


if __name__ == "__main__":
    main()
