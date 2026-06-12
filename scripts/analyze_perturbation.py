#!/usr/bin/env python3
"""Analyze the single-token perturbation probe (results/perturbation/).

Per model, comparing the injected arm (greedy continuation after forcing the rank-20
candidate at position 48) against the control arm (greedy continuation of the unedited
prefix):

  GSM8K cap%        fraction of continuations that ran to the 256-token cap. GSM8K
                    terminates well under 256 tokens when healthy, so cap = derailed.
                    (MMLU-Pro saturates the cap in both arms; excluded from cap%.)
  d_ent32, d_esc32  injected minus control, mean over prompts, of the mean entropy
                    lower bound / T=1.3 escape hazard over the first 32 post-injection
                    positions (both tasks; measures distributional destabilization
                    even where termination saturates).

Spearman rank correlation of the injected-arm GSM8K cap% against the matrix MMLU-Pro
pure-temperature drop closes the chain: same trigger rate -> different response.
"""
from __future__ import annotations

import glob
import json

MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]
DROPS = {"llama-3.1-8b-instruct": 40.8, "mistral-7b-instruct-v0.3": 8.0,
         "qwen2.5-7b-instruct": 4.8, "gemma-3-12b-it": 3.7, "qwen3-8b": 1.0,
         "qwen3-4b": 0.7, "qwen3-1.7b": 0.7}


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, j in enumerate(order):
            r[j] = i
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    m = (n - 1) / 2
    num = sum((a - m) * (b - m) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - m) ** 2 for a in rx) * sum((b - m) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    rows = []
    for m in MODELS:
        files = glob.glob(f"results/perturbation/{m}.jsonl")
        if not files:
            continue
        recs = [json.loads(line) for line in open(files[0], encoding="utf-8")]
        if not recs:
            continue
        g = [r for r in recs if r["task"] == "gsm8k"]
        cap_inj = mean([r["inj"]["capped"] for r in g])
        cap_ctl = mean([r["ctrl"]["capped"] for r in g])
        d_ent = mean([r["inj"]["ent32"] - r["ctrl"]["ent32"] for r in recs
                      if "ent32" in r["inj"] and "ent32" in r["ctrl"]])
        d_esc = mean([r["inj"]["esc32"] - r["ctrl"]["esc32"] for r in recs
                      if "esc32" in r["inj"] and "esc32" in r["ctrl"]])
        rows.append((m, len(g), cap_inj, cap_ctl, d_ent, d_esc))
    print(f"{'model':<26} {'n':>4} {'cap_inj':>8} {'cap_ctl':>8} "
          f"{'d_ent32':>8} {'d_esc32':>9} {'drop pp':>8}")
    for m, n, ci, cc, de, ds in rows:
        print(f"{m:<26} {n:>4} {ci*100:7.1f}% {cc*100:7.1f}% "
              f"{de:+8.4f} {ds:+9.5f} {DROPS[m]:8.1f}")
    if len(rows) >= 4:
        x = [r[2] for r in rows]
        y = [DROPS[r[0]] for r in rows]
        print(f"\nSpearman(inj cap%, drop)        = {spearman(x, y):+.2f}")
        x = [r[2] - r[3] for r in rows]
        print(f"Spearman(cap% inj-ctl, drop)    = {spearman(x, y):+.2f}")
        x = [r[4] for r in rows]
        print(f"Spearman(d_ent32, drop)         = {spearman(x, y):+.2f}")
        x = [r[5] for r in rows]
        print(f"Spearman(d_esc32, drop)         = {spearman(x, y):+.2f}")


if __name__ == "__main__":
    main()
