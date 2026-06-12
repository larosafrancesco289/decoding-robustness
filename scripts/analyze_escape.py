#!/usr/bin/env python3
"""Temperature-scaled escape hazard from the v2 logit probe (results/logit_probe_v2/).

Question: why do Llama-3.1 and Mistral collapse five-fold differently at T=1.3 when every
T=1 top-20 summary ties them? Hypothesis: the difference appears only after temperature
scaling, because escape mass past the top-20 depends on the SHAPE of the tail, not its
T=1 size. analyze_recovery.py shows derailment is absorbing, so the per-token escape
probability is the whole game: accuracy ~ survival ~ (1 - escape)^length.

Per position with raw top-20 probs p1..p20 (descending) and tail mass t = 1 - sum(p):
  scaled top mass     S_top  = sum_i p_i^(1/T)
  tail, concentrated  S_con  = t^(1/T)                  (one tail token: lower bound)
  tail, continuation  S_cont = (t/p20) * p20^(1/T)      (tail continues at p20's level)
  escape(T) = S_tail / (S_top + S_tail)  for each tail assumption.

Reports per model: mean per-token escape at T=1.3 (both assumptions), expected escapes
per 256-token generation, and survival (1 - escape)^256 under the continuation estimate,
against the MMLU-Pro T0.7->1.3 drop. CPU, seconds.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

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
    mx = (n - 1) / 2
    num = sum((a - mx) * (b - mx) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - mx) ** 2 for b in ry)) ** 0.5
    return num / den


def escapes(probs: list[float], temp: float) -> tuple[float, float]:
    """(concentrated, continuation) escape probability past the top-20 at `temp`."""
    inv = 1.0 / temp
    s_top = sum(p ** inv for p in probs if p > 0)
    t = max(0.0, 1.0 - sum(probs))
    if t <= 0 or s_top <= 0:
        return 0.0, 0.0
    s_con = t ** inv
    p20 = probs[-1]
    s_cont = (t / p20) * (p20 ** inv) if p20 > 0 else s_con
    return s_con / (s_top + s_con), s_cont / (s_top + s_cont)


def main():
    per_model = defaultdict(lambda: [[], []])  # model -> [con list, cont list]
    for f in glob.glob("results/logit_probe_v2/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for pos in r["probs"]:
                con, cont = escapes(pos, 1.3)
                per_model[r["model"]][0].append(con)
                per_model[r["model"]][1].append(cont)
    print(f"{'model':<26} {'esc_con':>9} {'esc_cont':>9} {'exp/256':>8} "
          f"{'surv256':>8} {'drop pp':>8}")
    rows = []
    for m in MODELS:
        con, cont = per_model[m]
        if not con:
            continue
        mc = sum(con) / len(con)
        mt = sum(cont) / len(cont)
        surv = (1 - mt) ** 256
        rows.append((m, mt))
        print(f"{m:<26} {mc:9.5f} {mt:9.5f} {mt*256:8.2f} {surv:8.3f} {DROPS[m]:8.1f}")
    if len(rows) >= 4:
        x = [r[1] for r in rows]
        y = [DROPS[r[0]] for r in rows]
        print(f"\nSpearman(escape_cont@T1.3, drop) = {spearman(x, y):+.2f}")


if __name__ == "__main__":
    main()
