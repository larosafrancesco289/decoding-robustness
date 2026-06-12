#!/usr/bin/env python3
"""Preliminary cross-model analysis of the full decoding x quantization matrix.

Reads every shard in results/full_matrix/shards/ and prints, per task:
  1. Greedy baselines (model x quant) -- sanity.
  2. Sampler x temperature, pooled over quant, per model -- the headline view of the
     temperature-robustness claim (does pure temperature collapse while truncation holds?).
  3. Temperature-robustness ranking per model: drop from T=0.7 to T=1.3, pooled over quant,
     ranked best (most robust) to worst.
  4. Per-quant stability of that ranking (Spearman of the per-quant robustness vector vs the
     pooled one) -- the main RQ: is the ranking stable as bits drop?
  5. Quant effect: greedy accuracy delta Q8 -> Q3 per model.
  6. Temp-first vs temp-last ablation (top_p, min_p) at T=1.3 -- the min-p critique.
  7. Parse-failure rate by model and by sampler -- reported as a metric.

Pure stdlib. No generation, no model loading. Resilient to incomplete cells (uses whatever
records exist, prints the n).
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict

SHARDS = "results/full_matrix/shards/*.jsonl"
QUANT_ORDER = ["BF16", "Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
# canonical short labels for the sampler arms
SAMPLER_LABEL = {
    "greedy": "greedy",
    "temperature": "temperature",
    "top_p_top_p0.95": "top_p",
    "top_k_top_k40": "top_k",
    "min_p_min_p0.05": "min_p",
    "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
    "min_p_min_p0.05_tlast": "min_p (tlast)",
    "top_p_top_p0.95_tlast": "top_p (tlast)",
}
SAMPLER_ORDER = [
    "temperature", "top_k", "top_p", "min_p", "top_n_sigma", "top_p (tlast)", "min_p (tlast)",
]
MODEL_ORDER = [
    "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "gemma-3-12b-it",
    "qwen2.5-7b-instruct", "qwen3-8b", "qwen3-4b", "qwen3-1.7b",
]
TEMPS = [0.7, 1.0, 1.3]


def load():
    recs = []
    for f in glob.glob(SHARDS):
        for line in open(f):
            recs.append(json.loads(line))
    return recs


def acc(records):
    if not records:
        return None, 0
    return sum(r["correct"] for r in records) / len(records), len(records)


def fmt(a):
    return f"{a*100:4.0f}%" if a is not None else "  - "


def spearman(x, y):
    """Spearman rho on two equal-length numeric vectors (ranking similarity)."""
    n = len(x)
    if n < 2:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def main():
    recs = load()
    # index: by (model, quant, sampler_label, temp, task)
    by = defaultdict(list)
    quants_seen = defaultdict(set)
    for r in recs:
        lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
        by[(r["model"], r["quant"], lab, r["temperature"], r["task"])].append(r)
        quants_seen[r["model"]].add(r["quant"])

    def pool(model, sampler, temp, task, quant=None):
        out = []
        for r in recs:
            if r["model"] != model:
                continue
            if quant is not None and r["quant"] != quant:
                continue
            if SAMPLER_LABEL.get(r["sampler"], r["sampler"]) != sampler:
                continue
            if r["temperature"] != temp or r["task"] != task:
                continue
            out.append(r)
        return out

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in recs)]

    print(f"\nLoaded {len(recs):,} records across {len(models)} models.\n")

    for task in ["gsm8k", "mmlu_pro"]:
        print("=" * 78)
        print(f"TASK: {task}")
        print("=" * 78)

        # 1. greedy baseline grid
        print("\n[1] Greedy baseline (T=0) accuracy -- model x quant")
        hdr = "  " + f"{'model':<26}" + "".join(f"{q:>8}" for q in QUANT_ORDER)
        print(hdr)
        for m in models:
            cells = []
            for q in QUANT_ORDER:
                a, _ = acc(pool(m, "greedy", 0.0, task, q))
                cells.append(fmt(a))
            print("  " + f"{m:<26}" + "".join(f"{c:>8}" for c in cells))

        # 2. sampler x temp, pooled over quant
        print("\n[2] Sampler x temperature (pooled over quant) -- accuracy")
        for m in models:
            print(f"\n  {m}")
            print("    " + f"{'sampler':<16}" + "".join(f"{('T'+str(t)):>8}" for t in TEMPS))
            for s in SAMPLER_ORDER:
                cells = []
                ns = []
                for t in TEMPS:
                    a, n = acc(pool(m, s, t, task))
                    cells.append(fmt(a))
                    ns.append(n)
                if max(ns) == 0:
                    continue
                print("    " + f"{s:<16}" + "".join(f"{c:>8}" for c in cells))

        # 3 + 4. robustness ranking + per-quant stability
        print("\n[3] Temp-robustness: accuracy drop T0.7 -> T1.3 (pooled over quant)")
        print("    smaller drop = more robust; ranked most-robust first")
        for m in models:
            drops = {}
            for s in SAMPLER_ORDER:
                a07, n07 = acc(pool(m, s, 0.7, task))
                a13, n13 = acc(pool(m, s, 1.3, task))
                if a07 is None or a13 is None:
                    continue
                drops[s] = a07 - a13
            if not drops:
                continue
            ranked = sorted(drops.items(), key=lambda kv: kv[1])
            line = ", ".join(f"{s} {d*100:+.0f}pp" for s, d in ranked)
            print(f"    {m}: {line}")

        print("\n[4] Per-quant stability of the robustness ranking (Spearman rho vs pooled)")
        for m in models:
            qs = [q for q in QUANT_ORDER if q in quants_seen[m]]
            pooled = {}
            for s in SAMPLER_ORDER:
                a07, _ = acc(pool(m, s, 0.7, task))
                a13, _ = acc(pool(m, s, 1.3, task))
                if a07 is not None and a13 is not None:
                    pooled[s] = a07 - a13
            common = [s for s in SAMPLER_ORDER if s in pooled]
            if len(common) < 2:
                continue
            pvec = [pooled[s] for s in common]
            rhos = []
            for q in qs:
                qvec = []
                ok = True
                for s in common:
                    a07, _ = acc(pool(m, s, 0.7, task, q))
                    a13, _ = acc(pool(m, s, 1.3, task, q))
                    if a07 is None or a13 is None:
                        ok = False
                        break
                    qvec.append(a07 - a13)
                if ok:
                    rho = spearman(pvec, qvec)
                    rhos.append(f"{q}:{rho:+.2f}" if rho is not None else f"{q}:--")
            print(f"    {m}: " + "  ".join(rhos))

    # 5. quant effect on greedy (pooled tasks)
    print("\n" + "=" * 78)
    print("[5] Quant effect on greedy accuracy: Q8_0 -> Q3_K_M (pooled over both tasks)")
    print("=" * 78)
    for m in models:
        a8 = acc([r for r in recs if r["model"] == m and r["quant"] == "Q8_0"
                  and SAMPLER_LABEL.get(r["sampler"]) == "greedy"])[0]
        a3 = acc([r for r in recs if r["model"] == m and r["quant"] == "Q3_K_M"
                  and SAMPLER_LABEL.get(r["sampler"]) == "greedy"])[0]
        if a8 is None or a3 is None:
            print(f"  {m:<26} (incomplete)")
            continue
        print(f"  {m:<26} Q8={fmt(a8)}  Q3={fmt(a3)}  delta={ (a3-a8)*100:+.1f}pp")

    # 6. temp-first vs temp-last at T=1.3 (min-p critique)
    print("\n" + "=" * 78)
    print("[6] Temp-FIRST vs temp-LAST at T=1.3 (pooled quant) -- the min-p critique ablation")
    print("=" * 78)
    for task in ["gsm8k", "mmlu_pro"]:
        print(f"\n  {task}")
        for m in models:
            row = []
            for base, tlast in [("top_p", "top_p (tlast)"), ("min_p", "min_p (tlast)")]:
                af, _ = acc(pool(m, base, 1.3, task))
                al, _ = acc(pool(m, tlast, 1.3, task))
                if af is None or al is None:
                    continue
                row.append(f"{base}: first {fmt(af)} / last {fmt(al)} ({(al-af)*100:+.0f}pp)")
            if row:
                print(f"    {m:<26} " + " | ".join(row))

    # 7. parse-fail rates
    print("\n" + "=" * 78)
    print("[7] Parse-failure rate (parse_method == 'failed')")
    print("=" * 78)
    print("\n  by model x task:")
    for m in models:
        for task in ["gsm8k", "mmlu_pro"]:
            sub = [r for r in recs if r["model"] == m and r["task"] == task]
            if not sub:
                continue
            f = sum(r["parse_method"] == "failed" for r in sub) / len(sub)
            print(f"    {m:<26} {task:<10} {f*100:5.1f}%  (n={len(sub)})")
    print("\n  by sampler x temp (pooled models/tasks):")
    for s in SAMPLER_ORDER + ["greedy"]:
        cells = []
        for t in ([0.0] if s == "greedy" else TEMPS):
            sub = [
                r for r in recs
                if SAMPLER_LABEL.get(r["sampler"]) == s and r["temperature"] == t
            ]
            if not sub:
                cells.append("  - ")
                continue
            f = sum(r["parse_method"] == "failed" for r in sub) / len(sub)
            cells.append(f"{f*100:4.1f}%")
        print(f"    {s:<16}" + "".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
