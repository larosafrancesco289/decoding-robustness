#!/usr/bin/env python
"""Power / precision analysis for Phase-1 sizing (consumes a pilot JSONL).

Purpose: from the Phase-0 pilot, decide the Phase-1 item count N (and sanity-check the
rep count K) for the key comparisons of the study: *does quantization reorder the
temperature-robustness ranking of decoding strategies?*

Honest caveats baked in:
- The pilot is **1 rep/item**, so it estimates the **between-item** variance directly but
  cannot estimate the **within-item across-rep** variance (only one draw per item). So this
  sizes the **item count N** from the observed per-item Bernoulli outcomes, and treats reps
  **K** as reducing sampling noise only under the optimistic independent-draw bound (rho=0).
  The true benefit of K sits between "n scales as N*K" (rho=0) and "K adds nothing" (rho=1);
  K=3 is a standard default for stochastic-decoding studies and we refine once we have K>1.
- **Parse failures count as incorrect** (a generation that never emits an answer is not a
  correct answer; this is the scoring used throughout).

Usage:  uv run python scripts/power_analysis.py [--jsonl results/pilot_llama/pilot.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

TASKS = ("gsm8k", "mmlu_pro")  # GPQA dropped (near-chance for 8B models)
Z_ALPHA = 1.959964  # two-sided alpha = 0.05


def _phi(x: float) -> float:
    """Standard normal CDF via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_power(p1: float, p2: float, n_per_cell: float) -> float:
    """Approx power of a two-sided two-proportion z-test, n observations per cell."""
    if p1 == p2:
        return 0.05
    se = math.sqrt(p1 * (1 - p1) / n_per_cell + p2 * (1 - p2) / n_per_cell)
    if se == 0:
        return 1.0
    z = abs(p1 - p2) / se
    return _phi(z - Z_ALPHA)  # one tail of the two-sided test (the dominant one)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default="results/pilot_llama/pilot.jsonl")
    args = ap.parse_args()

    recs = []
    for line in Path(args.jsonl).read_text().splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # cell -> [correct booleans]; parse-fail already implies correct=False in the record.
    cells: dict[tuple, list[bool]] = defaultdict(list)
    for r in recs:
        if r["task"] not in TASKS:
            continue
        cells[(r["task"], r["quant"], r["sampler"], r["temperature"])].append(bool(r["correct"]))

    def acc(task, quant, sampler, temp):
        v = cells.get((task, quant, sampler, temp))
        return (sum(v) / len(v)) if v else None

    quants = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
    samplers = sorted({s for (_, _, s, _) in cells})
    stoch = [s for s in samplers if s != "greedy"]

    # --- characterise the three comparison types by their observed effect size ---
    # (A) headline robustness gap: at T=1.3, best vs worst stochastic sampler (pooled over quant)
    # (B) quant interaction: per sampler, |acc(Q8) - acc(Q3)| pooled over temps
    # (C) good-sampler gap: min_p vs top_n_sigma pooled over temps & quants
    print("=" * 74)
    print("OBSERVED EFFECT SIZES (GSM8K + MMLU-Pro; parse-fail = incorrect)")
    print("=" * 74)

    headline = {}
    for task in TASKS:
        accs = {s: acc(task, "Q8_0", s, 1.3) for s in stoch}
        accs = {s: a for s, a in accs.items() if a is not None}
        hi, lo = max(accs, key=accs.get), min(accs, key=accs.get)
        headline[task] = abs(accs[hi] - accs[lo])
        print(f"\n[{task}] (A) headline @T=1.3 (Q8): {hi}={accs[hi]:.0%} vs {lo}={accs[lo]:.0%}"
              f"  -> Δ={headline[task]:.0%}")

    quant_deltas = []
    for task in TASKS:
        ds = []
        for s in samplers:
            a8 = [acc(task, "Q8_0", s, t) for t in (0.0, 0.7, 1.0, 1.3)]
            a3 = [acc(task, "Q3_K_M", s, t) for t in (0.0, 0.7, 1.0, 1.3)]
            a8 = [x for x in a8 if x is not None]
            a3 = [x for x in a3 if x is not None]
            if a8 and a3:
                ds.append(abs(st.mean(a8) - st.mean(a3)))
        quant_deltas += ds
        print(f"[{task}] (B) quant Q8→Q3 |Δ| per sampler: "
              f"median={st.median(ds):.1%}  max={max(ds):.1%}")
    quant_typ = st.median(quant_deltas)

    # (C) good-sampler gap, pooled
    def pooled(task, s):
        vs = [acc(task, q, s, t) for q in quants for t in (0.7, 1.0, 1.3)]
        vs = [x for x in vs if x is not None]
        return st.mean(vs) if vs else None
    cgaps = []
    for task in TASKS:
        a, b = pooled(task, "min_p_min_p0.05"), pooled(task, "top_n_sigma_top_n_sigma1.0")
        if a and b:
            cgaps.append(abs(a - b))
    good_gap = st.mean(cgaps) if cgaps else 0.0
    print(f"\n(C) good-sampler gap (min_p vs top-nσ, pooled): Δ≈{good_gap:.1%}")

    # --- power across item counts N, for K=1 and K=3 (rho=0 optimistic bound) ---
    print("\n" + "=" * 74)
    print("POWER to detect each effect, by items/condition N  (α=0.05, two-sided)")
    print("  n_per_cell = N*K under independent-rep bound (rho=0); K=3 is optimistic.")
    print("=" * 74)
    # representative absolute accuracies near which each effect sits (for the SE term)
    head_d = max(headline.values())
    scenarios = [
        (f"headline robustness  Δ≈{head_d:.0%}", 0.55, 0.55 - head_d),
        (f"quant interaction    Δ≈{quant_typ:.0%}", 0.70, 0.70 - quant_typ),
        (f"good-sampler gap     Δ≈{good_gap:.0%}", 0.60, 0.60 - good_gap),
    ]
    header = f"{'effect':28}" + "".join(f"N={n:<4}".rjust(9) for n in (50, 100, 150, 200, 300))
    for K in (1, 3):
        print(f"\n-- K={K} reps --")
        print(header)
        for label, p1, p2 in scenarios:
            row = f"{label:28}"
            for N in (50, 100, 150, 200, 300):
                pw = two_prop_power(p1, p2, N * K)
                row += f"{pw:7.0%}".rjust(9)
            print(row)
    print("\nRead: ~80% power is the usual target. Effects that clear it at N=50 are 'firm';"
          "\nthose that need N≥150–300 are the 'report-with-CIs / motivates-the-full-study' ones.")


if __name__ == "__main__":
    main()
