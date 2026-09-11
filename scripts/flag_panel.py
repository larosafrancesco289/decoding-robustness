#!/usr/bin/env python3
"""Apply the prespecified flag of configs/temp_sweep_2026.yaml to the 2026 panel (and, with --dir,
to any temperature-only run): per model x task, the plain-temperature change T0.7->T1.3 in accuracy,
cap-hit rate and strict-parse rate with item-clustered bootstrap CIs (analyze_engine_check.py's
estimators), then the rule on the POINT estimates: accuracy drop >= 15 pp, or cap-hit rise >= 20 pp,
or strict-parse fall >= 20 pp, on either task => FLAGGED (full 8-configuration grid, configs/panel_grid.yaml).
Also prints greedy accuracy and per-condition record counts so incomplete runs are visible.
"""
from __future__ import annotations

import argparse
import glob
import sys

sys.path.insert(0, "scripts")
import analyze_engine_check as aec  # noqa: E402

RULE = {"acc": 0.15, "cap": 0.20, "strict": 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/temp_sweep_2026")
    args = ap.parse_args()
    for path in sorted(glob.glob(f"{args.dir}/*.jsonl")):
        by = aec.load(path)
        if not by:
            print(f"\n##### {path}: empty"); continue
        model = next(iter(next(iter(by.values())).values()))[0]["model"]
        quant = next(iter(by))[0]
        print(f"\n##### {model}  quant={quant}  ({path})")
        flagged = []
        for task in sorted({k[1] for k in by}):
            print(f"  [{task}]  {'cond':<8} {'n':>4} {'acc':>6} {'cap':>6} {'strict':>6} {'fail':>6} {'toks':>6}")
            for sampler, temp in aec.CONDS:
                s = aec.stats(by.get((quant, task, sampler, temp), {}))
                lab = "greedy" if sampler == "greedy" else f"T{temp}"
                if s:
                    print(f"    {lab:<8} {s['n']:>4} {s['acc']*100:6.1f} {s['cap']*100:6.1f} {s['strict']*100:6.1f} {s['fail']*100:6.1f} {s['toks']:6.0f}")
                else:
                    print(f"    {lab:<8}    - (missing)")
            hi, lo = by.get((quant, task, "temperature", 1.3), {}), by.get((quant, task, "temperature", 0.7), {})
            trips = []
            for m in aec.METRICS:
                d = aec.boot_diff(hi, lo, m)
                if d is None:
                    continue
                point = d[0]
                sign = -1 if m in ("acc", "strict") else 1  # loss for acc/strict, rise for cap
                hit = sign * point >= RULE[m]
                print(f"    change T0.7->T1.3 {m:<6}: {aec.fmt_ci(d)}{'   <-- trips the rule' if hit else ''}")
                if hit:
                    trips.append(m)
            if trips:
                flagged.append((task, trips))
        verdict = "FLAGGED (" + "; ".join(f"{t}: {', '.join(ms)}" for t, ms in flagged) + ")" if flagged else "not flagged"
        print(f"  ==> {model}: {verdict}")


if __name__ == "__main__":
    main()
