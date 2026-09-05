#!/usr/bin/env python3
"""Analyze the perturbation sensitivity sweep (results/perturbation_sweep/).

For every arm (injection position 16/48/80, injected rank 2/5/10/20, injection count
1/2/4) and every model, reports the same statistics as analyze_perturbation.py:

  cap%      GSM8K continuations that ran to the 256-token cap (injected / control)
  d_esc32   injected minus control, mean over both tasks, of the T=1.3 escape hazard
            over the 32 positions after the (last) injection
  mult      injected / control escape hazard (the "multiplier" quoted in section 4.6)
  d_ent32   same for the entropy lower bound

then the Spearman rank correlation of each arm's d_esc32 and multiplier with the matrix
MMLU-Pro pure-temperature drop, so the reader can see whether rho(+0.86 at the anchor)
survives each knob. The pos48 anchor is also compared with results/perturbation/ (the
original single-slot run) as a determinism check.

  uv run python scripts/analyze_perturbation_sweep.py [--dir results/perturbation_sweep]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from analyze_perturbation import DROPS, MODELS, mean, spearman

ARMS = ["pos16", "pos48", "pos80", "rank2", "rank5", "rank10", "rank20",
        "count1", "count2", "count4"]
CTRL_OF = {"pos16": "pos16", "pos48": "pos48", "pos80": "pos80"}  # others use pos48 ctrl


def load(dirname: str) -> dict[str, list[dict]]:
    out = {}
    for m in MODELS:
        f = os.path.join(dirname, f"{m}.jsonl")
        if os.path.isfile(f):
            out[m] = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
    return out


def arm_rows(recs: list[dict], arm: str) -> dict | None:
    ctrl_arm = CTRL_OF.get(arm, "pos48")
    pairs = [(r["arms"][arm], r["arms"][ctrl_arm], r["task"]) for r in recs
             if r["arms"].get(arm) and r["arms"].get(ctrl_arm)]
    if not pairs:
        return None
    g = [(a, c) for a, c, t in pairs if t == "gsm8k"]
    esc = [(a["inj"]["esc32"], c["ctrl"]["esc32"]) for a, c, _ in pairs
           if "esc32" in a["inj"] and "esc32" in c["ctrl"]]
    ent = [(a["inj"]["ent32"], c["ctrl"]["ent32"]) for a, c, _ in pairs
           if "ent32" in a["inj"] and "ent32" in c["ctrl"]]
    esc_i, esc_c = mean([x for x, _ in esc]), mean([y for _, y in esc])
    return {
        "n": len(pairs), "n_gsm8k": len(g),
        "cap_inj": mean([a["inj"]["capped"] for a, _ in g]) if g else float("nan"),
        "cap_ctl": mean([c["ctrl"]["capped"] for _, c in g]) if g else float("nan"),
        "d_esc": mean([x - y for x, y in esc]),
        "mult": esc_i / esc_c if esc_c else float("nan"),
        "d_ent": mean([x - y for x, y in ent]),
        "count_done": mean([a.get("count_done", 1) for a, _, _ in pairs]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/perturbation_sweep")
    ap.add_argument("--orig", default="results/perturbation")
    args = ap.parse_args()
    data = load(args.dir)
    if not data:
        print("no sweep results found")
        return

    summary: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        print(f"\n=== arm {arm}" + ("" if arm in CTRL_OF else "   (control = pos48)"))
        print(f"{'model':<26} {'n':>4} {'nG':>3} {'cap_inj':>8} {'cap_ctl':>8} "
              f"{'d_esc32':>9} {'mult':>6} {'d_ent32':>8} {'drop':>6}")
        for m, recs in data.items():
            row = arm_rows(recs, arm)
            if row is None:
                continue
            summary.setdefault(arm, {})[m] = row
            extra = f"  inj/prompt={row['count_done']:.2f}" if arm.startswith("count") else ""
            print(f"{m:<26} {row['n']:>4} {row['n_gsm8k']:>3} {row['cap_inj']*100:7.1f}% "
                  f"{row['cap_ctl']*100:7.1f}% {row['d_esc']:+9.5f} {row['mult']:6.2f} "
                  f"{row['d_ent']:+8.4f} {DROPS[m]:6.1f}{extra}")

    print("\n=== Spearman with the MMLU-Pro pure-temperature drop, per arm")
    print(f"{'arm':<8} {'n':>3} {'rho(d_esc)':>11} {'rho(mult)':>10} {'rho(d_ent)':>11} "
          f"{'Llama mult':>11} {'max robust mult':>16}")
    for arm, rows in summary.items():
        ms = [m for m in MODELS if m in rows]
        if len(ms) < 4:
            continue
        y = [DROPS[m] for m in ms]
        r_esc = spearman([rows[m]["d_esc"] for m in ms], y)
        r_mult = spearman([rows[m]["mult"] for m in ms], y)
        r_ent = spearman([rows[m]["d_ent"] for m in ms], y)
        llama = rows.get("llama-3.1-8b-instruct", {}).get("mult", float("nan"))
        robust = max((rows[m]["mult"] for m in ms if m != "llama-3.1-8b-instruct"),
                     default=float("nan"))
        print(f"{arm:<8} {len(ms):>3} {r_esc:+11.2f} {r_mult:+10.2f} {r_ent:+11.2f} "
              f"{llama:11.2f} {robust:16.2f}")

    # Determinism check: anchor arm vs the original single-slot run, per prompt.
    orig = load(args.orig)
    if orig:
        print("\n=== anchor (pos48) vs original results/perturbation, per model")
        print(f"{'model':<26} {'shared':>6} {'same inj tok':>12} {'same len':>9} "
              f"{'|d_esc diff|':>12}")
        for m, recs in data.items():
            if m not in orig:
                continue
            o = {r["key"]: r for r in orig[m]}
            shared = [(r["arms"]["pos48"], o[r["key"]]) for r in recs
                      if r["key"] in o and r["arms"].get("pos48")]
            if not shared:
                continue
            same_tok = mean([a["inj_token"] == b["inj_token"] for a, b in shared])
            same_len = mean([a["inj"]["len"] == b["inj"]["len"] for a, b in shared])
            d_new = mean([a["inj"].get("esc32", 0) - a["ctrl"].get("esc32", 0)
                          for a, _ in shared])
            d_old = mean([b["inj"].get("esc32", 0) - b["ctrl"].get("esc32", 0)
                          for _, b in shared])
            print(f"{m:<26} {len(shared):>6} {same_tok*100:11.0f}% {same_len*100:8.0f}% "
                  f"{abs(d_new - d_old):12.5f}")


if __name__ == "__main__":
    main()
