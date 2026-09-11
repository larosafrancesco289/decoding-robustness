#!/usr/bin/env python3
"""Robustness check on the pre-registered parsers: re-grade every record with a strict
rule that tolerates markdown emphasis around the answer ("The final answer is **230**",
"The answer is **J**"), which the lm-eval-harness-faithful strict regex does not match.
Reports, per model x task, the largest accuracy change in any (quant, sampler, T) cell,
and the change in the headline plain-temperature T0.7 -> T1.3 drops and in the
deployment-band sampler spreads (max minus min over the 8 arms, T <= 1.3, Q8).
CPU, ~1 min. Writes nothing; the pre-registered grades stay primary.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, "src")
from decoding_robustness.tasks.parsers import (  # noqa: E402  # noqa: E402
    _NUMBER,
    normalize_number,
    parse_gsm8k_numeric,
    parse_mmlu_pro_letter,
)

_EMPH = r"[\*_\s]*"
LET = re.compile(r"answer\s+is\s*:?\s*" + _EMPH + r"\(?\s*" + _EMPH + r"([A-Za-z])\s*" + _EMPH + r"\)?", re.I)
NUM = re.compile(r"final answer is\s*:?\s*" + _EMPH + r"\(?\s*\$?\s*" + _EMPH + r"\\?(?:boxed\{)?\s*(" + _NUMBER + ")", re.I)


def regrade(r):
    o = r["raw_output"]
    if r["task"] == "gsm8k":
        m = NUM.findall(o)
        if m and normalize_number(m[-1]) is not None:
            return normalize_number(m[-1]), "strict"
        p = parse_gsm8k_numeric(o)
        return p.answer, p.method
    m = [x.upper() for x in LET.findall(o) if x.upper() in "ABCDEFGHIJ"]
    if m:
        return m[-1], "strict"
    p = parse_mmlu_pro_letter(o)
    return p.answer, p.method


def main():
    cells = defaultdict(lambda: [0, 0, 0])  # (model,quant,task,sampler,T) -> [n, correct_old, correct_new]
    changed = defaultdict(int)
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            ans, meth = regrade(r)
            new_correct = ans is not None and ans == r["gold"]
            k = (r["model"], r["quant"], r["task"], r["sampler"], r["temperature"])
            c = cells[k]
            c[0] += 1
            c[1] += int(r["correct"])
            c[2] += int(new_correct)
            if new_correct != r["correct"]:
                changed[(r["model"], r["task"])] += 1
    print("records whose grade changes, by model x task:")
    for k in sorted(changed):
        print(f"  {k[0]:<26} {k[1]:<9} {changed[k]}")
    print("\nlargest per-cell accuracy change (pp), by model x task:")
    worst = defaultdict(float)
    for k, (n, a, b) in cells.items():
        d = (b - a) / n * 100
        if abs(d) > abs(worst[(k[0], k[2])]):
            worst[(k[0], k[2])] = d
    for k in sorted(worst):
        print(f"  {k[0]:<26} {k[1]:<9} {worst[k]:+6.2f}")

    def acc(model, task, sampler, T, which, quant=None):
        n = c = 0
        for k, v in cells.items():
            if k[0] == model and k[2] == task and k[3] == sampler and k[4] == T and (quant is None or k[1] == quant):
                n += v[0]
                c += v[which]
        return c / n * 100 if n else float("nan")

    models = sorted({k[0] for k in cells})
    print("\nplain-temperature drop T0.7->T1.3 (pp, pooled over quant): old -> new")
    for m in models:
        for t in ("gsm8k", "mmlu_pro"):
            old = acc(m, t, "temperature", 0.7, 1) - acc(m, t, "temperature", 1.3, 1)
            new = acc(m, t, "temperature", 0.7, 2) - acc(m, t, "temperature", 1.3, 2)
            print(f"  {m:<26} {t:<9} {old:6.1f} -> {new:6.1f}")
    print("\nsampler spread max-min over arms at Q8, T<=1.3 (pp): old -> new")
    for m in models:
        for t in ("gsm8k", "mmlu_pro"):
            for T in (0.7, 1.0, 1.3):
                arms = sorted({k[3] for k in cells if k[0] == m and k[2] == t and k[4] == T and k[1] == "Q8_0"})
                if len(arms) < 2:
                    continue
                o = [acc(m, t, s, T, 1, "Q8_0") for s in arms]
                nw = [acc(m, t, s, T, 2, "Q8_0") for s in arms]
                print(f"  {m:<26} {t:<9} T{T}: {max(o)-min(o):5.1f} -> {max(nw)-min(nw):5.1f}")


if __name__ == "__main__":
    main()
