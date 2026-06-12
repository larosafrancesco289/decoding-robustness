#!/usr/bin/env python3
"""Decompose the temperature collapse into degeneration vs wrong-answer (paper artifact).

The headline drop (pure temperature, T0.7 -> T1.3) conflates two failure modes:
  * degeneration -- the model never terminates a well-formed answer (parse failure on MC;
    on GSM8K the flexible last-number fallback masks this as a "wrong answer", so we also
    report strict-format rate and the token-cap rate as degeneration proxies);
  * answer-selection error -- a well-formed final answer that is simply wrong.

For each model x task x temperature (pure temperature sampler, pooled over quant + reps):
  acc        overall accuracy (what STATS.txt reports)
  strict     fraction parsed via the pinned final-answer sentence (well-formed answers)
  failed     parse-failure fraction (no answer found at all)
  cap        fraction that ran to the max_new_tokens cap (degeneration proxy; includes
             parse-error-recovered records with 0 logged tokens, counted separately)
  acc|strict accuracy conditional on a well-formed (strict-parsed) answer
  acc|parsed accuracy conditional on any parse (strict or flexible)

Also runs the same table for the high-T smoke (results/hightemp_smoke/), where the "0%"
cells need the same degeneration-vs-wrong reading.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

CAP = {"gsm8k": 512, "mmlu_pro": 1024}


def table(paths: list[str], samplers: tuple[str, ...] = ("temperature",)) -> None:
    cells = defaultdict(list)  # (model, task, sampler, temp) -> [records]
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["sampler"] in samplers:
                cells[(r["model"], r["task"], r["sampler"], r["temperature"])].append(r)

    header = (
        f"{'model':<26} {'task':<9} {'sampler':<12} {'T':<5} {'n':>4} "
        f"{'acc':>6} {'strict':>7} {'flex':>6} {'failed':>7} {'cap':>6} "
        f"{'acc|strict':>10} {'acc|parsed':>10}"
    )
    print(header)
    print("-" * len(header))
    for (model, task, sampler, temp), recs in sorted(cells.items()):
        n = len(recs)
        acc = sum(r["correct"] for r in recs) / n
        strict = [r for r in recs if r["parse_method"] == "strict"]
        flex = [r for r in recs if r["parse_method"] == "flexible"]
        failed = sum(r["parse_method"] == "failed" for r in recs)
        cap = sum(
            r["n_completion_tokens"] >= CAP.get(task, 10**9) or r["n_completion_tokens"] == 0
            for r in recs
        )
        parsed = strict + flex
        acc_strict = sum(r["correct"] for r in strict) / len(strict) if strict else float("nan")
        acc_parsed = sum(r["correct"] for r in parsed) / len(parsed) if parsed else float("nan")
        print(
            f"{model:<26} {task:<9} {sampler:<12} {temp:<5} {n:>4} "
            f"{acc:>6.1%} {len(strict)/n:>7.1%} {len(flex)/n:>6.1%} {failed/n:>7.1%} {cap/n:>6.1%} "
            f"{acc_strict:>10.1%} {acc_parsed:>10.1%}"
        )


print("=" * 110)
print("FULL MATRIX -- pure temperature, pooled over quant + reps")
print("=" * 110)
table(sorted(glob.glob("results/full_matrix/shards/*.jsonl")))

print()
print("=" * 110)
print("HIGH-T SMOKE (T=2/3, Q8 only) -- all four samplers")
print("=" * 110)
table(
    sorted(glob.glob("results/hightemp_smoke/*.jsonl")),
    samplers=("temperature", "top_p_top_p0.95", "min_p_min_p0.05", "top_n_sigma_top_n_sigma1.0"),
)
