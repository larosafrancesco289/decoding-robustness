#!/usr/bin/env python3
"""Do robust models derail-and-recover, or never derail? (plain temperature, main matrix)

If a model derails and recovers, its *terminated* generations at T=1.3 should be inflated
relative to T=0.7 (a long detour leaves extra tokens). If a robust model simply never
takes the first bad step, the terminated-length distributions should match.

Per model x task, pooled over quants, plain temperature only:
  cap%   = fraction of generations that ran to the token cap
           (n_completion_tokens >= n_predict; `stopped` is true even at the cap)
  median/p90 completion tokens among terminated generations, T0.7 vs T1.3
  inflation = p90(T1.3) / p90(T0.7) among terminated generations
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    lens = defaultdict(list)   # (model, task, T) -> terminated completion lengths
    caps = defaultdict(lambda: [0, 0])  # (model, task, T) -> [capped, total]
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["sampler"] != "temperature" or r["temperature"] not in (0.7, 1.3):
                continue
            k = (r["model"], r["task"], r["temperature"])
            caps[k][1] += 1
            # server-recovered degenerate streams carry no timings (n_prompt_tokens 0);
            # they are run-to-cap generations, same as the decomposition analysis
            recovered = r["n_prompt_tokens"] == 0
            if not recovered and r["n_completion_tokens"] < r["params"]["n_predict"]:
                lens[k].append(r["n_completion_tokens"])
            else:
                caps[k][0] += 1
    for task in ("gsm8k", "mmlu_pro"):
        print(f"\n== {task} (plain temperature, pooled quants) ==")
        print(f"{'model':<26} {'cap%07':>7} {'cap%13':>7} {'med07':>6} {'med13':>6} "
              f"{'p90_07':>7} {'p90_13':>7} {'p90 infl':>9}")
        for m in MODELS:
            k7, k13 = (m, task, 0.7), (m, task, 1.3)
            if not caps[k7][1] or not caps[k13][1]:
                continue
            c7 = 100 * caps[k7][0] / caps[k7][1]
            c13 = 100 * caps[k13][0] / caps[k13][1]
            l7, l13 = lens[k7], lens[k13]
            if not l7 or not l13:
                print(f"{m:<26} {c7:6.1f}% {c13:6.1f}%   (no terminated gens)")
                continue
            p7, p13 = pctl(l7, 0.9), pctl(l13, 0.9)
            print(f"{m:<26} {c7:6.1f}% {c13:6.1f}% {pctl(l7,0.5):6d} {pctl(l13,0.5):6d} "
                  f"{p7:7d} {p13:7d} {p13/p7:8.2f}x")


if __name__ == "__main__":
    main()
