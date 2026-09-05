#!/usr/bin/env python3
"""Model-based companion to scripts/sampler_bound.py (ARR revision, Tier 2).

Per model x task, fit a generalized estimating equation over the full deployment-band grid
(8 stochastic arms x T in {0.7, 1.0, 1.3} x quantization levels x 50 items x K=3):

    correct ~ C(arm):C(T) + C(quant),   identity link, clusters = item, exchangeable correlation

The identity link makes every arm coefficient a risk difference in accuracy points against
plain temperature at the same T, and the item clustering with a robust (sandwich) covariance
absorbs the item difficulty that the K repetitions and quantization levels share. Reports each
truncation arm's contrast with its 95% CI, and the largest upper limit per model x task, for
direct comparison with the bootstrap bound. CPU, ~1 min. Requires `uv sync --extra analysis`.
"""
from __future__ import annotations

import glob
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ARMS = ["top_p_top_p0.95", "top_k_top_k40", "min_p_min_p0.05", "top_n_sigma_top_n_sigma1.0"]
SHORT = {"top_p_top_p0.95": "top-p", "top_k_top_k40": "top-k", "min_p_min_p0.05": "min-p",
         "top_n_sigma_top_n_sigma1.0": "top-nσ"}
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]


def main():
    rows = []
    for f in glob.glob("results/full_matrix/shards/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["sampler"] == "greedy":
                continue
            rows.append((r["model"], r["task"], r["temperature"], r["sampler"], r["quant"],
                         r["item_id"], int(r["correct"])))
    df = pd.DataFrame(rows, columns=["model", "task", "T", "arm", "quant", "item", "correct"])
    df["cell"] = df["arm"] + "@" + df["T"].astype(str)
    out = []
    warnings.filterwarnings("ignore")
    for m in MODELS:
        for task in ("gsm8k", "mmlu_pro"):
            d = df[(df.model == m) & (df.task == task)].copy()
            # reference level per T: plain temperature at that T. Build a design with one
            # column per (arm, T) minus the temperature@T columns, plus a T main effect.
            d["is_ref"] = d["arm"] == "temperature"
            cell_dummies = pd.get_dummies(d["cell"], dtype=float)
            keep = [c for c in cell_dummies.columns if not c.startswith("temperature@")]
            X = pd.concat([pd.get_dummies(d["T"].astype(str), prefix="T", dtype=float),
                           pd.get_dummies(d["quant"], prefix="q", drop_first=True, dtype=float),
                           cell_dummies[keep]], axis=1)
            model = sm.GEE(d["correct"].values, X.values, groups=d["item"].values,
                           family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable())
            res = model.fit()
            names = list(X.columns)
            for T in (0.7, 1.0, 1.3):
                for arm in ARMS:
                    c = f"{arm}@{T}"
                    k = names.index(c)
                    est = res.params[k] * 100
                    se = res.bse[k] * 100
                    out.append(dict(model=m, task=task, T=T, arm=arm, est=float(est),
                                    lo=float(est - 1.96 * se), hi=float(est + 1.96 * se)))
    json.dump(out, open("results/sampler_gee.json", "w"), indent=1)
    boot = {(o["model"], o["task"], o["T"], o["arm"]): o
            for o in json.load(open("results/sampler_bound.json"))}
    print("GEE risk difference vs plain temperature (pp) [95% CI]  |  bootstrap for comparison")
    for m in MODELS:
        for task in ("gsm8k", "mmlu_pro"):
            rr = [o for o in out if o["model"] == m and o["task"] == task]
            hi = max(rr, key=lambda o: o["hi"])
            bhi = max((boot[(m, task, o["T"], o["arm"])] for o in rr), key=lambda o: o["hi"])
            print(f"{m:<26}{task:<9} GEE max upper {hi['hi']:+5.1f} ({SHORT[hi['arm']]}, T{hi['T']}) "
                  f"| bootstrap max upper {bhi['hi']:+5.1f} ({SHORT[bhi['arm']]}, T{bhi['T']})")
    diffs = [abs(o["hi"] - boot[(o["model"], o["task"], o["T"], o["arm"])]["hi"]) for o in out]
    print(f"\nmax |GEE upper - bootstrap upper| over all {len(out)} contrasts: {max(diffs):.2f} pp; "
          f"median {np.median(diffs):.2f} pp")
    ests = [abs(o["est"] - boot[(o["model"], o["task"], o["T"], o["arm"])]["mean"]) for o in out]
    print(f"max |GEE estimate - paired mean|: {max(ests):.2f} pp (should be ~0 with identity link)")


if __name__ == "__main__":
    main()
