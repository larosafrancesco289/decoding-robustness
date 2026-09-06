#!/usr/bin/env python3
"""tables/tab_collapse.tex and the numbers behind Section 4.1: plain-temperature accuracy at Q8_0 for every model
that has the temperature axis (main grid, Llama lineage, the 2026 panel in results/temp_sweep_2026 when present),
at T=0.7 and T=1.3, the drop with an item-clustered bootstrap CI (paper convention: positive = loss), and the
change in cap-hit and strict-parse rates. Q8 is the primary estimand (pooled-over-quantization drops per level
are in the appendix). Pure stdlib + numpy."""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B, RNG = 2000, np.random.default_rng(0)
CAP = {"gsm8k": 512, "mmlu_pro": 1024}
SOURCES = ["results/full_matrix/shards/*__Q8_0.jsonl", "results/llama_lineage/*.jsonl", "results/temp_sweep_2026/*.jsonl"]
# display order and labels; models not listed are appended in file order
ORDER = ["llama-3.1-8b-instruct", "llama-3-8b-instruct", "llama-3.2-3b-instruct", "hermes-3-llama-3.1-8b",
         "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", "qwen3-8b", "qwen3-4b", "qwen3-1.7b", "gemma-3-12b-it",
         "qwen3.5-9b", "gemma-4-e4b-it", "olmo-3-7b-instruct"]
LABEL = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "llama-3-8b-instruct": "Llama-3-8B", "llama-3.2-3b-instruct": "Llama-3.2-3B",
         "hermes-3-llama-3.1-8b": "Hermes-3 (Llama-3.1-8B)", "mistral-7b-instruct-v0.3": "Mistral-7B-v0.3",
         "qwen2.5-7b-instruct": "Qwen2.5-7B", "qwen3-8b": "Qwen3-8B", "qwen3-4b": "Qwen3-4B", "qwen3-1.7b": "Qwen3-1.7B",
         "gemma-3-12b-it": "Gemma-3-12B", "qwen3.5-9b": "Qwen3.5-9B", "gemma-4-e4b-it": "Gemma-4-E4B", "olmo-3-7b-instruct": "OLMo-3-7B"}
GROUP = {"hermes-3-llama-3.1-8b": "same base, other post-training", "qwen3.5-9b": "2026 panel", "gemma-4-e4b-it": "2026 panel", "olmo-3-7b-instruct": "2026 panel"}
TASK = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}


def load():
    by = defaultdict(lambda: defaultdict(list))  # (model, task, temp) -> item -> records
    for pat in SOURCES:
        for path in glob.glob(pat):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["quant"] != "Q8_0" or r["sampler"] != "temperature":
                    continue
                by[(r["model"], r["task"], float(r["temperature"]))][r["item_id"]].append(r)
    return by


def rate(imap, items, f):
    num = den = 0
    for it in items:
        v = imap.get(it)
        if v:
            num += sum(f(r) for r in v)
            den += len(v)
    return num / den if den else float("nan")


def boot(hi, lo, f):
    items = np.array(sorted(set(hi) & set(lo)), dtype=object)
    if not len(items):
        return None
    pt = rate(lo, items, f) - rate(hi, items, f)  # loss = acc(0.7) - acc(1.3)
    draws = [rate(lo, s, f) - rate(hi, s, f) for s in (RNG.choice(items, len(items), replace=True) for _ in range(B))]
    return pt, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    by = load()
    models = [m for m in ORDER if any(k[0] == m for k in by)] + sorted({k[0] for k in by} - set(ORDER))
    is_cap = lambda r: r["n_completion_tokens"] >= CAP[r["task"]] or r["n_completion_tokens"] == 0
    is_strict = lambda r: r["parse_method"] == "strict"
    correct = lambda r: r["correct"]
    lines = ["\\begin{tabular}{llcccrr}", "\\toprule",
             " & & \\multicolumn{2}{c}{acc (\\%) at $T$} & & \\multicolumn{2}{c}{change at 1.3 (pp)} \\\\",
             "Model & Task & 0.7 & 1.3 & drop [95\\% CI] & cap & strict \\\\"]
    summary = {}
    last_group = None
    for m in models:
        g = GROUP.get(m)
        if g != last_group:
            lines.append("\\midrule")
            if g:
                lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{g}}}}} \\\\")
            last_group = g
        first = True
        for task in ("gsm8k", "mmlu_pro"):
            lo, hi = by.get((m, task, 0.7), {}), by.get((m, task, 1.3), {})
            if not (lo and hi):
                continue
            items = sorted(set(lo) & set(hi))
            a7, a13 = rate(lo, items, correct), rate(hi, items, correct)
            d = boot(hi, lo, correct)
            dcap = rate(hi, items, is_cap) - rate(lo, items, is_cap)
            dstr = rate(hi, items, is_strict) - rate(lo, items, is_strict)
            summary[(m, task)] = dict(a7=a7, a13=a13, drop=d, dcap=dcap, dstrict=dstr, n_items=len(items))
            bold = d[1] > 0.15
            dtxt = f"$\\mathbf{{{d[0]*100:+.1f}}}$" if bold else f"${d[0]*100:+.1f}$"
            lines.append(f"{LABEL.get(m, m) if first else ''} & {TASK[task]} & {a7*100:.1f} & {a13*100:.1f} & "
                         f"{dtxt} [{d[1]*100:.1f}, {d[2]*100:.1f}] & ${dcap*100:+.0f}$ & ${dstr*100:+.0f}$ \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/tab_collapse.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump({f"{m}|{t}": v for (m, t), v in summary.items()}, open("results/collapse_q8_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
