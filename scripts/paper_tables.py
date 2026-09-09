#!/usr/bin/env python3
"""Tables for the 2026-09 rewrite, from results/paper_records.parquet and results/sampler_bound_all*.json.

  uv run python scripts/paper_tables.py

Main text:   tables/tab_bound.tex (Table 2). Table 1 (tab_design.tex) is hand-written.
Appendix:    tab_temp_full, tab_survivor, tab_decomp, tab_quant_drops, tab_strat, tab_engine_full,
             tab_grid_gsm8k, tab_grid_mmlu_pro, tab_contrasts, tab_ladder
Also writes results/paper_numbers.json with every number the prose cites.
Bootstrap: item-clustered paired, B=2000 seed 0 (shared with paper_figures.py); the bound family uses B=5000
from sampler_bound_all.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from paper_figures import LABEL, RULES, TASK_NAME, TASKS, drops, item_acc, load, model_order, paired_boot, temp_slice  # noqa: E402

OUT = Path("paper/tables")
TOL = 1e-9
NUM = {}
GRID7 = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", "gemma-3-12b-it", "qwen3-8b", "qwen3-4b", "qwen3-1.7b"]
PANEL3 = ["hermes-3-llama-3.1-8b", "qwen3.5-9b", "olmo-3-7b-instruct"]
CFG_ORDER = [("greedy", "tfirst"), ("temperature", "tfirst"), ("top_p", "tfirst"), ("top_k", "tfirst"), ("min_p", "tfirst"),
             ("top_n_sigma", "tfirst"), ("top_p", "tlast"), ("min_p", "tlast")]
CFG_TEX = {("greedy", "tfirst"): "greedy", ("temperature", "tfirst"): "temperature", ("top_p", "tfirst"): "top-$p$",
           ("top_k", "tfirst"): "top-$k$", ("min_p", "tfirst"): "min-$p$", ("top_n_sigma", "tfirst"): "top-$n\\sigma$",
           ("top_p", "tlast"): "top-$p$ (temp.\\ last)", ("min_p", "tlast"): "min-$p$ (temp.\\ last)"}


def ci(pt, lo, hi, bold=False):
    s = f"${pt:+.1f}$ [{lo:.1f}, {hi:.1f}]"
    return f"$\\mathbf{{{pt:+.1f}}}$ [{lo:.1f}, {hi:.1f}]" if bold else s


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", name)


# ------------------------------------------------------------------ Table 2: bound
def tab_bound():
    S = json.load(open("results/sampler_bound_all_summary.json"))
    rows = [
        ("six robust grid models", "0.7, 1.0", "pooled", S["main grid robust T<=1.0"]),
        ("six robust grid models, Q8\\_0 only", "0.7, 1.0", "Q8", S["main grid robust Q8 T<=1.0"]),
        ("six robust grid models", "1.3", "pooled", None),
        ("Llama-3.1-8B", "0.7, 1.0", "pooled", S["main grid llama T<=1.0"]),
        ("three flagged panel models", "0.7, 1.0", "Q8", S["flagged-model grid Q8 T<=1.0"]),
        ("three flagged panel models", "1.3", "Q8", S["flagged-model grid Q8 T=1.3"]),
    ]
    # robust six at 1.3 is not in the summary; compute from the json
    out = json.load(open("results/sampler_bound_all.json"))
    r13 = [o for o in out if o["source"] == "main_grid" and o["T"] == 1.3 and o["model"] != "llama-3.1-8b-instruct"]
    rows[2] = ("six robust grid models", "1.3", "pooled", dict(n=len(r13), max_gain=max(o["mean"] for o in r13),
                                                              simultaneous=float("nan"), above=sum(o["lo"] > TOL for o in r13)))
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Models & $T$ & contrasts & largest gain (pp) & 95\\% bound (pp) & intervals $>0$ \\\\", "\\midrule"]
    for name, T, prec, s in rows:
        sim = "--" if s["simultaneous"] != s["simultaneous"] else f"{s['simultaneous']:.1f}"
        lines.append(f"{name} & {T} & {s['n']} & ${s['max_gain']:+.1f}$ & {sim} & {s['above']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_bound", lines)
    NUM["bound"] = {k: v for k, v in S.items()}
    NUM["bound"]["robust six T1.3"] = rows[2][3]


# ------------------------------------------------------------------ appendix: temperature table (13 models)
def tab_temp_full(df):
    d = temp_slice(df)
    dr = drops(d)
    order, frag = model_order(dr)
    lines = ["\\begin{tabular}{llcccrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{accuracy (\\%) at $T$} & & \\multicolumn{2}{c}{change 0.7$\\to$1.3 (pp)} \\\\",
             "Model & Task & 0.7 & 1.0 & 1.3 & drop [95\\% CI] & capped & strict \\\\", "\\midrule"]
    rec = {}
    for m in order:
        for i, task in enumerate(TASKS):
            g = d[(d.model == m) & (d.task == task) & (d.sampler == "temperature")]
            acc = {T: g[g["T"] == T].correct.mean() * 100 for T in (0.7, 1.0, 1.3)}
            cap = {T: g[g["T"] == T].at_cap.mean() * 100 for T in (0.7, 1.3)}
            st = {T: (g[g["T"] == T].parse == "strict").mean() * 100 for T in (0.7, 1.3)}
            r = dr[(dr.model == m) & (dr.task == task)].iloc[0]
            rec[(m, task)] = dict(acc=acc, cap=cap, strict=st, drop=r.d, lo=r.lo, hi=r.hi)
            lines.append(f"{LABEL[m] if i == 0 else ''} & {TASK_NAME[task]} & {acc[0.7]:.1f} & {acc[1.0]:.1f} & {acc[1.3]:.1f} & "
                         f"{ci(r.d, r.lo, r.hi, bold=r.lo >= 15)} & ${cap[1.3] - cap[0.7]:+.0f}$ & ${st[1.3] - st[0.7]:+.0f}$ \\\\")
        if m == order[len(frag) - 1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_temp_full", lines)
    NUM["temp"] = {f"{m}|{t}": v for (m, t), v in rec.items()}
    NUM["order"] = order
    NUM["fragile"] = sorted(frag)
    return order, frag


# ------------------------------------------------------------------ appendix: survivor table
def tab_survivor(df, order):
    d = temp_slice(df)
    lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
             " & & \\multicolumn{3}{c}{$T{=}0.7$} & \\multicolumn{4}{c}{$T{=}1.3$} \\\\",
             "Model & Task & strict \\% & acc$\\mid$strict & greedy, same items & strict \\% & $n$ & acc$\\mid$strict & greedy, same items \\\\", "\\midrule"]
    rec = {}
    for m in order:
        for i, task in enumerate(TASKS):
            g = d[(d.model == m) & (d.task == task)]
            greedy = g[g.sampler == "greedy"].set_index("item_id").correct
            vals = {}
            for T in (0.7, 1.3):
                s = g[(g.sampler == "temperature") & (g["T"] == T)]
                st = s[s.parse == "strict"]
                vals[T] = dict(share=len(st) / len(s) * 100, n=len(st), acc=st.correct.mean() * 100, greedy=greedy.reindex(st.item_id).mean() * 100)
            rec[(m, task)] = vals
            lines.append(f"{LABEL[m] if i == 0 else ''} & {TASK_NAME[task]} & {vals[0.7]['share']:.0f} & {vals[0.7]['acc']:.1f} & {vals[0.7]['greedy']:.1f} & "
                         f"{vals[1.3]['share']:.0f} & {vals[1.3]['n']} & {vals[1.3]['acc']:.1f} & {vals[1.3]['greedy']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_survivor", lines)
    NUM["survivor"] = {f"{m}|{t}": v for (m, t), v in rec.items()}


# ------------------------------------------------------------------ appendix: decomposition by parse path
def tab_decomp(df, order):
    d = temp_slice(df)
    lines = ["\\begin{tabular}{ll" + "rrrr" * 2 + "}", "\\toprule",
             " & & \\multicolumn{4}{c}{GSM8K} & \\multicolumn{4}{c}{MMLU-Pro} \\\\",
             "Model & $T$ & strict & flex. & failed & capped & strict & flex. & failed & capped \\\\", "\\midrule"]
    for m in order:
        for i, T in enumerate((0.7, 1.0, 1.3)):
            cells = []
            for task in TASKS:
                s = d[(d.model == m) & (d.task == task) & (d.sampler == "temperature") & (d["T"] == T)]
                cells += [f"{(s.parse == p).mean() * 100:.0f}" for p in ("strict", "flexible", "failed")] + [f"{s.at_cap.mean() * 100:.0f}"]
            lines.append(f"{LABEL[m] if i == 0 else ''} & {T} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_decomp", lines)


# ------------------------------------------------------------------ appendix: per-quantization drops
def tab_quant_drops(df):
    d = df[df.source == "main_grid"]
    levels = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
    lines = ["\\begin{tabular}{ll" + "r" * 4 + "}", "\\toprule",
             "Model & Task & " + " & ".join(l.replace("_", "\\_") for l in levels) + " \\\\", "\\midrule"]
    rec = {}
    for m in GRID7:
        for i, task in enumerate(TASKS):
            cells = []
            for q in levels:
                g = d[(d.model == m) & (d.task == task) & (d.quant == q) & (d.sampler == "temperature")]
                pt, lo, hi = paired_boot(item_acc(g[g["T"] == 0.7]), item_acc(g[g["T"] == 1.3]))
                cells.append(f"${pt:+.1f}$")
                rec[f"{m}|{task}|{q}"] = pt
            lines.append(f"{LABEL[m] if i == 0 else ''} & {TASK_NAME[task]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_quant_drops", lines)
    NUM["quant_drops"] = rec
    # greedy cost of quantization Q8 -> Q3, pooled tasks
    cost = {}
    for m in GRID7:
        g = d[(d.model == m) & (d.sampler == "greedy")]
        cost[m] = (g[g.quant == "Q8_0"].correct.mean() - g[g.quant == "Q3_K_M"].correct.mean()) * 100
    NUM["quant_greedy_cost_Q8_to_Q3"] = cost


# ------------------------------------------------------------------ appendix: stratified
def tab_strat(df):
    head = df[(df.source == "main_grid") & (df.quant == "Q8_0") & (df.task == "mmlu_pro")]
    strat = df[df.source == "strat"]
    lines = ["\\begin{tabular}{llccl}", "\\toprule", "Model & Subset & acc.\\ 0.7 & acc.\\ 1.3 & drop [95\\% CI] \\\\", "\\midrule"]
    rec = {}
    for m in ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]:
        for name, src in (("business (head)", head), ("stratified", strat)):
            g = src[(src.model == m) & (src.sampler == "temperature")]
            a07, a13 = item_acc(g[g["T"] == 0.7]), item_acc(g[g["T"] == 1.3])
            pt, lo, hi = paired_boot(a07, a13)
            rec[f"{m}|{name}"] = dict(acc07=a07.mean() * 100, acc13=a13.mean() * 100, drop=pt, lo=lo, hi=hi)
            lines.append(f"{LABEL[m] if name.startswith('business') else ''} & {name} & {a07.mean() * 100:.1f} & {a13.mean() * 100:.1f} & {ci(pt, lo, hi)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_strat", lines)
    NUM["strat"] = rec


# ------------------------------------------------------------------ appendix: engine and release checks
def tab_engine_full(df):
    arms = [("llama-3.2-3b-instruct", "BF16-hf", "transformers", "BF16, transformers"),
            ("llama-3.2-3b-instruct", "F16", "llama.cpp", "F16 GGUF, llama.cpp"),
            ("llama-3.2-3b-instruct", "Q8_0", "llama.cpp-v040", "Q8 GGUF, llama.cpp v0.4.0"),
            ("llama-3.2-3b-instruct", "Q8_0", "llama.cpp", "Q8 GGUF, llama.cpp (reference)"),
            ("llama-3.1-8b-instruct", "INT8-hf", "transformers", "INT8, transformers"),
            ("llama-3.1-8b-instruct", "Q8_0", "llama.cpp", "Q8 GGUF, llama.cpp (reference)"),
            ("qwen3-4b", "BF16-hf", "transformers", "BF16, transformers"),
            ("qwen3-4b", "BF16", "llama.cpp", "BF16 GGUF, llama.cpp"),
            ("qwen3-4b", "Q8_0", "llama.cpp-v040", "Q8 GGUF, llama.cpp v0.4.0"),
            ("qwen3-4b", "Q8_0", "llama.cpp", "Q8 GGUF, llama.cpp (reference)")]
    lines = ["\\begin{tabular}{ll" + "cclcl" * 1 + "cclcl}", "\\toprule",
             " & & \\multicolumn{5}{c}{GSM8K} & \\multicolumn{5}{c}{MMLU-Pro} \\\\",
             "Model & Precision, engine & 0.7 & 1.3 & drop [CI] & cap & vs.\\ ref.\\ [CI] & 0.7 & 1.3 & drop [CI] & cap & vs.\\ ref.\\ [CI] \\\\", "\\midrule"]
    rec = {}
    last = None
    for m, q, eng, name in arms:
        g = df[(df.model == m) & (df.quant == q) & (df.engine == eng) & (df.sampler == "temperature") & df.source.isin(["engine", "bridge", "main_grid", "lineage"])]
        ref = df[(df.model == m) & (df.quant == "Q8_0") & (df.engine == "llama.cpp") & (df.sampler == "temperature") & df.source.isin(["main_grid", "lineage"])]
        cells = []
        for task in TASKS:
            gt, rt = g[g.task == task], ref[ref.task == task]
            a07, a13 = item_acc(gt[gt["T"] == 0.7]), item_acc(gt[gt["T"] == 1.3])
            pt, lo, hi = paired_boot(a07, a13)
            cap = gt[gt["T"] == 1.3].at_cap.mean() * 100
            r07, r13 = item_acc(rt[rt["T"] == 0.7]), item_acc(rt[rt["T"] == 1.3])
            dd = (a07 - a13) - (r07 - r13)  # difference in drops, per item
            zero = pd.Series(0.0, index=dd.index)
            dpt, dlo, dhi = paired_boot(dd, zero)
            isref = "reference" in name
            cells += [f"{a07.mean() * 100:.1f}", f"{a13.mean() * 100:.1f}", ci(pt, lo, hi), f"{cap:.0f}", "--" if isref else ci(dpt, dlo, dhi)]
            rec[f"{m}|{name}|{task}"] = dict(acc07=a07.mean() * 100, acc13=a13.mean() * 100, drop=pt, lo=lo, hi=hi, cap13=cap, dd=dpt, dlo=dlo, dhi=dhi)
        if last and last != m:
            lines.append("\\midrule")
        lines.append(f"{LABEL[m] if last != m else ''} & {name} & " + " & ".join(cells) + " \\\\")
        last = m
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_engine_full", lines)
    NUM["engine"] = rec


# ------------------------------------------------------------------ appendix: per-configuration grids
def tab_grids(df):
    full = pd.read_parquet("results/paper_records.parquet")  # includes the temperature-last ablations
    for task in TASKS:
        halves = []
        # left half: first five grid models; right half: last two grid models + three flagged models
        groups = [[("main_grid", m) for m in GRID7[:5]], [("main_grid", m) for m in GRID7[5:]] + [("panel_grid", m) for m in PANEL3]]
        for grp in groups:
            rows = []
            for src, m in grp:
                d = full[(full.source == src) & (full.task == task) & (full.model == m)]
                for i, (samp, chain) in enumerate(CFG_ORDER):
                    g = d[(d.sampler == samp) & (d.chain == chain)]
                    if samp == "greedy":
                        cells = [f"{g.correct.mean() * 100:.1f}", "--", "--"]
                    else:
                        cells = [f"{g[g['T'] == T].correct.mean() * 100:.1f}" for T in (0.7, 1.0, 1.3)]
                    rows.append(f"{LABEL[m] if i == 0 else ''} & {CFG_TEX[(samp, chain)]} & " + " & ".join(cells))
                rows.append("\\midrule")
            halves.append(rows[:-1])
        n = max(len(h) for h in halves)
        for h in halves:
            h += [" & & & & "] * (n - len(h))
        lines = ["\\begin{tabular}{llccc@{\\hspace{14pt}}llccc}", "\\toprule",
                 "Model & Configuration & 0.7 & 1.0 & 1.3 & Model & Configuration & 0.7 & 1.0 & 1.3 \\\\", "\\midrule"]
        for a, b in zip(*halves):
            if a == "\\midrule" and b == "\\midrule":
                lines.append("\\midrule"); continue
            aa = " & & & & " if a == "\\midrule" else a
            bb = " & & & & " if b == "\\midrule" else b
            lines.append(f"{aa} & {bb} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"tab_grid_{task}", lines)


# ------------------------------------------------------------------ appendix: all paired contrasts incl. ablations
def tab_contrasts(df):
    full = pd.read_parquet("results/paper_records.parquet")  # includes tlast
    rec = {}
    for task in TASKS:
        lines = ["\\begin{tabular}{lc" + "l" * 6 + "}", "\\toprule",
                 "Model & $T$ & top-$p$ & top-$k$ & min-$p$ & top-$n\\sigma$ & top-$p$ (temp.\\ last) & min-$p$ (temp.\\ last) \\\\", "\\midrule"]
        for src, models in (("main_grid", GRID7), ("panel_grid", PANEL3)):
            s = full[(full.source == src) & (full.task == task)]
            for m in models:
                for i, T in enumerate((0.7, 1.0, 1.3)):
                    g = s[(s.model == m) & (s["T"] == T)]
                    base = item_acc(g[(g.sampler == "temperature") & (g.chain == "tfirst")])
                    cells = []
                    for samp, chain in [(r, "tfirst") for r in RULES] + [("top_p", "tlast"), ("min_p", "tlast")]:
                        x = item_acc(g[(g.sampler == samp) & (g.chain == chain)])
                        pt, lo, hi = paired_boot(x, base)
                        cells.append(ci(pt, lo, hi, bold=lo > TOL))
                        rec[f"{m}|{task}|{T}|{samp}|{chain}"] = dict(pt=pt, lo=lo, hi=hi)
                    lines.append(f"{LABEL[m] if i == 0 else ''} & {T} & " + " & ".join(cells) + " \\\\")
                lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        write(f"tab_contrasts_{task}", lines)
    NUM["contrasts"] = rec


# ------------------------------------------------------------------ appendix: ladder drops
def tab_ladder(df):
    models = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]
    grid = df[(df.source == "main_grid") & (df.quant == "Q8_0")]
    lad = df[df.source == "ladder"]
    lines = ["\\begin{tabular}{lc" + "l" * 3 + "}", "\\toprule", "Task & $T$ & " + " & ".join(LABEL[m] for m in models) + " \\\\", "\\midrule"]
    rec = {}
    for task in TASKS:
        for T in (1.3, 1.5, 1.7, 2.0):
            cells = []
            for m in models:
                b = item_acc(grid[(grid.model == m) & (grid.task == task) & (grid["T"] == 0.7) & (grid.sampler == "temperature")])
                x = item_acc(lad[(lad.model == m) & (lad.task == task) & (lad["T"] == T) & (lad.sampler == "temperature")])
                pt, lo, hi = paired_boot(b, x)
                cells.append(ci(pt, lo, hi))
                rec[f"{m}|{task}|{T}"] = dict(drop=pt, lo=lo, hi=hi, acc=x.mean() * 100, base=b.mean() * 100)
            lines.append(f"{TASK_NAME[task] if T == 1.3 else ''} & {T} & " + " & ".join(cells) + " \\\\")
        if task == TASKS[0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_ladder", lines)
    NUM["ladder"] = rec
    # cliff: first tested T at which plain temperature loses more than half of its T0.7 accuracy; per rule
    cliff = {}
    for m in models:
        for task in TASKS:
            b = grid[(grid.model == m) & (grid.task == task) & (grid["T"] == 0.7) & (grid.sampler == "temperature")].correct.mean()
            for rule in ("temperature", "top_p", "min_p", "top_n_sigma"):
                first = None
                for T in (1.0, 1.3, 1.5, 1.7, 2.0):
                    src = grid if T <= 1.0 else lad
                    a = src[(src.model == m) & (src.task == task) & (src["T"] == T) & (src.sampler == rule)].correct.mean()
                    if a < b / 2:
                        first = T
                        break
                cliff[f"{m}|{task}|{rule}"] = first
    NUM["cliff"] = cliff


def main():
    df = load()
    tab_bound()
    order, frag = tab_temp_full(df)
    tab_survivor(df, order)
    tab_decomp(df, order)
    tab_quant_drops(df)
    tab_strat(df)
    tab_engine_full(df)
    tab_grids(df)
    tab_contrasts(df)
    tab_ladder(df)
    json.dump(NUM, open("results/paper_numbers.json", "w"), indent=1, default=float)
    print("wrote results/paper_numbers.json")


if __name__ == "__main__":
    main()
