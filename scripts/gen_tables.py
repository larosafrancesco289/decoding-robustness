#!/usr/bin/env python3
"""Generate the appendix LaTeX tables from raw result JSONLs -> paper/tables/*.tex.

  uv run python scripts/gen_tables.py

Tables:
  tab_grid_gsm8k.tex / tab_grid_mmlu_pro.tex  sampler x temperature accuracy per model (App. A)
  tab_decomp.tex                              degeneration decomposition (App. B)
  tab_strat.tex                               stratified vs business MMLU-Pro subset (App. C)
  tab_hightemp.tex                            T=2-3 probe (App. D)
  tab_tlast.tex                               temperature-first vs -last arms (App. E)
  tab_selfcons.tex                            self-consistency, distinct answers/item (App. G)
  tab_quantdrop.tex                           plain-temp drop per quant level (App. quant)
  tab_recovery.tex                            terminated-length censoring (App. B)
  tab_mechanism.tex                           probe + escape + perturbation per model (App. H)
  tab_lineage.tex                             Llama lineage cell vs Q8 Llama-3.1 (App. D)

Pure stdlib. Accuracies pooled over quant unless stated. Booktabs format, \\input-able.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLER_LABEL = {
    "greedy": "greedy", "temperature": "temperature",
    "top_p_top_p0.95": "top_p", "top_k_top_k40": "top_k",
    "min_p_min_p0.05": "min_p", "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
    "min_p_min_p0.05_tlast": "top_p_tlast_minp", "top_p_top_p0.95_tlast": "top_p_tlast",
}
# careful: distinct labels for the two tlast arms
SAMPLER_LABEL["min_p_min_p0.05_tlast"] = "min_p_tlast"
TEX_SAMPLER = {"greedy": "greedy", "temperature": "temperature", "top_p": "top-$p$",
               "top_k": "top-$k$", "min_p": "min-$p$", "top_n_sigma": "top-$n\\sigma$",
               "top_p_tlast": "top-$p$ (temp-last)", "min_p_tlast": "min-$p$ (temp-last)"}
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
SHORT = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "mistral-7b-instruct-v0.3": "Mistral-7B",
         "qwen2.5-7b-instruct": "Qwen2.5-7B", "gemma-3-12b-it": "Gemma-3-12B",
         "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
         "llama-3-8b-instruct": "Llama-3-8B", "llama-3.2-3b-instruct": "Llama-3.2-3B"}
TASKS = ["gsm8k", "mmlu_pro"]
TASK_NAME = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}
TEMPS = [0.7, 1.0, 1.3]
CAP = {"gsm8k": 512, "mmlu_pro": 1024}
GRID_ORDER = ["greedy", "temperature", "top_k", "top_p", "min_p", "top_n_sigma",
              "top_p_tlast", "min_p_tlast"]


def load(pattern):
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def write(name, lines):
    path = OUT / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def pct(x, nd=1):
    return f"{x*100:.{nd}f}" if x == x else "--"


# ------------------------------------------------------------------ load matrix
cells = defaultdict(lambda: [0, 0])      # (model,task,sampler,temp) -> [n_correct, n]
cells_q8 = defaultdict(lambda: [0, 0])   # same, Q8_0 only (for the strat comparison)
lin31 = defaultdict(lambda: defaultdict(list))  # (task,T) -> item -> [correct]; Q8 Llama-3.1
decomp = defaultdict(lambda: defaultdict(int))  # same key -> counters
ans = defaultdict(lambda: defaultdict(list))    # selfcons: (m,task,T)->(item,quant)->[ans]
qd = defaultdict(lambda: [0, 0])         # (model,task,quant,temp) -> [n_correct, n]; plain temp
term_lens = defaultdict(list)            # (model,task,temp) -> terminated completion lengths
for r in load("results/full_matrix/shards/*.jsonl"):
    lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
    key = (r["model"], r["task"], lab, r["temperature"])
    cells[key][0] += int(r["correct"])
    cells[key][1] += 1
    if r["quant"] == "Q8_0":
        cells_q8[key][0] += int(r["correct"])
        cells_q8[key][1] += 1
        if r["model"] == "llama-3.1-8b-instruct" and lab == "temperature":
            lin31[(r["task"], r["temperature"])][r["item_id"]].append(int(r["correct"]))
    if lab == "temperature":
        if r["temperature"] in (0.7, 1.3):
            k = (r["model"], r["task"], r["quant"], r["temperature"])
            qd[k][0] += int(r["correct"])
            qd[k][1] += 1
        d = decomp[key]
        d["n"] += 1
        d["acc"] += int(r["correct"])
        d[r["parse_method"]] += 1
        if r["parse_method"] == "strict":
            d["acc_strict"] += int(r["correct"])
        # n_completion_tokens == 0 marks a server-recovered degenerate stream (run-to-cap)
        if r["n_completion_tokens"] >= CAP[r["task"]] or r["n_completion_tokens"] == 0:
            d["cap"] += 1
        elif r["temperature"] in (0.7, 1.3):
            term_lens[(r["model"], r["task"], r["temperature"])].append(
                r["n_completion_tokens"])
        ans[(r["model"], r["task"], r["temperature"])][(r["item_id"], r["quant"])].append(
            r["parsed_answer"])

# ------------------------------------------------------------------ A: full grids
for task in TASKS:
    lines = [
        "\\begin{tabular}{llccc}", "\\toprule",
        "Model & Sampler & $T{=}0.7$ & $T{=}1.0$ & $T{=}1.3$ \\\\",
    ]
    for m in MODELS:
        lines.append("\\midrule")
        first = True
        for s in GRID_ORDER:
            if s == "greedy":
                n_c, n = cells[(m, task, s, 0.0)]
                if not n:
                    continue
                row = ["greedy ($T{=}0$)", pct(n_c / n), "--", "--"]
            else:
                vals = []
                for t in TEMPS:
                    n_c, n = cells[(m, task, s, t)]
                    vals.append(pct(n_c / n) if n else "--")
                if all(v == "--" for v in vals):
                    continue
                row = [TEX_SAMPLER[s]] + vals
            head = SHORT[m] if first else ""
            first = False
            lines.append(f"{head} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"tab_grid_{task}.tex", lines)

# ------------------------------------------------------------------ B: decomposition
lines = [
    "\\begin{tabular}{llcccccc}", "\\toprule",
    "Model & Task & $T$ & acc & strict & failed & at cap & acc$|$strict \\\\",
]
for m in MODELS:
    lines.append("\\midrule")
    first = True
    for task in TASKS:
        for t in TEMPS:
            d = decomp[(m, task, "temperature", t)]
            n = d["n"]
            if not n:
                continue
            acs = d["acc_strict"] / d["strict"] if d["strict"] else float("nan")
            lines.append(
                f"{SHORT[m] if first else ''} & {TASK_NAME[task]} & {t} & "
                f"{pct(d['acc']/n)} & {pct(d['strict']/n)} & {pct(d['failed']/n)} & "
                f"{pct(d['cap']/n)} & {pct(acs)} \\\\")
            first = False
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_decomp.tex", lines)

# ------------------------------------------------------------------ C: stratified check
strat = defaultdict(lambda: [0, 0])
for r in load("results/mmlu_strat_check/*.jsonl"):
    lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
    strat[(r["model"], lab, r["temperature"])][0] += int(r["correct"])
    strat[(r["model"], lab, r["temperature"])][1] += 1
lines = [
    "\\begin{tabular}{llcccc}", "\\toprule",
    " & & \\multicolumn{3}{c}{$T$} & \\\\",
    "Model & Subset & 0.7 & 1.0 & 1.3 & drop \\\\",
]
for m in ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]:
    lines.append("\\midrule")
    for name, get in (
        ("stratified", lambda t, m=m: strat[(m, "temperature", t)]),
        ("head-50", lambda t, m=m: cells_q8[(m, "mmlu_pro", "temperature", t)]),
    ):
        vals = []
        for t in TEMPS:
            c, n = get(t)
            vals.append(c / n if n else float("nan"))
        drop = (vals[0] - vals[2]) * 100 if vals[0] == vals[0] else float("nan")
        lines.append(
            f"{SHORT[m]} & {name} & " + " & ".join(pct(v) for v in vals)
            + f" & ${drop:+.1f}$\\,pp \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_strat.tex", lines)

# ------------------------------------------------------------------ D: high-T probe
ht = defaultdict(lambda: [0, 0])
for r in load("results/hightemp_smoke/*.jsonl"):
    lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
    ht[(r["model"], r["task"], lab, r["temperature"])][0] += int(r["correct"])
    ht[(r["model"], r["task"], lab, r["temperature"])][1] += 1
lines = [
    "\\begin{tabular}{llcccc}", "\\toprule",
    " & & \\multicolumn{2}{c}{Llama-3.1-8B} & \\multicolumn{2}{c}{Gemma-3-12B} \\\\",
    "Task & Sampler & $T{=}2$ & $T{=}3$ & $T{=}2$ & $T{=}3$ \\\\",
]
for task in TASKS:
    lines.append("\\midrule")
    for s in ["temperature", "top_p", "min_p", "top_n_sigma"]:
        row = [TASK_NAME[task] if s == "temperature" else "", TEX_SAMPLER[s]]
        for m in ["llama-3.1-8b-instruct", "gemma-3-12b-it"]:
            for t in (2.0, 3.0):
                c, n = ht[(m, task, s, t)]
                row.append(pct(c / n, 0) if n else "--")
        lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_hightemp.tex", lines)

# ------------------------------------------------------------------ E: tlast arms
lines = [
    "\\begin{tabular}{llcccccc}", "\\toprule",
    " & & \\multicolumn{3}{c}{temperature-first} & \\multicolumn{3}{c}{temperature-last} \\\\",
    "Model & Method & 0.7 & 1.0 & 1.3 & 0.7 & 1.0 & 1.3 \\\\",
]
for task in TASKS:
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{TASK_NAME[task]}}}}} \\\\")
    for m in MODELS:
        for base, tl in (("top_p", "top_p_tlast"), ("min_p", "min_p_tlast")):
            row = [SHORT[m], TEX_SAMPLER[base]]
            ok = False
            for s in (base, tl):
                for t in TEMPS:
                    c, n = cells[(m, task, s, t)]
                    row.append(pct(c / n) if n else "--")
                    ok = ok or n > 0
            if ok:
                lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_tlast.tex", lines)

# ------------------------------------------------------------------ G: self-consistency
lines = [
    "\\begin{tabular}{lcccc}", "\\toprule",
    " & \\multicolumn{2}{c}{GSM8K} & \\multicolumn{2}{c}{MMLU-Pro} \\\\",
    "Model & $T{=}0.7$ & $T{=}1.3$ & $T{=}0.7$ & $T{=}1.3$ \\\\", "\\midrule",
]
for m in MODELS:
    row = [SHORT[m]]
    for task in TASKS:
        for t in (0.7, 1.3):
            groups = ans[(m, task, t)]
            if groups:
                vals = [len(set(v)) for v in groups.values()]
                row.append(f"{sum(vals)/len(vals):.2f}")
            else:
                row.append("--")
    lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_selfcons.tex", lines)

# ------------------------------------------------------------------ quant drops
QUANTS = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
lines = [
    "\\begin{tabular}{lcccc}", "\\toprule",
    "Model & Q8\\_0 & Q6\\_K & Q4\\_K\\_M & Q3\\_K\\_M \\\\",
]
for task in TASKS:
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{TASK_NAME[task]}}}}} \\\\")
    for m in MODELS:
        row = [SHORT[m]]
        for q in QUANTS:
            a07, a13 = qd.get((m, task, q, 0.7)), qd.get((m, task, q, 1.3))
            if a07 and a13:
                d = 100 * (a07[0] / a07[1] - a13[0] / a13[1])
                row.append(f"${d:+.1f}$")
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_quantdrop.tex", lines)

# ------------------------------------------------------------------ B2: length censoring
def pctl(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


lines = [
    "\\begin{tabular}{llccccc}", "\\toprule",
    " & & \\multicolumn{2}{c}{at cap (\\%)} & \\multicolumn{2}{c}{p90 length} & \\\\",
    "Model & Task & 0.7 & 1.3 & 0.7 & 1.3 & ratio \\\\",
]
for m in MODELS:
    lines.append("\\midrule")
    first = True
    for task in TASKS:
        d7 = decomp[(m, task, "temperature", 0.7)]
        d13 = decomp[(m, task, "temperature", 1.3)]
        l7 = term_lens[(m, task, 0.7)]
        l13 = term_lens[(m, task, 1.3)]
        if not d7["n"] or not l7 or not l13:
            continue
        p7, p13 = pctl(l7, 0.9), pctl(l13, 0.9)
        lines.append(
            f"{SHORT[m] if first else ''} & {TASK_NAME[task]} & "
            f"{pct(d7['cap']/d7['n'])} & {pct(d13['cap']/d13['n'])} & "
            f"{p7} & {p13} & {p13/p7:.2f}$\\times$ \\\\")
        first = False
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_recovery.tex", lines)

# ------------------------------------------------------------------ H: mechanism probes
def esc_cont(probs, temp=1.3):
    inv = 1.0 / temp
    s_top = sum(p ** inv for p in probs if p > 0)
    t = max(0.0, 1.0 - sum(probs))
    if t <= 0 or s_top <= 0:
        return 0.0
    p20 = probs[-1]
    s_tail = (t / p20) * (p20 ** inv) if p20 > 0 else t ** inv
    return s_tail / (s_top + s_tail)


probe = defaultdict(lambda: defaultdict(list))
for r in load("results/logit_probe/*.jsonl"):
    probe[r["model"]]["ent"].append(r["mean_entropy_lb"])
    probe[r["model"]]["flat"].append(r["frac_flat_pos"])
for r in load("results/logit_probe_v2/*.jsonl"):
    for pos in r["probs"]:
        probe[r["model"]]["esc"].append(esc_cont(pos))
for r in load("results/perturbation/*.jsonl"):
    if "ent32" in r["inj"] and "ent32" in r["ctrl"]:
        probe[r["model"]]["d_ent"].append(r["inj"]["ent32"] - r["ctrl"]["ent32"])
        probe[r["model"]]["d_esc"].append(r["inj"]["esc32"] - r["ctrl"]["esc32"])

drop_t = defaultdict(lambda: [0, 0])
for (m, task, lab, t), (c, n) in cells.items():
    if task == "mmlu_pro" and lab == "temperature" and t in (0.7, 1.3):
        drop_t[(m, t)] = [c, n]
lines = [
    "\\begin{tabular}{lcccccc}", "\\toprule",
    "Model & $H_{lb}$ & flat\\% & esc & $\\Delta$esc & $\\Delta H$ & drop \\\\",
    "\\midrule",
]
for m in MODELS:
    d = probe[m]
    if not d.get("ent"):
        continue
    lo, hi = drop_t[(m, 0.7)], drop_t[(m, 1.3)]
    drop = (lo[0] / lo[1] - hi[0] / hi[1]) * 100
    mean_ = lambda v: sum(v) / len(v)  # noqa: E731
    lines.append(
        f"{SHORT[m]} & {mean_(d['ent']):.3f} & {mean_(d['flat'])*100:.1f} & "
        f"{mean_(d['esc'])*1000:.2f} & {mean_(d['d_esc'])*1000:+.2f} & "
        f"{mean_(d['d_ent']):+.3f} & {drop:+.1f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_mechanism.tex", lines)

# ------------------------------------------------------------------ L: llama lineage cell
import random  # noqa: E402


def drop_ci(by7, by13, B=2000, seed=0):
    """Paired item-clustered bootstrap of the T0.7 -> T1.3 accuracy drop, in pp."""
    rng = random.Random(seed)
    items = sorted(set(by7) & set(by13))
    d = [100 * (sum(by7[i]) / len(by7[i]) - sum(by13[i]) / len(by13[i])) for i in items]
    n = len(items)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, boots[int(0.025 * B)], boots[int(0.975 * B) - 1]


def ci_num(x):
    return f"$-{abs(x):.1f}$" if x < 0 else f"{x:.1f}"


lin = defaultdict(lambda: defaultdict(list))  # (model,task,T) -> item -> [correct]
for r in load("results/llama_lineage/lineage.jsonl"):
    if SAMPLER_LABEL.get(r["sampler"], r["sampler"]) == "temperature":
        lin[(r["model"], r["task"], r["temperature"])][r["item_id"]].append(int(r["correct"]))
for (task, t), by in lin31.items():
    lin[("llama-3.1-8b-instruct", task, t)] = by

lines = [
    "\\begin{tabular}{llccc}", "\\toprule",
    " & & \\multicolumn{2}{c}{acc (\\%) at $T$} & \\\\",
    "Model & Task & 0.7 & 1.3 & drop [95\\% CI] \\\\",
]
for m in ["llama-3-8b-instruct", "llama-3.1-8b-instruct", "llama-3.2-3b-instruct"]:
    lines.append("\\midrule")
    first = True
    for task in TASKS:
        accs = []
        for t in (0.7, 1.3):
            vals = [c for lst in lin[(m, task, t)].values() for c in lst]
            accs.append(pct(sum(vals) / len(vals)))
        d, lo, hi = drop_ci(lin[(m, task, 0.7)], lin[(m, task, 1.3)])
        lines.append(
            f"{SHORT[m] if first else ''} & {TASK_NAME[task]} & " + " & ".join(accs)
            + f" & ${d:+.1f}$\\,[{ci_num(lo)},\\,{ci_num(hi)}] \\\\")
        first = False
lines += ["\\bottomrule", "\\end{tabular}"]
write("tab_lineage.tex", lines)

print("done.")
