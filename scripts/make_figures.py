#!/usr/bin/env python3
"""Generate the paper figures from the raw result JSONLs (no cached intermediates).

  uv sync --extra figures
  uv run python scripts/make_figures.py        # writes figures/*.pdf + .png

Figures (house style: serif, despined, direct labels, Okabe-Ito palette, vector PDF):
  fig1_collapse       forest plot: pure-temperature drop T0.7->T1.3 per model x task
  fig2_decomposition  accuracy vs accuracy|strict-parse across T (the degeneration story)
  fig3_cliff          cliff ladder: accuracy vs T per sampler; deployment band shaded
  fig4_spread         forest plot: sampler max-min spread at T=1.3 per model
  fig5_quant          forest plot: greedy accuracy drop Q8_0 -> Q3_K_M per model

All CIs are item-clustered nonparametric bootstrap (B=2000), matching scripts/stats_matrix.py.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

B = 2000
RNG = np.random.default_rng(0)
OUT = Path("figures")

# ---------------------------------------------------------------- house style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

SAMPLER_LABEL = {
    "greedy": "greedy", "temperature": "temperature",
    "top_p_top_p0.95": "top_p", "top_k_top_k40": "top_k",
    "min_p_min_p0.05": "min_p", "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
    "min_p_min_p0.05_tlast": "min_p(tlast)", "top_p_top_p0.95_tlast": "top_p(tlast)",
}
NONGREEDY = ["temperature", "top_k", "top_p", "min_p", "top_n_sigma",
             "top_p(tlast)", "min_p(tlast)"]
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
          "gemma-3-12b-it", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
SHORT = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "mistral-7b-instruct-v0.3": "Mistral-7B",
         "qwen2.5-7b-instruct": "Qwen2.5-7B", "gemma-3-12b-it": "Gemma-3-12B",
         "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}
TASKS = ["gsm8k", "mmlu_pro"]
TASK_NAME = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}
# Okabe-Ito colorblind-safe palette
C_FRAGILE = "#D55E00"   # vermillion: Llama / temperature
C_ROBUST = "#0072B2"    # blue
SAMPLER_COLOR = {"temperature": "#D55E00", "top_p": "#E69F00", "top_k": "#999999",
                 "min_p": "#0072B2", "top_n_sigma": "#009E73", "greedy": "#000000"}
SAMPLER_TEX = {"temperature": "temperature", "top_p": "top-$p$", "top_k": "top-$k$",
               "min_p": "min-$p$", "top_n_sigma": "top-$n\\sigma$"}
LADDER_SAMPLERS = ["temperature", "top_p", "min_p", "top_n_sigma"]


# ---------------------------------------------------------------- data + stats
def load_records(pattern, q8_only=False):
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if q8_only and r["quant"] != "Q8_0":
                continue
            yield r


def acc(item_map, items):
    num = den = 0
    for it in items:
        vals = item_map.get(it)
        if vals:
            num += sum(vals)
            den += len(vals)
    return num / den if den else float("nan")


def boot_drop(map_a, map_b):
    items = np.array(sorted(set(map_a) | set(map_b)), dtype=object)
    if not len(items):
        return None
    point = acc(map_a, items) - acc(map_b, items)
    draws = [acc(map_a, s) - acc(map_b, s)
             for s in (RNG.choice(items, len(items), replace=True) for _ in range(B))]
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def boot_spread(maps):
    items = np.array(sorted(set().union(*[set(m) for m in maps.values()])), dtype=object)

    def spread(it):
        vals = [acc(m, it) for m in maps.values()]
        vals = [v for v in vals if v == v]
        return max(vals) - min(vals) if vals else float("nan")

    point = spread(items)
    draws = [spread(RNG.choice(items, len(items), replace=True)) for _ in range(B)]
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def collect_matrix():
    by = defaultdict(lambda: defaultdict(list))
    by_q8 = defaultdict(lambda: defaultdict(list))
    strict = defaultdict(lambda: [0, 0])
    acc_strict = defaultdict(lambda: [0, 0])
    quant_greedy = defaultdict(lambda: defaultdict(list))
    for r in load_records("results/full_matrix/shards/*.jsonl"):
        lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
        key = (r["model"], r["task"], lab, r["temperature"])
        by[key][r["item_id"]].append(int(r["correct"]))
        if r["quant"] == "Q8_0":
            by_q8[key][r["item_id"]].append(int(r["correct"]))
        strict[key][0] += int(r["parse_method"] == "strict")
        strict[key][1] += 1
        if r["parse_method"] == "strict":
            acc_strict[key][0] += int(r["correct"])
            acc_strict[key][1] += 1
        if lab == "greedy":
            quant_greedy[(r["model"], r["quant"])][f"{r['task']}|{r['item_id']}"].append(
                int(r["correct"]))
    return by, by_q8, strict, acc_strict, quant_greedy


def collect_ladder():
    by = defaultdict(lambda: defaultdict(list))
    for r in load_records("results/cliff_ladder/*.jsonl"):
        lab = SAMPLER_LABEL.get(r["sampler"], r["sampler"])
        by[(r["model"], r["task"], lab, r["temperature"])][r["item_id"]].append(int(r["correct"]))
    return by


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def forest(ax, rows, xlabel, *, zero_line=True, highlight=lambda m: m.startswith("llama")):
    """One forest panel: rows = [(model, point_pp, lo_pp, hi_pp)] top-to-bottom."""
    n = len(rows)
    for i, (m, p, lo, hi) in enumerate(rows):
        y = n - 1 - i
        color = C_FRAGILE if highlight(m) else C_ROBUST
        ax.plot([lo, hi], [y, y], "-", color=color, lw=1.3, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.16, y + 0.16], "-", color=color, lw=1.3)
        ax.plot([hi, hi], [y - 0.16, y + 0.16], "-", color=color, lw=1.3)
        ax.plot(p, y, "o", color=color, ms=4.5, zorder=3)
    if zero_line:
        ax.axvline(0, color="#444444", lw=0.7, ls=":")
    ax.set_yticks(range(n), [SHORT[m] for m, *_ in rows][::-1])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, n - 0.4)


# ---------------------------------------------------------------- figures
def fig1_collapse(by):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True,
                             constrained_layout=True)
    for ax, task in zip(axes, TASKS, strict=True):
        rows = []
        for m in MODELS:
            r = boot_drop(by[(m, task, "temperature", 0.7)], by[(m, task, "temperature", 1.3)])
            if r:
                d, lo, hi = r
                rows.append((m, d * 100, lo * 100, hi * 100))
        forest(ax, rows, "accuracy drop $T0.7 \\to T1.3$ (pp)")
        ax.set_title(TASK_NAME[task])
    save(fig, "fig1_collapse")


def fig2_decomposition(by, acc_strict):
    temps = [0.7, 1.0, 1.3]
    show = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it",
            "mistral-7b-instruct-v0.3"]
    palette = {"llama-3.1-8b-instruct": C_FRAGILE, "qwen2.5-7b-instruct": C_ROBUST,
               "gemma-3-12b-it": "#009E73", "mistral-7b-instruct-v0.3": "#CC79A7"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True,
                             constrained_layout=True)
    for ax, task in zip(axes, TASKS, strict=True):
        label_pos = []
        for m in show:
            overall = [acc(by[(m, task, "temperature", t)],
                           list(by[(m, task, "temperature", t)])) * 100 for t in temps]
            cond = []
            for t in temps:
                num, den = acc_strict[(m, task, "temperature", t)]
                cond.append(num / den * 100 if den else np.nan)
            ax.plot(temps, overall, "-o", color=palette[m], ms=3.4, lw=1.5)
            ax.plot(temps, cond, "--s", color=palette[m], alpha=0.5, ms=3.2, lw=1.2)
            # direct label at the right end of the solid line (left panel only labels too)
            if task == "mmlu_pro":
                label_pos.append((m, overall[-1]))
        # dodge direct labels: line endpoints can sit ~2pp apart (Gemma vs Qwen2.5)
        if label_pos:
            min_gap = 7.0
            label_pos.sort(key=lambda t: t[1])
            ys = [y for _, y in label_pos]
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i - 1] + min_gap)
            for (m, _), y in zip(label_pos, ys):
                ax.annotate(SHORT[m], (temps[-1], y),
                            xytext=(5, 0), textcoords="offset points",
                            color=palette[m], fontsize=7.5, va="center")
        ax.set_title(TASK_NAME[task])
        ax.set_xlabel("temperature")
        ax.set_xticks(temps)
        ax.set_ylim(0, 102)
        ax.set_xlim(0.65, 1.35 if task == "gsm8k" else 1.62)
    axes[0].set_ylabel("accuracy (%)")
    # style legend (solid vs dashed), not model legend
    solid = plt.Line2D([], [], color="#444444", ls="-", marker="o", ms=3.4, lw=1.5)
    dashed = plt.Line2D([], [], color="#444444", ls="--", marker="s", ms=3.2, lw=1.2,
                        alpha=0.6)
    axes[0].legend([solid, dashed],
                   ["overall accuracy", "accuracy $|$ well-formed answer"],
                   loc="lower left", frameon=False, handlelength=2.4)
    save(fig, "fig2_decomposition")


def fig3_cliff(matrix_q8, ladder):
    anchors = [0.7, 1.0]
    rungs = [1.3, 1.5, 1.7, 2.0]
    show = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]
    fig, axes = plt.subplots(2, len(show), figsize=(6.6, 4.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, m in enumerate(show):
        for row, task in enumerate(TASKS):
            ax = axes[row][col]
            ax.axvspan(0.65, 1.3, color="#F2F2F2", zorder=0)  # deployment range
            for s in LADDER_SAMPLERS:
                xs, ys = [], []
                for t in anchors:
                    k = matrix_q8.get((m, task, s, t))
                    if k:
                        xs.append(t)
                        ys.append(acc(k, list(k)) * 100)
                for t in rungs:
                    k = ladder.get((m, task, s, t))
                    if k:
                        xs.append(t)
                        ys.append(acc(k, list(k)) * 100)
                ax.plot(xs, ys, "-o", color=SAMPLER_COLOR[s], ms=2.8, lw=1.4,
                        label=SAMPLER_TEX[s])
            if row == 0:
                ax.set_title(SHORT[m])
            if col == 0:
                ax.set_ylabel(f"{TASK_NAME[task]}\naccuracy (%)")
            if row == 1:
                ax.set_xlabel("temperature")
            ax.set_ylim(-2, 100)
            ax.set_xlim(0.62, 2.06)
            ax.set_xticks([0.7, 1.0, 1.3, 1.5, 1.7, 2.0])
            ax.tick_params(labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "fig3_cliff")


def fig4_spread(by):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True,
                             constrained_layout=True)
    for ax, task in zip(axes, TASKS, strict=True):
        rows = []
        for m in MODELS:
            maps = {s: by[(m, task, s, 1.3)] for s in NONGREEDY if by[(m, task, s, 1.3)]}
            if len(maps) >= 2:
                p, lo, hi = boot_spread(maps)
                rows.append((m, p * 100, lo * 100, hi * 100))
        forest(ax, rows, "sampler max$-$min at $T{=}1.3$ (pp)", zero_line=False)
        ax.set_title(TASK_NAME[task])
        ax.set_xlim(left=0)
    save(fig, "fig4_spread")


def fig5_quant(quant_greedy):
    fig, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    rows = []
    for m in MODELS:
        a8, a3 = quant_greedy.get((m, "Q8_0")), quant_greedy.get((m, "Q3_K_M"))
        if a8 and a3:
            d, lo, hi = boot_drop(a8, a3)
            rows.append((m, d * 100, lo * 100, hi * 100))
    forest(ax, rows, "greedy drop Q8_0 $\\to$ Q3_K_M (pp)",
           highlight=lambda m: m == "qwen3-1.7b")
    save(fig, "fig5_quant")


def fig6_mechanism(by):
    """Greedy-path next-token entropy vs MMLU-Pro temperature drop (the probe)."""
    stats = defaultdict(list)
    for r in load_records("results/logit_probe/*.jsonl"):
        stats[r["model"]].append(r["mean_entropy_lb"])
    if not stats:
        print("  (no logit probe data; skipping fig6)")
        return
    fig, ax = plt.subplots(figsize=(3.8, 2.8), constrained_layout=True)
    for m in MODELS:
        if m not in stats:
            continue
        ent = float(np.mean(stats[m]))
        r = boot_drop(by[(m, "mmlu_pro", "temperature", 0.7)],
                      by[(m, "mmlu_pro", "temperature", 1.3)])
        d, lo, hi = r
        color = C_FRAGILE if m.startswith("llama") else C_ROBUST
        ax.errorbar(ent, d * 100, yerr=[[(d - lo) * 100], [(hi - d) * 100]],
                    fmt="o", color=color, ms=4.5, lw=1.1, capsize=2)
        dx, dy = (4, 4) if m != "qwen3-1.7b" else (4, -9)
        ax.annotate(SHORT[m], (ent, d * 100), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.2, color=color)
    ax.set_xlabel("greedy-path next-token entropy (nats, lower bound)")
    ax.set_ylabel("MMLU-Pro drop $T0.7 \\to T1.3$ (pp)")
    save(fig, "fig6_mechanism")


def main():
    print("loading matrix ...")
    by, by_q8, strict, acc_strict, quant_greedy = collect_matrix()
    print("loading ladder ...")
    ladder = collect_ladder()
    fig1_collapse(by)
    fig2_decomposition(by, acc_strict)
    fig3_cliff(by_q8, ladder)
    fig4_spread(by)
    fig5_quant(quant_greedy)
    fig6_mechanism(by)
    print("done.")


if __name__ == "__main__":
    main()
