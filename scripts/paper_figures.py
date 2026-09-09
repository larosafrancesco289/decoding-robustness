#!/usr/bin/env python3
"""Figures for the 2026-09 rewrite, from results/paper_records.parquet (scripts/paper_data.py).

  uv run python scripts/paper_figures.py            # all
  uv run python scripts/paper_figures.py fig1 fig3  # some

Main text
  fig1_drops        forest: plain-temperature drop T0.7->1.3 per model x task, Q8, sorted by MMLU-Pro drop
  fig2_outcomes     stacked composition (correct / answered wrong / no answer) at T0.7,1.0,1.3, MMLU-Pro, 13 models
  figA_gain_loss    scatter (appendix): gain of each truncation rule over plain temperature vs loss of plain temperature vs greedy
  fig4_ladder       accuracy vs T to 2.0, three models x two tasks, four rules, half-baseline line
Appendix
  figA_outcomes_gsm8k  as fig2 for GSM8K
  fig3_recovery        forest (main): each rule at T1.3 minus plain at T1.3 and minus plain at T0.7, four fragile models with grids

Conventions: Q8_0 throughout; item-clustered paired bootstrap (B=2000, seed 0); cap = completion tokens >= budget or 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = 2000
OUT = Path("figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.dpi": 300, "legend.frameon": False,
})

LABEL = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "llama-3-8b-instruct": "Llama-3-8B", "llama-3.2-3b-instruct": "Llama-3.2-3B",
         "hermes-3-llama-3.1-8b": "Hermes-3-8B", "mistral-7b-instruct-v0.3": "Mistral-7B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
         "qwen3-8b": "Qwen3-8B", "qwen3-4b": "Qwen3-4B", "qwen3-1.7b": "Qwen3-1.7B", "gemma-3-12b-it": "Gemma-3-12B",
         "qwen3.5-9b": "Qwen3.5-9B", "gemma-4-e4b-it": "Gemma-4-E4B", "olmo-3-7b-instruct": "OLMo-3-7B"}
TASKS = ["gsm8k", "mmlu_pro"]
TASK_NAME = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}
RULES = ["top_p", "top_k", "min_p", "top_n_sigma"]
RULE_TEX = {"temperature": "plain temperature", "top_p": "top-$p$", "top_k": "top-$k$", "min_p": "min-$p$",
            "top_n_sigma": "top-$n\\sigma$"}
# Okabe-Ito, fixed assignment
C_FRAGILE, C_ROBUST, C_GRAY = "#D55E00", "#0072B2", "#8C8C8C"
RULE_COLOR = {"temperature": "#D55E00", "top_p": "#E69F00", "top_k": "#8C8C8C", "min_p": "#0072B2", "top_n_sigma": "#009E73"}
RULE_MARK = {"top_p": "o", "top_k": "s", "min_p": "D", "top_n_sigma": "^"}
T_COLOR = {0.7: "#9ECAE1", 1.0: "#4292C6", 1.3: "#08306B"}
RNG = np.random.default_rng(0)
IDX = RNG.integers(0, 50, (B, 50))


def load():
    df = pd.read_parquet("results/paper_records.parquet")
    return df[df.chain == "tfirst"]


def temp_slice(df):
    """Plain temperature + greedy at Q8 through llama.cpp for the 13 models (main grid, lineage, panel)."""
    d = df[(df.quant == "Q8_0") & df.source.isin(["main_grid", "lineage", "panel"])]
    return d


def item_acc(d):
    return d.groupby("item_id").correct.mean().sort_index()


def paired_boot(a, b):
    """mean(a-b) with item-clustered bootstrap CI; a, b item-indexed series on the same 50 items."""
    assert list(a.index) == list(b.index) and len(a) == 50
    diff = (a.values - b.values) * 100
    boots = diff[IDX].mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def drops(d):
    rows = []
    for m in LABEL:
        for task in TASKS:
            g = d[(d.model == m) & (d.task == task) & (d.sampler == "temperature")]
            if g.empty:
                continue
            a07, a13 = item_acc(g[g["T"] == 0.7]), item_acc(g[g["T"] == 1.3])
            pt, lo, hi = paired_boot(a07, a13)
            rows.append(dict(model=m, task=task, acc07=a07.mean() * 100, acc13=a13.mean() * 100, d=pt, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def model_order(dr):
    """Sorted by MMLU-Pro drop, largest first. Fragile = drop >= 15 on either task (the prespecified flag's accuracy criterion)."""
    mm = dr[dr.task == "mmlu_pro"].set_index("model")["d"]
    order = list(mm.sort_values(ascending=False).index)
    frag = {m for m in order if (dr[dr.model == m]["d"] >= 15).any()}
    return order, frag


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ fig 1
def fig1_drops(df):
    d = temp_slice(df)
    dr = drops(d)
    order, frag = model_order(dr)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, task in zip(axes, TASKS):
        s = dr[dr.task == task].set_index("model").loc[order]
        for yi, (m, r) in zip(y, s.iterrows()):
            c = C_FRAGILE if m in frag else C_ROBUST
            ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.2, solid_capstyle="butt")
            ax.plot(r.d, yi, "o", color=c, ms=4.2, mec="white", mew=0.6)
            ax.text(51.5, yi, f"{r.acc07:.0f}$\\rightarrow${r.acc13:.0f}", va="center", ha="left", fontsize=6.8, color="#444444")
        ax.axvline(0, color="#555555", lw=0.6, zorder=0)
        ax.set_xlim(-12, 51)
        ax.set_xticks([-10, 0, 10, 20, 30, 40, 50])
        ax.set_title(TASK_NAME[task], fontsize=9, pad=3)
        ax.set_xlabel("accuracy drop, $T$ 0.7 $\\rightarrow$ 1.3 (pp)")
        ax.text(51.5, len(order) - 0.15, "acc. (%)\n0.7$\\rightarrow$1.3", ha="left", va="bottom", fontsize=6.5, color="#444444")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[m] for m in order])
    for lab, m in zip(axes[0].get_yticklabels(), order):
        lab.set_color(C_FRAGILE if m in frag else "#222222")
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    # separator between fragile and robust groups
    n_frag = sum(m in frag for m in order)
    for ax in axes:
        ax.axhline(len(order) - n_frag - 0.5, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=0)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig1_drops")
    return order, frag


# ------------------------------------------------------------------ fig 2 (+ appendix GSM8K)
def outcome_shares(d, task):
    g = d[(d.task == task) & (d.sampler == "temperature")].copy()
    g["cat"] = np.where(g.correct, "correct", np.where(g.at_cap | (g.parse == "failed"), "noanswer", "wrong"))
    t = g.groupby(["model", "T"]).cat.value_counts(normalize=True).unstack().fillna(0) * 100
    return t


def fig_outcomes(df, task, name, order, frag):
    d = temp_slice(df)
    t = outcome_shares(d, task)
    cats = [("correct", C_ROBUST, "correct"), ("wrong", "#C9C9C9", "answered, wrong"), ("noanswer", C_FRAGILE, "capped or unparseable")]
    Ts = [0.7, 1.0, 1.3]
    fig, ax = plt.subplots(figsize=(3.3, 4.6))
    h, gap, group = 0.24, 0.05, 1.0
    ylab, ypos = [], []
    for gi, m in enumerate(order):
        base = -gi * group
        for ti, T in enumerate(Ts):
            yi = base + (1 - ti) * (h + gap)
            left = 0
            for c, col, _ in cats:
                v = t.loc[(m, T), c] if (m, T) in t.index else 0
                ax.barh(yi, v, left=left, height=h, color=col, edgecolor="white", linewidth=0.5)
                left += v
            if gi == 0:
                ax.text(101.5, yi, f"$T$={T}", va="center", ha="left", fontsize=6.3, color="#555555")
        ylab.append(LABEL[m]); ypos.append(base)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab)
    for lab, m in zip(ax.get_yticklabels(), order):
        lab.set_color(C_FRAGILE if m in frag else "#222222")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"share of generations (%), {TASK_NAME[task]}")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    n_frag = sum(m in frag for m in order)
    ax.axhline(-(n_frag - 0.5) * group, color="#BBBBBB", lw=0.6, ls=(0, (2, 2)), zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in cats]
    ax.legend(handles, [lab for _, _, lab in cats], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handlelength=1.0, handleheight=0.8, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0, fontsize=7)
    ax.set_ylim(-(len(order) - 1) * group - 0.45, 0.45)
    save(fig, name)


# ------------------------------------------------------------------ fig 3
def grid_cells(df):
    """Per (model, task, T): greedy acc, plain acc, and each rule's acc; Q8 for the grid, the flagged-model grid as is."""
    rows = []
    for src in ("main_grid", "panel_grid"):
        d = df[df.source == src]
        if src == "main_grid":
            d = d[d.quant == "Q8_0"]
        for (m, task), g in d.groupby(["model", "task"]):
            greedy = g[g.sampler == "greedy"].correct.mean() * 100
            for T in (0.7, 1.0, 1.3):
                plain = item_acc(g[(g["T"] == T) & (g.sampler == "temperature")])
                for rule in RULES:
                    x = item_acc(g[(g["T"] == T) & (g.sampler == rule)])
                    pt, lo, hi = paired_boot(x, plain)
                    rows.append(dict(source=src, model=m, task=task, T=T, rule=rule, greedy=greedy, plain=plain.mean() * 100,
                                     loss=greedy - plain.mean() * 100, gain=pt, lo=lo, hi=hi, acc=x.mean() * 100))
    return pd.DataFrame(rows)


def fig3_gain_loss(df):
    c = grid_cells(df)
    c.to_csv("results/gain_vs_loss_cells.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    lim_lo, lim_hi = -10, 46
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#BBBBBB", lw=0.8, zorder=0)
    ax.text(lim_hi - 1, lim_hi - 4.5, "full recovery\nto greedy", ha="right", va="top", fontsize=6.8, color="#666666")
    ax.axhline(0, color="#555555", lw=0.6, zorder=0)
    ax.axvline(0, color="#555555", lw=0.6, zorder=0)
    for T in (0.7, 1.0, 1.3):
        s = c[c["T"] == T]
        for rule in RULES:
            ss = s[s.rule == rule]
            ax.scatter(ss.loss, ss.gain, s=15, marker=RULE_MARK[rule], color=T_COLOR[T], alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
    # legend: temperature by colour, rule by marker
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], marker="o", ls="", color=T_COLOR[T], ms=5, label=f"$T$ = {T}") for T in (0.7, 1.0, 1.3)]
    h2 = [Line2D([], [], marker=RULE_MARK[r], ls="", color="#444444", ms=4.5, label=RULE_TEX[r]) for r in RULES]
    leg1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, 0.80), handletextpad=0.3, labelspacing=0.3)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("loss under plain temperature (pp)\ngreedy accuracy $-$ plain-temperature accuracy")
    ax.set_ylabel("gain from truncation (pp)\nrule accuracy $-$ plain-temperature accuracy")
    save(fig, "figA_gain_loss")
    b = c.sort_values("gain", ascending=False).drop_duplicates(["model", "task", "T"])
    print("  r(best gain, loss) =", np.corrcoef(b.loss, b.gain)[0, 1].round(3), "; r(all rules) =", np.corrcoef(c.loss, c.gain)[0, 1].round(3))


# ------------------------------------------------------------------ fig 4
def fig4_ladder(df):
    models = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-3-12b-it"]
    rules = ["temperature", "top_p", "min_p", "top_n_sigma"]
    grid = df[(df.source == "main_grid") & (df.quant == "Q8_0")]
    lad = df[df.source == "ladder"]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, sharey=True)
    for ci, m in enumerate(models):
        for ri, task in enumerate(TASKS):
            ax = axes[ri, ci]
            base07 = grid[(grid.model == m) & (grid.task == task) & (grid["T"] == 0.7) & (grid.sampler == "temperature")].correct.mean() * 100
            ax.axhline(base07 / 2, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=0)
            for rule in rules:
                xs, ys = [], []
                for T in (0.7, 1.0):
                    g = grid[(grid.model == m) & (grid.task == task) & (grid["T"] == T) & (grid.sampler == rule)]
                    xs.append(T); ys.append(g.correct.mean() * 100)
                for T in (1.3, 1.5, 1.7, 2.0):
                    g = lad[(lad.model == m) & (lad.task == task) & (lad["T"] == T) & (lad.sampler == rule)]
                    xs.append(T); ys.append(g.correct.mean() * 100)
                ax.plot(xs, ys, "-", color=RULE_COLOR[rule], lw=1.3, marker="o", ms=2.8, mec="white", mew=0.4)
            if ri == 0:
                ax.set_title(LABEL[m], fontsize=9, pad=3)
            if ci == 0:
                ax.set_ylabel(f"{TASK_NAME[task]}\naccuracy (%)")
            if ri == 1:
                ax.set_xlabel("temperature")
            ax.set_xticks([0.7, 1.0, 1.3, 1.5, 1.7, 2.0])
            ax.set_xticklabels(["0.7", "1.0", "1.3", "1.5", "1.7", "2.0"])
            ax.set_ylim(-3, 100)
    b = grid[(grid.model == models[2]) & (grid.task == "mmlu_pro") & (grid["T"] == 0.7) & (grid.sampler == "temperature")].correct.mean() * 100
    axes[1, 2].text(0.72, b / 2 - 3, "half of the $T$0.7\naccuracy", ha="left", va="top", fontsize=6.5, color="#777777")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=RULE_COLOR[r], lw=1.5, marker="o", ms=3, label=RULE_TEX[r]) for r in rules]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), handlelength=1.6, columnspacing=1.6)
    fig.subplots_adjust(hspace=0.15, wspace=0.12)
    save(fig, "fig4_ladder")


# ------------------------------------------------------------------ appendix recovery forest
def fig3_recovery(df):
    c = grid_cells(df)
    fragile = ["llama-3.1-8b-instruct", "hermes-3-llama-3.1-8b", "qwen3.5-9b", "olmo-3-7b-instruct"]
    # second reference: rule at 1.3 minus plain at 0.7
    rows = []
    for src in ("main_grid", "panel_grid"):
        d = df[df.source == src]
        if src == "main_grid":
            d = d[d.quant == "Q8_0"]
        for m in fragile:
            for task in TASKS:
                g = d[(d.model == m) & (d.task == task)]
                if g.empty:
                    continue
                p07 = item_acc(g[(g["T"] == 0.7) & (g.sampler == "temperature")])
                for rule in RULES:
                    x = item_acc(g[(g["T"] == 1.3) & (g.sampler == rule)])
                    pt, lo, hi = paired_boot(x, p07)
                    rows.append(dict(model=m, task=task, rule=rule, d07=pt, lo07=lo, hi07=hi))
    r2 = pd.DataFrame(rows).set_index(["model", "task", "rule"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.7), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
    labels, ypos = [], []
    y = 0
    for m in fragile:
        for task in TASKS:
            for rule in RULES:
                s = c[(c.model == m) & (c.task == task) & (c["T"] == 1.3) & (c.rule == rule)].iloc[0]
                q = r2.loc[(m, task, rule)]
                col = RULE_COLOR[rule]
                axes[0].plot([s.lo, s.hi], [y, y], color=col, lw=1.1); axes[0].plot(s.gain, y, "o", color=col, ms=3.6, mec="white", mew=0.5)
                axes[1].plot([q.lo07, q.hi07], [y, y], color=col, lw=1.1); axes[1].plot(q.d07, y, "o", color=col, ms=3.6, mec="white", mew=0.5)
                labels.append(f"{LABEL[m]}, {TASK_NAME[task]}" if rule == RULES[0] else ""); ypos.append(y)
                y -= 1
            y -= 0.8
    for ax, xl, tt in zip(axes, ["difference (pp)", "difference (pp)"],
                          ["gain over plain temperature at $T$ 1.3", "difference from plain temperature at $T$ 0.7"]):
        ax.axvline(0, color="#555555", lw=0.6, zorder=0)
        ax.set_xlabel(xl)
        ax.set_title(tt, fontsize=8, pad=4)
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    axes[0].set_yticks([p for p, l in zip(ypos, labels) if l])
    axes[0].set_yticklabels([l for l in labels if l], fontsize=7.5)
    for p, l in zip(ypos, labels):
        if l:
            for ax in axes:
                ax.axhline(p + 0.6, color="#DDDDDD", lw=0.5, zorder=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=RULE_COLOR[r], lw=1.5, marker="o", ms=3, label=RULE_TEX[r]) for r in RULES]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=1.6, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig3_recovery")


def main(which):
    df = load()
    order, frag = fig1_drops(df) if "fig1" in which or not which else model_order(drops(temp_slice(df)))
    if not which or "fig2" in which:
        fig_outcomes(df, "mmlu_pro", "fig2_outcomes", order, frag)
        fig_outcomes(df, "gsm8k", "figA_outcomes_gsm8k", order, frag)
    if not which or "fig3" in which:
        fig3_gain_loss(df)
    if not which or "fig4" in which:
        fig4_ladder(df)
    if not which or "figA" in which:
        fig3_recovery(df)


if __name__ == "__main__":
    main(sys.argv[1:])
