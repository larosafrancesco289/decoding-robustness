#!/usr/bin/env python3
"""Render results/sampler_bound.json as paper/tables/tab_bound.tex (Appendix: sampler bound)."""
import json

ARMS = ["top_p_top_p0.95", "top_k_top_k40", "min_p_min_p0.05", "top_n_sigma_top_n_sigma1.0",
        "top_p_top_p0.95_tlast", "min_p_min_p0.05_tlast"]
HEAD = {"top_p_top_p0.95": "top-$p$", "top_k_top_k40": "top-$k$", "min_p_min_p0.05": "min-$p$",
        "top_n_sigma_top_n_sigma1.0": "top-$n\\sigma$",
        "top_p_top_p0.95_tlast": "top-$p$ (T last)", "min_p_min_p0.05_tlast": "min-$p$ (T last)"}
MODELS = [("llama-3.1-8b-instruct", "Llama-3.1-8B"), ("mistral-7b-instruct-v0.3", "Mistral-7B"),
          ("qwen2.5-7b-instruct", "Qwen2.5-7B"), ("gemma-3-12b-it", "Gemma-3-12B"),
          ("qwen3-8b", "Qwen3-8B"), ("qwen3-4b", "Qwen3-4B"), ("qwen3-1.7b", "Qwen3-1.7B")]
TASK = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}


def fmt(o):
    s = f"${o['mean']:+.1f}$ [{o['lo']:+.1f}, {o['hi']:+.1f}]"
    return f"\\textbf{{{s}}}" if o["lo"] > 0 else s


def main():
    d = {(o["model"], o["task"], o["T"], o["arm"]): o for o in json.load(open("results/sampler_bound.json"))}
    lines = ["\\begin{tabular}{llr" + "r" * len(ARMS) + "}", "\\toprule",
             "Model & Task & $T$ & " + " & ".join(HEAD[a] for a in ARMS) + " \\\\", "\\midrule"]
    for m, name in MODELS:
        first = True
        for task in ("gsm8k", "mmlu_pro"):
            for T in (0.7, 1.0, 1.3):
                cells = " & ".join(fmt(d[(m, task, T, a)]) for a in ARMS)
                lines.append(f"{name if first else ''} & {TASK[task] if T == 0.7 else ''} & {T} & {cells} \\\\")
                first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open("paper/tables/tab_bound.tex", "w").write("\n".join(lines) + "\n")
    print("wrote paper/tables/tab_bound.tex")


if __name__ == "__main__":
    main()
