#!/usr/bin/env python3
"""tables/tab_engine.tex: the engine/precision replication (results/engine_check/*.jsonl) next to the paper's
llama.cpp cells. One row per model x engine/precision x task: accuracy at T0.7 and T1.3, the drop with its
item-clustered bootstrap CI (paper convention: drop = acc(0.7) - acc(1.3), positive = loss), the cap-hit rate at
T1.3, and the difference in drops relative to the Q8 llama.cpp cell with its CI. Pure stdlib + numpy."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_engine_check import REFS, boot_diff, boot_diff_in_drops, load, stats  # noqa: E402

MODEL_LABEL = {"llama-3.2-3b-instruct": "Llama-3.2-3B", "llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen3-4b": "Qwen3-4B"}
ENGINE_LABEL = {"BF16-hf": "BF16, transformers", "INT8-hf": "INT8, transformers",
                "F16": "F16 GGUF, llama.cpp", "BF16": "BF16 GGUF, llama.cpp", "Q8_0": "Q8 GGUF, llama.cpp"}
TASK_LABEL = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro"}
ORDER = ["llama-3.2-3b-instruct", "llama-3.1-8b-instruct", "qwen3-4b"]


def ci(t, flip=False):
    if t is None:
        return "--"
    p, lo, hi, _ = t
    if flip:
        p, lo, hi = -p, -hi, -lo
    return f"${p*100:+.1f}$\\,[${lo*100:.1f}$,\\,${hi*100:.1f}$]"


def main():
    rows_by_model: dict[str, list[tuple[str, dict]]] = {}
    for path in sorted(glob.glob("results/engine_check/*.jsonl")):
        eng = load(path)
        if not eng:
            continue
        model = next(iter(next(iter(eng.values())).values()))[0]["model"]
        quant = next(iter(eng))[0]
        tag = "offload" if "offload" in path else ""
        rows_by_model.setdefault(model, []).append((quant + (" (CPU offload)" if tag else ""), eng))
    lines = ["\\begin{tabular}{lllccccc}", "\\toprule",
             " & & & \\multicolumn{2}{c}{acc (\\%) at $T$} & & cap & \\\\",
             "Model & Engine, precision & Task & 0.7 & 1.3 & drop [95\\% CI] & at 1.3 & drop $-$ drop(Q8) \\\\"]
    for model in ORDER:
        if model not in rows_by_model:
            continue
        lines.append("\\midrule")
        refs = {}
        for rpath, quants in REFS[model]:
            refs.update(load(rpath, model=model, quants=quants))
        ref_q8 = {k: v for k, v in refs.items() if k[0] == "Q8_0"}
        sources = rows_by_model[model] + [(rq, {k: v for k, v in refs.items() if k[0] == rq}) for rq in sorted({k[0] for k in refs}, reverse=True)]
        first = True
        for label, src in sources:
            quant = label.split(" ")[0]
            for task in ("gsm8k", "mmlu_pro"):
                lo, hi = src.get((quant, task, "temperature", 0.7), {}), src.get((quant, task, "temperature", 1.3), {})
                s7, s13 = stats(lo), stats(hi)
                if not (s7 and s13):
                    continue
                drop = boot_diff(hi, lo)
                dd = "--" if quant == "Q8_0" else ci(boot_diff_in_drops(hi, lo, ref_q8.get(("Q8_0", task, "temperature", 1.3), {}),
                                                                        ref_q8.get(("Q8_0", task, "temperature", 0.7), {})), flip=True)
                lines.append(f"{MODEL_LABEL[model] if first else ''} & {ENGINE_LABEL.get(quant, quant) + (' (CPU offload)' if 'offload' in label else '')} & "
                             f"{TASK_LABEL[task]} & {s7['acc']*100:.1f} & {s13['acc']*100:.1f} & {ci(drop, flip=True)} & {s13['cap']*100:.0f} & {dd} \\\\")
                first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/tab_engine_full.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    # Compact main-text version: one row per model x engine, both tasks side by side.
    comp = ["\\begin{tabular}{llccc}", "\\toprule",
            " & & \\multicolumn{2}{c}{drop $T0.7 \\to 1.3$ [95\\% CI]} & cap at 1.3 \\\\",
            "Model & Engine, precision & GSM8K & MMLU-Pro & G / M \\\\"]
    for model in ORDER:
        if model not in rows_by_model:
            continue
        comp.append("\\midrule")
        refs = {}
        for rpath, quants in REFS[model]:
            refs.update(load(rpath, model=model, quants=quants))
        sources = rows_by_model[model] + [(rq, {k: v for k, v in refs.items() if k[0] == rq}) for rq in sorted({k[0] for k in refs}, reverse=True)]
        first = True
        for label, src in sources:
            quant = label.split(" ")[0]
            cells, caps = [], []
            for task in ("gsm8k", "mmlu_pro"):
                lo, hi = src.get((quant, task, "temperature", 0.7), {}), src.get((quant, task, "temperature", 1.3), {})
                s13 = stats(hi)
                if not (stats(lo) and s13):
                    cells.append("--"); caps.append("--"); continue
                cells.append(ci(boot_diff(hi, lo), flip=True)); caps.append(f"{s13['cap']*100:.0f}")
            eng_label = ENGINE_LABEL.get(quant, quant) + (" (CPU offload)" if "offload" in label else "")
            comp.append(f"{MODEL_LABEL[model] if first else ''} & {eng_label} & {cells[0]} & {cells[1]} & {caps[0]} / {caps[1]} \\\\")
            first = False
    comp += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables/tab_engine.tex").write_text("\n".join(comp) + "\n")
    print("\n".join(comp))


if __name__ == "__main__":
    main()
