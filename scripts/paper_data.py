#!/usr/bin/env python3
"""One flat table over every result source the rewrite uses (2026-09-09).

  uv run python scripts/paper_data.py          # writes results/paper_records.parquet

Columns: source, model, quant, engine, sampler (short), chain (tfirst/tlast), T, task, item_id, rep,
correct, parse (strict/flexible/failed), n_out, at_cap. Definitions match the paper: cap = completion
tokens >= budget or == 0 (server-recovered degenerate stream); parse path from the record.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

CAP = {"gsm8k": 512, "mmlu_pro": 1024}
SAMPLER = {"greedy": "greedy", "temperature": "temperature",
           "top_p_top_p0.95": "top_p", "top_k_top_k40": "top_k",
           "min_p_min_p0.05": "min_p", "top_n_sigma_top_n_sigma1.0": "top_n_sigma",
           "min_p_min_p0.05_tlast": "min_p", "top_p_top_p0.95_tlast": "top_p"}
SOURCES = {
    "main_grid": "results/full_matrix/shards/*.jsonl",
    "lineage": "results/llama_lineage/*.jsonl",
    "panel": "results/temp_sweep_2026/*.jsonl",
    "panel_grid": "results/panel_grid/*.jsonl",
    "engine": "results/engine_check/*.jsonl",
    "bridge": "results/bridge_v040/*.jsonl",
    "ladder": "results/cliff_ladder/*.jsonl",
    "strat": "results/mmlu_strat_check/*.jsonl",
}


def rows():
    for src, pat in SOURCES.items():
        for path in sorted(glob.glob(pat)):
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                n = r.get("n_completion_tokens", 0) or 0
                p = r.get("params", {})
                yield dict(
                    source=src, model=r["model"], quant=r["quant"],
                    engine=p.get("engine", "llama.cpp") + ("-v040" if str(r.get("server_commit", "")).startswith("5266f24") else ""),
                    sampler=SAMPLER[r["sampler"]],
                    chain="tlast" if r["sampler"].endswith("_tlast") else "tfirst",
                    T=float(r["temperature"]), task=r["task"], item_id=r["item_id"],
                    rep=int(r.get("repetition", 1)), correct=bool(r["correct"]),
                    parse=r.get("parse_method") or "failed", n_out=int(n),
                    at_cap=bool(n >= CAP[r["task"]] or n == 0),
                )


def main():
    df = pd.DataFrame(rows())
    Path("results").mkdir(exist_ok=True)
    df.to_parquet("results/paper_records.parquet", index=False)
    print(df.groupby(["source"]).size())
    print(df.groupby(["source", "model"]).size().to_string())


if __name__ == "__main__":
    main()
