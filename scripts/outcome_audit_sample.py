#!/usr/bin/env python3
"""Draw the blinded outcome-audit sample (Astra 09-06 item C): ~150 plain-temperature generations across
Llama-3.1-8B, Mistral-7B, Qwen2.5-7B and Qwen3-8B (Q8), T in {0.7, 1.3}, both tasks, oversampling the
parse/cap paths where the parser's reading is least certain. Writes two files:
  results/outcome_audit/blind.jsonl   audit_id, task, prompt (short), raw_output   (labeler sees ONLY this)
  results/outcome_audit/key.jsonl     audit_id -> model, T, parse_method, cap, correct, gold, parsed_answer
Labels to assign (one per output): completed-correct, completed-wrong, cut-off-coherent (reasoning still coherent
at the cap, no answer yet), degenerate (token salad / loops / language drift), misformatted (an identifiable
final answer the pinned sentence would miss), other."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

CAP = {"gsm8k": 512, "mmlu_pro": 1024}
MODELS = ["llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", "qwen3-8b"]
rng = random.Random(20260906)
pool = defaultdict(list)
for m in MODELS:
    for line in open(f"results/full_matrix/shards/{m}__Q8_0.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["sampler"] != "temperature" or r["temperature"] not in (0.7, 1.3):
            continue
        cap = r["n_completion_tokens"] >= CAP[r["task"]] or r["n_completion_tokens"] == 0
        path = "cap" if cap else r["parse_method"]
        pool[(m, r["task"], r["temperature"], path)].append(r)
# quota: every (model, task, T, path) stratum contributes up to 4, strict paths only 2 (they are the easy case)
sample = []
for key in sorted(pool):
    k = 2 if key[3] == "strict" else 4
    sample += rng.sample(pool[key], min(k, len(pool[key])))
rng.shuffle(sample)
out = Path("results/outcome_audit"); out.mkdir(parents=True, exist_ok=True)
with open(out / "blind.jsonl", "w") as fb, open(out / "key.jsonl", "w") as fk:
    for i, r in enumerate(sample, 1):
        aid = f"A{i:03d}"
        fb.write(json.dumps({"audit_id": aid, "task": r["task"], "raw_output": r["raw_output"]}) + "\n")
        fk.write(json.dumps({"audit_id": aid, "model": r["model"], "temperature": r["temperature"], "task": r["task"],
                             "parse_method": r["parse_method"], "cap": r["n_completion_tokens"] >= CAP[r["task"]] or r["n_completion_tokens"] == 0,
                             "correct": r["correct"], "gold": r["gold"], "parsed_answer": r["parsed_answer"], "id": r["id"]}) + "\n")
from collections import Counter

print(len(sample), "records;", Counter((r["model"].split("-")[0], r["temperature"], "cap" if (r["n_completion_tokens"] >= CAP[r["task"]] or r["n_completion_tokens"] == 0) else r["parse_method"]) for r in sample))
