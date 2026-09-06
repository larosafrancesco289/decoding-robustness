#!/usr/bin/env python3
"""Independent audit of the full-matrix shards (no reuse of analysis code).

Checks, in order:
  1. Duplicate record ids ACROSS shard files (RecordStore dedups only within one file).
  2. Cell completeness: expected vs actual record counts per (model, quant, task).
  3. Identity-field consistency: id string vs explicit fields; greedy K=1 at T=0 only.
  4. Params sanity: exactly one truncation param; chain order matches the label
     (_tlast arms reversed); seeds vary across reps; temperature forced 0.0 for greedy.
  5. Independent recompute of the headline numbers (pure-temp drop T0.7->T1.3 per
     model/task, pooled over quant) to compare against STATS.txt.
  6. Parse-method / stopped / empty-output rates per model/task (degeneration signals).
  7. MMLU-Pro N=50 subset: category composition (head-of-split bias check).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict

SHARDS = sorted(glob.glob("results/full_matrix/shards/*.jsonl"))

records = []
id_to_files = defaultdict(list)
for path in SHARDS:
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"!! corrupt line {path}:{ln}")
                continue
            r["_file"] = path
            records.append(r)
            id_to_files[r["id"]].append(path)

print(f"shards: {len(SHARDS)}   records: {len(records)}   unique ids: {len(id_to_files)}")

# -- 1. cross-shard duplicates ------------------------------------------------
dupes = {i: fs for i, fs in id_to_files.items() if len(fs) > 1}
print(f"\n[1] duplicate ids: {len(dupes)}")
for i, fs in list(dupes.items())[:10]:
    print(f"    {i}  in  {[f.split('/')[-1] for f in fs]}")
if len(dupes) > 10:
    cross = Counter()
    for fs in dupes.values():
        cross[tuple(sorted({f.split('/')[-1] for f in fs}))] += 1
    print("    file-pair summary:")
    for pair, n in cross.most_common(20):
        print(f"      {n:6d}  {pair}")

# -- 2. cell completeness -----------------------------------------------------
# expected per (model,quant,task): greedy 50 + 7 samplers x 3 temps x 50 items x 3 reps = 3200
cell = Counter((r["model"], r["quant"], r["task"]) for r in records)
print(f"\n[2] cells: {len(cell)}")
bad = {k: v for k, v in cell.items() if v != 3200}
if bad:
    for k, v in sorted(bad.items()):
        print(f"    UNEXPECTED {k}: {v} (want 3200)")
else:
    print("    all cells exactly 3200 records ✓")

# -- 3. identity consistency --------------------------------------------------
mismatch = 0
greedy_bad = 0
rep_count = defaultdict(set)
for r in records:
    want = (
        f"{r['model']}|{r['quant']}|{r['sampler']}|t{r['temperature']}"
        f"|{r['task']}|{r['item_id']}|r{r['repetition']}"
    )
    if r["id"] != want:
        mismatch += 1
    if r["sampler"].startswith("greedy"):
        if r["temperature"] != 0.0 or r["repetition"] != 1:
            greedy_bad += 1
    group = (r["model"], r["quant"], r["sampler"], r["temperature"], r["task"], r["item_id"])
    rep_count[group].add(r["repetition"])
print(f"\n[3] id/field mismatches: {mismatch}   greedy not (T=0, r=1): {greedy_bad}")
repdist = Counter(frozenset(v) for v in rep_count.values())
for reps, n in sorted(repdist.items(), key=lambda kv: -kv[1])[:6]:
    print(f"    rep-set {sorted(reps)}: {n} (item x condition) groups")

# -- 4. params sanity ----------------------------------------------------------
TRUNC = ("top_p", "top_k", "min_p", "top_n_sigma")
param_bad = []
seed_by_group = defaultdict(set)
for r in records:
    p = r["params"]
    s = r["sampler"]
    present = [t for t in TRUNC if t in p]
    if s.startswith(("greedy", "temperature")):
        ok = not present and p["samplers"] == ["temperature"]
    else:
        meth = next(t for t in TRUNC if s.startswith(t))
        tlast = s.endswith("_tlast")
        want_chain = [meth, "temperature"] if tlast else ["temperature", meth]
        ok = present == [meth] and p["samplers"] == want_chain
    if s.startswith("greedy") and p["temperature"] != 0.0:
        ok = False
    if not ok:
        param_bad.append((r["id"], p.get("samplers"), present))
    group = (r["model"], r["quant"], r["sampler"], r["temperature"], r["task"], r["item_id"])
    seed_by_group[group].add(p["seed"])
print(f"\n[4] param violations: {len(param_bad)}")
for b in param_bad[:5]:
    print(f"    {b}")
const_seed = sum(1 for k, v in seed_by_group.items() if len(rep_count[k]) > 1 and len(v) == 1)
print(f"    groups with >1 rep but a single seed: {const_seed}")

# -- 5. independent headline recompute ----------------------------------------
print("\n[5] pure-temp drop T0.7 -> T1.3 (pooled over quant+reps), independent recompute")
acc = defaultdict(lambda: [0, 0])  # (model,task,temp) -> [num,den]
for r in records:
    if r["sampler"] == "temperature":
        a = acc[(r["model"], r["task"], r["temperature"])]
        a[0] += int(r["correct"])
        a[1] += 1
models = sorted({r["model"] for r in records})
for task in ("gsm8k", "mmlu_pro"):
    print(f"  {task}:")
    for m in models:
        lo, hi = acc[(m, task, 0.7)], acc[(m, task, 1.3)]
        if lo[1] and hi[1]:
            d = lo[0] / lo[1] - hi[0] / hi[1]
            print(
                f"    {m:<26} acc@0.7={lo[0]/lo[1]:.3f} (n={lo[1]})  "
                f"acc@1.3={hi[0]/hi[1]:.3f} (n={hi[1]})  drop={d*100:+.1f}pp"
            )

# -- 6. degeneration signals ----------------------------------------------------
print("\n[6] parse_method / stopped / empty rates (temperature sampler @T=1.3, pooled quant)")
sig = defaultdict(lambda: Counter())
for r in records:
    if r["sampler"] == "temperature" and r["temperature"] == 1.3:
        k = (r["model"], r["task"])
        sig[k]["n"] += 1
        sig[k][r["parse_method"]] += 1
        # llama.cpp sets stopped=True on cap hits too; cap = ran to the task budget, or a server-recovered
        # degenerate stream (n_completion_tokens == 0), as in gen_tables.py.
        if r["n_completion_tokens"] >= {"gsm8k": 512, "mmlu_pro": 1024}[r["task"]] or r["n_completion_tokens"] == 0:
            sig[k]["hit_token_cap"] += 1
        if not r["raw_output"].strip():
            sig[k]["empty"] += 1
for (m, t), c in sorted(sig.items()):
    n = c["n"]
    print(
        f"    {m:<26} {t:<9} strict={c['strict']/n:5.1%} flex={c['flexible']/n:5.1%} "
        f"failed={c['failed']/n:5.1%} cap={c['hit_token_cap']/n:5.1%} empty={c['empty']/n:5.1%}"
    )

# -- 7. MMLU-Pro subset composition ---------------------------------------------
print("\n[7] MMLU-Pro item ids in the matrix (head-of-split bias check)")
mmlu_items = sorted({r["item_id"] for r in records if r["task"] == "mmlu_pro"})
print(f"    {len(mmlu_items)} items: {mmlu_items[0]} .. {mmlu_items[-1]}")
try:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    cats = Counter(ds[i]["category"] for i in range(len(mmlu_items)))
    print(f"    categories in first {len(mmlu_items)} rows: {dict(cats)}")
    g50 = Counter()
    print(f"    (full split has {len(ds)} rows, {len(set(ds['category']))} categories)")
except Exception as e:  # noqa: BLE001
    print(f"    (dataset check skipped: {e})")

print("\naudit done.")
sys.exit(0)
