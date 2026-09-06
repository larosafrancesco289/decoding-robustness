#!/usr/bin/env python3
"""Engine/precision check (2026-09-06): compare results/engine_check/*.jsonl (transformers BF16/INT8,
llama.cpp F16 GGUF) against the study's llama.cpp Q8 (and Qwen3-4B BF16) cells on the same frozen
items, same prompts, same budgets, same parser.

For every model x task x condition (greedy, T0.7, T1.0, T1.3) it prints, engine vs reference:
accuracy, cap-hit rate (stopped=False), strict-parse rate, parse-failure rate, mean completion
tokens; the engine-minus-reference accuracy gap with an item-clustered bootstrap CI; and the
T0.7->T1.3 drop within each engine with its own CI. Prompt-hash agreement is checked per item so a
template mismatch cannot hide behind an accuracy number.

Pure stdlib + numpy; CPU, seconds.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

B = 2000
RNG = np.random.default_rng(0)
REFS = {  # model -> list of (path glob, allowed quants)
    "llama-3.2-3b-instruct": [("results/llama_lineage/lineage.jsonl", {"Q8_0"})],
    "llama-3-8b-instruct": [("results/llama_lineage/lineage.jsonl", {"Q8_0"})],
    "llama-3.1-8b-instruct": [("results/full_matrix/shards/llama-3.1-8b-instruct__Q8_0.jsonl", {"Q8_0"})],
    "qwen3-4b": [("results/full_matrix/shards/qwen3-4b__Q8_0.jsonl", {"Q8_0"}),
                 ("results/full_matrix/shards/qwen3-4b__BF16.jsonl", {"BF16"})],
}
CONDS = [("greedy", 0.0), ("temperature", 0.7), ("temperature", 1.0), ("temperature", 1.3)]


def load(path, model=None, quants=None):
    """(quant, task, sampler, temp) -> item -> list of record dicts."""
    by = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(path):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if model and r["model"] != model:
                continue
            if quants and r["quant"] not in quants:
                continue
            if r["sampler"] not in ("greedy", "temperature"):
                continue
            by[(r["quant"], r["task"], r["sampler"], float(r["temperature"]))][r["item_id"]].append(r)
    return by


def stats(item_map):
    recs = [r for v in item_map.values() for r in v]
    n = len(recs)
    if not n:
        return None
    return dict(
        n=n, items=len(item_map),
        acc=np.mean([r["correct"] for r in recs]),
        # same definition as gen_tables.py / decompose_collapse.py: ran to the task cap, or a
        # server-recovered empty stream. (llama.cpp's "stopped" is True for cap hits too.)
        cap=np.mean([_is_cap(r) for r in recs]),
        strict=np.mean([r["parse_method"] == "strict" for r in recs]),
        fail=np.mean([r["parse_method"] == "failed" for r in recs]),
        # mean length over records with a real count (llama.cpp writes 0 for server-recovered degenerate streams)
        toks=np.mean([r["n_completion_tokens"] for r in recs if r["n_completion_tokens"]] or [float("nan")]),
    )


def acc_items(item_map, items):
    num = den = 0
    for it in items:
        v = item_map.get(it)
        if v:
            num += sum(r["correct"] for r in v)
            den += len(v)
    return num / den if den else float("nan")


def _is_cap(r):
    return r["n_completion_tokens"] >= r["params"]["n_predict"] or r["n_completion_tokens"] == 0


METRICS = {"acc": lambda r: r["correct"], "cap": _is_cap, "strict": lambda r: r["parse_method"] == "strict"}


def rate_items(item_map, items, f):
    num = den = 0
    for it in items:
        v = item_map.get(it)
        if v:
            num += sum(f(r) for r in v)
            den += len(v)
    return num / den if den else float("nan")


def boot_diff(map_a, map_b, metric="acc"):
    """rate(a) - rate(b) with an item-clustered bootstrap CI over the items both maps share."""
    f = METRICS[metric]
    items = np.array(sorted(set(map_a) & set(map_b)), dtype=object)
    if not len(items):
        return None
    point = rate_items(map_a, items, f) - rate_items(map_b, items, f)
    draws = [rate_items(map_a, s, f) - rate_items(map_b, s, f)
             for s in (RNG.choice(items, size=len(items), replace=True) for _ in range(B))]
    return point, np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5), len(items)


def boot_diff_in_drops(e13, e07, r13, r07):
    """(engine drop) - (reference drop) on accuracy, T0.7->T1.3, same item resample for all four cells."""
    f = METRICS["acc"]
    items = np.array(sorted(set(e13) & set(e07) & set(r13) & set(r07)), dtype=object)
    if not len(items):
        return None
    def dd(s):
        return (rate_items(e13, s, f) - rate_items(e07, s, f)) - (rate_items(r13, s, f) - rate_items(r07, s, f))
    draws = [dd(RNG.choice(items, size=len(items), replace=True)) for _ in range(B)]
    return dd(items), np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5), len(items)


def fmt_ci(t):
    if t is None:
        return "n/a"
    p, lo, hi, n = t
    return f"{p*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}] (n_items={n})"


def hash_agreement(eng, ref):
    """Per item: does the engine record's prompt hash / prompt token count match the reference's?
    Llama-3.x templates stamp the run date into the system header ("Today Date: 06 Sep 2026"), so a
    run on a different day changes the hash but not the token count. Verified 2026-09-06 by re-rendering
    every item of both tasks from the GGUF sidecar template with date_string="04 Jul 2026" (lineage run
    day) and "06 Sep 2026": the July render reproduces every lineage hash, the September render every
    engine-check hash (see the session log; the check is a 15-line script, not repeated here)."""
    seen = agree = tok = 0
    for key, imap in eng.items():
        for it, recs in imap.items():
            for rkey, rmap in ref.items():
                if rkey[1] != key[1] or it not in rmap:
                    continue
                # llama.cpp reports tokens_evaluated=0 on a full prompt-cache hit; skip those.
                rtok = next((r["n_prompt_tokens"] for r in rmap[it] if r["n_prompt_tokens"]), 0)
                if not rtok:
                    break
                etok = next((r["n_prompt_tokens"] for r in recs if r["n_prompt_tokens"]), 0)
                if not etok:
                    break
                seen += 1
                agree += recs[0]["prompt_hash"] == rmap[it][0]["prompt_hash"]
                tok += etok == rtok
                break
    return agree, tok, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/engine_check")
    ap.add_argument("--label", default=None, help="display label for the engine-side rows (e.g. 'v0.4.0' for the commit bridge, whose quant is Q8_0 like the reference)")
    args = ap.parse_args()
    for path in sorted(glob.glob(f"{args.dir}/*.jsonl")):
        eng = load(path)
        if not eng:
            print(f"\n##### {path}: empty"); continue
        model = next(iter(next(iter(eng.values())).values()))[0]["model"]
        equant = next(iter(eng))[0]
        engine = next(iter(next(iter(eng.values())).values()))[0]["params"].get("engine", "llama.cpp")
        shown = args.label or equant
        print(f"\n##### {model}  engine={engine}  quant={equant}  ({path})")
        refs = {}
        for rpath, quants in REFS.get(model, []):
            refs.update(load(rpath, model=model, quants=quants))
        if not refs:
            print("  no reference cells configured"); continue
        a, t, s = hash_agreement(eng, refs)
        note = "" if a == s else ("   (hashes differ only by the template's run-date stamp; token counts agree)" if t == s
                                  else "   <-- MISMATCH beyond the date stamp, check template")
        print(f"  prompt-hash agreement with reference: {a}/{s}; prompt-token-count agreement: {t}/{s}{note}")
        for task in sorted({k[1] for k in eng}):
            print(f"\n  [{task}]  {'cond':<9} {'src':<9} {'n':>4} {'acc':>6} {'cap':>6} {'strict':>6} {'fail':>6} {'toks':>6}")
            for sampler, temp in CONDS:
                ek = (equant, task, sampler, temp)
                e = stats(eng.get(ek, {}))
                lab = "greedy" if sampler == "greedy" else f"T{temp}"
                if e:
                    print(f"  {'':9}{lab:<9} {shown:<9} {e['n']:>4} {e['acc']*100:6.1f} {e['cap']*100:6.1f} {e['strict']*100:6.1f} {e['fail']*100:6.1f} {e['toks']:6.0f}")
                else:
                    print(f"  {'':9}{lab:<9} {shown:<9}    - (not yet)")
                for rq in sorted({k[0] for k in refs}):
                    rk = (rq, task, sampler, temp)
                    r = stats(refs.get(rk, {}))
                    if not r:
                        continue
                    gap = fmt_ci(boot_diff(eng.get(ek, {}), refs[rk])) if e else "n/a"
                    print(f"  {'':9}{'':<9} {rq+'(ref)':<9} {r['n']:>4} {r['acc']*100:6.1f} {r['cap']*100:6.1f} {r['strict']*100:6.1f} {r['fail']*100:6.1f} {r['toks']:6.0f}   gap engine-ref: {gap}")
            # within-engine change 0.7 -> 1.3 in accuracy, cap-hit rate, strict-parse rate
            for q, src, name in [(equant, eng, shown)] + [(rq, refs, rq) for rq in sorted({k[0] for k in refs})]:
                hi, lo = src.get((q, task, "temperature", 1.3), {}), src.get((q, task, "temperature", 0.7), {})
                parts = [f"{m} {fmt_ci(boot_diff(hi, lo, m))}" for m in METRICS if boot_diff(hi, lo, m)]
                if parts:
                    print(f"  {'':9}change T0.7->T1.3 [{name}]: " + " | ".join(parts))
            # paired difference in accuracy drops, engine minus each reference
            for rq in sorted({k[0] for k in refs}):
                d = boot_diff_in_drops(eng.get((equant, task, "temperature", 1.3), {}), eng.get((equant, task, "temperature", 0.7), {}),
                                       refs.get((rq, task, "temperature", 1.3), {}), refs.get((rq, task, "temperature", 0.7), {}))
                if d:
                    print(f"  {'':9}drop({shown}) - drop({rq}): {fmt_ci(d)}")


if __name__ == "__main__":
    main()
