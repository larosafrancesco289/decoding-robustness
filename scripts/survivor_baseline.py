#!/usr/bin/env python3
"""Survivor-matched greedy baseline for Section 4.3.

Among plain-temperature generations at T=1.3 that reach a strict parse (the survivors),
accuracy is high partly because easy items survive. The matched baseline is greedy decoding
on the same (quantization level, item) pairs, weighted like the survivors. Prints, per task:
accuracy among survivors, greedy on the survivors, and greedy on all items (pooled over
quantization). CPU, seconds.
"""
import glob
import json


def main(model="llama-3.1-8b-instruct"):
    recs = [json.loads(l) for f in glob.glob(f"results/full_matrix/shards/{model}__*.jsonl")
            for l in open(f, encoding="utf-8")]
    for task in ("gsm8k", "mmlu_pro"):
        greedy = {(r["quant"], r["item_id"]): r["correct"] for r in recs
                  if r["task"] == task and r["sampler"] == "greedy"}
        surv = [r for r in recs if r["task"] == task and r["sampler"] == "temperature"
                and r["temperature"] == 1.3 and r["parse_method"] == "strict"]
        acc = sum(r["correct"] for r in surv) / len(surv) * 100
        matched = sum(greedy[(r["quant"], r["item_id"])] for r in surv) / len(surv) * 100
        allg = sum(greedy.values()) / len(greedy) * 100
        print(f"{task}: n_survivors={len(surv)} acc|strict={acc:.1f} greedy|survivors={matched:.1f} greedy|all={allg:.1f}")


if __name__ == "__main__":
    main()
