#!/usr/bin/env python
"""Logit probe v2: store the raw per-position top-20 probabilities (not just summaries).

Motivation (2026-06-10 night): no top-20 summary statistic at T=1 separates Llama-3.1
from Mistral (entropy 0.329 vs 0.322, flat% 7.1 vs 6.8, tail mass ~1e-3 both), yet their
T1.3 MMLU-Pro drops differ five-fold, and analyze_recovery.py shows derailment is
ABSORBING (no recover-and-terminate footprint in any model). So the difference must be in
the temperature-scaled shape of the distribution: how much mass escapes past the plausible
set once probs are raised to 1/T. That is computable offline from the raw top-20 arrays,
which v1 threw away. This run stores them (~4 MB/model).

Same protocol as v1: frozen N=50 GSM8K + MMLU-Pro prompts, greedy path, Q8, n_probs=20,
post_sampling_probs=false. Output: results/logit_probe_v2/<model>.jsonl, one line per
prompt with `probs` = [[p1..p20] per position, descending] rounded to 6 significant
digits. Analysis: scripts/analyze_escape.py.

  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
  export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/logit_probe2.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoding_robustness.config import load_experiment_config  # noqa: E402
from decoding_robustness.config.schema import QuantLevel  # noqa: E402
from decoding_robustness.inference import ChatTemplate, llama_server  # noqa: E402
from decoding_robustness.tasks import load_task  # noqa: E402

MAX_NEW = 256
TOPN = 20


def position_probs(entry: dict) -> list[float]:
    if "top_logprobs" in entry:
        return [math.exp(t["logprob"]) for t in entry["top_logprobs"]]
    if "top_probs" in entry:
        return [t["prob"] for t in entry["top_probs"]]
    if "probs" in entry:
        return [t["prob"] for t in entry["probs"]]
    raise KeyError(f"unrecognized completion_probabilities entry: {list(entry)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_matrix.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--out-dir", default="results/logit_probe_v2")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=16384)
    args = ap.parse_args()

    config = load_experiment_config(args.config)
    config = config.model_copy(
        update={
            "server": config.server.model_copy(
                update={"parallel": args.parallel, "n_ctx": args.n_ctx}
            )
        }
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    for ckpt in config.checkpoints:
        if args.only_model and ckpt.name not in args.only_model.split(","):
            continue
        fname = ckpt.quant_files.get(QuantLevel.Q8_0)
        path = model_dir / fname if fname else None
        if path is None or not path.is_file():
            print(f"-- {ckpt.name}: no Q8_0 GGUF, skipped")
            continue
        out_path = out_dir / f"{ckpt.name}.jsonl"
        done = set()
        if out_path.is_file():
            done = {json.loads(line)["key"] for line in open(out_path) if line.strip()}

        meta = path.with_suffix(path.suffix + ".meta.json")
        template = ChatTemplate.from_meta_file(meta)
        print(f"== {ckpt.name} (Q8_0) -> {out_path}", flush=True)
        with llama_server(path, config.server) as client:
            wrote = 0
            for task in config.tasks:
                for item in load_task(task):
                    key = f"{ckpt.name}|{task.name}|{item.item_id}"
                    if key in done:
                        continue
                    rendered = template.render(
                        [{"role": "user", "content": item.prompt}], enable_thinking=False
                    )
                    params = {
                        "samplers": ["temperature"],
                        "temperature": 0.0,
                        "seed": 0,
                        "n_predict": MAX_NEW,
                        "cache_prompt": True,
                        "n_probs": TOPN,
                        "post_sampling_probs": False,
                    }
                    result = client.completion(rendered, params)
                    entries = result.raw.get("completion_probabilities", [])
                    if not entries:
                        print(f"   !! {key}: no completion_probabilities; skipped")
                        continue
                    positions = []
                    for e in entries:
                        probs = sorted(position_probs(e), reverse=True)[:TOPN]
                        positions.append([float(f"{p:.6g}") for p in probs])
                    if wrote == 0 and positions:
                        near1 = sum(p[0] > 0.999 for p in positions) / len(positions)
                        if near1 > 0.99:
                            raise SystemExit(
                                "top-1 ~1.0 everywhere: post-sampler probs; aborting."
                            )
                    record = {"key": key, "model": ckpt.name, "task": task.name,
                              "item_id": item.item_id, "probs": positions}
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    wrote += 1
                    if wrote % 20 == 0:
                        print(f"   ... {wrote} prompts", flush=True)
            print(f"   done: {wrote} new prompts", flush=True)


if __name__ == "__main__":
    main()
