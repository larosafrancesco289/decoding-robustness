#!/usr/bin/env python
"""Per-token next-token-distribution probe (the SQ4 mechanism analysis).

Hypothesis from DECOMP.txt: the temperature collapse is degeneration — one bad tail-token
draw derails the sequence, and the per-token tail mass compounds over length. If so, the
fragile model (Llama-3.1) should put systematically MORE probability mass in the tail of
its next-token distribution along its own greedy path than robust models do, and the
model ranking on tail mass should predict the cliff ordering.

Method: for every matrix model (Q8), decode the frozen N=50 GSM8K + MMLU-Pro prompts
GREEDILY (the model's modal path; no sampling confound) with ``n_probs=20`` and
``post_sampling_probs=false``, i.e. the raw softmax top-20 per position. Per position we
take: top-1 prob, top1-top2 margin, tail mass (1 - sum of top-20), and the entropy lower
bound H = -sum p ln p - tail*ln(tail). Per prompt we store means + the fraction of "flat"
positions (top-1 < 0.5). One JSONL line per prompt -> results/logit_probe/<model>.jsonl.

Sanity check: if the reported top-1 prob is ~1.0 at >99% of positions, the server gave us
post-sampler (argmax-collapsed) probs and the run aborts with a clear message.

  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
  export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/logit_probe.py                       # all models with a Q8 GGUF
  uv run python scripts/logit_probe.py --only-model qwen3-8b --parallel 4 --n-ctx 12288
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

MAX_NEW = 256  # enough positions for stable per-prompt stats; probe, not eval
TOPN = 20


def position_probs(entry: dict) -> list[float]:
    """Extract the top-N raw probs for one position, across server response formats."""
    if "top_logprobs" in entry:  # OAI-style (current servers): logprob fields
        return [math.exp(t["logprob"]) for t in entry["top_logprobs"]]
    if "top_probs" in entry:  # post_sampling_probs=true style
        return [t["prob"] for t in entry["top_probs"]]
    if "probs" in entry:  # legacy: [{tok_str, prob}]
        return [t["prob"] for t in entry["probs"]]
    raise KeyError(f"unrecognized completion_probabilities entry: {list(entry)}")


def prompt_stats(positions: list[list[float]]) -> dict:
    tops, margins, tails, ents = [], [], [], []
    for probs in positions:
        probs = sorted(probs, reverse=True)[:TOPN]
        if not probs:
            continue
        top1 = probs[0]
        tail = max(0.0, 1.0 - sum(probs))
        h = -sum(p * math.log(p) for p in probs if p > 0)
        if tail > 0:
            h += -tail * math.log(tail)  # lump the tail: lower bound on true entropy
        tops.append(top1)
        margins.append(top1 - (probs[1] if len(probs) > 1 else 0.0))
        tails.append(tail)
        ents.append(h)
    n = len(tops)
    if not n:
        return {}
    return {
        "n_pos": n,
        "mean_top1": sum(tops) / n,
        "mean_margin": sum(margins) / n,
        "mean_tail_mass": sum(tails) / n,
        "mean_entropy_lb": sum(ents) / n,
        "frac_flat_pos": sum(t < 0.5 for t in tops) / n,
        "frac_top1_near1": sum(t > 0.999 for t in tops) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_matrix.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--out-dir", default="results/logit_probe")
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
        print(f"== {ckpt.name} (Q8_0) -> {out_path}")
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
                        "temperature": 0.0,  # greedy: probe the modal path
                        "seed": 0,
                        "n_predict": MAX_NEW,
                        "cache_prompt": True,
                        "n_probs": TOPN,
                        "post_sampling_probs": False,
                    }
                    result = client.completion(rendered, params)
                    entries = result.raw.get("completion_probabilities", [])
                    if not entries:
                        print(f"   !! {key}: no completion_probabilities "
                              f"(parse_error_recovered={result.parse_error_recovered}) — skipped")
                        continue
                    stats = prompt_stats([position_probs(e) for e in entries])
                    if not stats:
                        continue
                    if wrote == 0 and stats["frac_top1_near1"] > 0.99:
                        raise SystemExit(
                            "top-1 prob ~1.0 everywhere: server returned post-sampler "
                            "(argmax-collapsed) probs; re-run with a T=1.0 chain instead."
                        )
                    record = {"key": key, "model": ckpt.name, "task": task.name,
                              "item_id": item.item_id, **stats}
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    wrote += 1
                    if wrote % 20 == 0:
                        print(f"   ... {wrote} prompts")
            print(f"   done: {wrote} new prompts")


if __name__ == "__main__":
    main()
