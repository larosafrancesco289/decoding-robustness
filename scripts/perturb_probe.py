#!/usr/bin/env python
"""Single-token perturbation probe: the causal test behind section 4.6.

Finding chain so far: derailment is absorbing (analyze_recovery.py), and Llama-3.1 ties
Mistral on every distributional statistic including the temperature-scaled beyond-top-20
escape hazard (analyze_escape.py) yet collapses five-fold harder. Hypothesis: fragility
is the model's instability to a SINGLE off-distribution token in its own context.

Protocol, per model (Q8) and frozen prompt:
  1. Greedy-decode a 128-token prefix with n_probs=20 (token strings + candidates).
     Skip the prompt if it terminates before INJECT_AT+1 tokens.
  2. Take the first INJECT_AT selected tokens as the prefix text; inject the RANK-20
     candidate at that position (the model's own realistic tail draw).
  3. Continue greedily for 256 tokens from (rendered prompt + prefix + injected) and,
     as control, from (rendered prompt + prefix). Greedy continuation isolates the
     deterministic response to the corruption.
  4. Record: capped (ran to 256) vs terminated, continuation length, and over the first
     32 post-injection positions the mean entropy lower bound and the T=1.3 escape
     hazard (continuation tail assumption), for both arms.

Prediction if the hypothesis holds: injected-arm cap rate Llama >> Mistral, control arms
~equal and low; post-injection entropy blows up for Llama only.

  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
  export LLAMA_SERVER_BIN=$PWD/vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/perturb_probe.py [--only-model M] [--limit N]
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

PREFIX_LEN = 128
INJECT_AT = 48
CONT_LEN = 256
HEAD_POS = 32  # continuation positions used for entropy/escape stats
TOPN = 20


def cand_probs(entry: dict) -> list[float]:
    return sorted((math.exp(t["logprob"]) for t in entry["top_logprobs"]), reverse=True)


def head_stats(entries: list[dict]) -> dict:
    """Mean entropy lower bound + T=1.3 escape hazard over the first HEAD_POS positions."""
    ents, escs = [], []
    for e in entries[:HEAD_POS]:
        probs = cand_probs(e)[:TOPN]
        if not probs:
            continue
        tail = max(0.0, 1.0 - sum(probs))
        h = -sum(p * math.log(p) for p in probs if p > 0)
        if tail > 0:
            h += -tail * math.log(tail)
        ents.append(h)
        inv = 1.0 / 1.3
        s_top = sum(p ** inv for p in probs if p > 0)
        p20 = probs[-1]
        s_tail = (tail / p20) * (p20 ** inv) if (tail > 0 and p20 > 0) else tail ** inv
        escs.append(s_tail / (s_top + s_tail) if s_top + s_tail > 0 else 0.0)
    if not ents:
        return {}
    return {"ent32": sum(ents) / len(ents), "esc32": sum(escs) / len(escs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_matrix.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--only-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max prompts per model (smoke)")
    ap.add_argument("--out-dir", default="results/perturbation")
    ap.add_argument("--n-ctx", type=int, default=8192)
    args = ap.parse_args()

    config = load_experiment_config(args.config)
    config = config.model_copy(
        update={"server": config.server.model_copy(update={"parallel": 1,
                                                           "n_ctx": args.n_ctx})}
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
                items = list(load_task(task))
                if args.limit:
                    items = items[: args.limit]
                for item in items:
                    key = f"{ckpt.name}|{task.name}|{item.item_id}"
                    if key in done:
                        continue
                    rendered = template.render(
                        [{"role": "user", "content": item.prompt}], enable_thinking=False
                    )
                    greedy = {"samplers": ["temperature"], "temperature": 0.0, "seed": 0,
                              "cache_prompt": True, "n_probs": TOPN,
                              "post_sampling_probs": False}
                    r = client.completion(rendered, {**greedy, "n_predict": PREFIX_LEN})
                    entries = r.raw.get("completion_probabilities", [])
                    if len(entries) <= INJECT_AT:
                        continue  # terminated before the injection point
                    prefix = "".join(e["token"] for e in entries[:INJECT_AT])
                    cands = entries[INJECT_AT]["top_logprobs"]
                    inj = min(cands, key=lambda t: t["logprob"])  # rank-20 candidate
                    arms = {}
                    for arm, text in (("inj", prefix + inj["token"]), ("ctrl", prefix)):
                        c = client.completion(rendered + text,
                                              {**greedy, "n_predict": CONT_LEN})
                        ce = c.raw.get("completion_probabilities", [])
                        arms[arm] = {
                            "len": len(ce),
                            "capped": len(ce) >= CONT_LEN,
                            **head_stats(ce),
                            "head_text": "".join(e["token"] for e in ce[:24]),
                        }
                    record = {"key": key, "model": ckpt.name, "task": task.name,
                              "item_id": item.item_id, "inject_at": INJECT_AT,
                              "inj_token": inj["token"],
                              "inj_prob": math.exp(inj["logprob"]),
                              "inj": arms["inj"], "ctrl": arms["ctrl"]}
                    with out_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    wrote += 1
                    if wrote % 20 == 0:
                        print(f"   ... {wrote} prompts", flush=True)
            print(f"   done: {wrote} new prompts", flush=True)


if __name__ == "__main__":
    main()
