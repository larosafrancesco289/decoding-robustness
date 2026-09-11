#!/usr/bin/env python
"""Engine and precision replication of the temperature axis through HF transformers.

The main grid runs every model through llama.cpp on imatrix GGUF quantizations. This script
repeats the collapse axis (greedy plus plain temperature at the config's temperatures) on a
model served by a different engine (transformers) at a chosen precision (bf16, or 8-bit via
bitsandbytes), with everything else held identical to the grid:

  * the same frozen items (loaded through the study's task loaders and config),
  * the same rendered prompt string (the GGUF's embedded chat template, rendered client-side
    from its .meta.json sidecar, so the prompt hash matches the grid's records),
  * the same token budgets, the same parser and grader, the same record schema, so every
    existing analysis script reads the output unchanged.

Sampling is plain temperature: do_sample=True, temperature=T, top_k=0, top_p=1.0, no repetition
penalty. Greedy is do_sample=False. Records carry quant="BF16-hf" (or "INT8-hf") and
params.engine="transformers" so they can never be confused with llama.cpp records.

Usage (on the GPU box, after `uv sync --extra engine`):

  uv run python scripts/run_transformers_axis.py \
      --config configs/llama_lineage.yaml --model-name llama-3.2-3b-instruct \
      --hf-model meta-llama/Llama-3.2-3B-Instruct \
      --meta models/Llama-3.2-3B-Instruct-Q8_0.gguf.meta.json \
      --out results/engine_check/llama-3.2-3b-instruct__BF16-hf.jsonl

  --load-in-8bit uses bitsandbytes int8 (fits Llama-3.1-8B on 16 GB); --device-map auto lets
  accelerate offload to CPU when bf16 weights exceed VRAM (slow, but exact).
  Resumable: existing record ids in --out are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoding_robustness.config.loader import load_experiment_config  # noqa: E402
from decoding_robustness.inference.templating import ChatTemplate, prompt_sha256  # noqa: E402
from decoding_robustness.runner.records import (  # noqa: E402
    GenerationRecord,
    RecordStore,
    make_record_id,
)
from decoding_robustness.seeding import derive_seed  # noqa: E402
from decoding_robustness.tasks.grading import grade  # noqa: E402
from decoding_robustness.tasks.loaders import load_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML (tasks, temperatures, repetitions)")
    ap.add_argument("--model-name", required=True, help="record model name, e.g. llama-3.2-3b-instruct")
    ap.add_argument("--hf-model", required=True, help="HF repo id of the instruct model")
    ap.add_argument("--meta", default=None, help="GGUF .meta.json sidecar carrying the chat template; if omitted, the HF tokenizer's own template is used (smoke tests only)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--quant-label", default=None, help="record quant label (default BF16-hf / INT8-hf)")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--device-map", default="cuda", help="'cuda' or 'auto' (CPU offload)")
    ap.add_argument("--max-memory", default=None,
                    help="with --device-map auto: accelerate max_memory, e.g. '0=13GiB,cpu=26GiB' (leave VRAM headroom for the KV cache)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperatures", type=float, nargs="*", default=None, help="override config")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names")
    ap.add_argument("--greedy", dest="greedy", action="store_true", default=True)
    ap.add_argument("--no-greedy", dest="greedy", action="store_false")
    ap.add_argument("--limit-items", type=int, default=None, help="smoke test: first N items only")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_experiment_config(args.config)
    temps = args.temperatures if args.temperatures is not None else list(cfg.temperatures)
    reps = cfg.generation.repetitions
    global_seed = cfg.generation.global_seed
    quant = args.quant_label or ("INT8-hf" if args.load_in_8bit else "BF16-hf")
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if args.meta:
        template = ChatTemplate.from_meta_file(args.meta)
    else:
        print("[engine-check] WARNING: no --meta given; using the HF tokenizer's chat template (prompt hashes will not match the grid)")
        template = ChatTemplate(template=tok.chat_template, bos_token=tok.bos_token or "", eos_token=tok.eos_token or "")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    load_kw: dict = {"device_map": args.device_map}
    if args.max_memory:
        mm = {}
        for part in args.max_memory.split(","):
            k, v = part.split("=")
            mm[int(k) if k.isdigit() else k] = v
        load_kw["max_memory"] = mm
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kw)
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tok.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    import transformers

    engine_tag = f"transformers=={transformers.__version__}; torch=={torch.__version__}"
    # Provenance and the resolved generation defaults (anything not overridden below is inherited from
    # the checkpoint's generation_config.json; we neutralise the penalties explicitly and log the rest).
    hf_revision = getattr(model.config, "_commit_hash", None)
    gen_defaults = {k: v for k, v in model.generation_config.to_dict().items()
                    if k in ("temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram_size",
                             "typical_p", "min_p", "eos_token_id", "pad_token_id", "bos_token_id", "do_sample", "num_beams")}
    device_map_used = getattr(model, "hf_device_map", None)
    print(f"[engine-check] hf_revision={hf_revision} eos_ids={eos_ids} generation_config={gen_defaults}")
    if device_map_used:
        print(f"[engine-check] device_map={device_map_used}")

    store = RecordStore(args.out)
    done = store.existing_ids()
    print(f"[engine-check] {args.model_name} {quant} via {engine_tag}; {len(done)} records already done")

    conditions: list[tuple[str, float]] = []
    if args.greedy:
        conditions.append(("greedy", 0.0))
    conditions += [("temperature", t) for t in temps]

    for task in cfg.tasks:
        if args.tasks and task.name not in args.tasks:
            continue
        items = load_task(task)
        if args.limit_items:
            items = items[: args.limit_items]
        rendered = {
            it.item_id: template.render([{"role": "user", "content": it.prompt}], enable_thinking=False)
            for it in items
        }
        for sampler, temp in conditions:
            n_reps = 1 if sampler == "greedy" else reps
            units = []
            for it in items:
                for r in range(1, n_reps + 1):
                    rid = make_record_id(model=args.model_name, quant=quant, sampler=sampler,
                                         temperature=temp, task=task.name, item_id=it.item_id, repetition=r)
                    if rid in done:
                        continue
                    units.append((rid, it, r))
            print(f"[engine-check] {task.name} {sampler} T={temp}: {len(units)} to generate")
            for start in range(0, len(units), args.batch_size):
                batch = units[start:start + args.batch_size]
                prompts = [rendered[u[1].item_id] for u in batch]
                # The rendered prompt already contains the BOS text from the template; do not add it again.
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                seed = derive_seed(global_seed=global_seed, model=args.model_name, quant=quant,
                                   sampler=sampler, temperature=temp, task=task.name,
                                   item_id=batch[0][1].item_id, repetition=batch[0][2])
                torch.manual_seed(seed)
                gen_kw: dict = dict(max_new_tokens=task.max_new_tokens, eos_token_id=eos_ids,
                                    pad_token_id=tok.pad_token_id, repetition_penalty=1.0,
                                    no_repeat_ngram_size=0, num_beams=1)
                if sampler == "greedy":
                    gen_kw.update(do_sample=False)
                else:
                    gen_kw.update(do_sample=True, temperature=float(temp), top_k=0, top_p=1.0)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**enc, **gen_kw)
                latency = time.perf_counter() - t0
                n_prompt = enc["input_ids"].shape[1]
                for bi, ((rid, it, r), row) in enumerate(zip(batch, out)):
                    comp = row[n_prompt:]
                    # strip padding and count real completion tokens; stopped = an EOS was produced
                    comp_list = comp.tolist()
                    stopped = any(t in eos_ids for t in comp_list)
                    end_token = None
                    if stopped:
                        first_eos = next(i for i, t in enumerate(comp_list) if t in eos_ids)
                        end_token = comp_list[first_eos]
                        comp_list = comp_list[:first_eos]
                    comp_list = [t for t in comp_list if t != tok.pad_token_id or tok.pad_token_id in eos_ids]
                    text = tok.decode(comp_list, skip_special_tokens=True)
                    g = grade(text, it.gold, task.parser)
                    rec = GenerationRecord(
                        id=rid, model=args.model_name, quant=quant, sampler=sampler,
                        params={"engine": "transformers", "dtype": "int8" if args.load_in_8bit else "bf16",
                                "hf_model": args.hf_model, "hf_revision": hf_revision,
                                "samplers": ["temperature"], "temperature": temp, "top_k": 0, "top_p": 1.0,
                                "repetition_penalty": 1.0, "eos_ids": list(eos_ids), "end_token": end_token,
                                "generation_config": gen_defaults, "device_map": args.device_map,
                                "n_predict": task.max_new_tokens, "batch_size": len(batch),
                                "batch_index": start // args.batch_size, "batch_row": bi},
                        temperature=temp, task=task.name, item_id=it.item_id, repetition=r, seed=seed,
                        prompt_hash=prompt_sha256(rendered[it.item_id]), raw_output=text,
                        parsed_answer=g.parsed_answer, parse_method=g.parse_method, gold=g.gold,
                        correct=g.correct, n_prompt_tokens=int((enc["attention_mask"][bi] == 1).sum()),
                        n_completion_tokens=len(comp_list), latency_s=latency,
                        tokens_per_second=None, stopped=stopped, server_commit=engine_tag,
                        timestamp=dt.datetime.now(dt.UTC).isoformat(),
                    )
                    store.append(rec)
                print(f"  batch {start // args.batch_size + 1}: {latency:.1f}s")


if __name__ == "__main__":
    main()
