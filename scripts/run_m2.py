#!/usr/bin/env python
"""Task end-to-end driver: one model, one quant, GSM8K, greedy single-shot.

Loads a GSM8K subset at the config's pinned revision, launches llama-server for the
chosen quant, runs greedy decoding over the items through the resumable loop, grades each
with the deterministic numeric parser, appends JSONL records, and prints a check:

  * accuracy vs. the known Llama-3.1-8B-Instruct GSM8K number,
  * parse-failure rate,
  * a few graded transcripts to eyeball.

Re-running resumes from the JSONL (completed ids are skipped).

  export LLAMA_SERVER_BIN=vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/run_m2.py --config configs/pilot_llama.yaml --quant Q4_K_M --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoding_robustness.config import QuantLevel, load_experiment_config  # noqa: E402
from decoding_robustness.config.schema import DecodingMethod, SamplerSpec  # noqa: E402
from decoding_robustness.inference import ChatTemplate, llama_server  # noqa: E402
from decoding_robustness.runner import RecordStore, run_conditions  # noqa: E402
from decoding_robustness.tasks import load_task  # noqa: E402

RULE = "-" * 78


def _resolve_template(ckpt, model_path: Path) -> ChatTemplate:
    if ckpt.chat_template is not None:
        return ChatTemplate(template=ckpt.chat_template)
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if meta_path.is_file():
        return ChatTemplate.from_meta_file(meta_path)
    raise SystemExit(
        f"no chat template: set Checkpoint.chat_template or fetch {model_path.name} "
        f"(fetch_models.py writes {meta_path.name})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_llama.yaml")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--checkpoint", default=None, help="checkpoint name (default: first)")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--binary", default=None, help="llama-server path (or $LLAMA_SERVER_BIN)")
    parser.add_argument("--limit", type=int, default=20, help="number of items (default: 20)")
    parser.add_argument(
        "--out", default=None, help="JSONL path (default: <output_dir>/m2_<quant>.jsonl)"
    )
    parser.add_argument("--show", type=int, default=3, help="transcripts to print")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    ckpt = next(
        (c for c in config.checkpoints if c.name == args.checkpoint),
        config.checkpoints[0],
    )
    quant = QuantLevel(args.quant)
    task = next(t for t in config.tasks if t.name == args.task)
    # Uses the run's --limit, not the config subset_size (which is the full pilot size).
    task = task.model_copy(update={"subset_size": args.limit})

    model_path = Path(args.model_dir) / ckpt.gguf_filename(quant)
    template = _resolve_template(ckpt, model_path)
    stop = [template.eos_token] if template.eos_token else None

    out_path = Path(args.out) if args.out else Path(config.output_dir) / f"m2_{quant.value}.jsonl"
    store = RecordStore(out_path)

    greedy = next(
        (s for s in config.samplers if s.method == DecodingMethod.GREEDY),
        SamplerSpec(method=DecodingMethod.GREEDY),
    )

    print(f"model      : {ckpt.name} @ {quant.value}")
    print(f"task       : {task.name} (revision={task.revision}, split={task.split})")
    print(f"out        : {out_path}")
    print(RULE)
    print(f"loading {task.name} ...")
    items = load_task(task)
    print(f"  {len(items)} items")
    print(RULE)

    shown = 0

    def _on_record(record):
        nonlocal shown
        if shown < args.show:
            shown += 1
            mark = "OK " if record.correct else "XX "
            print(
                f"[{mark}] {record.item_id}  gold={record.gold}  "
                f"pred={record.parsed_answer} ({record.parse_method})  "
                f"{record.n_completion_tokens} tok"
            )
            print(f"      {record.raw_output.strip()[-300:]}\n")

    with llama_server(model_path, config.server, binary=args.binary) as client:
        summary = run_conditions(
            client=client,
            template=template,
            checkpoint=ckpt,
            quant=quant,
            task=task,
            items=items,
            conditions=[(greedy, 0.0)],
            store=store,
            global_seed=config.generation.global_seed,
            server_commit=config.server.llama_cpp_commit,
            stop=stop,
            on_record=_on_record,
        )

    print(RULE)
    print("SANITY CHECK")
    print(f"  generated      : {summary.generated}  (skipped/resumed: {summary.skipped})")
    if summary.accuracy is not None:
        print(f"  accuracy       : {summary.correct}/{summary.generated} = {summary.accuracy:.1%}")
        print(f"  parse failures : {summary.parse_failures} ({summary.parse_failure_rate:.1%})")
    else:
        print("  (nothing new generated; all items already on disk)")


if __name__ == "__main__":
    main()
