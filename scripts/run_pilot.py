#!/usr/bin/env python
"""M3 pilot driver (SPEC §13): run the (quant × decode-grid × task) matrix for the canary
model and print the first sampler×quant accuracy table.

Runs incrementally — quants not yet downloaded are skipped (and reported), tasks whose
parser isn't registered yet are skipped. Everything appends to one resumable JSONL, so
re-running after more quants arrive just fills in the missing cells.

  export LLAMA_SERVER_BIN=vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/run_pilot.py --config configs/pilot_llama.yaml          # full subset
  uv run python scripts/run_pilot.py --config configs/pilot_llama.yaml --limit 10  # quick smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoding_robustness.analysis import SamplerQuantTable, SamplerTemperatureTable  # noqa: E402
from decoding_robustness.config import load_experiment_config  # noqa: E402
from decoding_robustness.runner.matrix import run_matrix  # noqa: E402

RULE = "-" * 78


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_llama.yaml")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--binary", default=None, help="llama-server path (or $LLAMA_SERVER_BIN)")
    parser.add_argument("--limit", type=int, default=None, help="override per-task subset size")
    parser.add_argument(
        "--out", default=None, help="JSONL path (default: <output_dir>/<phase>.jsonl)"
    )
    parser.add_argument(
        "--table-only", action="store_true", help="skip generation, just print the table"
    )
    parser.add_argument(
        "--only-model",
        default=None,
        help="comma-separated checkpoint name(s) to run (default: all) — for Slurm array tasks",
    )
    parser.add_argument(
        "--only-quant",
        default=None,
        help="comma-separated quant level(s) to run, e.g. 'Q4_K_M' (default: all)",
    )
    # Server overrides for Slurm array tasks: a distinct port avoids 8080
    # collisions when the scheduler packs several array tasks onto one shared GPU node, and
    # --parallel / --n-ctx let a VRAM-tight cell on an 11GB card dial back without editing
    # the config. These touch only infrastructure, never the locked decoding/sampling science.
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    parser.add_argument("--parallel", type=int, default=None, help="override server.parallel")
    parser.add_argument("--n-ctx", type=int, default=None, help="override server.n_ctx")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="render with enable_thinking=True (Qwen3 thinking-ON spot-check only; NOT the "
        "frozen matrix). Write to a separate --out so it doesn't collide with thinking-OFF ids.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    server_overrides = {
        k: v
        for k, v in (("port", args.port), ("parallel", args.parallel), ("n_ctx", args.n_ctx))
        if v is not None
    }
    if server_overrides:
        config = config.model_copy(
            update={"server": config.server.model_copy(update=server_overrides)}
        )
    out_path = Path(args.out) if args.out else Path(config.output_dir) / f"{config.phase}.jsonl"

    if not args.table_only:
        n_cond = len(list(config.decoding_conditions()))
        print(f"pilot: {config.name}")
        print(f"  quants     : {[q.value for q in config.quant_levels]}")
        print(f"  conditions : {n_cond} (sampler×temp)")
        print(f"  tasks      : {[t.name for t in config.tasks]}")
        print(f"  out        : {out_path}")
        print(RULE)

        done = {"n": 0}
        started = time.monotonic()

        def _tick(record):
            done["n"] += 1
            if done["n"] % 25 == 0:
                rate = done["n"] / (time.monotonic() - started)
                print(f"  ... {done['n']} generated ({rate:.1f}/s)")

        only_models = {m.strip() for m in args.only_model.split(",")} if args.only_model else None
        only_quants = {q.strip() for q in args.only_quant.split(",")} if args.only_quant else None
        progress = run_matrix(
            config,
            model_dir=args.model_dir,
            binary=args.binary,
            out_path=out_path,
            limit=args.limit,
            only_models=only_models,
            only_quants=only_quants,
            enable_thinking=args.enable_thinking,
            on_record=_tick,
        )

        print(RULE)
        print(f"server loads   : {progress.server_loads}")
        print(f"generated      : {progress.generated}  (skipped/resumed: {progress.skipped})")
        if progress.missing_quants:
            print(f"missing quants : {', '.join(progress.missing_quants)} (GGUF not on disk yet)")
        if progress.skipped_tasks:
            print(
                f"skipped tasks  : {', '.join(progress.skipped_tasks)} (no parser registered yet)"
            )

    print(RULE)
    print("SAMPLER × QUANT ACCURACY (pooled over temperature)")
    print(SamplerQuantTable.from_jsonl(out_path).render())

    print(RULE)
    print("SAMPLER × TEMPERATURE ACCURACY (pooled over quant) — the SQ1 headline")
    print(SamplerTemperatureTable.from_jsonl(out_path).render())


if __name__ == "__main__":
    main()
