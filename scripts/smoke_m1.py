#!/usr/bin/env python
"""Inference smoke test: one model, one quant, a handful of completions.

Runs each decoding method in the config against a live llama-server and prints the raw
text side by side, so you can eyeball the sanity checks:
  * do top-p / min-p / top-nσ produce sane, *different* outputs?
  * does seed determinism hold under single-stream serving?

This is the artifact you run on the GPU (after build_llamacpp.sh + fetch_models.py); it
does no grading; grading lives in src/decoding_robustness/tasks.

  export LLAMA_SERVER_BIN=vendor/llama.cpp/build/bin/llama-server
  uv run python scripts/smoke_m1.py --config configs/pilot_llama.yaml --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoding_robustness.config import QuantLevel, load_experiment_config  # noqa: E402
from decoding_robustness.config.schema import DecodingMethod  # noqa: E402
from decoding_robustness.inference import (  # noqa: E402
    ChatTemplate,
    completion_params,
    llama_server,
    prompt_sha256,
)
from decoding_robustness.inference.sampling import chain_label  # noqa: E402
from decoding_robustness.seeding import derive_seed  # noqa: E402

PROMPTS = [
    "A robot has 3 boxes. Each box holds 4 red balls and 5 blue balls. "
    "How many balls are there in total? Think step by step.",
    "In two sentences, explain why the sky is blue.",
]

RULE = "-" * 78


def run_condition(
    client, template, ckpt_name, quant, sampler, temperature, prompt, *, n_predict, item_id, stop
):
    """Render the prompt, derive the paired seed, and run one completion."""
    rendered = template.render([{"role": "user", "content": prompt}])
    seed = derive_seed(
        global_seed=0,
        model=ckpt_name,
        quant=quant,
        sampler=sampler.label,
        temperature=temperature,
        task="smoke",
        item_id=item_id,
        repetition=1,
    )
    params = completion_params(sampler, temperature, seed, n_predict=n_predict, stop=stop)
    result = client.completion(rendered, params)
    return rendered, seed, params, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_llama.yaml")
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--checkpoint", default=None, help="checkpoint name (default: first)")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--binary", default=None, help="llama-server path (or $LLAMA_SERVER_BIN)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-predict", type=int, default=200)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    ckpt = next(
        (c for c in config.checkpoints if c.name == args.checkpoint),
        config.checkpoints[0],
    )
    quant = QuantLevel(args.quant)
    filename = ckpt.gguf_filename(quant)
    model_path = Path(args.model_dir) / filename
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")

    if ckpt.chat_template is not None:
        template = ChatTemplate(template=ckpt.chat_template)
    elif meta_path.is_file():
        template = ChatTemplate.from_meta_file(meta_path)
    else:
        raise SystemExit(
            f"no chat template: set Checkpoint.chat_template or fetch {filename} "
            f"(fetch_models.py writes {meta_path.name})"
        )
    stop = [template.eos_token] if template.eos_token else None

    print(f"model      : {ckpt.name} @ {quant.value}")
    print(f"file       : {model_path}")
    print(f"temperature: {args.temperature} (greedy forced to 0.0)")
    print(RULE)

    with llama_server(model_path, config.server, binary=args.binary) as client:
        props = client.props()
        print(f"server     : {props.get('model_path', '?')}")
        print(RULE)

        for prompt in PROMPTS:
            print(f"\nPROMPT: {prompt}\n")
            for sampler in config.samplers:
                temp = 0.0 if sampler.method == DecodingMethod.GREEDY else args.temperature
                rendered, seed, params, result = run_condition(
                    client,
                    template,
                    ckpt.name,
                    quant.value,
                    sampler,
                    temp,
                    prompt,
                    n_predict=args.n_predict,
                    item_id=prompt[:16],
                    stop=stop,
                )
                tps = result.predicted_per_second
                print(
                    f"[{sampler.label}]  chain={chain_label(params['samplers'])}  "
                    f"seed={seed}  prompt_sha={prompt_sha256(rendered)[:12]}  "
                    f"tok={result.tokens_predicted}"
                    f"{f'  {tps:.1f} tok/s' if tps else ''}"
                )
                print(f"  {result.content.strip()}\n")
            print(RULE)

        # Seed determinism: same seed -> identical text; different seed -> different text.
        print("\nDETERMINISM CHECK (single-stream)")
        stochastic = next(s for s in config.samplers if s.method == DecodingMethod.MIN_P)
        rendered = template.render([{"role": "user", "content": PROMPTS[0]}])
        params_a = completion_params(
            stochastic, args.temperature, 12345, n_predict=args.n_predict, stop=stop
        )
        out_a1 = client.completion(rendered, params_a).content
        out_a2 = client.completion(rendered, params_a).content
        params_b = completion_params(
            stochastic, args.temperature, 67890, n_predict=args.n_predict, stop=stop
        )
        out_b = client.completion(rendered, params_b).content
        print(f"  same seed reproducible : {out_a1 == out_a2}")
        print(f"  different seed differs  : {out_a1 != out_b}")


if __name__ == "__main__":
    main()
