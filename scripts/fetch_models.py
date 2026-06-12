#!/usr/bin/env python
"""Fetch a GGUF checkpoint and record what we need to reproduce it (SPEC §7, §12).

For each model file this:
  1. downloads the GGUF from its Hugging Face repo (Bartowski's imatrix quants),
  2. records the SHA256 (pinned in the manifest), and
  3. extracts the embedded chat template + BOS/EOS tokens into a ``<file>.meta.json``
     sidecar, so prompts are rendered client-side from the model's own template.

Usage (M1 canary):
  uv run python scripts/fetch_models.py \
      --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
      --file Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Or resolve repo/file from a config's checkpoint:
  uv run python scripts/fetch_models.py --config configs/pilot_llama.yaml --quant Q4_K_M
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_value(reader, key):  # noqa: ANN001 - gguf types are version-fluid
    """Best-effort read of a GGUF metadata field value across gguf versions."""
    field = reader.get_field(key)
    if field is None:
        return None
    # Newer gguf exposes .contents(); fall back to a manual single-value decode.
    contents = getattr(field, "contents", None)
    if callable(contents):
        return field.contents()
    return str(bytes(field.parts[field.data[0]]), encoding="utf-8")


def extract_metadata(gguf_path: Path) -> dict:
    """Pull chat template + BOS/EOS token strings out of the GGUF. Best-effort."""
    from gguf import GGUFReader  # imported lazily so download works without parsing

    reader = GGUFReader(str(gguf_path))
    meta: dict = {}
    template = _field_value(reader, "tokenizer.chat_template")
    if template:
        meta["chat_template"] = template

    tokens = _field_value(reader, "tokenizer.ggml.tokens")
    for name, key in (
        ("bos_token", "tokenizer.ggml.bos_token_id"),
        ("eos_token", "tokenizer.ggml.eos_token_id"),
    ):
        token_id = _field_value(reader, key)
        if tokens is not None and token_id is not None:
            meta[name] = tokens[int(token_id)]
    return meta


def resolve_from_config(config_path: str, quant: str, checkpoint: str | None):
    """Return (repo, filename) for a checkpoint+quant in an experiment config."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from decoding_robustness.config import QuantLevel, load_experiment_config

    config = load_experiment_config(config_path)
    ckpts = config.checkpoints
    if checkpoint is not None:
        ckpts = [c for c in ckpts if c.name == checkpoint]
        if not ckpts:
            raise SystemExit(f"checkpoint '{checkpoint}' not found in {config_path}")
    ckpt = ckpts[0]
    return ckpt.hf_repo, ckpt.gguf_filename(QuantLevel(quant))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="HF GGUF repo, e.g. bartowski/...-GGUF")
    parser.add_argument("--file", help="GGUF filename within the repo")
    parser.add_argument("--config", help="resolve --repo/--file from this experiment config")
    parser.add_argument("--checkpoint", help="checkpoint name (with --config; default: first)")
    parser.add_argument("--quant", help="quant level (with --config), e.g. Q4_K_M")
    parser.add_argument("--out-dir", default="models", help="download directory (default: models)")
    parser.add_argument("--revision", default=None, help="HF repo revision to pin")
    parser.add_argument("--skip-template", action="store_true", help="don't extract chat template")
    parser.add_argument(
        "--local",
        help="path to an already-downloaded GGUF: skip the download, just hash + extract metadata",
    )
    args = parser.parse_args()

    repo = filename = None
    if args.config and args.quant:
        repo, filename = resolve_from_config(args.config, args.quant, args.checkpoint)
    elif args.repo and args.file:
        repo, filename = args.repo, args.file

    if args.local:
        # File fetched out-of-band (e.g. via the HF web UI): no download, just record it.
        local_path = Path(args.local)
        if not local_path.is_file():
            raise SystemExit(f"--local file not found: {local_path}")
        filename = filename or local_path.name
    else:
        if repo is None or filename is None:
            raise SystemExit(
                "provide --repo and --file, or --config and --quant, or --local <path>"
            )
        from huggingface_hub import hf_hub_download

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f">> downloading {repo}/{filename} (rev={args.revision or 'main'})")
        local_path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=args.revision,
                local_dir=str(out_dir),
            )
        )

    print(">> hashing (SHA256)...")
    sha = _sha256(local_path)

    meta = {
        "repo": repo,
        "filename": filename,
        "revision": args.revision,
        "sha256": sha,
    }
    if not args.skip_template:
        try:
            meta.update(extract_metadata(local_path))
        except ImportError as exc:
            # Hard failure, not "best effort": the sidecar would be written without a
            # template and silently break the runner later. (`uv sync` drops the optional
            # gguf dep unless --extra fetch is passed.)
            raise SystemExit(
                f"chat-template extraction needs the 'gguf' package ({exc}); "
                "run: uv sync --extra fetch"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - a genuinely template-less GGUF is tolerated
            print(
                f"!! could not extract chat template ({exc}); "
                "set Checkpoint.chat_template in the config instead"
            )

    meta_path = local_path.with_suffix(local_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f">> file   : {local_path}")
    print(f">> sha256 : {sha}")
    print(f">> meta   : {meta_path}{' (with chat template)' if 'chat_template' in meta else ''}")


if __name__ == "__main__":
    main()
