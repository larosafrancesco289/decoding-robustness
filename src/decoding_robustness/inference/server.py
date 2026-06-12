"""llama-server process lifecycle (SPEC §7): launch, wait for health, shut down.

One (model, quant) pair is one server load. The runner loops conditions (sampler,
temperature) as request params against a single running server, which is the throughput
lever (SPEC §8) — so launching/stopping the server is deliberately separate from issuing
completions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config.schema import ServerConfig
from .client import LlamaServerClient


def resolve_binary(binary: str | None = None) -> str:
    """Locate the llama-server executable (arg → $LLAMA_SERVER_BIN → PATH)."""
    candidate = binary or os.environ.get("LLAMA_SERVER_BIN") or "llama-server"
    found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
    if found is None:
        raise FileNotFoundError(
            f"llama-server not found ('{candidate}'). Build it via scripts/build_llamacpp.sh "
            "and pass --binary or set $LLAMA_SERVER_BIN."
        )
    return found


def build_command(binary: str, model_path: str | Path, config: ServerConfig) -> list[str]:
    """Assemble the llama-server launch command from a ServerConfig."""
    cmd = [
        binary,
        "--model",
        str(model_path),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--ctx-size",
        str(config.n_ctx),
        "--n-gpu-layers",
        str(config.n_gpu_layers),
        "--parallel",
        str(config.parallel),
    ]
    if config.disable_server_add_bos:
        # Stop the server prepending its own BOS; the client-rendered template owns it.
        cmd.extend(["--override-kv", "tokenizer.ggml.add_bos_token=bool:false"])
    cmd.extend(config.extra_args)
    return cmd


@contextmanager
def llama_server(
    model_path: str | Path,
    config: ServerConfig,
    *,
    binary: str | None = None,
    startup_timeout: float = 300.0,
    log_file: str | Path | None = None,
) -> Iterator[LlamaServerClient]:
    """Launch llama-server for ``model_path``, yield a ready client, stop it on exit.

    Raises TimeoutError if the server does not become healthy within ``startup_timeout``,
    and RuntimeError if the process exits early (e.g. model file / VRAM problem).
    """
    binary = resolve_binary(binary)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")

    cmd = build_command(binary, model_path, config)
    base_url = f"http://{config.host}:{config.port}"

    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None  # noqa: SIM115
    stdout = log_handle if log_handle else None
    process = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT if stdout else None)

    client = LlamaServerClient(base_url)
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {process.returncode}) before becoming "
                    f"healthy; command: {' '.join(cmd)}"
                )
            if client.health():
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"llama-server not healthy after {startup_timeout:.0f}s at {base_url}"
                )
            time.sleep(0.5)

        yield client
    finally:
        client.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log_handle:
            log_handle.close()
