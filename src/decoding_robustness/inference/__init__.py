"""Inference layer (M1): llama-server lifecycle, the native /completion client,
client-side chat templating, and sampler-chain + seed plumbing."""

from __future__ import annotations

from .client import CompletionResult, LlamaServerClient
from .sampling import chain_label, completion_params, default_chain
from .server import build_command, llama_server, resolve_binary
from .templating import ChatTemplate, prompt_sha256

__all__ = [
    "ChatTemplate",
    "CompletionResult",
    "LlamaServerClient",
    "build_command",
    "chain_label",
    "completion_params",
    "default_chain",
    "llama_server",
    "prompt_sha256",
    "resolve_binary",
]
