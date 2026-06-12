"""Tests for llama-server command assembly (build_command)."""

from __future__ import annotations

from decoding_robustness.config.schema import ServerConfig
from decoding_robustness.inference.server import build_command


def test_build_command_has_core_flags():
    cmd = build_command("llama-server", "models/m.gguf", ServerConfig())
    assert cmd[0] == "llama-server"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "models/m.gguf"
    assert cmd[cmd.index("--ctx-size") + 1] == "4096"
    assert cmd[cmd.index("--parallel") + 1] == "1"


def test_disable_server_add_bos_emits_override_by_default():
    cmd = build_command("llama-server", "m.gguf", ServerConfig())
    assert "--override-kv" in cmd
    assert "tokenizer.ggml.add_bos_token=bool:false" in cmd


def test_add_bos_override_omitted_when_disabled():
    cmd = build_command("llama-server", "m.gguf", ServerConfig(disable_server_add_bos=False))
    assert "tokenizer.ggml.add_bos_token=bool:false" not in cmd


def test_extra_args_appended_last():
    cfg = ServerConfig(extra_args=["--flash-attn", "1"])
    cmd = build_command("llama-server", "m.gguf", cfg)
    assert cmd[-2:] == ["--flash-attn", "1"]
