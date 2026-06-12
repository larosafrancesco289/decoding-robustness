"""Tests for client-side chat-template rendering and prompt hashing."""

from __future__ import annotations

import json

from decoding_robustness.inference.templating import ChatTemplate, prompt_sha256

# A minimal Llama-3-style template: exercises bos_token injection, per-role headers,
# and the add_generation_prompt trailing assistant header.
LLAMA_LIKE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    " + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)


def _template() -> ChatTemplate:
    return ChatTemplate(template=LLAMA_LIKE, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")


def test_render_includes_special_tokens_and_content():
    out = _template().render([{"role": "user", "content": "Hello"}])
    assert out.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>user<|end_header_id|>" in out
    assert "Hello<|eot_id|>" in out


def test_add_generation_prompt_appends_assistant_header():
    msgs = [{"role": "user", "content": "Hi"}]
    with_prompt = _template().render(msgs, add_generation_prompt=True)
    without = _template().render(msgs, add_generation_prompt=False)
    assert with_prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert not without.endswith("assistant<|end_header_id|>\n\n")


def test_render_is_deterministic():
    msgs = [{"role": "user", "content": "same"}]
    assert _template().render(msgs) == _template().render(msgs)


def test_prompt_hash_is_stable_and_sensitive():
    a = prompt_sha256("abc")
    assert a == prompt_sha256("abc")
    assert a != prompt_sha256("abd")
    assert len(a) == 64


def test_from_meta_file_roundtrip(tmp_path):
    meta = {
        "chat_template": LLAMA_LIKE,
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|eot_id|>",
    }
    path = tmp_path / "model.gguf.meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    template = ChatTemplate.from_meta_file(path)
    assert template.bos_token == "<|begin_of_text|>"
    out = template.render([{"role": "user", "content": "Hello"}])
    assert "Hello<|eot_id|>" in out
