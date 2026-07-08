"""Client-side chat-prompt rendering.

Each model's *official* chat template is rendered here, on the client, with the same
Jinja2 engine HuggingFace uses (`apply_chat_template`), so the rendered string matches
what the model was trained to expect. We render client-side (rather than letting the
server apply the template) so the exact prompt string exists before the request and can
be hashed and logged per generation.

The template itself is the one embedded in the GGUF (extracted by scripts/fetch_models.py
into a ``<model>.meta.json`` sidecar). Carrying the model's own template avoids hand-
transcribing a different one per family and keeps the rendered prompt faithful.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

# A chat message is the usual {"role": ..., "content": ...} mapping.
Message = dict[str, str]


def _raise_exception(message: str) -> None:
    """`raise_exception(...)` helper that chat templates call to reject bad input."""
    raise TemplateError(message)


def _build_env() -> ImmutableSandboxedEnvironment:
    # trim_blocks/lstrip_blocks mirror transformers' settings so whitespace matches.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


@dataclass(frozen=True)
class ChatTemplate:
    """A model's chat template plus the special tokens it references."""

    template: str
    bos_token: str = ""
    eos_token: str = ""

    @classmethod
    def from_meta_file(cls, path: str | Path) -> ChatTemplate:
        """Load a template from a ``*.meta.json`` sidecar written by fetch_models.py."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        template = data.get("chat_template")
        if not template:
            raise ValueError(f"{path}: no 'chat_template' in metadata sidecar")
        return cls(
            template=template,
            bos_token=data.get("bos_token", ""),
            eos_token=data.get("eos_token", ""),
        )

    def render(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool = True,
        **extra: object,
    ) -> str:
        """Render ``messages`` to the exact prompt string fed to /completion."""
        env = _build_env()
        compiled = env.from_string(self.template)
        return compiled.render(
            messages=messages,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=add_generation_prompt,
            **extra,
        )


def prompt_sha256(prompt: str) -> str:
    """Stable hash of a rendered prompt, logged per generation."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
