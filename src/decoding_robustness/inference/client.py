"""Thin HTTP client for the llama-server native ``/completion`` endpoint (SPEC §7).

We use the native endpoint (not the OpenAI-compatible one) so the sampler chain, seed,
and per-method truncation params map straight onto the request, and the response carries
llama.cpp's token counts and timings for the throughput/length covariates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

# llama-server runs a chat/tool-call PEG parser over the generated text even on the native
# /completion path, and it throws a 500 on degenerate token-salad — exactly what pure
# temperature at high T produces, a result we must capture, not crash on. No server flag
# disables this reliably. But the server embeds the full generated text in the error body
# ("Failed to parse input at pos N: <text>"), so we recover it and return it as content.
_PARSE_ERROR_RE = re.compile(r"Failed to parse input at pos \d+:\s?(.*)", re.S)
# llama.cpp v0.4.0 (tag, 2026-09-04) rejects the same streams with this message and no text:
# its PEG output parser fails on the invalid UTF-8 a cap-length degenerate stream ends in.
_FORMAT_ERROR_RE = re.compile(
    r"The model produced output that does not match the expected .* format"
)


@dataclass
class CompletionResult:
    """One /completion response, with the fields the JSONL record needs (SPEC §7.1)."""

    content: str
    tokens_predicted: int
    tokens_evaluated: int
    stopped: bool
    timings: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    # True when the text was recovered from a server parse-error 500 (degenerate output).
    # Such records have no token counts/timings — identifiable for analysis exclusion.
    parse_error_recovered: bool = False

    @property
    def predicted_per_second(self) -> float | None:
        """Decode throughput (tok/s) as reported by the server, if present."""
        value = self.timings.get("predicted_per_second")
        return float(value) if value is not None else None


def _recover_parse_error(resp: httpx.Response) -> CompletionResult | None:
    """If a 500 is a chat-parse failure, pull the generated text out of the error body.

    Returns None if the response is some other 500 (which the caller then raises on).
    Token counts/timings are unavailable from the error, so they are left at 0 and the
    result is flagged ``parse_error_recovered``.
    """
    try:
        message = resp.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        return None
    match = _PARSE_ERROR_RE.search(message)
    if match is None:
        if _FORMAT_ERROR_RE.search(message) is None:
            return None
        content = ""  # v0.4.0 does not return the text; the record is a zero-token cap hit
    else:
        content = match.group(1)
    return CompletionResult(
        content=content,
        tokens_predicted=0,
        tokens_evaluated=0,
        stopped=True,
        parse_error_recovered=True,
    )


class LlamaServerClient:
    """Minimal client: ``/health``, ``/props``, and ``/completion``."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> bool:
        """True once the server reports it is ready to serve completions."""
        try:
            resp = self._client.get("/health")
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 and resp.json().get("status") == "ok"

    def props(self) -> dict:
        """Server properties (model path, embedded template, build info) for the manifest."""
        resp = self._client.get("/props")
        resp.raise_for_status()
        return resp.json()

    def completion(self, prompt: str, params: dict) -> CompletionResult:
        """Run one completion. ``params`` comes from sampling.completion_params().

        Recovers the generated text from a server parse-error 500 (degenerate output)
        rather than raising, so one pathological condition can't abort a batched run.
        """
        body = {**params, "prompt": prompt, "stream": False}
        resp = self._client.post("/completion", json=body)
        if resp.status_code == 500:
            recovered = _recover_parse_error(resp)
            if recovered is not None:
                return recovered
        resp.raise_for_status()
        data = resp.json()
        return CompletionResult(
            content=data.get("content", ""),
            tokens_predicted=int(data.get("tokens_predicted", 0)),
            tokens_evaluated=int(data.get("tokens_evaluated", 0)),
            stopped=bool(data.get("stop", data.get("stopped_eos", False))),
            timings=data.get("timings", {}) or {},
            raw=data,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaServerClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
