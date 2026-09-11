"""Tests for the /completion client's parse-error recovery (degenerate-output 500s)."""

from __future__ import annotations

import httpx

from decoding_robustness.inference.client import _recover_parse_error


def _resp(status, json_body):
    return httpx.Response(
        status, json=json_body, request=httpx.Request("POST", "http://x/completion")
    )


def test_recovers_generated_text_from_parse_error_500():
    text = "Let's break down the problem... then it degenerates into Gand thresh foo bar"
    resp = _resp(
        500,
        {
            "error": {
                "code": 500,
                "message": f"Failed to parse input at pos 0: {text}",
                "type": "server_error",
            }
        },
    )
    result = _recover_parse_error(resp)
    assert result is not None
    assert result.content == text
    assert result.parse_error_recovered is True
    assert result.tokens_predicted == 0


def test_returns_none_for_unrelated_500():
    resp = _resp(500, {"error": {"message": "out of memory", "type": "server_error"}})
    assert _recover_parse_error(resp) is None


def test_returns_none_for_non_json_body():
    resp = httpx.Response(
        500, text="<html>bad gateway</html>", request=httpx.Request("POST", "http://x/completion")
    )
    assert _recover_parse_error(resp) is None


def test_recovers_v040_format_error_500_as_zero_token_record():
    # llama.cpp v0.4.0 rejects the same degenerate streams with no text in the body.
    resp = _resp(
        500,
        {
            "error": {
                "code": 500,
                "message": (
                    "The model produced output that does not match the expected "
                    "Content-only format"
                ),
                "type": "server_error",
            }
        },
    )
    result = _recover_parse_error(resp)
    assert result is not None
    assert result.content == ""
    assert result.parse_error_recovered is True
    assert result.tokens_predicted == 0
