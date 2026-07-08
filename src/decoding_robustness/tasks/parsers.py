"""Deterministic answer parsers.

No LLM judge: every answer is extracted by literature-faithful regexes and compared by
exact normalised match. Each parser returns a ``ParseResult`` that records *how* the
answer was found (strict format vs. lenient fallback vs. failure) so the parse-failure
rate is a logged metric and the pre-registered exclusion rule can be applied later.

The GSM8K parser mirrors lm-eval-harness ``gsm8k-cot-llama``: it first looks for the
pinned "The final answer is ..." sentence, then falls back to the last number in the text.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# A signed number with optional thousands separators / currency / decimals, e.g.
# "1,024", "-3.5", "$42". Kept deliberately permissive; normalize_number tidies it up.
_NUMBER = r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?"

# Primary: the pinned final-answer sentence the GSM8K prompt asks the model to emit.
_FINAL_ANSWER_RE = re.compile(
    r"final answer is\s*:?\s*\(?(" + _NUMBER + r")\)?",
    re.IGNORECASE,
)
# Fallback: every standalone number, so we can take the last one.
_ANY_NUMBER_RE = re.compile(_NUMBER)


@dataclass(frozen=True)
class ParseResult:
    """The outcome of running a parser on one raw completion."""

    answer: str | None  # normalised answer, or None if nothing parseable was found
    method: str  # "strict" | "flexible" | "failed"; logged as the parse-path metric

    @property
    def ok(self) -> bool:
        return self.answer is not None


def normalize_number(text: str) -> str | None:
    """Canonicalise a numeric string for exact match (strip $, commas, trailing '.0').

    Returns None if ``text`` holds no parseable number. Integers render without a decimal
    point so "18", "18.0" and "$18" all normalise to "18".
    """
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return repr(value)


def parse_gsm8k_numeric(output: str) -> ParseResult:
    """Extract the numeric answer from a GSM8K completion.

    Strict path: the last "The final answer is N" sentence. Flexible path: the last number
    anywhere in the output (lm-eval-harness flexible-extract). Failure: no number at all.
    """
    matches = _FINAL_ANSWER_RE.findall(output)
    if matches:
        normalized = normalize_number(matches[-1])
        if normalized is not None:
            return ParseResult(answer=normalized, method="strict")

    numbers = _ANY_NUMBER_RE.findall(output)
    if numbers:
        normalized = normalize_number(numbers[-1])
        if normalized is not None:
            return ParseResult(answer=normalized, method="flexible")

    return ParseResult(answer=None, method="failed")


# Multiple-choice tasks (MMLU-Pro, GPQA) ask the model to end with the pinned
# "The answer is (X)" sentence. The strict path keys on that sentence (mirroring the
# MMLU-Pro / lm-eval-harness extraction); the flexible path falls back to the last
# parenthesised option letter anywhere in the text. Letters outside the task's valid
# range are ignored so a stray capital (or an option label echoed mid-reasoning) doesn't
# get mistaken for the final answer.
_ANSWER_LETTER_RE = re.compile(r"answer\s+is\s*:?\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE)
_PAREN_LETTER_RE = re.compile(r"\(\s*([A-Za-z])\s*\)")


def _parse_mc_letter(output: str, valid: str) -> ParseResult:
    """Extract a multiple-choice letter restricted to ``valid`` (e.g. 'ABCD' or 'ABCDEFGHIJ').

    Strict: the last pinned "answer is (X)" sentence. Flexible: the last parenthesised
    letter '(X)' anywhere. Failure: no in-range letter found by either path.
    """
    valid_set = set(valid)
    for letter in reversed(_ANSWER_LETTER_RE.findall(output)):
        if letter.upper() in valid_set:
            return ParseResult(answer=letter.upper(), method="strict")
    for letter in reversed(_PAREN_LETTER_RE.findall(output)):
        if letter.upper() in valid_set:
            return ParseResult(answer=letter.upper(), method="flexible")
    return ParseResult(answer=None, method="failed")


def parse_mmlu_pro_letter(output: str) -> ParseResult:
    """MMLU-Pro answer letter (up to ten options, A–J)."""
    return _parse_mc_letter(output, "ABCDEFGHIJ")


def parse_gpqa_letter(output: str) -> ParseResult:
    """GPQA answer letter (four options, A–D)."""
    return _parse_mc_letter(output, "ABCD")


# parser name (TaskSpec.parser) -> callable.
_PARSERS: dict[str, Callable[[str], ParseResult]] = {
    "gsm8k_numeric": parse_gsm8k_numeric,
    "mmlu_pro_letter": parse_mmlu_pro_letter,
    "gpqa_letter": parse_gpqa_letter,
}


def get_parser(name: str) -> Callable[[str], ParseResult]:
    """Resolve a parser by its registered name (TaskSpec.parser)."""
    try:
        return _PARSERS[name]
    except KeyError as exc:
        raise KeyError(
            f"no parser registered for '{name}' (known: {', '.join(sorted(_PARSERS))})"
        ) from exc
