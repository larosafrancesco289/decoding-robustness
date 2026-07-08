"""Deterministic grading: exact normalised match, no LLM judge.

Grading is split from parsing on purpose: the parser decides *what the model answered*
(and whether it answered parseably), the grader decides *whether that answer is correct*.
A parse failure is never silently counted as wrong-with-an-answer; ``correct`` is False
and the parse path is logged so the failure rate can be reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parsers import ParseResult, get_parser


@dataclass(frozen=True)
class GradeResult:
    """The graded outcome for one completion."""

    parsed_answer: str | None
    parse_method: str  # "strict" | "flexible" | "failed"
    gold: str
    correct: bool


def grade(output: str, gold: str, parser_name: str) -> GradeResult:
    """Parse ``output`` with the named parser and compare to ``gold`` by exact match.

    Numeric tasks have both sides already normalised (loaders normalise the gold,
    parsers normalise the prediction), so equality is a plain string compare.
    """
    result: ParseResult = get_parser(parser_name)(output)
    correct = result.answer is not None and result.answer == gold
    return GradeResult(
        parsed_answer=result.answer,
        parse_method=result.method,
        gold=gold,
        correct=correct,
    )
