"""Tests for the M2 task layer: numeric parsing, grading, and the GSM8K gold extractor.

All offline — no dataset download, no server. The dataset *loader* itself is exercised
by the M2 driver (scripts/run_m2.py), which needs the network and a pinned revision.
"""

from __future__ import annotations

import pytest

from decoding_robustness.tasks.grading import grade
from decoding_robustness.tasks.loaders import _format_options, _gsm8k_gold
from decoding_robustness.tasks.parsers import (
    normalize_number,
    parse_gpqa_letter,
    parse_gsm8k_numeric,
    parse_mmlu_pro_letter,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18", "18"),
        ("18.0", "18"),
        ("1,024", "1024"),
        ("$42", "42"),
        ("  -3 ", "-3"),
        ("3.5", "3.5"),
        ("72.", "72"),
        ("not a number", None),
        ("", None),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


def test_parse_strict_final_answer_sentence():
    out = "First 4+5=9, then 9*3=27.\nThe final answer is 27."
    result = parse_gsm8k_numeric(out)
    assert result.answer == "27"
    assert result.method == "strict"


def test_parse_strict_takes_last_final_answer_and_strips_punctuation():
    out = "The final answer is 10. Wait, recompute. The final answer is 27."
    result = parse_gsm8k_numeric(out)
    assert result.answer == "27"
    assert result.method == "strict"


def test_parse_flexible_falls_back_to_last_number():
    out = "Each box has 9 balls, three boxes, so 27 total."
    result = parse_gsm8k_numeric(out)
    assert result.answer == "27"
    assert result.method == "flexible"


def test_parse_handles_thousands_separator_in_final_answer():
    out = "The final answer is 1,024"
    result = parse_gsm8k_numeric(out)
    assert result.answer == "1024"
    assert result.method == "strict"


def test_parse_failure_when_no_number():
    result = parse_gsm8k_numeric("I cannot solve this problem.")
    assert result.answer is None
    assert result.method == "failed"
    assert not result.ok


def test_grade_correct_and_incorrect():
    correct = grade("The final answer is 27.", gold="27", parser_name="gsm8k_numeric")
    assert correct.correct is True
    assert correct.parse_method == "strict"

    wrong = grade("The final answer is 26.", gold="27", parser_name="gsm8k_numeric")
    assert wrong.correct is False
    assert wrong.parsed_answer == "26"


def test_grade_parse_failure_is_not_correct():
    result = grade("no answer here at all", gold="27", parser_name="gsm8k_numeric")
    assert result.correct is False
    assert result.parse_method == "failed"
    assert result.parsed_answer is None


def test_grade_normalises_both_sides():
    # gold "1024" vs model "1,024" must match after normalisation
    assert grade("The final answer is 1,024", gold="1024", parser_name="gsm8k_numeric").correct


def test_gsm8k_gold_extracts_after_marker():
    answer = "Janet sells 16 - 3 - 4 = 9 eggs.\n9 * 2 = 18\n#### 18"
    assert _gsm8k_gold(answer) == "18"


def test_gsm8k_gold_strips_commas():
    assert _gsm8k_gold("... #### 1,024") == "1024"


# --- multiple-choice (MMLU-Pro / GPQA) parsing ---


def test_mc_parse_strict_answer_sentence():
    result = parse_mmlu_pro_letter("Working through it, option C fits.\nThe answer is (C).")
    assert result.answer == "C"
    assert result.method == "strict"


def test_mc_parse_strict_without_parens_and_case_insensitive():
    result = parse_mmlu_pro_letter("the answer is d")
    assert result.answer == "D"
    assert result.method == "strict"


def test_mc_parse_takes_last_answer_sentence():
    out = "The answer is (A). Wait, reconsider. The answer is (D)."
    assert parse_mmlu_pro_letter(out).answer == "D"


def test_mc_parse_flexible_falls_back_to_last_paren_letter():
    result = parse_mmlu_pro_letter("Between the choices I'll go with (B).")
    assert result.answer == "B"
    assert result.method == "flexible"


def test_mc_parse_failure_when_no_letter():
    result = parse_mmlu_pro_letter("I am not sure which option is correct.")
    assert result.answer is None
    assert result.method == "failed"
    assert not result.ok


def test_gpqa_ignores_out_of_range_letter_but_mmlu_pro_accepts_it():
    # 'G' is valid for MMLU-Pro (A–J) but not GPQA (A–D).
    out = "The answer is (G)."
    assert parse_mmlu_pro_letter(out).answer == "G"
    assert parse_gpqa_letter(out).answer is None


def test_gpqa_skips_out_of_range_to_find_valid_earlier_letter():
    # A late out-of-range letter must not shadow an in-range strict answer.
    out = "The answer is (C). Some models would say the answer is (Z)."
    assert parse_gpqa_letter(out).answer == "C"


def test_mc_grade_integration():
    assert grade("The answer is (C).", gold="C", parser_name="mmlu_pro_letter").correct
    assert not grade("The answer is (A).", gold="D", parser_name="gpqa_letter").correct


def test_format_options_labels_in_order():
    assert _format_options(["foo", "bar", "baz"]) == "A. foo\nB. bar\nC. baz"
