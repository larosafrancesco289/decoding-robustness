"""Task layer: dataset loaders, deterministic answer parsers, and graders.

Implements GSM8K (CoT, numeric exact-match), MMLU-Pro, and GPQA (letter exact-match).
"""

from __future__ import annotations

from .grading import GradeResult, grade
from .loaders import TaskItem, load_task
from .parsers import (
    ParseResult,
    get_parser,
    normalize_number,
    parse_gpqa_letter,
    parse_gsm8k_numeric,
    parse_mmlu_pro_letter,
)

__all__ = [
    "GradeResult",
    "ParseResult",
    "TaskItem",
    "get_parser",
    "grade",
    "load_task",
    "normalize_number",
    "parse_gpqa_letter",
    "parse_gsm8k_numeric",
    "parse_mmlu_pro_letter",
]
