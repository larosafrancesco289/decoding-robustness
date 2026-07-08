"""Task dataset loaders.

Each loader turns a ``TaskSpec`` into a deterministic, ordered list of ``TaskItem``s:
a stable ``item_id``, the user-message ``prompt`` (the content fed into the chat
template, *before* rendering), and the normalised ``gold`` answer the grader compares
against. Datasets are pulled from the HF hub at the spec's ``revision``, and the exact
rendered prompt of every item is hashed and logged per generation record.

Subsetting is deterministic head-of-split (first ``subset_size`` items), or a
deterministic round-robin over categories for the stratified replication. Either way
the same items appear across every condition, which a fixed
revision + fixed slice guarantees.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from datasets import load_dataset

from ..config.schema import TaskSpec
from .parsers import normalize_number

# The Llama-3.1 GSM8K eval prompt (Meta's released evals / lm-eval-harness
# ``gsm8k-cot-llama``). It asks for free-form chain-of-thought and pins the final-answer
# format the parser keys on, which is what keeps the parse-failure rate low.
GSM8K_PROMPT = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] is the '
    "response to the problem."
)


@dataclass(frozen=True)
class TaskItem:
    """One benchmark item: its stable id, the (pre-template) prompt, and the gold answer."""

    item_id: str
    prompt: str
    gold: str
    meta: dict = field(default_factory=dict)


def _gsm8k_gold(answer: str) -> str:
    """Extract the reference number from a GSM8K ``answer`` field ('... #### 18')."""
    marker = answer.rfind("####")
    raw = answer[marker + 4 :] if marker != -1 else answer
    normalized = normalize_number(raw)
    if normalized is None:
        raise ValueError(f"could not parse GSM8K gold answer from: {answer!r}")
    return normalized


def load_gsm8k(task: TaskSpec) -> list[TaskItem]:
    """Load the GSM8K split named in ``task`` into TaskItems (CoT, numeric gold)."""
    dataset = load_dataset(
        task.hf_dataset,
        task.hf_config or "main",
        split=task.split,
        revision=task.revision,
    )
    if task.subset_size is not None:
        dataset = dataset.select(range(min(task.subset_size, len(dataset))))

    items: list[TaskItem] = []
    for index, row in enumerate(dataset):
        items.append(
            TaskItem(
                item_id=f"{task.name}-{task.split}-{index:05d}",
                prompt=GSM8K_PROMPT.format(question=row["question"].strip()),
                gold=_gsm8k_gold(row["answer"]),
                meta={"index": index},
            )
        )
    return items


# A 0-shot multiple-choice CoT prompt shared by MMLU-Pro and GPQA. It mirrors the GSM8K
# setup (free-form reasoning) and pins the same "The answer is (X)" final-answer format
# the MC parser keys on. The official MMLU-Pro eval is 5-shot; we stay 0-shot to match the
# rest of the study and keep the prompt a controlled variable across tasks.
MCQ_PROMPT = (
    "Given the following multiple-choice question, reason step by step and then give a "
    "final answer.\n"
    "Question: {question}\n"
    "Options:\n{options}\n"
    'Your response should end with "The answer is (X)" where X is the letter of the '
    "correct option."
)
# Option labels A..; MMLU-Pro has up to ten options, GPQA exactly four.
_OPTION_LETTERS = "ABCDEFGHIJ"


def _format_options(options: list[str]) -> str:
    """Render an option list as 'A. ...\\nB. ...' for the MCQ prompt."""
    return "\n".join(f"{_OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def _stratified_indices(categories: list[str], size: int) -> list[int]:
    """Deterministic stratified row pick: round-robin over categories in row order.

    Depth d takes the d-th row of every category (alphabetical category order) until
    ``size`` rows are collected. No RNG; the pick is a pure function of the split.
    """
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        by_cat.setdefault(cat, []).append(i)
    picked: list[int] = []
    depth = 0
    while len(picked) < size:
        advanced = False
        for cat in sorted(by_cat):
            rows = by_cat[cat]
            if depth < len(rows):
                picked.append(rows[depth])
                advanced = True
                if len(picked) == size:
                    break
        if not advanced:  # size exceeds the split
            break
        depth += 1
    return sorted(picked)


def load_mmlu_pro(task: TaskSpec) -> list[TaskItem]:
    """Load MMLU-Pro into TaskItems (multiple choice, up to ten options, letter gold).

    The dataset pads short option lists with the sentinel "N/A"; we drop those (they only
    ever trail the real options, so the answer letter is unaffected) and take the released
    ``answer`` letter as gold. ``item_id`` always carries the ORIGINAL row index, so head
    and stratified subsets draw from one shared id space.
    """
    dataset = load_dataset(
        task.hf_dataset,
        task.hf_config,
        split=task.split,
        revision=task.revision,
    )
    if task.subset_size is None:
        indices = list(range(len(dataset)))
    elif task.subset_strategy == "stratified_category":
        indices = _stratified_indices(list(dataset["category"]), task.subset_size)
    else:
        indices = list(range(min(task.subset_size, len(dataset))))

    items: list[TaskItem] = []
    for index in indices:
        row = dataset[index]
        options = [opt for opt in row["options"] if opt != "N/A"]
        gold = row["answer"].strip()
        if gold not in _OPTION_LETTERS[: len(options)]:
            raise ValueError(f"MMLU-Pro gold {gold!r} out of range for {len(options)} options")
        items.append(
            TaskItem(
                item_id=f"{task.name}-{task.split}-{index:05d}",
                prompt=MCQ_PROMPT.format(
                    question=row["question"].strip(), options=_format_options(options)
                ),
                gold=gold,
                meta={"index": index, "category": row.get("category")},
            )
        )
    return items


def load_gpqa_diamond(task: TaskSpec) -> list[TaskItem]:
    """Load GPQA into TaskItems (build a 4-option MCQ; letter gold).

    GPQA ships one correct + three incorrect answers per question, not a pre-built MCQ. We
    assemble the four options and shuffle them with a per-question deterministic RNG (seeded
    from the question text), so the option order, and therefore the gold letter, is stable
    across runs and identical across every decoding condition, yet not tied to row order.
    """
    dataset = load_dataset(
        task.hf_dataset,
        task.hf_config,
        split=task.split,
        revision=task.revision,
    )
    if task.subset_size is not None:
        dataset = dataset.select(range(min(task.subset_size, len(dataset))))

    items: list[TaskItem] = []
    for index, row in enumerate(dataset):
        question = row["Question"].strip()
        # Index 0 is the correct answer; record where it lands after the shuffle.
        choices = [
            row["Correct Answer"].strip(),
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]
        seed = int(hashlib.sha256(f"gpqa|{question}".encode()).hexdigest()[:8], 16)
        order = list(range(len(choices)))
        random.Random(seed).shuffle(order)
        shuffled = [choices[i] for i in order]
        gold = _OPTION_LETTERS[order.index(0)]
        items.append(
            TaskItem(
                item_id=f"{task.name}-{task.split}-{index:05d}",
                prompt=MCQ_PROMPT.format(
                    question=question, options=_format_options(shuffled)
                ),
                gold=gold,
                meta={"index": index},
            )
        )
    return items


# task name -> loader.
_LOADERS: dict[str, Callable[[TaskSpec], list[TaskItem]]] = {
    "gsm8k": load_gsm8k,
    "mmlu_pro": load_mmlu_pro,
    "gpqa_diamond": load_gpqa_diamond,
}


def load_task(task: TaskSpec) -> list[TaskItem]:
    """Dispatch to the loader registered for ``task.name``."""
    try:
        loader = _LOADERS[task.name]
    except KeyError as exc:
        raise KeyError(
            f"no loader registered for task '{task.name}' (known: {', '.join(sorted(_LOADERS))})"
        ) from exc
    return loader(task)
