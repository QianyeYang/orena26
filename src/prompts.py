"""Format-aware prompt construction (shared across FRAME methods).

The model must emit answers the strict ``focus`` formats can parse, so each
answer_format gets a tailored instruction that pins the output shape. Runners
pair this with :mod:`src.adapter` (post-hoc normalisation) for defence in depth:
prompt for terseness, then sanitise.

Moved here from ``track-frame/baseline/src/`` so the baseline and the
zero-shot sweep share one prompt strategy (single source of truth).
"""

from __future__ import annotations

from focus import FOType

from src.adapter import parse_mc_options

FO_NAMES: tuple[str, ...] = FOType.names()  # 9 canonical classes

SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a single endoscopic "
    "frame from colorectal surgery. Look carefully at the image and answer the "
    "question. Respond with ONLY the answer in the exact requested format — no "
    "explanation, no extra words."
)


def build_instruction(question: str, answer_format: str) -> tuple[str, list[str] | None]:
    """Return ``(instruction_text, mc_options)`` for a question + its answer_format.

    ``mc_options`` is the parsed choice list for multiple_choice (else ``None``);
    the runner forwards it to ``adapter.build_response`` to snap the output to a
    valid option.
    """
    options: list[str] | None = None

    if answer_format == "binary":
        instr = f"{question}\n\nAnswer with exactly one word: yes or no."
    elif answer_format == "number":
        instr = f"{question}\n\nAnswer with a single non-negative integer (digits only)."
    elif answer_format == "fo_class":
        names = ", ".join(FO_NAMES)
        instr = (
            f"{question}\n\nAnswer with EXACTLY ONE of these foreign-object names: "
            f"{names}. If none is visible, answer: none. Output only the name."
        )
    elif answer_format == "multiple_choice":
        options = parse_mc_options(question)
        instr = (
            f"{question}\n\nAnswer by copying EXACTLY ONE of the listed options, "
            f"verbatim, and nothing else."
        )
    elif answer_format == "percentage":
        instr = f"{question}\n\nAnswer with a single number (a percentage)."
    elif answer_format == "time":
        instr = f"{question}\n\nAnswer with a timestamp as hh:mm:ss."
    else:  # open_ended, matching, unknown
        instr = f"{question}\n\nAnswer concisely, in a few words."

    return instr, options
