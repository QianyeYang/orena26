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

from src.adapter import is_multi_select, parse_mc_options
from src.counting import CountMode, CountingQuestion, classify_counting_question

FO_NAMES: tuple[str, ...] = FOType.names()  # canonical classes
PROMPT_STRATEGIES: tuple[str, ...] = ("direct", "bbox-json")

SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a single endoscopic "
    "frame from colorectal surgery. Look carefully at the image and answer the "
    "question. Respond with ONLY the answer in the exact requested format — no "
    "explanation, no extra words."
)

BBOX_COUNTING_SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a single endoscopic "
    "frame from colorectal surgery. For a counting question, localize the "
    "requested visible foreign objects and respond with ONLY valid JSON matching "
    "the requested schema. Do not answer with a bare number or add an explanation; "
    "the caller will derive the count from your bounding boxes."
)

SEGMENT_SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a clip from a "
    "colorectal surgery video. The clip is shown as a sequence of frames in "
    "chronological order, each labelled with its absolute video timestamp "
    "(hh:mm:ss from the start of the video). Timestamps mentioned in the "
    "question refer to that same absolute video time. Look carefully at the "
    "frames and answer the question. Respond with ONLY the answer in the exact "
    "requested format — no explanation, no extra words."
)

PROCEDURE_SYSTEM_PROMPT = (
    "You are an expert surgical vision assistant analysing a colorectal surgery "
    "video from its start up to the current moment. The video is shown as a "
    "sequence of frames in chronological order, each labelled with its absolute "
    "video timestamp (hh:mm:ss from the start of the video). Timestamps "
    "mentioned in the question refer to that same absolute video time. Look "
    "carefully at the frames and answer the question. Respond with ONLY the "
    "answer in the exact requested format — no explanation, no extra words."
)


def system_prompt(track: str, prompt_strategy: str = "direct") -> str:
    """Return the per-track and per-strategy system prompt."""
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"unknown prompt strategy: {prompt_strategy}")
    if track == "frame" and prompt_strategy == "bbox-json":
        return BBOX_COUNTING_SYSTEM_PROMPT
    if track == "segment":
        return SEGMENT_SYSTEM_PROMPT
    if track == "procedure":
        return PROCEDURE_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def build_bbox_counting_instruction(
    question: str,
    count_question: CountingQuestion,
) -> str:
    """Build the one-shot localize-then-count instruction for one FRAME row."""

    if count_question.mode is CountMode.INSTANCES:
        selection = (
            "Include one object entry for every visible foreign-object instance, "
            "regardless of class."
        )
        reduction = "The caller will count all valid boxes."
    elif count_question.mode is CountMode.CLASSES:
        selection = (
            "Include one object entry for every visible foreign-object instance "
            "and label each one with its canonical class."
        )
        reduction = "The caller will count the distinct labels attached to valid boxes."
    else:
        selection = (
            f'Include only visible instances of the class "{count_question.target}" '
            "and omit objects from every other class."
        )
        reduction = (
            f'The caller will count valid boxes labeled "{count_question.target}".'
        )

    names = ", ".join(FO_NAMES)
    return (
        f"{question}\n\n"
        "Solve this by localizing objects before counting. Return exactly this "
        "JSON schema and nothing else:\n"
        '{"objects":[{"label":"<class>","bbox_2d":'
        "[x_min,y_min,x_max,y_max]}]}\n\n"
        "Coordinates must be integers normalized to 0-1000, ordered as "
        "[x_min,y_min,x_max,y_max]. Use only these canonical labels: "
        f"{names}.\n{selection}\n"
        'If no requested object is visible, return {"objects":[]}.\n\n'
        "One-shot format example:\n"
        "Question: How many Clips appear in this frame?\n"
        "Assistant: "
        '{"objects":[{"label":"Clip","bbox_2d":[88,140,166,230]},'
        '{"label":"Clip","bbox_2d":[612,355,701,438]}]}\n\n'
        f"{reduction} Do not include a count field or a final numeric answer."
    )


def build_instruction(
    question: str,
    answer_format: str,
    prompt_strategy: str = "direct",
) -> tuple[str, list[str] | None]:
    """Return ``(instruction_text, mc_options)`` for a question + its answer_format.

    ``mc_options`` is the parsed choice list for multiple_choice (else ``None``);
    the runner forwards it to ``adapter.build_response`` to snap the output to a
    valid option. ``bbox-json`` changes only recognized FRAME counting questions.
    """
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"unknown prompt strategy: {prompt_strategy}")

    options: list[str] | None = None
    count_question = classify_counting_question(question, answer_format)
    if prompt_strategy == "bbox-json" and count_question is not None:
        return build_bbox_counting_instruction(question, count_question), options

    if answer_format == "binary":
        instr = f"{question}\n\nAnswer with exactly one word: yes or no."
    elif answer_format == "number":
        instr = f"{question}\n\nAnswer with a single non-negative integer (digits only)."
    elif answer_format == "fo_class":
        names = ", ".join(FO_NAMES)
        instr = (
            f"{question}\n\nAnswer with one or more of these foreign-object names: "
            f"{names}. Separate multiple names with a comma and a space. If none "
            "is visible, answer: none. Output only the name(s)."
        )
    elif answer_format == "multiple_choice":
        options = parse_mc_options(question)
        if is_multi_select(question):  # window tracks only; frame questions never match
            instr = (
                f"{question}\n\nAnswer by copying the correct option(s) from the "
                f"list, verbatim; separate multiple options with a comma and a "
                f"space. If none apply, answer: none. Output nothing else."
            )
        else:
            instr = (
                f"{question}\n\nAnswer by copying EXACTLY ONE of the listed options, "
                f"verbatim, and nothing else."
            )
    elif answer_format == "percentage":
        instr = f"{question}\n\nAnswer with a single number (a percentage)."
    elif answer_format == "time":
        instr = (
            f"{question}\n\nAnswer with one or more timestamps as hh:mm:ss. "
            "Separate multiple timestamps with a comma and a space."
        )
    else:  # open_ended, matching, unknown
        instr = f"{question}\n\nAnswer concisely, in a few words."

    return instr, options
