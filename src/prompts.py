"""Format-aware prompt construction (shared across FRAME methods).

The model must emit answers the strict ``focus`` formats can parse, so each
answer_format gets a tailored instruction that pins the output shape. Runners
pair this with :mod:`src.adapter` (post-hoc normalisation) for defence in depth:
prompt for terseness, then sanitise.

Moved here from ``track-frame/baseline/src/`` so the baseline and the
zero-shot sweep share one prompt strategy (single source of truth).
"""

from __future__ import annotations

import re

from focus import FOType

from src.adapter import is_multi_select, parse_mc_options
from src.counting import CountMode, CountingQuestion, classify_counting_question

FO_NAMES: tuple[str, ...] = FOType.names()  # canonical classes
PROMPT_STRATEGIES: tuple[str, ...] = (
    "direct",
    "bbox-json",
    "v2",
    "v2-noscope",
    "v2-nodesc",
    "v3",
    "v3-desc",
)

#: Strategies sharing :func:`build_v2_instruction` and the procedure-aware
#: system prompt, differing only in which system-prompt blocks they carry.
V2_FAMILY: tuple[str, ...] = ("v2", "v2-noscope", "v2-nodesc", "v3", "v3-desc")

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


# ── v2 strategy ──────────────────────────────────────────────────────
#
# Three defects in the "direct" prompts, each measured against the official
# annotations (see docs/frame-track-v2-plan.md):
#
# 1. The system prompt asserts "colorectal surgery" for every row. That is
#    false for all 8,000 LapChole rows (Laparoscopic Cholecystectomy) and for
#    whatever procedure the hidden out-of-distribution split holds. v2 takes
#    the row's own ``procedure_type``.
# 2. ``fo_class`` always says "one or more … separate with a comma". 4,365 of
#    the 8,969 official FRAME fo_class rows (48.7%) are single-answer by
#    construction, which the question states as "Please provide a class name"
#    versus "the class names". v2 pins the arity from that phrasing — a rule
#    with zero violations over all 16,190 fo_class rows in all three tracks.
# 3. ``open_ended`` appends "Answer concisely, in a few words" even to the 646
#    positional-enumeration rows whose question already mandates a numbered
#    multi-item list ("1. Sponge: top/left 2. Needle: bottom/left"). v2 leaves
#    self-describing questions alone.
#
# v2 also names every canonical class the challenge defines, including the two
# (Mesh, Absorbable Hemostatic Agent) that never occur in either training set,
# so a fine-tune cannot silently drop them from its output vocabulary.

_PROCEDURE_FALLBACK = "minimally invasive"

#: Compact visual descriptors, faithful to the challenge's FO_definitions.txt
#: (the same text the official LLM judge is shown). Kept to one line per class
#: so it can ride in every system prompt without crowding out the image.
FO_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("Sponge", "soft absorbent pad, white when fresh, reddish-brown when blood-soaked"),
    ("Clip", "small metal or polymer vessel/duct clip; counts only once placed, not while still in the applier"),
    ("Specimen Bag", "sterile retrieval pouch; the pouch only, not its string"),
    ("Silicone Loop", "soft flexible band, usually white, encircling a vessel for traction"),
    ("External Drain", "clear or fluid-filled tube evacuating fluid out of the body"),
    ("Needle", "sharp straight or curved suture needle; suture thread alone does not count"),
    ("Gallstone", "roundish white/yellow calcified concretion from the gallbladder"),
    ("Specimen", "excised tissue or organ, fully detached, awaiting retrieval; not fat or blood"),
    ("Mesh", "screen-like implantable patch reinforcing a muscle wall; more open weave than a sponge"),
    ("Absorbable Hemostatic Agent", "resorbable white or pale-yellow frizzy mesh applied to a bleeding surface"),
)

V2_FO_SCOPE = (
    "A foreign object is an object fully introduced into the body cavity that "
    "must be retrieved or accounted for. Instruments that stay connected to the "
    "outside (graspers, scissors, trocars, staplers, the camera) are not foreign "
    "objects, and neither are detachable stapler anvils."
)

#: v3 keeps the definition but drops the exclusion list. Measured on the 4B,
#: zero-shot, 6,252 rows: under V2_FO_SCOPE the model answered "yes" to 2.1% of
#: binary questions (v1: 24.3%, truth: 44.1%) and answered 0 to 67.0% of
#: counting questions — where the FRAME ground truth is *never* 0. Telling a
#: model what does not count reads to it as licence to say nothing counts.
V3_FO_SCOPE = (
    "A foreign object is an object introduced into the body cavity that must be "
    "retrieved or accounted for, such as a sponge, clip, needle or drain. "
    "Several may be present at once."
)

#: Which system-prompt blocks each v2-family strategy carries. The instruction
#: text is identical across the family, so an A/B over these isolates the
#: system prompt.
_SYSTEM_VARIANTS: dict[str, tuple[str | None, bool]] = {
    # strategy: (scope paragraph or None, include per-class descriptions)
    "v2": (V2_FO_SCOPE, True),
    "v2-noscope": (None, True),
    "v2-nodesc": (V2_FO_SCOPE, False),
    "v3": (V3_FO_SCOPE, False),
    "v3-desc": (V3_FO_SCOPE, True),
}


def _v2_class_reference(include_descriptions: bool = True) -> str:
    """Return the canonical class roster, optionally with visual descriptors."""
    if not include_descriptions:
        return "The canonical classes are: " + ", ".join(FO_NAMES) + "."
    lines = "\n".join(f"- {name}: {desc}" for name, desc in FO_DESCRIPTIONS)
    return f"The canonical classes are exactly these ten:\n{lines}"


def v2_system_prompt(
    track: str,
    procedure_type: str | None = None,
    *,
    include_descriptions: bool | None = None,
    variant: str = "v2",
) -> str:
    """Procedure-aware system prompt naming every canonical class.

    ``procedure_type`` comes from the row (``Proctocolectomy``, ``Rectal
    Resection``, ``Sigmoid Resection``, ``Laparoscopic Cholecystectomy``, or
    anything the hidden split holds); it falls back to a neutral phrasing
    rather than to a guess.

    ``variant`` selects which system-prompt blocks to carry (see
    :data:`_SYSTEM_VARIANTS`); ``include_descriptions`` overrides the variant's
    own choice, and defaults to it when left unset.
    """
    scope, variant_descriptions = _SYSTEM_VARIANTS.get(variant, _SYSTEM_VARIANTS["v2"])
    if include_descriptions is None:
        include_descriptions = variant_descriptions
    procedure = (procedure_type or "").strip() or _PROCEDURE_FALLBACK
    if track == "frame":
        visual = f"a single endoscopic frame from a {procedure} procedure"
        temporal = ""
    else:
        scope = "clip" if track == "segment" else "video, from its start up to the current moment"
        visual = f"an endoscopic {scope} from a {procedure} procedure"
        temporal = (
            " The frames are in chronological order, each labelled with its "
            "absolute video timestamp (hh:mm:ss from the start of the video); "
            "timestamps in the question refer to that same absolute video time."
        )
    scope_block = f"{scope}\n\n" if scope else ""
    return (
        f"You are an expert surgical vision assistant analysing {visual}.{temporal}\n\n"
        f"{scope_block}"
        f"{_v2_class_reference(include_descriptions)}\n\n"
        "Look carefully, then answer in exactly the format the question asks "
        "for. Give the answer only — no explanation, no restating the question."
    )


#: Official questions state fo_class arity in their own wording. Validated over
#: every fo_class row in all three tracks: "single" never has a multi-class
#: answer (7,715/7,715), and the residual "unknown" templates are single too.
_FO_SINGLE_CUE = re.compile(r"provide a class name|a single class|one class name", re.I)
_FO_MULTI_CUE = re.compile(
    r"class names|class name\(s\)|classes are|objects are|all foreign|which combination", re.I
)

#: Questions that carry their own output contract ("Please provide the answer in
#: the following format: …"). Appending a generic terseness rule to these
#: contradicts them, so v2 stays out of the way.
_SELF_FORMATTING = re.compile(
    r"in the following format|please provide the answer in|for example:", re.I
)


def fo_class_is_multi(question: str) -> bool:
    """True when an fo_class question admits more than one class in its answer."""
    if _FO_SINGLE_CUE.search(question):
        return False
    return bool(_FO_MULTI_CUE.search(question))


def build_v2_instruction(
    question: str,
    answer_format: str,
) -> tuple[str, list[str] | None]:
    """v2 instruction: arity- and self-format-aware, otherwise as strict as v1.

    ``binary``/``number``/``fo_class``/``percentage``/``time`` are parsed by
    exact match, so they stay terse. ``open_ended``/``multiple_choice`` go to an
    LLM judge that is told to accept a correct answer even with extraneous text
    — but ``read`` still rejects anything over 300 characters first, so the
    length guard is stated rather than left to chance.
    """
    options: list[str] | None = None
    if answer_format == "binary":
        return f"{question}\n\nAnswer with exactly one word: yes or no.", options
    if answer_format == "number":
        return (
            f"{question}\n\nCount carefully, then answer with a single "
            "non-negative integer (digits only, no words).",
            options,
        )
    if answer_format == "fo_class":
        names = ", ".join(FO_NAMES)
        if fo_class_is_multi(question):
            instr = (
                f"{question}\n\nName every class that is visible, using these "
                f"names: {names}. Separate multiple names with a comma and a "
                "space, and name each class at most once. If none is visible, "
                "answer: none. Output only the name(s)."
            )
        else:
            instr = (
                f"{question}\n\nAnswer with exactly ONE of these names: {names}. "
                "If none is visible, answer: none. Output only the name."
            )
        return instr, options
    if answer_format == "multiple_choice":
        options = parse_mc_options(question)
        if is_multi_select(question):
            instr = (
                f"{question}\n\nAnswer by copying the correct option(s) from the "
                "list, verbatim; separate multiple options with a comma and a "
                "space. If none apply, answer: none. Output nothing else."
            )
        else:
            instr = (
                f"{question}\n\nAnswer by copying EXACTLY ONE of the listed "
                "options, verbatim, and nothing else."
            )
        return instr, options
    if answer_format == "percentage":
        return f"{question}\n\nAnswer with a single number (a percentage).", options
    if answer_format == "time":
        return (
            f"{question}\n\nAnswer with one or more timestamps as hh:mm:ss. "
            "Separate multiple timestamps with a comma and a space.",
            options,
        )
    # open_ended / matching / unknown
    if _SELF_FORMATTING.search(question):
        # The question already specifies its output contract; only guard length.
        return f"{question}\n\nFollow that format exactly, in under 300 characters.", options
    return f"{question}\n\nAnswer concisely, in a few words (under 300 characters).", options


def system_prompt(
    track: str,
    prompt_strategy: str = "direct",
    *,
    procedure_type: str | None = None,
) -> str:
    """Return the per-track and per-strategy system prompt.

    ``procedure_type`` is used only by the v2-family strategies; the other
    strategies ignore it, so existing callers are unaffected.
    """
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(f"unknown prompt strategy: {prompt_strategy}")
    if prompt_strategy in V2_FAMILY:
        return v2_system_prompt(track, procedure_type, variant=prompt_strategy)
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
    if prompt_strategy in V2_FAMILY:
        return build_v2_instruction(question, answer_format)

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
