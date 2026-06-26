"""Output adapter: constrain raw VLM text into a string the focus answer-format
parser accepts, so ``focus.Evaluator`` scores it correctly.

The Evaluator calls ``fmt.read(Response.content)`` on the RAW content, and the
format classes are strict:
- ``binary``          -> exactly ``"yes"`` / ``"no"``
- ``number``          -> a bare non-negative integer
- ``fo_class``        -> ONE canonical FO name (``FOType.names()``, case-insensitive) or ``"none"``
- ``percentage``      -> a bare number
- ``time``            -> ``hh:mm:ss``
- ``open_ended`` / ``multiple_choice`` / ``matching`` -> judged by an LLM, but
  ``read`` still verifies ``len <= max_length`` (300) first, so we strip + truncate.

Unparseable content is simply scored wrong (the Evaluator catches ``ValueError``),
so normalisers fall back to cleaned text rather than raising.

NOTE: ``fo_class`` is single-label by design here — multi-label ground truth
(e.g. ``"Clip, Sponge"``) is unscorable by the stock format (see
``docs/issues-tbd.md`` item 4).
"""

from __future__ import annotations

import re

from focus import FOType, Response, get_format_class

MAX_TEXT_LEN = 300  # OpenEnded / MultipleChoice / Matching default max_length

_FO_NAMES = FOType.names()  # 9 canonical names
_NONE_CUES = (
    "none", "no foreign object", "no object", "nothing",
    "absent", "not present", "n/a",
)


def _clean(text) -> str:
    return " ".join(str(text).split())


def normalize_binary(text) -> str:
    low = _clean(text).lower()
    if low in ("yes", "no"):
        return low
    if re.search(r"\b(yes|yeah|yep|true|correct|present|affirmative)\b", low) or low.startswith("yes"):
        return "yes"
    if re.search(r"\b(no|nope|false|incorrect|absent|negative)\b", low) or low.startswith("no"):
        return "no"
    return low  # unparseable -> scored wrong


def normalize_number(text) -> str:
    m = re.search(r"\d+", str(text))
    return m.group(0) if m else _clean(text)


def normalize_fo_class(text) -> str:
    low = _clean(text).lower()
    if not low:
        return "none"
    # prefer a concrete class; on overlap (Specimen vs Specimen Bag) the longer wins,
    # otherwise the earliest-mentioned class wins.
    best: tuple[tuple[int, int], str] | None = None
    for n in _FO_NAMES:
        i = low.find(n.lower())
        if i != -1:
            key = (i, -len(n))
            if best is None or key < best[0]:
                best = (key, n)
    if best is not None:
        return best[1]
    if any(re.search(rf"(^|\W){re.escape(c)}(\W|$)", low) for c in _NONE_CUES):
        return "none"
    return _clean(text)


def parse_mc_options(question: str) -> list[str]:
    """Extract the choices from an MC question (``"... select one answer: a; b; c"``)."""
    m = re.search(r"select one answer[:\s]*(.+)$", question, flags=re.IGNORECASE | re.DOTALL)
    tail = m.group(1) if m else question
    return [o.strip(" .") for o in re.split(r"[;\n]", tail) if o.strip(" .")]


def normalize_text(text, options: list[str] | None = None, max_len: int = MAX_TEXT_LEN) -> str:
    """Strip/collapse whitespace and truncate; if ``options`` given, snap to a match."""
    t = _clean(text)
    if options:
        low = t.lower()
        for opt in sorted(options, key=len, reverse=True):
            if opt.lower() in low:
                return opt[:max_len]
    return t[:max_len]


def normalize_percentage(text) -> str:
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return m.group(0) if m else _clean(text)


def normalize_time(text) -> str:
    m = re.search(r"(\d{1,2}):([0-5]?\d):([0-5]?\d)", str(text))
    if m:
        h, mn, s = (int(x) for x in m.groups())
        return f"{h:02d}:{mn:02d}:{s:02d}"
    return _clean(text)


def normalize_answer(answer_format: str, text, *, options: list[str] | None = None) -> str:
    """Dispatch raw model text to the normaliser for ``answer_format``."""
    if answer_format == "binary":
        return normalize_binary(text)
    if answer_format == "number":
        return normalize_number(text)
    if answer_format == "fo_class":
        return normalize_fo_class(text)
    if answer_format == "percentage":
        return normalize_percentage(text)
    if answer_format == "time":
        return normalize_time(text)
    if answer_format == "multiple_choice":
        return normalize_text(text, options=options)
    return normalize_text(text)  # open_ended, matching, unknown


def build_response(qID: str, raw_text, answer_format: str, *, latency: float = 0.0,
                   options: list[str] | None = None) -> Response:
    """Wrap normalised model output as a focus ``Response`` ready for evaluation."""
    return Response(
        qID=qID,
        content=normalize_answer(answer_format, raw_text, options=options),
        latency=latency,
    )


def is_parseable(answer_format: str, content: str, format_kwargs: dict | None = None) -> bool:
    """Whether ``content`` would survive ``fmt.read`` (debug/parse-rate helper)."""
    try:
        get_format_class(answer_format)(**(format_kwargs or {})).read(content)
        return True
    except Exception:
        return False
