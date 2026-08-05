"""Output adapter: constrain raw VLM text into a string the focus answer-format
parser accepts, so ``focus.Evaluator`` scores it correctly.

The Evaluator calls ``fmt.read(Response.content)`` on the RAW content, and the
format classes are strict:
- ``binary``          -> exactly ``"yes"`` / ``"no"``
- ``number``          -> a bare non-negative integer
- ``fo_class``        -> one or more comma-separated canonical FO names
  (case- and order-insensitive) or ``"none"``
- ``percentage``      -> a bare number
- ``time``            -> one or more comma-separated ``hh:mm:ss`` timestamps
- ``open_ended`` / ``multiple_choice`` / ``matching`` -> judged by an LLM, but
  ``read`` still verifies ``len <= max_length`` (300) first, so we strip + truncate.

Unparseable content is simply scored wrong (the Evaluator catches ``ValueError``),
so normalisers fall back to cleaned text rather than raising.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from focus import FOType, Response, get_format_class

MAX_TEXT_LEN = 300  # OpenEnded / MultipleChoice / Matching default max_length

_FO_NAMES = FOType.names()
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


def normalize_fo_class(
    text,
    *,
    fo_names: Sequence[str] | None = None,
) -> str:
    """Return every recognized FO class as a parser-compatible set string.

    The official ``FOClass`` format accepts comma-separated class names and
    compares them as an order-insensitive set. Match longer names first and
    mask them before looking for shorter names so ``"Specimen Bag"`` does not
    also become ``"Specimen"``. ``fo_names`` lets submission runtimes pass the
    definitions supplied with the evaluation batch.
    """
    low = _clean(text).lower()
    if not low:
        return "none"

    names = tuple(dict.fromkeys(_FO_NAMES if fo_names is None else fo_names))
    matched: set[str] = set()
    masked = low
    for name in sorted(names, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        if re.search(pattern, masked):
            matched.add(name)
            masked = re.sub(pattern, " ", masked)
    if matched:
        return ", ".join(name for name in names if name in matched)

    if any(re.search(rf"(^|\W){re.escape(c)}(\W|$)", low) for c in _NONE_CUES):
        return "none"
    return _clean(text)


# matches both the single-select ("select one answer:") and the SEGMENT/PROCEDURE
# multi-select ("select [none, ]one or multiple answers:") question tails
_MC_TAIL_RE = re.compile(
    r"select (?:one answer|(?:none, )?one or multiple answers?)[:\s]*(.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_mc_options(question: str) -> list[str]:
    """Extract the choices from an MC question (``"... select one answer: a; b; c"``)."""
    m = _MC_TAIL_RE.search(question)
    tail = m.group(1) if m else question
    return [o.strip(" .") for o in re.split(r"[;\n]", tail) if o.strip(" .")]


def is_multi_select(question: str) -> bool:
    """Whether an MC question allows several answers (window-track variant)."""
    return bool(
        re.search(r"select (?:none, )?one or multiple answers?", question, flags=re.IGNORECASE)
    )


def normalize_text(text, options: list[str] | None = None, max_len: int = MAX_TEXT_LEN,
                   *, multi: bool = False) -> str:
    """Strip/collapse whitespace and truncate; if ``options`` given, snap to a match.

    ``multi=True`` (multi-select MC): return every matched option, comma+space
    joined in listed order — the ground-truth convention. Longest options are
    matched first and masked out so e.g. ``"top/right"`` can't also count as a
    hit for a shorter overlapping option.
    """
    t = _clean(text)
    if options:
        low = t.lower()
        if multi:
            matched: set[str] = set()
            masked = low
            for opt in sorted(options, key=len, reverse=True):
                if opt.lower() in masked:
                    matched.add(opt)
                    masked = masked.replace(opt.lower(), " ")
            if matched:
                return ", ".join(o for o in options if o in matched)[:max_len]
            if re.search(r"(^|\W)none(\W|$)", low):
                return "none"
            return t[:max_len]
        for opt in sorted(options, key=len, reverse=True):
            if opt.lower() in low:
                return opt[:max_len]
    return t[:max_len]


def normalize_percentage(text) -> str:
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return m.group(0) if m else _clean(text)


def normalize_time(text) -> str:
    matches = re.findall(r"(\d{1,2}):([0-5]?\d):([0-5]?\d)", str(text))
    if matches:
        return ", ".join(
            f"{int(h):02d}:{int(mn):02d}:{int(s):02d}"
            for h, mn, s in matches
        )
    return _clean(text)


def normalize_answer(answer_format: str, text, *, options: list[str] | None = None,
                     multi: bool = False,
                     fo_names: Sequence[str] | None = None) -> str:
    """Dispatch raw model text to the normaliser for ``answer_format``."""
    if answer_format == "binary":
        return normalize_binary(text)
    if answer_format == "number":
        return normalize_number(text)
    if answer_format == "fo_class":
        return normalize_fo_class(text, fo_names=fo_names)
    if answer_format == "percentage":
        return normalize_percentage(text)
    if answer_format == "time":
        return normalize_time(text)
    if answer_format == "multiple_choice":
        return normalize_text(text, options=options, multi=multi)
    return normalize_text(text)  # open_ended, matching, unknown


def build_response(qID: str, raw_text, answer_format: str, *, latency: float = 0.0,
                   options: list[str] | None = None, multi: bool = False,
                   fo_names: Sequence[str] | None = None) -> Response:
    """Wrap normalised model output as a focus ``Response`` ready for evaluation."""
    return Response(
        qID=qID,
        content=normalize_answer(
            answer_format,
            raw_text,
            options=options,
            multi=multi,
            fo_names=fo_names,
        ),
        latency=latency,
    )


def is_parseable(answer_format: str, content: str, format_kwargs: dict | None = None) -> bool:
    """Whether ``content`` would survive ``fmt.read`` (debug/parse-rate helper)."""
    try:
        get_format_class(answer_format)(**(format_kwargs or {})).read(content)
        return True
    except Exception:
        return False
