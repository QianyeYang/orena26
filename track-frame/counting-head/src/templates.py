"""The ten FRAME counting question templates, as a dense integer id.

FRAME asks exactly ten distinct counting questions across all 6,356 ``number``
rows of both datasets and both splits: two aggregate forms plus one per named
class. That makes "which object is this question about?" a lookup rather than a
language problem, which is what lets the head be conditioned on a small
embedding instead of having to read the question itself.

``src.counting.classify_counting_question`` already does the parse; this module
only assigns the result a stable id and supplies the per-template answer bounds
that ground truth is known to obey.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from focus import FOType  # noqa: E402

from src.counting import CountMode, classify_counting_question  # noqa: E402

#: Canonical class order, shared with :class:`focus.FOType`.
FO_NAMES: tuple[str, ...] = tuple(FOType.names())

#: ids 0 and 1 are the aggregate questions; 2+i is "how many <FO_NAMES[i]>".
N_TEMPLATES: int = 2 + len(FO_NAMES)

#: Ground-truth bounds, verified over every FRAME counting row (see
#: ``docs``/session notes): counts are never 0 in this track — unlike segment and
#: procedure, which do contain zeros — and the distinct-class question never
#: exceeds 4. Applying these at prediction time is free accuracy.
MIN_COUNT: int = 1
MAX_COUNT: int = 16
CLASSES_MAX: int = 4


def template_id(question: str) -> int | None:
    """Return the template id for *question*, or ``None`` if it is not counting.

    Deliberately does not take an ``answer_format``: the official ``Request``
    carries only ``qID``, ``videoID``, ``start_time``, ``end_time``,
    ``procedure_type`` and ``question``, so at submission time the question text
    is the only thing available to route on.
    """
    parsed = classify_counting_question(question, "number")
    if parsed is None:
        return None
    if parsed.mode is CountMode.INSTANCES:
        return 0
    if parsed.mode is CountMode.CLASSES:
        return 1
    if parsed.target is None:
        return None
    return 2 + FO_NAMES.index(parsed.target)


def template_name(tid: int) -> str:
    if tid == 0:
        return "instances"
    if tid == 1:
        return "classes"
    return FO_NAMES[tid - 2]


def clamp(count: int, tid: int) -> int:
    """Apply the ground-truth bounds this template is known to obey."""
    hi = CLASSES_MAX if tid == 1 else MAX_COUNT
    return max(MIN_COUNT, min(int(count), hi))
