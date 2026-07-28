"""Structured localization helpers for FRAME counting questions.

The bounding-box prompting experiment asks the VLM to emit detections rather
than a final number.  This module classifies the three counting semantics in
the official FRAME data, parses a deliberately small JSON schema, and derives
the numeric answer deterministically from valid boxes.

The experiment intentionally does not fall back to a bare number found in the
raw model output: doing so would mix direct-answer prompting with the
localize-then-count intervention being measured.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import json
import math
import re
from typing import Any

from focus import FOType


class CountMode(str, Enum):
    """How detections should be reduced to the requested count."""

    INSTANCES = "instances"
    CLASSES = "classes"
    TARGET = "target"


@dataclass(frozen=True)
class CountingQuestion:
    """Parsed semantics of one official FRAME counting question."""

    mode: CountMode
    target: str | None = None


@dataclass(frozen=True)
class Detection:
    """One parsed labeled box."""

    label: str
    bbox_2d: tuple[float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "bbox_2d": list(self.bbox_2d)}


@dataclass(frozen=True)
class StructuredCountResult:
    """Detections plus the numeric answer derived from them."""

    answer: str
    count: int
    detections: tuple[Detection, ...]
    schema_valid: bool
    status: str

    @property
    def detections_json(self) -> str:
        return json.dumps(
            {"objects": [detection.as_dict() for detection in self.detections]},
            separators=(",", ":"),
        )


_HOW_MANY_FRAME_RE = re.compile(
    r"^\s*how many\s+(?P<subject>.+?)\s+appear\s+in\s+(?:this|the)\s+frame\b",
    flags=re.IGNORECASE,
)
_INSTANCE_SUBJECTS = {
    "foreign object instance",
    "different foreign object instance",
}
_CLASS_SUBJECTS = {
    "foreign object class",
    "different foreign object class",
}
_BBOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box")
_LABEL_KEYS = ("label", "class", "class_name", "name", "object")
_OBJECT_LIST_KEYS = ("objects", "detections", "instances", "items")
_FO_NAMES = tuple(FOType.names())


def _singular_key(value: str) -> str:
    """Normalize a label/subject and singularize its final token."""

    tokens = re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))
    if not tokens:
        return ""
    if tokens[-1].endswith("sses"):
        tokens[-1] = tokens[-1][:-2]
    elif tokens[-1].endswith("ies"):
        tokens[-1] = f"{tokens[-1][:-3]}y"
    elif tokens[-1].endswith("s") and not tokens[-1].endswith("ss"):
        tokens[-1] = tokens[-1][:-1]
    return " ".join(tokens)


_FO_BY_KEY = {_singular_key(name): name for name in _FO_NAMES}


def canonicalize_fo_label(value: str) -> str:
    """Map common label variants such as ``"surgical clips"`` to an FO name."""

    key = _singular_key(value)
    if not key:
        return ""
    if key in _FO_BY_KEY:
        return _FO_BY_KEY[key]

    # Models sometimes add a generic modifier ("surgical clip") even when the
    # prompt supplies the canonical vocabulary. Prefer the longest contained
    # canonical name so Specimen Bag cannot collapse to Specimen.
    matches = [
        (len(canonical_key), canonical_name)
        for canonical_key, canonical_name in _FO_BY_KEY.items()
        if re.search(rf"(^|\s){re.escape(canonical_key)}($|\s)", key)
    ]
    if matches:
        return max(matches)[1]
    return " ".join(str(value).split())


def classify_counting_question(
    question: str,
    answer_format: str,
) -> CountingQuestion | None:
    """Classify an official FRAME count question, or return ``None``."""

    if answer_format != "number":
        return None
    match = _HOW_MANY_FRAME_RE.search(str(question))
    if match is None:
        return None

    subject = _singular_key(match.group("subject"))
    if subject in _INSTANCE_SUBJECTS:
        return CountingQuestion(CountMode.INSTANCES)
    if subject in _CLASS_SUBJECTS:
        return CountingQuestion(CountMode.CLASSES)

    target = _FO_BY_KEY.get(subject)
    if target is None:
        return None
    return CountingQuestion(CountMode.TARGET, target=target)


def _json_candidates(raw_text: Any) -> list[str]:
    text = str(raw_text).strip()
    if not text:
        return []

    candidates = [text]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json|python)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            _, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[start : start + length])

    # Preserve order while avoiding repeated work on identical snippets.
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _load_payload(raw_text: Any) -> tuple[Any | None, str]:
    candidates = _json_candidates(raw_text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if _looks_like_detection_payload(payload):
                return payload, "json"

        # ``literal_eval`` safely recovers frequent JSON-like deviations such
        # as single quotes. It cannot execute model-produced code.
        try:
            payload = ast.literal_eval(candidate)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            pass
        else:
            if _looks_like_detection_payload(payload):
                return payload, "python_literal"
    return None, "invalid_json"


def _looks_like_detection_payload(payload: Any) -> bool:
    if isinstance(payload, list):
        return not payload or all(isinstance(item, dict) for item in payload)
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in (*_OBJECT_LIST_KEYS, *_BBOX_KEYS))


def _object_items(payload: Any) -> tuple[list[Any] | None, bool]:
    if isinstance(payload, list):
        return payload, True
    if not isinstance(payload, dict):
        return None, False

    for key in _OBJECT_LIST_KEYS:
        if key in payload:
            items = payload[key]
            return (items, True) if isinstance(items, list) else (None, False)
    if any(key in payload for key in _BBOX_KEYS):
        return [payload], True
    return None, False


def _bbox_value(item: dict[str, Any]) -> Any:
    for key in _BBOX_KEYS:
        if key in item:
            return item[key]
    return None


def _label_value(item: dict[str, Any]) -> str:
    for key in _LABEL_KEYS:
        value = item.get(key)
        if value is not None:
            return canonicalize_fo_label(str(value))
    return ""


def _valid_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        key_sets = (
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("x1", "y1", "x2", "y2"),
        )
        for keys in key_sets:
            if all(key in value for key in keys):
                value = [value[key] for key in keys]
                break

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(number, bool) for number in value):
        return None
    try:
        coords = tuple(float(number) for number in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(number) for number in coords):
        return None

    x_min, y_min, x_max, y_max = coords
    if x_min < 0 or y_min < 0 or x_max > 1000 or y_max > 1000:
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return coords


def _parse_detections(items: list[Any]) -> tuple[list[Detection], int]:
    detections: list[Detection] = []
    invalid = 0
    for item in items:
        if not isinstance(item, dict):
            invalid += 1
            continue
        bbox = _valid_bbox(_bbox_value(item))
        label = _label_value(item)
        if bbox is None or not label:
            invalid += 1
            continue
        detections.append(Detection(label=label, bbox_2d=bbox))
    return detections, invalid


def _regex_detections(raw_text: Any) -> list[Detection]:
    """Recover labeled boxes from mildly malformed object dictionaries."""

    text = str(raw_text)
    detections: list[Detection] = []
    for object_match in re.finditer(r"\{[^{}]{0,1000}\}", text, flags=re.DOTALL):
        fragment = object_match.group(0)
        bbox_match = re.search(
            r"""["']?(?:bbox_2d|bbox|box|bounding_box)["']?\s*:\s*
                \[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*
                ([-+]?\d+(?:\.\d+)?)\s*,\s*
                ([-+]?\d+(?:\.\d+)?)\s*,\s*
                ([-+]?\d+(?:\.\d+)?)\s*\]""",
            fragment,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        label_match = re.search(
            r"""["']?(?:label|class|class_name|name|object)["']?\s*:\s*
                ["']([^"']+)["']""",
            fragment,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        if bbox_match is None or label_match is None:
            continue
        bbox = _valid_bbox(list(bbox_match.groups()))
        label = canonicalize_fo_label(label_match.group(1))
        if bbox is not None and label:
            detections.append(Detection(label=label, bbox_2d=bbox))
    return detections


def _derive_count(
    detections: list[Detection],
    question: CountingQuestion,
) -> int:
    if question.mode is CountMode.INSTANCES:
        return len(detections)
    if question.mode is CountMode.CLASSES:
        return len({canonicalize_fo_label(detection.label) for detection in detections})
    if question.target is None:
        raise ValueError("target counting mode requires a target label")
    target = canonicalize_fo_label(question.target)
    return sum(
        canonicalize_fo_label(detection.label) == target
        for detection in detections
    )


def derive_structured_count(
    raw_text: Any,
    question: CountingQuestion,
) -> StructuredCountResult:
    """Parse detections and derive a number without reading a bare count."""

    payload, parse_status = _load_payload(raw_text)
    items, schema_found = _object_items(payload)
    if schema_found and items is not None:
        detections, invalid = _parse_detections(items)
        count = _derive_count(detections, question)
        if invalid:
            status = f"{parse_status}_partial_invalid_objects"
            schema_valid = False
        else:
            status = parse_status
            schema_valid = True
        return StructuredCountResult(
            answer=str(count),
            count=count,
            detections=tuple(detections),
            schema_valid=schema_valid,
            status=status,
        )

    recovered = _regex_detections(raw_text)
    if recovered:
        count = _derive_count(recovered, question)
        return StructuredCountResult(
            answer=str(count),
            count=count,
            detections=tuple(recovered),
            schema_valid=False,
            status="recovered_bbox_regex",
        )

    return StructuredCountResult(
        answer="0",
        count=0,
        detections=(),
        schema_valid=False,
        status="missing_objects" if payload is not None else "invalid_json",
    )
