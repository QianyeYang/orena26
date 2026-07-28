"""Server-side normalisation and completeness checks for Frame annotations."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from guide import CLASS_NAMES, canonical_class, classes_in_answer


VALID_STATUSES = {
    "unstarted",
    "in_progress",
    "complete",
    "needs_review",
    "skipped",
}
CHECK_KEYS = (
    "full_frame_scanned",
    "every_instance_boxed",
    "classes_reviewed",
    "qa_compared",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class AnnotationError(ValueError):
    """An annotation payload is malformed or cannot enter its requested state."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_annotation",
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or []


def empty_annotation(case_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "revision": 0,
        "status": "unstarted",
        "no_foreign_objects": False,
        "boxes": [],
        "checks": {key: False for key in CHECK_KEYS},
        "notes": "",
        "created_at": None,
        "updated_at": None,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AnnotationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise AnnotationError(f"{label} must be a finite number")
    return round(number, 3)


def _normalise_box(
    raw: Any, *, width: int, height: int, seen_ids: set[str], index: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AnnotationError(f"box {index + 1} must be an object")
    box_id = str(raw.get("id", "")).strip()
    if not _SAFE_ID.fullmatch(box_id):
        raise AnnotationError(
            f"box {index + 1} has an invalid id; use letters, numbers, _ or -"
        )
    if box_id in seen_ids:
        raise AnnotationError(f"duplicate box id: {box_id}")
    seen_ids.add(box_id)

    class_name = canonical_class(raw.get("class_name"))
    if class_name not in CLASS_NAMES:
        raise AnnotationError(
            f"box {index + 1} has an unknown FOCUS class"
        )
    x = _finite_number(raw.get("x"), f"box {index + 1} x")
    y = _finite_number(raw.get("y"), f"box {index + 1} y")
    box_width = _finite_number(raw.get("width"), f"box {index + 1} width")
    box_height = _finite_number(raw.get("height"), f"box {index + 1} height")
    if x < 0 or y < 0 or box_width <= 1 or box_height <= 1:
        raise AnnotationError(
            f"box {index + 1} must have non-negative coordinates and be "
            "larger than one pixel"
        )
    tolerance = 0.01
    if x + box_width > width + tolerance or y + box_height > height + tolerance:
        raise AnnotationError(
            f"box {index + 1} extends outside the {width}×{height} image"
        )
    # Remove tiny floating-point overshoots introduced by canvas scaling.
    box_width = min(box_width, width - x)
    box_height = min(box_height, height - y)
    return {
        "id": box_id,
        "class_name": class_name,
        "x": x,
        "y": y,
        "width": round(box_width, 3),
        "height": round(box_height, 3),
        "difficult": bool(raw.get("difficult", False)),
        "uncertain": bool(raw.get("uncertain", False)),
    }


def normalise_annotation(
    raw: Any,
    case: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AnnotationError("annotation payload must be a JSON object")
    status = str(raw.get("status", "in_progress"))
    if status not in VALID_STATUSES:
        raise AnnotationError(f"unknown status: {status}")
    if status == "unstarted":
        status = "in_progress"

    boxes_raw = raw.get("boxes", [])
    if not isinstance(boxes_raw, list):
        raise AnnotationError("boxes must be an array")
    seen_ids: set[str] = set()
    boxes = [
        _normalise_box(
            item,
            width=int(case["image_width"]),
            height=int(case["image_height"]),
            seen_ids=seen_ids,
            index=index,
        )
        for index, item in enumerate(boxes_raw)
    ]
    no_objects = bool(raw.get("no_foreign_objects", False))
    if no_objects and boxes:
        raise AnnotationError(
            "a case cannot have boxes and “no foreign objects” at the same time"
        )

    raw_checks = raw.get("checks", {})
    if not isinstance(raw_checks, dict):
        raise AnnotationError("checks must be an object")
    checks = {key: bool(raw_checks.get(key, False)) for key in CHECK_KEYS}
    notes = str(raw.get("notes", "")).strip()
    if len(notes) > 10_000:
        raise AnnotationError("notes cannot exceed 10,000 characters")
    if status == "skipped" and not notes:
        raise AnnotationError("add a reason in notes before skipping a case")

    record = {
        "schema_version": 1,
        "case_id": case["id"],
        "revision": int((previous or {}).get("revision", 0)),
        "status": status,
        "no_foreign_objects": no_objects,
        "boxes": boxes,
        "checks": checks,
        "notes": notes,
        "created_at": (previous or {}).get("created_at"),
        "updated_at": (previous or {}).get("updated_at"),
    }
    report = validate_annotation(record, case)

    if status in {"complete", "needs_review"} and report["structural_errors"]:
        raise AnnotationError(
            "finish the annotation decision and completion checklist before "
            f"saving as {status.replace('_', ' ')}",
            code="completion_blocked",
            details=report["structural_errors"],
        )
    if status == "complete":
        blockers = report["qa_mismatches"]
        if blockers:
            raise AnnotationError(
                "this case is not complete; resolve the listed checks or save "
                "it as needs review",
                code="completion_blocked",
                details=blockers,
            )
        if report["uncertain_boxes"]:
            raise AnnotationError(
                "uncertain boxes must be resolved or saved as needs review",
                code="completion_blocked",
                details=[
                    {
                        "kind": "uncertain_box",
                        "message": (
                            f"{report['uncertain_boxes']} uncertain box(es) remain"
                        ),
                    }
                ],
            )
    return record


def _mismatch(
    question: dict[str, Any],
    *,
    expected: Any,
    observed: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "kind": "qa_mismatch",
        "question_id": question["id"],
        "question_kind": question["kind"],
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def _qa_mismatches(
    annotation: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    boxes = annotation["boxes"]
    counts = Counter(box["class_name"] for box in boxes)
    observed_classes = set(counts)
    mismatches: list[dict[str, Any]] = []

    for question in case["questions"]:
        kind = question["kind"]
        expected_number = question.get("numeric_answer")
        if kind == "total_instance_count" and expected_number is not None:
            if len(boxes) != expected_number:
                mismatches.append(
                    _mismatch(
                        question,
                        expected=expected_number,
                        observed=len(boxes),
                        message=(
                            f"QA says {expected_number} total instance(s), but "
                            f"{len(boxes)} box(es) are drawn"
                        ),
                    )
                )
        elif kind == "distinct_class_count" and expected_number is not None:
            if len(observed_classes) != expected_number:
                mismatches.append(
                    _mismatch(
                        question,
                        expected=expected_number,
                        observed=len(observed_classes),
                        message=(
                            f"QA says {expected_number} distinct class(es), but "
                            f"boxes contain {len(observed_classes)}"
                        ),
                    )
                )
        elif kind == "class_count" and question.get("target_class"):
            class_name = question["target_class"]
            observed = counts[class_name]
            if expected_number is not None and observed != expected_number:
                mismatches.append(
                    _mismatch(
                        question,
                        expected=expected_number,
                        observed=observed,
                        message=(
                            f"QA says {expected_number} {class_name} instance(s), "
                            f"but boxes contain {observed}"
                        ),
                    )
                )
        elif kind == "class_inventory":
            expected_classes = set(question.get("answer_classes", []))
            answer_is_none = str(question["answer"]).strip().casefold() in {
                "none",
                "no",
                "no foreign object",
                "no foreign objects",
            }
            if answer_is_none:
                expected_classes = set()
            if expected_classes != observed_classes:
                mismatches.append(
                    _mismatch(
                        question,
                        expected=sorted(expected_classes),
                        observed=sorted(observed_classes),
                        message=(
                            "QA inventory and boxed class set differ: "
                            f"QA={', '.join(sorted(expected_classes)) or 'none'}; "
                            f"boxes={', '.join(sorted(observed_classes)) or 'none'}"
                        ),
                    )
                )
        elif kind == "single_object_identity":
            expected_classes = set(question.get("answer_classes", []))
            if len(boxes) != 1 or observed_classes != expected_classes:
                mismatches.append(
                    _mismatch(
                        question,
                        expected={
                            "instances": 1,
                            "classes": sorted(expected_classes),
                        },
                        observed={
                            "instances": len(boxes),
                            "classes": sorted(observed_classes),
                        },
                        message=(
                            "single-object QA disagrees with the current boxes"
                        ),
                    )
                )
        elif kind == "closest_to_center":
            expected_classes = set(classes_in_answer(question["answer"]))
            if expected_classes and boxes:
                center_x = float(case["image_width"]) / 2
                center_y = float(case["image_height"]) / 2
                closest = min(
                    boxes,
                    key=lambda box: (
                        box["x"] + box["width"] / 2 - center_x
                    )
                    ** 2
                    + (
                        box["y"] + box["height"] / 2 - center_y
                    )
                    ** 2,
                )
                if closest["class_name"] not in expected_classes:
                    mismatches.append(
                        _mismatch(
                            question,
                            expected=sorted(expected_classes),
                            observed=closest["class_name"],
                            message=(
                                "the box centre nearest the frame centre has "
                                f"class {closest['class_name']}, not the QA class"
                            ),
                        )
                    )
    return mismatches


def validate_annotation(
    annotation: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    boxes = annotation.get("boxes", [])
    no_objects = bool(annotation.get("no_foreign_objects"))
    checks = annotation.get("checks", {})
    structural_errors: list[dict[str, Any]] = []

    if not no_objects and not boxes:
        structural_errors.append(
            {
                "kind": "missing_decision",
                "message": (
                    "draw at least one box or explicitly choose “no foreign "
                    "objects”"
                ),
            }
        )
    for key in CHECK_KEYS:
        if not checks.get(key, False):
            structural_errors.append(
                {
                    "kind": "unchecked_completion_item",
                    "check": key,
                    "message": f"completion check is unfinished: {key}",
                }
            )

    counts = Counter(box["class_name"] for box in boxes)
    qa_mismatches = _qa_mismatches(annotation, case)
    uncertain = sum(bool(box.get("uncertain")) for box in boxes)
    return {
        "structural_errors": structural_errors,
        "qa_mismatches": qa_mismatches,
        "uncertain_boxes": uncertain,
        "box_count": len(boxes),
        "class_count": len(counts),
        "inventory": [
            {"class_name": class_name, "count": counts[class_name]}
            for class_name in CLASS_NAMES
            if counts[class_name]
        ],
        "can_complete": not structural_errors
        and not qa_mismatches
        and uncertain == 0,
    }
