"""Completeness and QA-consistency tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from validation import AnnotationError, normalise_annotation  # noqa: E402


CASE = {
    "id": "heico-0123456789abcdef",
    "image_width": 960,
    "image_height": 540,
    "questions": [
        {
            "id": "1",
            "kind": "total_instance_count",
            "numeric_answer": 2,
            "answer": "2",
        },
        {
            "id": "2",
            "kind": "class_count",
            "target_class": "Clip",
            "numeric_answer": 2,
            "answer": "2",
        },
        {
            "id": "3",
            "kind": "class_inventory",
            "answer_classes": ["Clip"],
            "answer": "Clip",
        },
    ],
}


def payload(status: str = "complete") -> dict:
    return {
        "status": status,
        "no_foreign_objects": False,
        "boxes": [
            {
                "id": "box-one",
                "class_name": "Clip",
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
            },
            {
                "id": "box-two",
                "class_name": "Clip",
                "x": 100,
                "y": 120,
                "width": 25,
                "height": 35,
            },
        ],
        "checks": {
            "full_frame_scanned": True,
            "every_instance_boxed": True,
            "classes_reviewed": True,
            "qa_compared": True,
        },
        "notes": "",
    }


class ValidationTest(unittest.TestCase):
    def test_consistent_case_can_complete(self) -> None:
        result = normalise_annotation(payload(), CASE)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["boxes"]), 2)

    def test_count_mismatch_blocks_complete(self) -> None:
        value = payload()
        value["boxes"].pop()
        with self.assertRaises(AnnotationError) as context:
            normalise_annotation(value, CASE)
        self.assertEqual(context.exception.code, "completion_blocked")
        self.assertTrue(context.exception.details)

    def test_mismatch_can_be_sent_to_review(self) -> None:
        value = payload("needs_review")
        value["boxes"].pop()
        result = normalise_annotation(value, CASE)
        self.assertEqual(result["status"], "needs_review")

    def test_review_still_requires_completion_checklist(self) -> None:
        value = payload("needs_review")
        value["checks"]["qa_compared"] = False
        with self.assertRaises(AnnotationError):
            normalise_annotation(value, CASE)

    def test_uncertain_box_blocks_complete(self) -> None:
        value = payload()
        value["boxes"][0]["uncertain"] = True
        with self.assertRaises(AnnotationError):
            normalise_annotation(value, CASE)

    def test_out_of_bounds_box_is_rejected(self) -> None:
        value = payload("in_progress")
        value["boxes"][0]["x"] = 950
        value["boxes"][0]["width"] = 30
        with self.assertRaises(AnnotationError):
            normalise_annotation(value, CASE)

    def test_skipped_case_requires_note(self) -> None:
        value = payload("skipped")
        with self.assertRaises(AnnotationError):
            normalise_annotation(value, CASE)
        value["notes"] = "image is unreadable"
        self.assertEqual(normalise_annotation(value, CASE)["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
