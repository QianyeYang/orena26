from __future__ import annotations

import json
import unittest

from src.counting import (
    CountMode,
    CountingQuestion,
    canonicalize_fo_label,
    classify_counting_question,
    derive_structured_count,
)
from src.prompts import build_instruction, system_prompt


class CountingQuestionTest(unittest.TestCase):
    def test_classifies_every_official_question_shape(self) -> None:
        cases = {
            "How many different foreign object instances appear in this frame? "
            "Please provide a number.": (CountMode.INSTANCES, None),
            "How many different foreign object classes appear in this frame? "
            "Please provide a number.": (CountMode.CLASSES, None),
            "How many Clips appear in this frame? Please provide a number.": (
                CountMode.TARGET,
                "Clip",
            ),
            "How many Sponges appear in this frame? Please provide a number.": (
                CountMode.TARGET,
                "Sponge",
            ),
            "How many External drains appear in this frame? Please provide a number.": (
                CountMode.TARGET,
                "External Drain",
            ),
            "How many Specimen bags appear in this frame? Please provide a number.": (
                CountMode.TARGET,
                "Specimen Bag",
            ),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = classify_counting_question(question, "number")
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual((result.mode, result.target), expected)

    def test_rejects_non_number_and_non_count_questions(self) -> None:
        question = "How many Clips appear in this frame?"
        self.assertIsNone(classify_counting_question(question, "open_ended"))
        self.assertIsNone(
            classify_counting_question(
                "What foreign object is visible in this frame?", "number"
            )
        )

    def test_canonicalizes_plural_and_modified_labels(self) -> None:
        self.assertEqual(canonicalize_fo_label("surgical clips"), "Clip")
        self.assertEqual(canonicalize_fo_label("Specimen Bags"), "Specimen Bag")


class CountingPromptTest(unittest.TestCase):
    def test_bbox_prompt_contains_schema_one_shot_and_no_count_instruction(self) -> None:
        question = "How many Clips appear in this frame? Please provide a number."
        prompt, options = build_instruction(question, "number", "bbox-json")
        self.assertIsNone(options)
        self.assertIn('"bbox_2d":[x_min,y_min,x_max,y_max]', prompt)
        self.assertIn("One-shot format example:", prompt)
        self.assertIn('"label":"Clip"', prompt)
        self.assertIn("Do not include a count field", prompt)

    def test_bbox_system_prompt_forbids_bare_number(self) -> None:
        self.assertIn("Do not answer with a bare number", system_prompt("frame", "bbox-json"))

    def test_direct_prompt_remains_the_existing_digits_instruction(self) -> None:
        question = "How many Clips appear in this frame? Please provide a number."
        prompt, _ = build_instruction(question, "number")
        self.assertEqual(
            prompt,
            f"{question}\n\nAnswer with a single non-negative integer (digits only).",
        )

    def test_non_counting_prompt_is_unchanged_under_bbox_strategy(self) -> None:
        question = "Is a Clip visible?"
        direct, _ = build_instruction(question, "binary", "direct")
        bbox_strategy, _ = build_instruction(question, "binary", "bbox-json")
        self.assertEqual(direct, bbox_strategy)


class StructuredCountTest(unittest.TestCase):
    def test_counts_all_instance_boxes(self) -> None:
        raw = json.dumps(
            {
                "objects": [
                    {"label": "Clip", "bbox_2d": [10, 20, 30, 40]},
                    {"label": "Sponge", "bbox_2d": [100, 200, 300, 400]},
                ]
            }
        )
        result = derive_structured_count(
            raw, CountingQuestion(CountMode.INSTANCES)
        )
        self.assertEqual(result.answer, "2")
        self.assertTrue(result.schema_valid)

    def test_class_count_deduplicates_labels_not_boxes(self) -> None:
        raw = json.dumps(
            {
                "objects": [
                    {"label": "Clip", "bbox_2d": [10, 20, 30, 40]},
                    {"label": "clips", "bbox_2d": [50, 60, 70, 80]},
                    {"label": "Sponge", "bbox_2d": [100, 200, 300, 400]},
                ]
            }
        )
        result = derive_structured_count(raw, CountingQuestion(CountMode.CLASSES))
        self.assertEqual(result.answer, "2")

    def test_target_count_uses_only_matching_labeled_boxes(self) -> None:
        raw = json.dumps(
            {
                "objects": [
                    {"label": "Clip", "bbox_2d": [10, 20, 30, 40]},
                    {"label": "Sponge", "bbox_2d": [100, 200, 300, 400]},
                ]
            }
        )
        result = derive_structured_count(
            raw, CountingQuestion(CountMode.TARGET, target="Clip")
        )
        self.assertEqual(result.answer, "1")

    def test_valid_empty_object_list_is_zero(self) -> None:
        result = derive_structured_count(
            '{"objects":[]}', CountingQuestion(CountMode.INSTANCES)
        )
        self.assertEqual(result.answer, "0")
        self.assertTrue(result.schema_valid)

    def test_accepts_fenced_json_and_json_like_single_quotes(self) -> None:
        fenced = derive_structured_count(
            '```json\n{"objects":[{"label":"Needle","bbox":[1,2,3,4]}]}\n```',
            CountingQuestion(CountMode.INSTANCES),
        )
        literal = derive_structured_count(
            "{'objects': [{'label': 'Needle', 'bbox_2d': [1, 2, 3, 4]}]}",
            CountingQuestion(CountMode.INSTANCES),
        )
        self.assertEqual(fenced.answer, "1")
        self.assertEqual(literal.answer, "1")
        self.assertTrue(fenced.schema_valid)
        self.assertTrue(literal.schema_valid)

    def test_skips_invalid_boxes_and_marks_partial_schema(self) -> None:
        raw = json.dumps(
            {
                "objects": [
                    {"label": "Clip", "bbox_2d": [10, 20, 30, 40]},
                    {"label": "Clip", "bbox_2d": [30, 20, 10, 40]},
                ]
            }
        )
        result = derive_structured_count(
            raw, CountingQuestion(CountMode.INSTANCES)
        )
        self.assertEqual(result.answer, "1")
        self.assertFalse(result.schema_valid)
        self.assertIn("partial_invalid_objects", result.status)

    def test_does_not_fall_back_to_a_bare_model_number(self) -> None:
        result = derive_structured_count(
            "There are 7 clips.", CountingQuestion(CountMode.TARGET, target="Clip")
        )
        self.assertEqual(result.answer, "0")
        self.assertFalse(result.schema_valid)
        self.assertEqual(result.status, "invalid_json")

    def test_recovers_boxes_from_unquoted_json_keys(self) -> None:
        result = derive_structured_count(
            "{label: 'Clip', bbox_2d: [10, 20, 30, 40]}",
            CountingQuestion(CountMode.INSTANCES),
        )
        self.assertEqual(result.answer, "1")
        self.assertFalse(result.schema_valid)
        self.assertEqual(result.status, "recovered_bbox_regex")


if __name__ == "__main__":
    unittest.main()
