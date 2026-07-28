from __future__ import annotations

import unittest

from focus import get_format_class

from src.adapter import build_response, normalize_answer, normalize_fo_class


class FOClassAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.format = get_format_class("fo_class")()

    def assert_same_set(self, actual: str, expected: str) -> None:
        self.assertEqual(self.format.read(actual), self.format.read(expected))

    def test_preserves_multiple_classes(self) -> None:
        self.assert_same_set(
            normalize_fo_class("The answer is Clip, Sponge."),
            "Clip, Sponge",
        )

    def test_masks_overlapping_class_names(self) -> None:
        self.assert_same_set(
            normalize_fo_class("Specimen Bag"),
            "Specimen Bag",
        )

    def test_keeps_explicit_overlapping_classes(self) -> None:
        self.assert_same_set(
            normalize_fo_class("Specimen Bag and Specimen"),
            "Specimen Bag, Specimen",
        )

    def test_deduplicates_without_losing_other_classes(self) -> None:
        self.assert_same_set(
            normalize_answer("fo_class", "Clip, Clip and Sponge"),
            "Clip, Sponge",
        )

    def test_supports_runtime_defined_classes(self) -> None:
        names = ("Clip", "Marker Band")
        dynamic_format = get_format_class("fo_class")(valid_names=names)
        actual = normalize_fo_class(
            "Marker Band and Clip",
            fo_names=names,
        )
        self.assertEqual(
            dynamic_format.read(actual),
            dynamic_format.read("Clip, Marker Band"),
        )

    def test_build_response_keeps_the_complete_set(self) -> None:
        response = build_response(
            "q001",
            "Sponge and Clip",
            "fo_class",
        )
        self.assert_same_set(response.content, "Clip, Sponge")


if __name__ == "__main__":
    unittest.main()
