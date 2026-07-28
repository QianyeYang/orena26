"""Integrity checks for the generated 4,000-case annotation pool."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PATH = REPO_ROOT / "data/annotations/frame-annotation-tool/pool.json"


class PoolIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.pool["cases"]

    def test_size_rank_and_balance(self) -> None:
        self.assertEqual(len(self.cases), 4_000)
        self.assertEqual(
            [case["rank"] for case in self.cases], list(range(1, 4_001))
        )
        self.assertEqual(
            Counter(case["dataset"] for case in self.cases),
            {"heico": 2_000, "lapchole": 2_000},
        )

    def test_unique_training_cases_in_score_order(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(case["split"] == "train" for case in self.cases))
        scores = [case["priority_score"] for case in self.cases]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_selected_images_and_qa(self) -> None:
        for case in self.cases:
            image_path = REPO_ROOT / case["image_relative_path"]
            self.assertTrue(image_path.is_file(), image_path)
            self.assertGreater(image_path.stat().st_size, 0, image_path)
            self.assertGreater(case["image_width"], 0)
            self.assertGreater(case["image_height"], 0)
            self.assertGreaterEqual(len(case["questions"]), 1)

    def test_guide_examples_are_released_cases(self) -> None:
        known = {case["id"] for case in self.cases}
        known.update(
            case["id"] for case in self.pool.get("extra_guide_cases", [])
        )
        for examples in self.pool["guide_examples"].values():
            self.assertTrue(set(examples).issubset(known))


if __name__ == "__main__":
    unittest.main()
