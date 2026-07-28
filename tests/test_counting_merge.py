from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
MERGE_PATH = (
    REPO
    / "track-frame"
    / "counting-bbox-prompt"
    / "src"
    / "merge_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("counting_merge_outputs", MERGE_PATH)
assert SPEC is not None and SPEC.loader is not None
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class CountingMergeTest(unittest.TestCase):
    def test_response_merge_replaces_by_id_and_preserves_order(self) -> None:
        baseline = [
            {"qID": "a", "content": "1", "latency": 0.1},
            {"qID": "b", "content": "2", "latency": 0.2},
        ]
        replacements = [{"qID": "b", "content": "3", "latency": 0.3}]
        merged = MERGE.merge_responses(baseline, replacements)
        self.assertEqual([row["qID"] for row in merged], ["a", "b"])
        self.assertEqual([row["content"] for row in merged], ["1", "3"])

    def test_prediction_merge_retains_direct_audit_fields(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "model_name": "direct",
                    "prompt": "direct prompt a",
                    "raw_model_output": "1",
                    "prediction": "1",
                    "latency_sec": 0.1,
                },
                {
                    "sample_id": "b",
                    "model_name": "direct",
                    "prompt": "direct prompt b",
                    "raw_model_output": "2",
                    "prediction": "2",
                    "latency_sec": 0.2,
                },
            ]
        )
        replacements = pd.DataFrame(
            [
                {
                    "sample_id": "b",
                    "model_name": "counting-only",
                    "prompt": "bbox prompt",
                    "prompt_strategy": "bbox-json",
                    "raw_model_output": '{"objects":[]}',
                    "prediction": "0",
                    "latency_sec": 0.4,
                }
            ]
        )
        merged = MERGE.merge_predictions(
            baseline,
            replacements,
            model_name="hybrid",
        ).set_index("sample_id")

        self.assertFalse(bool(merged.loc["a", "changed_by_prompt"]))
        self.assertTrue(bool(merged.loc["b", "changed_by_prompt"]))
        self.assertEqual(merged.loc["b", "prediction"], "0")
        self.assertEqual(merged.loc["b", "baseline_prediction"], "2")
        self.assertEqual(merged.loc["b", "baseline_prompt"], "direct prompt b")
        self.assertEqual(set(merged["model_name"]), {"hybrid"})


if __name__ == "__main__":
    unittest.main()
