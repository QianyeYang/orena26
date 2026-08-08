"""Unified-SFT parity tests: each track's items must equal its specialist's.

Data-dependent tests need the annotation parquets (and the ``focus`` package),
so this suite is meant to run on the training cluster:

    cd /datasets/engs2732/orena && python -m unittest tests.test_unified_parity -v

The geometry tests are pure and run anywhere.
"""

from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

from PIL import Image

from src.videovqa_unified import (
    UnifiedSFTDataset,
    budget_resize,
    build_frame_messages,
    smart_resize_dims,
)

REPO = Path(__file__).resolve().parents[1]

TRACKS_CFG = {
    "frame": {"stride": {"heico": 25, "lapchole": 30}, "max_frames": 1,
              "max_pixels": 602112},
    "segment": {"stride": {"heico": 25, "lapchole": 30}, "max_frames": 64,
                "max_pixels": 262144},
    "procedure": {"stride": {"heico": 250, "lapchole": 300}, "max_frames": 96,
                  "max_pixels": 131072},
}
N = 4  # rows per (track, dataset) shard under test


def _load_frame_module(name: str):
    path = REPO / "track-frame" / "lora-finetune" / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"frame_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parquets_available() -> bool:
    from src.paths import parquet_path

    return parquet_path("frame", "train", "heico").is_file()


class GeometryTest(unittest.TestCase):
    FACTOR = 32  # Qwen3-VL: patch 16 x merge 2
    MIN_PIXELS = 65536  # processor default shortest_edge

    def test_budgets_are_factor_aligned(self) -> None:
        for tc in TRACKS_CFG.values():
            self.assertEqual(tc["max_pixels"] % (self.FACTOR * self.FACTOR), 0)

    def test_resize_within_budget_aligned_and_idempotent(self) -> None:
        for w, h in ((1920, 1080), (854, 480), (640, 512)):
            for tc in TRACKS_CFG.values():
                budget = tc["max_pixels"]
                img = Image.new("RGB", (w, h))
                out = budget_resize(
                    img, factor=self.FACTOR, min_pixels=self.MIN_PIXELS,
                    max_pixels=budget,
                )
                self.assertEqual(out.width % self.FACTOR, 0)
                self.assertEqual(out.height % self.FACTOR, 0)
                self.assertLessEqual(out.width * out.height, budget)
                self.assertGreaterEqual(out.width * out.height, self.MIN_PIXELS)
                again = budget_resize(
                    out, factor=self.FACTOR, min_pixels=self.MIN_PIXELS,
                    max_pixels=budget,
                )
                self.assertEqual((again.width, again.height), (out.width, out.height))
                # the processor's own pass (global budget 602112) must be a no-op
                relaxed = smart_resize_dims(
                    out.height, out.width, factor=self.FACTOR,
                    min_pixels=self.MIN_PIXELS, max_pixels=602112,
                )
                self.assertEqual(relaxed, (out.height, out.width))

    def test_matches_transformers_smart_resize(self) -> None:
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                smart_resize,
            )
        except Exception:  # noqa: BLE001 — optional dependency on test host
            self.skipTest("transformers qwen2_vl smart_resize not importable")
        for h, w in ((1080, 1920), (480, 854), (512, 640), (2160, 3840)):
            for tc in TRACKS_CFG.values():
                ours = smart_resize_dims(
                    h, w, factor=self.FACTOR, min_pixels=self.MIN_PIXELS,
                    max_pixels=tc["max_pixels"],
                )
                theirs = smart_resize(
                    h, w, factor=self.FACTOR, min_pixels=self.MIN_PIXELS,
                    max_pixels=tc["max_pixels"],
                )
                self.assertEqual(ours, tuple(theirs))


@unittest.skipUnless(_parquets_available(), "annotation parquets not on this host")
class FrameParityTest(unittest.TestCase):
    """Unified FRAME items == specialist FrameSFTDataset + SFTCollator messages."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame_dataset_mod = _load_frame_module("dataset")
        cls.specialist = cls.frame_dataset_mod.FrameSFTDataset(
            "frame", "train", datasets=("heico",), limit=N
        )
        cls.unified = UnifiedSFTDataset(
            TRACKS_CFG, "train", datasets=("heico",), limit_per_track=N
        )

    def test_items_match(self) -> None:
        from src import prompts as P

        for i in range(N):
            spec_ex = self.specialist.examples[i]
            uni_ex = self.unified[i]  # frame items come first in concat order
            self.assertEqual(uni_ex["track"], "frame")
            self.assertEqual(uni_ex["image_paths"], [spec_ex["image_path"]])
            self.assertEqual(uni_ex["answer"], spec_ex["answer"])
            expected_msgs = self.frame_dataset_mod.SFTCollator._messages(
                types.SimpleNamespace(system_prompt=P.SYSTEM_PROMPT),
                spec_ex["instruction"],
            )
            self.assertEqual(uni_ex["messages"], expected_msgs)


@unittest.skipUnless(_parquets_available(), "annotation parquets not on this host")
class WindowParityTest(unittest.TestCase):
    """Unified SEGMENT/PROCEDURE items == specialist VideoSFTDataset items."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.unified = UnifiedSFTDataset(
            TRACKS_CFG, "train", datasets=("heico",), limit_per_track=N
        )

    def _check_track(self, track: str, offset: int) -> None:
        from src.videovqa_sft import VideoSFTDataset

        tc = TRACKS_CFG[track]
        specialist = VideoSFTDataset(
            track, "train", datasets=("heico",), stride=tc["stride"],
            max_frames=tc["max_frames"], limit=N,
        )
        for i in range(N):
            spec_ex = specialist[i]
            uni_ex = self.unified[offset + i]
            self.assertEqual(uni_ex["track"], track)
            self.assertEqual(uni_ex["messages"], spec_ex["messages"])
            self.assertEqual(uni_ex["image_paths"], spec_ex["image_paths"])
            self.assertEqual(uni_ex["answer"], spec_ex["answer"])
            self.assertEqual(uni_ex["max_pixels"], tc["max_pixels"])

    def test_segment_items_match(self) -> None:
        self._check_track("segment", offset=N)

    def test_procedure_items_match(self) -> None:
        self._check_track("procedure", offset=2 * N)


class FrameMessageShapeTest(unittest.TestCase):
    """Structural spec of the frame-style prompt (runs anywhere)."""

    def test_structure(self) -> None:
        msgs = build_frame_messages("SYS", "Q?")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], [{"type": "text", "text": "SYS"}])
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(
            msgs[1]["content"],
            [{"type": "image"}, {"type": "text", "text": "Q?"}],
        )


if __name__ == "__main__":
    unittest.main()
