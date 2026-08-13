from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "track-frame" / "v2" / "src"))

from counting_v2 import (  # noqa: E402
    TILE_NAMES,
    COMBINE_RULES,
    build_view_instruction,
    build_views,
    combine_class_sets,
    combine_counts,
)
from generate import strip_thinking  # noqa: E402
from src.counting import CountMode, CountingQuestion  # noqa: E402


class StripThinkingTest(unittest.TestCase):
    def test_removes_a_closed_reasoning_block(self) -> None:
        self.assertEqual(strip_thinking("<think>\nlet me look\n</think>\n\nClip"), "Clip")

    def test_removes_a_template_prefilled_empty_block(self) -> None:
        self.assertEqual(strip_thinking("<think>\n\n</think>\n\n3"), "3")

    def test_truncated_reasoning_yields_no_answer(self) -> None:
        """A cut-off reasoning block contains no answer, so it must not leak."""
        self.assertEqual(strip_thinking("<think>\nThe user wants me to identify"), "")

    def test_passes_plain_answers_through(self) -> None:
        self.assertEqual(strip_thinking("  Clip, Sponge  "), "Clip, Sponge")

    def test_drops_an_unmatched_closing_tag(self) -> None:
        self.assertEqual(strip_thinking("reasoning without opener</think>\n\nyes"), "yes")


class CombineCountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tiles_5 = {f"tile:{n}": v for n, v in zip(TILE_NAMES, (2, 1, 2, 0))}

    def test_whole_rule_ignores_the_extra_views(self) -> None:
        views = {"whole": 3, "hflip": 6, **self.tiles_5}
        self.assertEqual(combine_counts(views, "whole"), 3)

    def test_tiles_rule_sums_the_quadrants(self) -> None:
        self.assertEqual(combine_counts({"whole": 3, **self.tiles_5}, "tiles"), 5)

    def test_vote_takes_the_modal_view(self) -> None:
        views = {"whole": 3, "hflip": 4, "vflip": 4, "zoom": 2}
        self.assertEqual(combine_counts(views, "vote"), 4)

    def test_vote_never_invents_an_unobserved_value(self) -> None:
        """A tie must resolve to one of the observed counts, not their mean."""
        views = {"whole": 2, "hflip": 4}
        self.assertIn(combine_counts(views, "vote"), {2, 4})

    def test_max_rule_prefers_the_tile_sum_when_it_is_higher(self) -> None:
        """Tiling exists to break undercounting, so it wins ties upward."""
        views = {"whole": 3, "hflip": 3, **self.tiles_5}
        self.assertEqual(combine_counts(views, "max_whole_tiles"), 5)

    def test_max_rule_keeps_the_vote_when_tiles_undercount(self) -> None:
        views = {"whole": 4, "hflip": 4, **{f"tile:{n}": 0 for n in TILE_NAMES}}
        self.assertEqual(combine_counts(views, "max_whole_tiles"), 4)

    def test_tiles_if_high_leaves_low_counts_alone(self) -> None:
        views = {"whole": 1, "hflip": 1, **{f"tile:{n}": v for n, v in zip(TILE_NAMES, (1, 1, 0, 0))}}
        self.assertEqual(combine_counts(views, "tiles_if_high", high_threshold=4), 1)

    def test_tiles_if_high_takes_over_above_the_threshold(self) -> None:
        views = {"whole": 3, "hflip": 3, **{f"tile:{n}": 2 for n in TILE_NAMES}}
        self.assertEqual(combine_counts(views, "tiles_if_high", high_threshold=4), 8)

    def test_partial_tiles_are_not_summed(self) -> None:
        """Three of four tiles is not a count of the frame — fall back."""
        views = {"whole": 4, "tile:top-left": 1, "tile:top-right": 1, "tile:bottom-left": 1}
        self.assertEqual(combine_counts(views, "tiles"), 4)

    def test_unknown_rule_raises(self) -> None:
        with self.assertRaises(ValueError):
            combine_counts({"whole": 1}, "median")

    def test_every_declared_rule_is_implemented(self) -> None:
        views = {"whole": 2, "hflip": 2, "vflip": 3, "zoom": 2, **self.tiles_5}
        for rule in COMBINE_RULES:
            self.assertIsInstance(combine_counts(views, rule), int, rule)


class CombineClassSetsTest(unittest.TestCase):
    def test_unions_rather_than_sums(self) -> None:
        views = {"tile:top-left": frozenset({"Clip"}), "tile:top-right": frozenset({"Clip", "Sponge"})}
        self.assertEqual(combine_class_sets(views), frozenset({"Clip", "Sponge"}))

    def test_none_is_dropped_when_another_view_saw_something(self) -> None:
        views = {"whole": frozenset({"None"}), "hflip": frozenset({"Clip"})}
        self.assertEqual(combine_class_sets(views), frozenset({"Clip"}))

    def test_all_empty_stays_empty(self) -> None:
        self.assertEqual(combine_class_sets({"whole": frozenset({"None"})}), frozenset())


class ViewTest(unittest.TestCase):
    def setUp(self) -> None:
        from PIL import Image

        self.tmp = Path(__file__).resolve().parent / "_v2_view_probe.png"
        Image.new("RGB", (960, 540), "black").save(self.tmp)

    def tearDown(self) -> None:
        self.tmp.unlink(missing_ok=True)

    def test_whole_frame_is_always_first(self) -> None:
        views = build_views(str(self.tmp))
        self.assertEqual(views[0].view_id, "whole")

    def test_four_tiles_are_produced_and_upscaled(self) -> None:
        views = build_views(str(self.tmp), tiles=True, augment=False)
        tiles = [v for v in views if v.kind == "tile"]
        self.assertEqual(len(tiles), 4)
        for tile in tiles:
            # A raw 480x270 quadrant would carry fewer tokens than the whole
            # frame; upscaling is the point of tiling here.
            self.assertGreater(tile.image.size[0] * tile.image.size[1], 480 * 270)

    def test_tiles_can_be_disabled(self) -> None:
        views = build_views(str(self.tmp), tiles=False, augment=False)
        self.assertEqual([v.view_id for v in views], ["whole"])

    def test_tile_instruction_states_the_centre_rule_and_the_quadrant(self) -> None:
        views = build_views(str(self.tmp), tiles=True, augment=False)
        tile = next(v for v in views if v.kind == "tile")
        instr = build_view_instruction(
            "How many Clips appear in this frame?",
            CountingQuestion(CountMode.TARGET, "Clip"),
            tile,
        )
        self.assertIn("centre lies inside this crop", instr)
        self.assertIn(tile.tile_name, instr)
        self.assertIn("Clip", instr)

    def test_whole_view_keeps_the_original_question(self) -> None:
        views = build_views(str(self.tmp), tiles=False, augment=False)
        question = "How many Clips appear in this frame? Please provide a number."
        instr = build_view_instruction(question, CountingQuestion(CountMode.TARGET, "Clip"), views[0])
        self.assertIn(question, instr)

    def test_class_mode_asks_for_names_not_a_number(self) -> None:
        views = build_views(str(self.tmp), tiles=True, augment=False)
        tile = next(v for v in views if v.kind == "tile")
        instr = build_view_instruction("How many classes?", CountingQuestion(CountMode.CLASSES), tile)
        self.assertIn("Name only classes", instr)
        self.assertNotIn("non-negative integer", instr)


if __name__ == "__main__":
    unittest.main()
