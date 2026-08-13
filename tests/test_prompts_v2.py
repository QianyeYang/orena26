from __future__ import annotations

import unittest

from src.prompts import (
    FO_DESCRIPTIONS,
    FO_NAMES,
    PROMPT_STRATEGIES,
    SYSTEM_PROMPT,
    build_instruction,
    fo_class_is_multi,
    system_prompt,
    v2_system_prompt,
)

# Verbatim official FRAME questions, one per fo_class template.
SINGLE_ANSWER_QUESTIONS = (
    "There is one surgical foreign object visible in the frame. What surgical "
    "foreign object is visible in this video frame? Please provide a class name.",
    "Which of the visible foreign objects has its centre closest to the centre "
    "of the image? Please provide a class name.",
    "What class is the foreign object located in the bottom/left relative to "
    "the image center? Please provide a class name.",
)
MULTI_ANSWER_QUESTIONS = (
    "List all foreign objects that are visible in this video frame. Please "
    "provide the class names or answer with none.",
    "Which combination of foreign object classes is visible in this frame? "
    "Please provide the class names or answer with none.",
)
POSITIONAL_ENUMERATION_QUESTION = (
    "At timepoint 00:09:39 please provide all relative central positions of "
    "foreign objects present in the frame. Please provide the answer in the "
    "following format: “number. object type: quadrant”, where number "
    "represents an enumeration starting with 1, object type is the type of the "
    "foreign object and quadrant is one of the following options: top/left, "
    "top/right, bottom/left, bottom/right. Respond “none” in case there "
    "are no foreign objects present at timepoint 00:09:39. For example: "
    "1. Sponge: top/left 2. Sponge: top/right 3. Needle: bottom/left"
)


class BackwardsCompatibilityTest(unittest.TestCase):
    """v2 is additive: every existing caller uses positional defaults."""

    def test_default_strategy_prompts_are_unchanged(self) -> None:
        self.assertEqual(system_prompt("frame"), SYSTEM_PROMPT)
        instr, opts = build_instruction("How many Clips appear?", "number")
        self.assertIn("single non-negative integer (digits only)", instr)
        self.assertIsNone(opts)

    def test_procedure_type_is_ignored_outside_v2(self) -> None:
        self.assertEqual(
            system_prompt("frame", procedure_type="Laparoscopic Cholecystectomy"),
            SYSTEM_PROMPT,
        )

    def test_v2_is_registered(self) -> None:
        self.assertIn("v2", PROMPT_STRATEGIES)


class V2SystemPromptTest(unittest.TestCase):
    def test_names_the_row_procedure_not_a_hardcoded_one(self) -> None:
        prompt = v2_system_prompt("frame", "Laparoscopic Cholecystectomy")
        self.assertIn("Laparoscopic Cholecystectomy", prompt)
        self.assertNotIn("colorectal", prompt.lower())

    def test_falls_back_rather_than_guessing(self) -> None:
        for missing in (None, "", "   "):
            prompt = v2_system_prompt("frame", missing)
            self.assertIn("minimally invasive", prompt)
            self.assertNotIn("colorectal", prompt.lower())

    def test_lists_every_canonical_class_including_untrained_ones(self) -> None:
        prompt = v2_system_prompt("frame", "Sigmoid Resection")
        for name in FO_NAMES:
            self.assertIn(name, prompt)
        # Absent from both official training sets, so the fine-tune can only
        # emit them if the prompt keeps them in scope.
        self.assertIn("Mesh", prompt)
        self.assertIn("Absorbable Hemostatic Agent", prompt)

    def test_descriptions_cover_the_canonical_roster_exactly(self) -> None:
        self.assertEqual(tuple(n for n, _ in FO_DESCRIPTIONS), tuple(FO_NAMES))

    def test_window_tracks_keep_the_timestamp_contract(self) -> None:
        for track in ("segment", "procedure"):
            prompt = v2_system_prompt(track, "Rectal Resection")
            self.assertIn("hh:mm:ss", prompt)
        self.assertNotIn("hh:mm:ss", v2_system_prompt("frame", "Rectal Resection"))


class V2ArityTest(unittest.TestCase):
    def test_single_answer_templates_demand_one_class(self) -> None:
        for question in SINGLE_ANSWER_QUESTIONS:
            self.assertFalse(fo_class_is_multi(question), question)
            instr, _ = build_instruction(question, "fo_class", "v2")
            self.assertIn("exactly ONE", instr)
            self.assertNotIn("Separate multiple names", instr)

    def test_multi_answer_templates_allow_a_class_set(self) -> None:
        for question in MULTI_ANSWER_QUESTIONS:
            self.assertTrue(fo_class_is_multi(question), question)
            instr, _ = build_instruction(question, "fo_class", "v2")
            self.assertIn("Separate multiple names", instr)
            self.assertNotIn("exactly ONE", instr)

    def test_direct_strategy_still_says_one_or_more_everywhere(self) -> None:
        """Documents the defect v2 fixes, so a regression is visible."""
        instr, _ = build_instruction(SINGLE_ANSWER_QUESTIONS[0], "fo_class")
        self.assertIn("one or more", instr)


class V2SelfFormattingTest(unittest.TestCase):
    def test_enumeration_question_keeps_its_own_contract(self) -> None:
        instr, _ = build_instruction(POSITIONAL_ENUMERATION_QUESTION, "open_ended", "v2")
        self.assertIn("Follow that format exactly", instr)
        self.assertNotIn("in a few words", instr)

    def test_plain_open_ended_still_asks_for_brevity(self) -> None:
        instr, _ = build_instruction(
            "Which foreign object class is partially occluded by an instrument "
            "in this frame?",
            "open_ended",
            "v2",
        )
        self.assertIn("in a few words", instr)

    def test_every_v2_instruction_states_the_300_character_limit(self) -> None:
        for question in (POSITIONAL_ENUMERATION_QUESTION, "Why is that clip there?"):
            instr, _ = build_instruction(question, "open_ended", "v2")
            self.assertIn("300 characters", instr)

    def test_direct_strategy_contradicts_the_enumeration_question(self) -> None:
        """Documents the defect v2 fixes."""
        instr, _ = build_instruction(POSITIONAL_ENUMERATION_QUESTION, "open_ended")
        self.assertIn("in a few words", instr)


class V2StrictFormatTest(unittest.TestCase):
    def test_exact_match_formats_stay_terse(self) -> None:
        binary, _ = build_instruction("Do Clips and Sponges co-occur?", "binary", "v2")
        self.assertIn("exactly one word: yes or no", binary)
        number, _ = build_instruction("How many Clips appear?", "number", "v2")
        self.assertIn("digits only", number)

    def test_multiple_choice_returns_parsed_options(self) -> None:
        question = (
            "Where is the center of the Needle located relative to the image "
            "center in this frame? Please select one answer: top/left; "
            "top/right; bottom/left; bottom/right"
        )
        instr, options = build_instruction(question, "multiple_choice", "v2")
        self.assertEqual(options, ["top/left", "top/right", "bottom/left", "bottom/right"])
        self.assertIn("EXACTLY ONE", instr)

    def test_unknown_strategy_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_instruction("q", "binary", "v9")
        with self.assertRaises(ValueError):
            system_prompt("frame", "v9")


class SystemVariantTest(unittest.TestCase):
    """The v2 family differs only in the system prompt — that is the A/B axis.

    Measured on Qwen3-VL-4B zero-shot over 6,252 rows: the v2 exclusion clause
    drove the binary "yes" rate to 2.1% (v1: 24.3%, truth: 44.1%) and put 67% of
    counting mass on 0, which is never a correct FRAME answer. v3 keeps the
    definition and the roster but drops the exclusion list and the descriptions.
    """

    QUESTION = "How many Clips appear in this frame?"

    def test_instruction_is_identical_across_the_family(self) -> None:
        baseline, _ = build_instruction(self.QUESTION, "number", "v2")
        for strategy in ("v2-noscope", "v2-nodesc", "v3"):
            with self.subTest(strategy=strategy):
                instr, _ = build_instruction(self.QUESTION, "number", strategy)
                self.assertEqual(instr, baseline)

    def test_noscope_drops_only_the_exclusion_paragraph(self) -> None:
        full = system_prompt("frame", "v2", procedure_type="Rectal Resection")
        noscope = system_prompt("frame", "v2-noscope", procedure_type="Rectal Resection")
        self.assertIn("not foreign objects", full)
        self.assertNotIn("not foreign objects", noscope)
        self.assertIn("blood-soaked", noscope)  # descriptions survive

    def test_nodesc_drops_only_the_descriptions(self) -> None:
        nodesc = system_prompt("frame", "v2-nodesc", procedure_type="Rectal Resection")
        self.assertIn("not foreign objects", nodesc)
        self.assertNotIn("blood-soaked", nodesc)

    def test_v3_drops_both_but_keeps_every_class_name(self) -> None:
        v3 = system_prompt("frame", "v3", procedure_type="Laparoscopic Cholecystectomy")
        self.assertNotIn("not foreign objects", v3)
        self.assertNotIn("blood-soaked", v3)
        self.assertIn("Several may be present at once", v3)
        self.assertIn("Laparoscopic Cholecystectomy", v3)
        for name in ("Mesh", "Absorbable Hemostatic Agent", "Gallstone"):
            self.assertIn(name, v3)

    def test_v3_is_much_shorter_than_v2(self) -> None:
        v2 = system_prompt("frame", "v2", procedure_type="Rectal Resection")
        v3 = system_prompt("frame", "v3", procedure_type="Rectal Resection")
        self.assertLess(len(v3), len(v2) / 2)

    def test_v1_default_is_still_byte_identical(self) -> None:
        from src.prompts import SYSTEM_PROMPT

        self.assertEqual(system_prompt("frame"), SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
