"""Deterministic fact extractor + four-state logic. Runs under pytest or unittest."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import facts as F  # noqa: E402


def types(sentence):
    return {(f["type"], f["norm"]) for f in F.extract(sentence)}


class Extract(unittest.TestCase):
    # 10 hand-made sentences covering numbers, dates, negation and names
    def test_1_digits_and_name(self):
        self.assertEqual(types("Please send 15 bags to Hurontario Street."),
                         {("number", "15"), ("name_place", "hurontario street")})

    def test_2_number_words(self):
        self.assertIn(("number", "40"), types("About forty people stayed in the yard."))

    def test_3_compound_number(self):
        self.assertIn(("number", "2000000"), types("The tower cost 2 million dollars."))

    def test_4_decimal_and_comma(self):
        t = types("The bill was 1,250.50 and the fee 3.5 percent.")
        self.assertIn(("number", "1250.5"), t); self.assertIn(("number", "3.5"), t)

    def test_5_weekday_month_year(self):
        t = types("It was cut down on Tuesday, 14 March 2021.")
        self.assertTrue({("date_time", "tuesday"), ("date_time", "march"), ("date_time", "year2021")} <= t)
        self.assertNotIn(("number", "2021"), t)

    def test_6_time(self):
        self.assertIn(("date_time", "15:00"), types("Pickup is at 3pm near the library."))
        self.assertIn(("date_time", "09:30"), types("The office opens at 9:30."))

    def test_7_negation(self):
        self.assertIn(("negation", "neg"), types("The garbage was not collected."))
        self.assertIn(("negation", "neg"), types("They didn't come."))

    def test_7b_double_negative_is_not_a_negation(self):
        self.assertNotIn(("negation", "neg"), types("It is not uncommon to be followed."))

    def test_7c_prefixed_negation_matches(self):
        self.assertTrue(F.match({"type": "negation", "norm": "neg"}, "the true number is still unknown")[0])

    def test_7d_one_of_is_not_a_number(self):
        self.assertNotIn(("number", "1"), types("It is one of the most important holidays."))

    def test_8_sentence_initial_word_not_a_name(self):
        self.assertEqual(types("Soon, officers entered the yard."), set())

    def test_9_acronym_and_initials(self):
        self.assertIn(("name_place", "us president george bush"),
                      types("U.S. President George W. Bush welcomed the announcement."))

    def test_10_may_as_verb(self):
        self.assertNotIn(("date_time", "may"), types("You may see the pyramids in the dark."))
        self.assertIn(("date_time", "may"), types("The festival starts in May."))


class Match(unittest.TestCase):
    def test_number_by_value(self):
        f = {"type": "number", "norm": "40"}
        self.assertEqual(F.match(f, "about 40 inmates")[0], True)
        self.assertEqual(F.match(f, "about forty inmates")[0], True)
        self.assertEqual(F.match(f, "about 14 inmates")[0], False)

    def test_lakh_equals_two_million(self):
        f = {"type": "number", "norm": "2000000"}
        self.assertTrue(F.match(f, "over the past 20 lakh years")[0])

    def test_negation_presence(self):
        f = {"type": "negation", "norm": "neg"}
        self.assertTrue(F.match(f, "it was never collected")[0])
        self.assertFalse(F.match(f, "it was collected")[0])

    def test_name_exact_and_fuzzy(self):
        f = {"type": "name_place", "norm": "nigeria"}
        self.assertEqual(F.match(f, "had been to Nigeria."), (True, False))
        self.assertEqual(F.match(f, "had been to Niger."), (False, False))
        g = {"type": "name_place", "norm": "galapagos"}
        self.assertEqual(F.match(g, "on the Galápagos islands"), (True, True))  # labelled fuzzy

    def test_no_candidate(self):
        self.assertEqual(F.match({"type": "number", "norm": "1"}, None), (None, False))

    def test_change_kind(self):
        f = {"type": "number", "norm": "15"}
        self.assertEqual(F.change_kind(f, "I heard 50 bags"), "changed")
        self.assertEqual(F.change_kind(f, "I heard some bags"), "missing")


class FourState(unittest.TestCase):
    def test_states(self):
        self.assertEqual(F.state(True, True), "survived")
        self.assertEqual(F.state(True, False), "asr_associated_break")
        self.assertEqual(F.state(False, False), "translation_baseline_break")
        self.assertEqual(F.state(False, True), "compensation_or_discordant")
        self.assertIsNone(F.state(True, None))

    def test_clip_priority(self):
        self.assertEqual(F.clip_state(["survived", "translation_baseline_break", "asr_associated_break"]),
                         "asr_associated_break")
        self.assertEqual(F.clip_state(["survived", "compensation_or_discordant"]), "compensation_or_discordant")
        self.assertIsNone(F.clip_state([]))

    def test_check_end_to_end(self):
        ref = "Please send 15 bags to Hurontario Street on Tuesday, not Monday."
        out = F.check(ref, t1="Please send 15 bags to Hurontario Street on Tuesday, not Monday.",
                      t2_auto="Please send 50 bags to Hurontario Street on Tuesday and Monday.")
        st = {f["type"] + ":" + f["norm"]: f["state_auto"] for f in out}
        self.assertEqual(st["number:15"], "asr_associated_break")
        self.assertEqual(st["negation:neg"], "asr_associated_break")
        self.assertEqual(st["name_place:hurontario street"], "survived")
        self.assertEqual(F.clip_state([f["state_auto"] for f in out]), "asr_associated_break")


if __name__ == "__main__":
    unittest.main()
