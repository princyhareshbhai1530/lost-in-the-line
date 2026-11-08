"""Unit tests (stdlib unittest; no keys, no network). Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import common as c  # noqa: E402
import meaning as m  # noqa: E402
from score import break_station  # noqa: E402


class TextMetrics(unittest.TestCase):
    def test_cer_ignores_case_punct_spaces(self):
        self.assertEqual(c.cer("Hello, World.", "hello world"), 0.0)

    def test_cer_counts_edits(self):
        self.assertAlmostEqual(c.cer("abcd", "abxd"), 0.25)

    def test_wer(self):
        self.assertAlmostEqual(c.wer("the cat sat", "the hat sat"), 1 / 3)

    def test_cer_chinese_has_no_word_boundaries(self):
        self.assertAlmostEqual(c.cer("我们去公园", "我们去公圆"), 0.2)

    def test_empty_hypothesis_is_full_error(self):
        self.assertEqual(c.cer("abc", ""), 1.0)

    def test_scripts(self):
        self.assertEqual(c.dominant_script("یہ اردو ہے"), "Arabic")
        self.assertEqual(c.dominant_script("यह हिंदी है"), "Devanagari")
        self.assertEqual(c.dominant_script("ਪੰਜਾਬੀ"), "Gurmukhi")
        self.assertEqual(c.dominant_script("hello"), "Latin")
        self.assertEqual(c.dominant_script("Tiếng Việt"), "Latin")

    def test_han_variant(self):
        self.assertEqual(c.han_variant("我們這裡"), "traditional")
        self.assertEqual(c.han_variant("我们这里"), "simplified")

    def test_scribe_codes(self):
        self.assertEqual(c.lang_code_1("urd"), "ur")
        self.assertEqual(c.lang_code_1("fil"), "tl")


class Facts(unittest.TestCase):
    def test_numbers_and_names(self):
        f = m.extract_facts("The patient had been to Nigeria, where 40 cases of Ebola occurred.")
        self.assertIn("40", f); self.assertIn("Nigeria", f); self.assertIn("Ebola", f)

    def test_sentence_initial_word_not_a_name(self):
        self.assertEqual(m.extract_facts("Soon, officers entered the yard."), [])

    def test_number_words(self):
        f = m.extract_facts("Yet eighty percent of goods were taxed.")
        self.assertIn("eighty", f)
        self.assertTrue(m.fact_present("eighty", "but 80% of the goods"))

    def test_acronym_and_initial(self):
        f = m.extract_facts("U.S. President George W. Bush welcomed the announcement.")
        self.assertIn("US President George Bush", f)

    def test_fact_lost(self):
        self.assertFalse(m.fact_present("Nigeria", "The patient had been to Niger."))


class BreakStation(unittest.TestCase):
    def test_none_when_both_high(self):
        self.assertEqual(break_station(100, 100), "none")

    def test_ear(self):
        self.assertEqual(break_station(100, 35), "ear")

    def test_translator(self):
        self.assertEqual(break_station(35, 35), "translator")


class Rubric(unittest.TestCase):
    def test_rubric(self):
        self.assertEqual(c.CONFIG["judge_rubric"], {"same": 100, "minor": 75, "changed": 35, "unrelated": 0})

    def test_empty_translation_is_unrelated(self):
        self.assertEqual(m.judge("Anything", "")["score"], 0)


class Census(unittest.TestCase):
    def test_answer_key_aligned(self):
        clips = c.fleurs_clips()
        self.assertEqual(len(clips), 560)
        self.assertEqual(len({x["lang_config"] for x in clips}), 14)
        en = {x["fleurs_id"]: x["raw_transcription"] for x in clips if x["lang_config"] == "en_us"}
        self.assertTrue(all(x["true_english"] == en[x["fleurs_id"]] for x in clips))


if __name__ == "__main__":
    unittest.main()
