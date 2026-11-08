"""Batched Gemini path, offline: fake Gemini, temporary cache/results (real cache never touched)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import common as c  # noqa: E402
import gemini  # noqa: E402
import meaning  # noqa: E402
import score  # noqa: E402


class FakeGemini:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, json_mode=False):
        self.calls += 1
        items = [json.loads(line) for line in prompt.split("Items:\n", 1)[1].splitlines() if line.strip()]
        if "reference" in items[0]:
            return json.dumps([{"id": it["id"], "label": "same", "reason": "fake"} for it in items])
        return json.dumps([{"id": it["id"], "english": "EN " + it["text"][:10]} for it in items])

    def embed_batch(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


class Batching(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = (c.cache_dir, c.results_dir, gemini.generate, gemini.embed_batch)
        c.cache_dir = lambda: self.tmp / "cache"
        c.results_dir = lambda: (self.tmp / "results").mkdir(exist_ok=True) or self.tmp / "results"
        self.fake = FakeGemini()
        gemini.generate, gemini.embed_batch = self.fake.generate, self.fake.embed_batch
        self.clips = c.fleurs_clips(["ur_pk", "en_us"], 12)
        for cl in self.clips:  # pretend Scribe already ran
            c.cache_put("scribe_autodetect", cl["id"], {"text": cl["raw_transcription"][:-3], "language_code": "urd"})

    def tearDown(self):
        c.cache_dir, c.results_dir, gemini.generate, gemini.embed_batch = self.saved

    def test_batched_run_uses_few_calls_and_scores_every_clip(self):
        rows = score.run(["ur_pk", "en_us"], 12, False)
        self.assertEqual(len(rows), 24)
        # 24 Urdu texts -> 3 translation batches; 48 pairs minus English identity dupes -> few judge batches; 1 embed batch
        self.assertLessEqual(self.fake.calls, 10)
        ur = [r for r in rows if r["lang_config"] == "ur_pk"]
        self.assertTrue(all(r["translated_from_true"].startswith("EN ") for r in ur))
        self.assertTrue(all(r["meaning_heard"] == 100 for r in rows))
        before = self.fake.calls
        score.run(["ur_pk", "en_us"], 12, False)  # second run: everything cached
        self.assertEqual(self.fake.calls, before)

    def test_bad_chunk_is_split_not_sent_one_by_one(self):
        calls = []

        def call(chunk):
            calls.append(len(chunk))
            return None if len(chunk) > 25 else chunk  # model chokes on big chunks

        gemini.run_batched(list(range(100)), 100, call)
        self.assertEqual(sum(n for n in calls if n <= 25), 100)  # every item handled exactly once
        self.assertLessEqual(len(calls), 7)  # 100 -> 50,50 -> 25x4

    def test_malformed_batch_falls_back(self):
        orig = gemini.generate
        gemini.generate = lambda p, json_mode=False: '[{"id": 1, "label": "same"}]'  # wrong count
        try:
            self.assertIsNone(gemini.generate_json_list("p", 2))
        finally:
            gemini.generate = orig


if __name__ == "__main__":
    unittest.main()
