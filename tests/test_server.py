"""Backend failure paths (no network, no keys): consent, empty audio, too long, unsupported card, replay, static.
Runs under pytest or unittest."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server")); sys.path.insert(0, str(ROOT / "src"))

import app as server  # noqa: E402


def form(fields, audio=None):
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if audio is not None:
        out += f'--{b}\r\nContent-Disposition: form-data; name="audio"; filename="a.webm"\r\nContent-Type: audio/webm\r\n\r\n'.encode() + audio + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


@unittest.skipUnless((ROOT / "app" / "data" / "cards.json").exists(), "run export_app_data first")
class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = {k: os.environ.pop(k, None) for k in ("ELEVENLABS_API_KEY", "GEMINI_API_KEY")}
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.card = next(iter(server.load_cards()))

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        for k, v in cls.saved.items():
            if v:
                os.environ[k] = v

    def call(self, path, body=None, ctype=None):
        req = urllib.request.Request(self.base + path, data=body, headers={"Content-Type": ctype} if ctype else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def live(self, fields, audio=b"x" * 2000):
        body, ct = form(fields, audio)
        s, b = self.call("/api/live", body, ct)
        return s, json.loads(b)

    def test_cards_have_facts_and_no_replay_payload(self):
        s, b = self.call("/api/cards")
        cards = json.loads(b)
        self.assertEqual(s, 200); self.assertTrue(3 <= len(cards) <= 5)
        self.assertTrue(all(cd["facts"] for cd in cards)); self.assertTrue(all("replay" not in cd for cd in cards))

    def test_replay(self):
        s, b = self.call(f"/api/replay/{self.card}")
        t = json.loads(b)
        self.assertEqual(s, 200); self.assertEqual(t["mode"], "REPLAY"); self.assertIn("facts", t)

    def test_no_consent(self):
        s, j = self.live({"card_id": self.card, "consent": "false", "duration_ms": "3000"})
        self.assertEqual((s, j["error"]), (400, "no_consent"))

    def test_empty_audio(self):
        s, j = self.live({"card_id": self.card, "consent": "true", "duration_ms": "3000"}, audio=b"")
        self.assertEqual(s, 400)

    def test_too_long(self):
        s, j = self.live({"card_id": self.card, "consent": "true", "duration_ms": "25000"})
        self.assertEqual((s, j["error"]), (400, "too_long"))

    def test_unsupported_card(self):
        s, j = self.live({"card_id": "xx_0", "consent": "true", "duration_ms": "3000"})
        self.assertEqual((s, j["error"]), (404, "unsupported"))

    def test_missing_key_degrades(self):
        s, j = self.live({"card_id": self.card, "consent": "true", "duration_ms": "3000"})
        self.assertEqual((s, j["error"]), (503, "no_api_key"))

    def test_static_and_no_traversal(self):
        self.assertEqual(self.call("/")[0], 200)
        self.assertEqual(self.call("/data/app_data.json")[0], 200)
        self.assertEqual(self.call("/../config.json")[0], 404)


if __name__ == "__main__":
    unittest.main()
