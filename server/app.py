"""Live demo backend (Play the Line) plus static hosting of app/. Python standard library only.

Run:  py server/app.py            (then open http://127.0.0.1:8000)
      py server/app.py --port 8080

Endpoints
  GET  /api/cards              3 to 5 prompt cards (FLEURS sentences with at least one critical fact)
  POST /api/live               multipart: audio, card_id, consent=true, duration_ms. Scribe (auto-detect), then Gemini
                               translation with the fixed prompt, then the deterministic fact check. Returns a trace.
                               The audio is never stored; only the text trace is saved to replays/.
  GET  /api/replay/{card_id}   a team-recorded verified trace (replays/verified_<card_id>.json) if present, else the
                               benchmark trace of the same sentence and language.
Keys are read from ELEVENLABS_API_KEY and GEMINI_API_KEY in this process only and never sent to the browser.
Each API step has a 12 s timeout; any failure returns an error code and the UI offers the replay.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common as c  # noqa: E402
import facts as F  # noqa: E402
from build_eval_table import ACCEPT  # noqa: E402
from translate import PROMPT  # noqa: E402

APP = ROOT / "app"
REPLAYS = ROOT / "replays"
STEP_TIMEOUT = 12
MAX_AUDIO_BYTES = 3_000_000
MAX_MS = 20_500
ISO_TO_CONFIG = {r["iso639_1"]: r["fleurs_config"] for r in c.crosswalk() if r["fleurs_config"]}
ISO1_TO_3 = {"ur": "urd", "pa": "pan", "hi": "hin", "ta": "tam", "es": "spa", "gu": "guj", "ar": "ara", "zh": "cmn", "tl": "fil"}


class StepError(Exception):
    def __init__(self, code: str, step: str, message: str, status: int = 502):
        super().__init__(message)
        self.code, self.step, self.message, self.status = code, step, message, status


def load_cards() -> dict:
    p = APP / "data" / "cards.json"
    return {cd["card_id"]: cd for cd in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}


def post(url: str, headers: dict, body: bytes, step: str) -> dict:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=STEP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        code = "quota" if e.code == 429 else "api_error"
        raise StepError(code, step, f"{step} returned HTTP {e.code}") from None
    except (socket.timeout, TimeoutError):
        raise StepError("timeout", step, f"{step} took longer than {STEP_TIMEOUT} s", 504) from None
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)):
            raise StepError("timeout", step, f"{step} took longer than {STEP_TIMEOUT} s", 504) from None
        raise StepError("network", step, f"{step} could not be reached") from None


def scribe(audio: bytes, filename: str, mime: str) -> dict:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise StepError("no_api_key", "scribe", "ELEVENLABS_API_KEY is not set on the server", 503)
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="model_id"\r\n\r\n{c.CONFIG["scribe_model"]}\r\n'.encode(),
             f'--{b}\r\nContent-Disposition: form-data; name="tag_audio_events"\r\n\r\nfalse\r\n'.encode(),
             f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode(),
             audio, f"\r\n--{b}--\r\n".encode()]
    return post("https://api.elevenlabs.io/v1/speech-to-text",
                {"xi-api-key": key, "Content-Type": f"multipart/form-data; boundary={b}"}, b"".join(parts), "scribe")


def translate(text: str, language: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise StepError("no_api_key", "translate", "GEMINI_API_KEY is not set on the server", 503)
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT.format(language=language, text=text)}]}],
            "generationConfig": {"temperature": c.CONFIG["temperature"]}}
    resp = post(f"https://generativelanguage.googleapis.com/v1beta/models/{c.CONFIG['gemini_model']}:generateContent",
                {"x-goog-api-key": key, "Content-Type": "application/json"}, json.dumps(body).encode(), "translate")
    try:
        return "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"] if not p.get("thought")).strip()
    except (KeyError, IndexError):
        raise StepError("api_error", "translate", "translation response was empty") from None


def live_trace(card: dict, audio: bytes, filename: str, mime: str) -> dict:
    heard = scribe(audio, filename, mime)
    text = (heard.get("text") or "").strip()
    if not text:
        raise StepError("empty_transcript", "scribe", "Scribe heard no speech", 422)
    det = (heard.get("language_code") or "").lower()
    english = translate(text, card["language"])
    cfg = ISO_TO_CONFIG.get(card["iso"])
    script_ok = c.dominant_script(text) == c.EXPECTED_SCRIPT.get(cfg) if cfg else None
    out_facts = []
    for f in card["facts"]:
        present, fuzzy = F.match(f, english)
        status = ("uncertain" if fuzzy else "survived") if present else F.change_kind(f, english)
        out_facts.append({"type": f["type"], "text": f["text"], "status": status})
    return {"mode": "LIVE", "source": "your recording", "card_id": card["card_id"], "language": card["language"],
            "expected_lang": card["iso"], "detected_lang": det, "lang_correct": det in ACCEPT.get(card["iso"], {card["iso"]}),
            "script_ok": script_ok, "asr_text": text, "english": english, "english_ref": card["english_ref_raw"],
            "facts": out_facts, "cer": round(c.cer(card["source_text"], text), 3), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}


class Handler(BaseHTTPRequestHandler):
    server_version = "LostInTheLine/1.0"

    def log_message(self, fmt, *args):  # quiet, and never logs headers (keys are server-side anyway)
        sys.stderr.write("%s %s\n" % (self.command, self.path.split("?")[0]))

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/cards":
            return self.send_json([{k: v for k, v in cd.items() if k != "replay"} for cd in load_cards().values()])
        if path.startswith("/api/replay/"):
            cid = path.rsplit("/", 1)[-1]
            cards = load_cards()
            if cid not in cards:
                return self.send_json({"error": "unsupported", "message": "unknown card"}, 404)
            verified = REPLAYS / f"verified_{cid}.json"
            if verified.exists():
                t = json.loads(verified.read_text(encoding="utf-8"))
                t["mode"] = "REPLAY"
                t.setdefault("source", "team-recorded verified example")
                return self.send_json(t)
            return self.send_json(cards[cid]["replay"])
        return self.static(path)

    def static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (APP / rel).resolve()
        if APP.resolve() not in target.parents and target != APP.resolve() or not target.is_file():
            self.send_response(404); self.end_headers(); return
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".flac":
            ctype = "audio/flac"
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/live":
            return self.send_json({"error": "not_found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_AUDIO_BYTES + 100_000:
                return self.send_json({"error": "bad_audio", "message": "audio missing or too large"}, 400)
            body = self.rfile.read(n)
            msg = BytesParser(policy=email_policy).parsebytes(
                b"Content-Type: " + (self.headers.get("Content-Type") or "").encode() + b"\r\n\r\n" + body)
            fields, audio, fname, mime = {}, None, "live.webm", "audio/webm"
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "audio":
                    audio = part.get_payload(decode=True)
                    fname = part.get_filename() or fname
                    mime = part.get_content_type() or mime
                elif name:
                    fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8", "replace").strip()
            if fields.get("consent") != "true":
                return self.send_json({"error": "no_consent", "message": "consent is required"}, 400)
            if not audio:
                return self.send_json({"error": "empty_audio", "message": "no audio received"}, 400)
            if len(audio) > MAX_AUDIO_BYTES or int(fields.get("duration_ms") or 0) > MAX_MS:
                return self.send_json({"error": "too_long", "message": "recordings are limited to 20 seconds"}, 400)
            card = load_cards().get(fields.get("card_id", ""))
            if not card:
                return self.send_json({"error": "unsupported", "message": "unknown card or language"}, 404)
            trace = live_trace(card, audio, fname, mime)
            REPLAYS.mkdir(exist_ok=True)
            (REPLAYS / f"live_{time.strftime('%Y%m%d_%H%M%S')}_{card['card_id']}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
            return self.send_json(trace)
        except StepError as e:
            return self.send_json({"error": e.code, "step": e.step, "message": e.message}, e.status)
        except Exception as e:  # never crash the demo
            return self.send_json({"error": "server_error", "message": type(e).__name__}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    keys = {k: bool(os.environ.get(k)) for k in ("ELEVENLABS_API_KEY", "GEMINI_API_KEY")}
    print(f"Lost in the Line: http://{a.host}:{a.port}  (live mode keys: {keys}; without keys the UI uses replays)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
