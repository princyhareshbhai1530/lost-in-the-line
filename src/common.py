"""Shared paths, settings, caching, HTTP and text utilities for the Broken Telephone pipeline.

Modes
-----
LIVE (default): calls ElevenLabs Scribe v2 and Gemini; results go to cache/ and results/.
DRY RUN (BT_MOCK=1 or --mock): deterministic fake providers, no network, no credits.
  Everything is written to cache_mock/ and results_mock/ and is labelled FAKE so it can
  never be confused with real numbers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

DEFAULT_PACK = Path.home() / "Downloads" / "Mississauga_Unheard_ElevenLabs_Track" / "Mississauga_Unheard_ElevenLabs_Track"
PACK = Path(os.environ.get("BT_DATA", CONFIG.get("data_pack") or DEFAULT_PACK))
FLEURS_DIR = PACK / "02_speech_home_languages_FLEURS"
CENSUS_DIR = PACK / "01_people_census"
CITY311_DIR = PACK / "05_city_311_requests"


def is_mock() -> bool:
    return os.environ.get("BT_MOCK") == "1" or "--mock" in sys.argv


def cache_dir() -> Path:
    d = ROOT / ("cache_mock" if is_mock() else "cache")
    d.mkdir(exist_ok=True)
    return d


def results_dir() -> Path:
    d = ROOT / ("results_mock" if is_mock() else "results")
    d.mkdir(exist_ok=True)
    return d


def data_status() -> str:
    return "DRY_RUN_FAKE" if is_mock() else "REAL"


def stage_tag(language_first: bool = False) -> str:
    return "langfirst" if language_first else "autodetect"


# --------------------------------------------------------------------------- data

def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0].keys()) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def crosswalk() -> list[dict]:
    return read_csv(PACK / "language_crosswalk.csv")


def fleurs_clips(langs: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """All FLEURS clips with the answer key attached (true English = en_us text of same fleurs_id)."""
    rows = read_csv(FLEURS_DIR / "metadata.csv")
    english = {r["fleurs_id"]: r["raw_transcription"] for r in rows if r["lang_config"] == "en_us"}
    out = []
    for r in rows:
        if langs and r["lang_config"] not in langs and r["language"] not in langs:
            continue
        r = dict(r)
        r["id"] = f'{r["lang_config"]}_{r["fleurs_id"]}'
        r["true_english"] = english[r["fleurs_id"]]
        r["audio_path"] = str(FLEURS_DIR / r["file"])
        out.append(r)
    if limit:
        by_lang: dict[str, list] = {}
        for r in out:
            by_lang.setdefault(r["lang_config"], []).append(r)
        out = [r for v in by_lang.values() for r in sorted(v, key=lambda x: int(x["fleurs_id"]))[:limit]]
    return out


def parse_langs(argv: list[str]) -> tuple[list[str] | None, int | None]:
    langs, limit = None, None
    if "--langs" in argv:
        langs = argv[argv.index("--langs") + 1].split(",")
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--signal" in argv:  # phase 1 gate: 40 clips across Urdu, Punjabi, Tagalog, English
        langs, limit = CONFIG["signal_test_langs"], CONFIG["signal_test_clips_per_lang"]
    return langs, limit


# --------------------------------------------------------------------------- cache

def key_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def cache_get(kind: str, key: str):
    p = cache_dir() / kind / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def cache_put(kind: str, key: str, value) -> None:
    p = cache_dir() / kind / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- http

def require_key(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"{name} is not set. Set it in your shell (never commit it), or use --mock for a dry run.")
    return v


def http_json(url: str, *, headers: dict, body: bytes | None = None, method: str = "POST",
              retries: int = 5, timeout: int = 180) -> dict:
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            if e.code == 429 and re.search(r"PerDay|per day|daily", detail, re.I):
                sys.exit("Daily API quota reached. Everything done so far is cached: rerun the same command "
                         "after the quota resets (or switch key/model).\n" + detail[:300])
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = e.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit() else (30 if e.code == 429 else 2 ** attempt * 2))
                continue
            raise RuntimeError(f"HTTP {e.code} from {url.split('?')[0]}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            raise


def multipart(fields: dict, file_field: str, file_path: Path, mime: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- text

def normalise(text: str) -> str:
    """NFKC, lower-case, drop punctuation/symbols, collapse whitespace. Script-agnostic."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalise(ref).replace(" ", "")), list(normalise(hyp).replace(" ", ""))
    return levenshtein(r, h) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref).split(), normalise(hyp).split()
    return levenshtein(r, h) / max(len(r), 1)


SCRIPT_RANGES = [
    ("Arabic", 0x0600, 0x06FF), ("Arabic", 0x0750, 0x077F), ("Arabic", 0xFB50, 0xFDFF), ("Arabic", 0xFE70, 0xFEFF),
    ("Devanagari", 0x0900, 0x097F), ("Gurmukhi", 0x0A00, 0x0A7F), ("Gujarati", 0x0A80, 0x0AFF),
    ("Tamil", 0x0B80, 0x0BFF), ("Han", 0x4E00, 0x9FFF), ("Han", 0x3400, 0x4DBF),
    ("Shahmukhi/Arabic", 0x08A0, 0x08FF),
]


def dominant_script(text: str) -> str:
    counts: dict[str, int] = {}
    for c in text or "":
        if not c.isalpha():
            continue
        cp = ord(c)
        name = next((n for n, lo, hi in SCRIPT_RANGES if lo <= cp <= hi), None)
        if name is None:
            name = "Latin" if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF else "Other"
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else "None"


# Common characters that exist only in Simplified Chinese (their Traditional forms differ).
SIMPLIFIED_ONLY = set("们这说时国会来为对发后门见长东车书学马风问还过们个么讯让认记话语资万与业从众优传们")
TRADITIONAL_ONLY = set("們這說時國會來為對發後門見長東車書學馬風問還過個麼訊讓認記話語資萬與業從眾優傳")


def han_variant(text: str) -> str:
    s = sum(c in SIMPLIFIED_ONLY for c in text or "")
    t = sum(c in TRADITIONAL_ONLY for c in text or "")
    if s == t == 0:
        return "unknown"
    return "simplified" if s > t else "traditional"


EXPECTED_SCRIPT = {
    "ur_pk": "Arabic", "ar_eg": "Arabic", "hi_in": "Devanagari", "pa_in": "Gurmukhi", "gu_in": "Gujarati",
    "ta_in": "Tamil", "cmn_hans_cn": "Han", "yue_hant_hk": "Han", "en_us": "Latin", "es_419": "Latin",
    "fil_ph": "Latin", "pl_pl": "Latin", "pt_br": "Latin", "vi_vn": "Latin",
}

# Scribe reports ISO 639-3 codes; map to the pack's iso639_1 codes.
ISO3_TO_1 = {
    "eng": "en", "urd": "ur", "ara": "ar", "arz": "ar", "cmn": "zh", "zho": "zh", "pan": "pa", "pol": "pl",
    "yue": "yue", "fil": "tl", "tgl": "tl", "spa": "es", "tam": "ta", "vie": "vi", "hin": "hi", "por": "pt",
    "guj": "gu", "ben": "bn",
}


def lang_code_1(code: str | None) -> str:
    c = (code or "").lower()
    return ISO3_TO_1.get(c, c)


def log(msg: str) -> None:
    tag = "[DRY RUN - FAKE] " if is_mock() else ""
    print(f"{tag}{msg}", flush=True)
