"""Deterministic FAKE providers for the no-credit dry run. Never used for reported numbers.

The fake error rates are arbitrary and exist only so every downstream script has
non-trivial input. Output lives in cache_mock/ and results_mock/.
"""
from __future__ import annotations

import hashlib
import math
import random
import re

from common import CONFIG


def _rng(*parts: str) -> random.Random:
    return random.Random(int(hashlib.md5("|".join(parts).encode()).hexdigest()[:12], 16))


def _rate(lang: str, salt: str, hi: float) -> float:
    return _rng(lang, salt).uniform(0.0, hi)


def scribe(clip: dict, language_first: bool) -> dict:
    lang = clip["lang_config"]
    rate = _rate(lang, "ear", 0.25) * (0.4 if language_first else 1.0)
    r = _rng(clip["id"], "ear", str(language_first))
    chars = list(clip["raw_transcription"])
    out = [c for c in chars if not (c.isalpha() and r.random() < rate * 0.5)]
    text = "".join(out)
    wrong_lang = (not language_first) and r.random() < rate * 0.3
    code = {"ur_pk": "hin", "pa_in": "hin"}.get(lang, "xxx") if wrong_lang else clip["iso639_1"]
    return {"text": text, "language_code": code, "language_probability": round(1 - rate, 3), "mock": True}


def translate(text: str, clip: dict, source: str) -> str:
    lang = clip["lang_config"]
    words = clip["true_english"].split()
    r = _rng(clip["id"], "tr", source, text[:40])
    rate = _rate(lang, "tr", 0.2)
    if source == "heard":  # garbled input -> more loss
        rate += 1 - min(1.0, len(text) / max(1, len(clip["raw_transcription"])))
        rate += _rate(lang, "ear", 0.25) * 0.8
    out = []
    for w in words:
        if r.random() < rate * 0.6:
            continue
        if re.fullmatch(r"\d+[\d,.]*", w) and r.random() < rate:
            w = str(int(re.sub(r"\D", "", w) or 0) + 1)
        out.append(w)
    return " ".join(out) or "..."


def judge(true_en: str, cand: str) -> str:
    a, b = set(true_en.lower().split()), set(cand.lower().split())
    j = len(a & b) / max(1, len(a | b))
    return "same" if j > 0.9 else "minor" if j > 0.7 else "changed" if j > 0.3 else "unrelated"


def embed(text: str, dim: int = 256) -> list[float]:
    v = [0.0] * dim
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16) % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


_ = CONFIG
