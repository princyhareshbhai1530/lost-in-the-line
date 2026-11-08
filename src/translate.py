"""Station 3, the Translator: Gemini translates home-language text to English, twice.

T1 = the true reference transcript   -> isolates translation loss
T2 = what Scribe heard               -> total loss (ear + translator)
English clips skip translation (the City desk reads them directly).

Usage: python src/translate.py [--signal|--langs ...] [--language-first] [--mock]
"""
from __future__ import annotations

import json
import sys

import common as c
import mock
from transcribe import transcribe

PROMPT = (
    "Translate the following {language} text into English. It is a message a resident said to a city "
    "phone line. Translate faithfully: keep every number, date and name. Output only the English "
    "translation, nothing else.\n\nText:\n{text}"
)


BATCH_PROMPT = (
    "Translate each numbered item below into English. Each item gives its source language and the text. "
    "Each item is a separate message a resident said to a city phone line: translate every item independently, "
    "never use one item to fix another. Translate faithfully: keep every number, date and name; do not correct "
    "or complete the text. Respond with JSON only: a list of objects "
    '{{"id": <number>, "english": "<translation>"}}, one per item, same ids.\n\nItems:\n{items}'
)


def _key(clip: dict, text: str) -> str:
    if c.is_mock():
        return c.key_hash("translate", "mock", clip["language"], text)
    import gemini
    return c.key_hash("translate", c.CONFIG["gemini_model"], gemini.settings_tag(), clip["language"], text)


def pending(clips: list[dict], language_first: bool = False) -> list[tuple]:
    """Unique (clip, text, source) translations not cached yet, grouped by language order."""
    out, seen = [], set()
    for clip in sorted(clips, key=lambda x: (x["lang_config"], int(x["fleurs_id"]))):
        if clip["lang_config"] == "en_us":
            continue
        heard = transcribe(clip, language_first).get("text", "") or ""
        for text, source in ((clip["raw_transcription"], "true"), (heard, "heard")):
            k = _key(clip, text)
            if text.strip() and k not in seen and c.cache_get("translate", k) is None:
                seen.add(k); out.append((clip, text, source))
    return out


def prefetch(clips: list[dict], language_first: bool = False) -> None:
    """Fill the translation cache with few, large requests (translate_batch_size items each; languages may be
    mixed, each item labelled with its language). Each item is cached under the key translate() uses."""
    if c.is_mock():
        return
    import gemini
    todo = pending(clips, language_first)
    if not todo:
        return
    size = int(c.CONFIG.get("translate_batch_size", 10))
    c.log(f"Translating {len(todo)} texts, up to {size} per request (~{-(-len(todo) // size)} Flash requests)")
    done = [0]

    def progress(n):
        done[0] += n
        c.log(f"  translated {done[0]}/{len(todo)}")

    def call(chunk):
        listing = "\n".join(json.dumps({"id": j + 1, "language": cl["language"], "text": t}, ensure_ascii=False)
                            for j, (cl, t, _) in enumerate(chunk))
        res = gemini.generate_json_list(BATCH_PROMPT.format(items=listing), len(chunk))
        if res is None:
            return None
        for (clip, text, source), r in zip(chunk, res):
            c.cache_put("translate", _key(clip, text), {
                "model": c.CONFIG["gemini_model"], "settings": gemini.settings_tag(), "language": clip["language"],
                "source_text": text, "english": str(r.get("english", "")).strip(), "clip": clip["id"],
                "source": source, "batched": True, "batch_size": len(chunk)})
        return res

    gemini.run_batched(todo, size, call, progress)


def translate(text: str, clip: dict, source: str) -> str:
    if clip["lang_config"] == "en_us":
        return text
    if not text.strip():
        return ""
    model = "mock" if c.is_mock() else c.CONFIG["gemini_model"]
    key = _key(clip, text)
    hit = c.cache_get("translate", key)
    if hit is not None:
        return hit["english"]
    if c.is_mock():
        out = mock.translate(text, clip, source)
    else:
        import gemini
        out = gemini.generate(PROMPT.format(language=clip["language"], text=text))
    c.cache_put("translate", key, {"model": model, "language": clip["language"], "source_text": text,
                                   "english": out, "clip": clip["id"], "source": source})
    return out


def translate_clip(clip: dict, language_first: bool = False) -> dict:
    heard = transcribe(clip, language_first)
    return {
        "translated_from_true": translate(clip["raw_transcription"], clip, "true"),
        "translated_from_heard": translate(heard.get("text", ""), clip, "heard"),
    }


def main() -> None:
    langs, limit = c.parse_langs(sys.argv)
    lf = "--language-first" in sys.argv
    clips = c.fleurs_clips(langs, limit)
    c.log(f"Translate T1+T2 ({c.stage_tag(lf)}): {len(clips)} clips")
    for i, clip in enumerate(clips, 1):
        translate_clip(clip, lf)
        if i % 20 == 0 or i == len(clips):
            c.log(f"  {i}/{len(clips)}")


if __name__ == "__main__":
    main()
