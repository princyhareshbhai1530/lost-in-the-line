"""Station 2, the Ear: ElevenLabs Scribe v2 on FLEURS clips. Every response is cached.

Usage:
  python src/transcribe.py --signal                 # phase 1 gate (40 clips)
  python src/transcribe.py                          # all 560 clips, auto-detect language
  python src/transcribe.py --language-first --langs ur_pk,pa_in,fil_ph   # measured fix rerun
  add --mock for the no-credit dry run
"""
from __future__ import annotations

import sys
from pathlib import Path

import common as c
import mock


# Scribe accepts ISO 639-1 or 639-3; send 639-3 so Tagalog/Filipino and Cantonese are unambiguous.
ISO1_TO_3 = {"en": "eng", "ur": "urd", "ar": "ara", "zh": "cmn", "pa": "pan", "pl": "pol", "yue": "yue",
             "tl": "fil", "es": "spa", "ta": "tam", "vi": "vie", "hi": "hin", "pt": "por", "gu": "guj"}


def scribe_live(clip: dict, language_first: bool) -> dict:
    key = c.require_key("ELEVENLABS_API_KEY")
    fields = {"model_id": c.CONFIG["scribe_model"], "tag_audio_events": "false", "diarize": "false",
              "timestamps_granularity": "none"}
    if language_first:
        fields["language_code"] = ISO1_TO_3.get(clip["iso639_1"], clip["iso639_1"])
    body, ctype = c.multipart(fields, "file", Path(clip["audio_path"]), "audio/flac")
    resp = c.http_json("https://api.elevenlabs.io/v1/speech-to-text", body=body,
                       headers={"xi-api-key": key, "Content-Type": ctype})
    resp.pop("words", None)
    return resp


def transcribe(clip: dict, language_first: bool = False) -> dict:
    kind = f"scribe_{c.stage_tag(language_first)}"
    hit = c.cache_get(kind, clip["id"])
    if hit is not None:
        return hit
    resp = mock.scribe(clip, language_first) if c.is_mock() else scribe_live(clip, language_first)
    resp["model"] = c.CONFIG["scribe_model"]
    resp["language_given"] = clip["iso639_1"] if language_first else None
    c.cache_put(kind, clip["id"], resp)
    return resp


def main() -> None:
    langs, limit = c.parse_langs(sys.argv)
    lf = "--language-first" in sys.argv
    clips = c.fleurs_clips(langs, limit)
    secs = sum(float(x["duration_sec"]) for x in clips)
    c.log(f"Scribe ({c.stage_tag(lf)}): {len(clips)} clips, {secs / 60:.1f} min of audio")
    for i, clip in enumerate(clips, 1):
        transcribe(clip, lf)
        if i % 20 == 0 or i == len(clips):
            c.log(f"  {i}/{len(clips)}")


if __name__ == "__main__":
    main()
