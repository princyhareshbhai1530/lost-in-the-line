"""Phase 0 / credit check. Verifies keys, that the configured Gemini models exist, ElevenLabs credits,
and estimates the cost of the full run from audio minutes.

Usage: python src/check_setup.py            (needs keys; makes only free metadata calls)
"""
from __future__ import annotations

import json
import os
import sys

import common as c


def main() -> None:
    print(f"Data pack: {c.PACK}  exists={c.PACK.exists()}")
    clips = c.fleurs_clips()
    mins = sum(float(x["duration_sec"]) for x in clips) / 60
    print(f"FLEURS clips: {len(clips)}  audio: {mins:.1f} min (full run). Signal test: "
          f"{sum(float(x['duration_sec']) for x in c.fleurs_clips(c.CONFIG['signal_test_langs'], c.CONFIG['signal_test_clips_per_lang'])) / 60:.1f} min")
    print(f"Fix rerun (3 languages x 40): ~{mins * 3 / 14:.1f} min more")
    for k in ("ELEVENLABS_API_KEY", "GEMINI_API_KEY"):
        print(f"{k}: {'set' if os.environ.get(k) else 'MISSING'}")
    if os.environ.get("GEMINI_API_KEY"):
        import gemini
        names = gemini.list_models()
        for m in (c.CONFIG["gemini_model"], c.CONFIG["embedding_model"]):
            print(f"Gemini model {m}: {'listed' if f'models/{m}' in names else 'NOT LISTED - pick another in config.json'}")
        flash = sorted(n.split("/", 1)[1] for n in names if "flash" in n and "gemini" in n)
        print(f"Flash models this key can see: {', '.join(flash)}")
        if "--live" in sys.argv:
            # Listed is not the same as usable: new accounts can be blocked from older models.
            print("Live test: 1 Flash request + 1 embedded text ...")
            try:
                out = gemini.generate('Reply with JSON only: {"ok": true}', json_mode=True)
                print(f"  {c.CONFIG['gemini_model']}: WORKS (reply: {out[:40]})")
            except (RuntimeError, SystemExit) as e:
                print(f"  {c.CONFIG['gemini_model']}: FAILS -> {str(e)[:300]}")
            try:
                v = gemini.embed_batch(["test"])
                print(f"  {c.CONFIG['embedding_model']}: WORKS ({len(v[0])} dims)")
            except (RuntimeError, SystemExit) as e:
                print(f"  {c.CONFIG['embedding_model']}: FAILS -> {str(e)[:300]}")
        else:
            print("Listed does not mean usable. Run `py src/check_setup.py --live` (costs 1 Flash request) to be sure.")
    if os.environ.get("ELEVENLABS_API_KEY"):
        sub = c.http_json("https://api.elevenlabs.io/v1/user/subscription", method="GET",
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]})
        used, limit = sub.get("character_count"), sub.get("character_limit")
        print(f"ElevenLabs tier={sub.get('tier')} credits used={used} limit={limit} remaining={None if used is None else limit - used}")
        snap = c.ROOT / "results" / "credits_snapshots.jsonl"
        snap.parent.mkdir(exist_ok=True)
        with open(snap, "a", encoding="utf-8") as f:
            f.write(json.dumps({"used": used, "limit": limit}) + "\n")
        print("Snapshot appended to results/credits_snapshots.jsonl. Run again after the signal test: "
              "(credits used during it) / 40 clips x 680 = full-run estimate.")


if __name__ == "__main__":
    main()
