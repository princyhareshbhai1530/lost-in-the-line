"""Score every clip: ear (CER/WER, script/language flags), translator (judge, embedding, facts).

Writes results/clips_scored_<stage>.csv and results/clips_scored_<stage>.jsonl (one row per clip).

Usage: python src/score.py [--signal|--langs ...] [--language-first] [--mock]
"""
from __future__ import annotations

import json
import sys

import common as c
import meaning
from transcribe import transcribe
from translate import prefetch as translate_prefetch
from translate import translate_clip

FIELDS = ["id", "language", "lang_config", "fleurs_id", "gender", "variety_note", "duration_sec", "stage",
          "cer", "wer", "detected_language", "language_correct", "expected_script", "heard_script",
          "script_correct", "han_variant", "meaning_true", "meaning_heard", "label_true", "label_heard",
          "embedding_true", "embedding_heard", "n_facts", "facts_survived_true", "facts_survived_heard",
          "translation_loss", "hearing_loss", "total_loss", "fact_loss_heard", "break_station",
          "said_native", "heard_native", "true_english", "translated_from_true", "translated_from_heard"]


def break_station(m_true: float, m_heard: float) -> str:
    thr = c.CONFIG["break_none_threshold"]
    if m_true >= thr and m_heard >= thr:
        return "none"
    hearing = m_true - m_heard
    translation = 100 - m_true
    return "ear" if hearing > translation else "translator"


def score_clip(clip: dict, language_first: bool = False) -> dict:
    heard = transcribe(clip, language_first)
    tr = translate_clip(clip, language_first)
    heard_text = heard.get("text", "") or ""
    det_raw = (heard.get("language_code") or "").lower()
    det = c.lang_code_1(det_raw)
    # Scribe reports Cantonese and Mandarin both as the generic Chinese code "zho": that is not a wrong language.
    # (Cantonese written in Simplified characters is caught separately, as a script flip.)
    lang_ok = det == clip["iso639_1"] or (clip["iso639_1"] in ("zh", "yue") and det_raw in ("zho", "chi", "zh"))
    exp_script = c.EXPECTED_SCRIPT[clip["lang_config"]]
    h_script = c.dominant_script(heard_text)
    han = c.han_variant(heard_text) if exp_script == "Han" else ""
    # Cantonese reference is Traditional (Hong Kong), Mandarin is Simplified: the wrong variant counts as a script flip
    wrong_han = (clip["lang_config"] == "yue_hant_hk" and han == "simplified") or \
                (clip["lang_config"] == "cmn_hans_cn" and han == "traditional")
    true_en = clip["true_english"]
    jt = meaning.judge(true_en, tr["translated_from_true"])
    jh = meaning.judge(true_en, tr["translated_from_heard"])
    facts = meaning.facts_check(true_en, tr["translated_from_true"], tr["translated_from_heard"])
    n = len(facts)
    ft = sum(f["survived_true"] for f in facts)
    fh = sum(f["survived_heard"] for f in facts)
    return {
        "id": clip["id"], "language": clip["language"], "lang_config": clip["lang_config"],
        "fleurs_id": clip["fleurs_id"], "gender": clip["gender"], "variety_note": clip["variety_note"],
        "duration_sec": float(clip["duration_sec"]), "stage": c.stage_tag(language_first),
        "cer": round(c.cer(clip["transcription"], heard_text), 4),
        "wer": round(c.wer(clip["transcription"], heard_text), 4),
        "detected_language": det_raw if det_raw in ("zho", "chi") else det, "language_correct": lang_ok,
        "expected_script": exp_script, "heard_script": h_script,
        "script_correct": h_script == exp_script and not wrong_han, "han_variant": han,
        "meaning_true": jt["score"], "meaning_heard": jh["score"],
        "label_true": jt["label"], "label_heard": jh["label"],
        "embedding_true": meaning.similarity(true_en, tr["translated_from_true"]),
        "embedding_heard": meaning.similarity(true_en, tr["translated_from_heard"]),
        "n_facts": n, "facts_survived_true": ft, "facts_survived_heard": fh,
        "translation_loss": round(1 - jt["score"] / 100, 4),
        "total_loss": round(1 - jh["score"] / 100, 4),
        "hearing_loss": round((jt["score"] - jh["score"]) / 100, 4),
        "fact_loss_heard": round(1 - fh / n, 4) if n else "",
        "break_station": break_station(jt["score"], jh["score"]),
        "said_native": clip["raw_transcription"], "heard_native": heard_text, "true_english": true_en,
        "translated_from_true": tr["translated_from_true"], "translated_from_heard": tr["translated_from_heard"],
        "facts": facts, "audio_file": clip["file"],
    }


def run(langs, limit, language_first: bool) -> list[dict]:
    clips = c.fleurs_clips(langs, limit)
    c.log(f"Scoring {len(clips)} clips ({c.stage_tag(language_first)})")
    # 1. Ear: every clip through Scribe (cached)
    for i, clip in enumerate(clips, 1):
        transcribe(clip, language_first)
        if i % 20 == 0 or i == len(clips):
            c.log(f"  Scribe {i}/{len(clips)}")
    # 2-3. Batched Gemini calls fill the cache, so the per-clip loop below makes no new requests
    translate_prefetch(clips, language_first)
    pairs, texts = [], []
    for clip in clips:
        tr = translate_clip(clip, language_first)
        for cand in (tr["translated_from_true"], tr["translated_from_heard"]):
            pairs.append((clip["true_english"], cand))
            texts += [clip["true_english"], cand]
    meaning.prefetch_judges(pairs)
    meaning.prefetch_embeddings(texts)
    rows = []
    for i, clip in enumerate(clips, 1):
        rows.append(score_clip(clip, language_first))
        if i % 20 == 0 or i == len(clips):
            c.log(f"  {i}/{len(clips)}")
    stage = c.stage_tag(language_first)
    suffix = "_signal" if "--signal" in sys.argv else ""
    out = c.results_dir()
    c.write_csv(out / f"clips_scored_{stage}{suffix}.csv", rows, FIELDS)
    with open(out / f"clips_scored_{stage}{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c.log(f"Wrote {out / f'clips_scored_{stage}{suffix}.csv'}")
    return rows


def estimate(langs, limit, language_first: bool) -> None:
    """Planned Gemini Flash requests, without calling Gemini (Scribe must already be cached)."""
    from translate import pending
    clips = c.fleurs_clips(langs, limit)
    missing = [x["id"] for x in clips if c.cache_get(f"scribe_{c.stage_tag(language_first)}", x["id"]) is None]
    if missing:
        print(f"{len(missing)} clips not transcribed yet (Scribe credits needed): e.g. {missing[:3]}")
        return
    tr = len(pending(clips, language_first))
    pairs_upper = 2 * len(clips)  # T1 + T2 per clip; duplicates and cached grades lower this
    ts, js = int(c.CONFIG["translate_batch_size"]), int(c.CONFIG["judge_batch_size"])
    t_req, j_req = -(-tr // ts) if tr else 0, -(-pairs_upper // js)
    print(f"Translations pending: {tr} -> {t_req} Flash requests")
    print(f"Judge grades: at most {pairs_upper} -> at most {j_req} Flash requests")
    print(f"Total: at most {t_req + j_req} Flash requests (+1-2 per batch if a reply is malformed and split)")


if __name__ == "__main__":
    langs, limit = c.parse_langs(sys.argv)
    if "--estimate" in sys.argv:
        estimate(langs, limit, "--language-first" in sys.argv)
    else:
        run(langs, limit, "--language-first" in sys.argv)
