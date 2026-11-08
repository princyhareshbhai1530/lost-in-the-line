"""One row per FLEURS clip with the English human answer key, both Scribe conditions, both translations,
secondary meaning signals and the deterministic four-state fact labels. Reads only cached results: 0 API calls.

Inputs : results/clips_scored_autodetect.jsonl (auto-detect, main condition)
         results/clips_scored_langfirst.jsonl  (language supplied to Scribe; subset of languages)
Outputs: results/eval_table.jsonl, results/eval_table.csv, results/run_config.json
"""
from __future__ import annotations

import json

import common as c
import config
import facts as F

# Accepted detected-language codes per expected language (ISO 639-1 or 639-3 variants).
# Scribe labels Cantonese and Mandarin with the generic Chinese code "zho"; that is accepted for both, and a
# wrong Chinese script is reported separately as a script error.
ACCEPT = {"en": {"en", "eng"}, "ur": {"ur", "urd"}, "ar": {"ar", "ara", "arz"}, "zh": {"zh", "zho", "cmn", "chi"},
          "pa": {"pa", "pan"}, "pl": {"pl", "pol"}, "yue": {"yue", "zho", "zh", "chi"}, "tl": {"tl", "tgl", "fil"},
          "es": {"es", "spa"}, "ta": {"ta", "tam"}, "vi": {"vi", "vie"}, "hi": {"hi", "hin"}, "pt": {"pt", "por"},
          "gu": {"gu", "guj"}}

COLUMNS = ["clip_id", "fleurs_id", "language", "lang_config", "iso639_1", "gender", "variety_note", "duration_sec",
           "audio_path", "source_text", "english_ref_raw", "english_ref_norm",
           "asr_auto_text", "asr_auto_detected_lang", "asr_auto_lang_correct", "asr_auto_script_correct",
           "asr_hint_text", "asr_hint_detected_lang", "asr_hint_script_correct",
           "cer_auto", "wer_auto", "cer_hint", "wer_hint",
           "t1_clean_translation", "t2_auto_translation", "t2_hint_translation",
           "judge_t1", "judge_t2_auto", "judge_t2_hint", "emb_t1", "emb_t2_auto", "emb_t2_hint",
           "n_facts", "clip_fact_state", "clip_fact_state_hint", "facts", "fact_states"]


def load(name: str) -> dict:
    p = c.ROOT / "results" / name
    if not p.exists():
        return {}
    return {r["id"]: r for r in (json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip())}


def raw_code(stage: str, clip_id: str) -> str:
    hit = c.cache_get(f"scribe_{stage}", clip_id) or {}
    return (hit.get("language_code") or "").lower()


def main() -> None:
    if c.is_mock():
        raise SystemExit("build_eval_table reads real cached results only.")
    auto, hint = load("clips_scored_autodetect.jsonl"), load("clips_scored_langfirst.jsonl")
    meta = {x["id"]: x for x in c.fleurs_clips()}
    en_norm = {x["fleurs_id"]: c.normalise(x["transcription"]) for x in meta.values() if x["lang_config"] == "en_us"}
    rows = []
    for cid, a in sorted(auto.items()):
        m, h = meta[cid], hint.get(cid)
        ref = a["true_english"]
        t1, t2, t2h = a["translated_from_true"], a["translated_from_heard"], (h["translated_from_heard"] if h else None)
        fs = F.check(ref, t1, t2, t2h)
        det = raw_code("autodetect", cid)
        row = {
            "clip_id": cid, "fleurs_id": a["fleurs_id"], "language": a["language"], "lang_config": a["lang_config"],
            "iso639_1": m["iso639_1"], "gender": a["gender"], "variety_note": a["variety_note"],
            "duration_sec": a["duration_sec"], "audio_path": m["file"],
            "source_text": m["raw_transcription"], "english_ref_raw": ref, "english_ref_norm": en_norm[a["fleurs_id"]],
            "asr_auto_text": a["heard_native"], "asr_auto_detected_lang": det,
            "asr_auto_lang_correct": det in ACCEPT[m["iso639_1"]], "asr_auto_script_correct": a["script_correct"],
            "asr_hint_text": h["heard_native"] if h else None,
            "asr_hint_detected_lang": raw_code("langfirst", cid) if h else None,
            "asr_hint_script_correct": h["script_correct"] if h else None,
            "cer_auto": a["cer"], "wer_auto": a["wer"], "cer_hint": h["cer"] if h else None, "wer_hint": h["wer"] if h else None,
            "t1_clean_translation": t1, "t2_auto_translation": t2, "t2_hint_translation": t2h,
            "judge_t1": a["label_true"], "judge_t2_auto": a["label_heard"], "judge_t2_hint": h["label_heard"] if h else None,
            "emb_t1": a["embedding_true"], "emb_t2_auto": a["embedding_heard"], "emb_t2_hint": h["embedding_heard"] if h else None,
            "n_facts": len(fs),
            "clip_fact_state": F.clip_state([f["state_auto"] for f in fs]),
            "clip_fact_state_hint": F.clip_state([f["state_hint"] for f in fs]) if h else None,
            "facts": fs,
            "fact_states": {f"{f['type']}:{f['norm']}": f["state_auto"] for f in fs},
        }
        rows.append(row)
    out = c.ROOT / "results"
    with open(out / "eval_table.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    flat = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows]
    c.write_csv(out / "eval_table.csv", flat, COLUMNS)

    import meaning
    import translate
    run_config = {
        "data_status": "REAL", "n_clips": len(rows), "n_hint_clips": sum(1 for r in rows if r["asr_hint_text"] is not None),
        "hint_languages": sorted({r["language"] for r in rows if r["asr_hint_text"] is not None}),
        "scribe_model": config.SCRIBE_MODEL, "gemini_model": config.GEMINI_MODEL, "embedding_model": config.EMBEDDING_MODEL,
        "temperature": config.TEMPERATURE,
        "thinking": "requested 'minimal'; the model rejected it, so it ran at its default thinking setting",
        "threshold": config.SEM_THRESHOLD, "threshold_sensitivity": config.SEM_SENSITIVITY,
        "bootstrap": {"unit": "fleurs_id (sentence) cluster, paired", "reps": config.REPS, "seed": config.SEED},
        "normalisation": config.NORMALISATION, "judge_prompt_version": c.CONFIG["judge_prompt_version"],
        "judge_rubric": c.CONFIG["judge_rubric"],
        "prompts": {"translate_single": translate.PROMPT, "translate_batch": translate.BATCH_PROMPT,
                    "judge_single": meaning.JUDGE_PROMPT, "judge_batch": meaning.BATCH_JUDGE_PROMPT},
        "batch_sizes": {"translate": c.CONFIG["translate_batch_size"], "judge": c.CONFIG["judge_batch_size"],
                        "embed": c.CONFIG["embed_batch_size"]},
        "english_reference": "en_us row with the same fleurs_id: raw_transcription for facts, transcription (normalised) for text",
        "language_code_acceptance": {k: sorted(v) for k, v in ACCEPT.items()},
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    nf = sum(1 for r in rows if r["n_facts"])
    c.log(f"eval_table: {len(rows)} clips, {nf} with facts, {sum(r['n_facts'] for r in rows)} facts, "
          f"{run_config['n_hint_clips']} hint clips ({', '.join(run_config['hint_languages'])})")


if __name__ == "__main__":
    main()
