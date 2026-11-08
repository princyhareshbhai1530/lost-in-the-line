"""Write app/data/app_data.json (build spec section 4) and app/data/app_data.js (same data, so the app opens from
file:// with no network), plus app/data/cards.json for the live mode. Copies the audio the app plays into
app/audio/. Every number comes from a results file; anything missing is null and the UI shows [pending].
"""
from __future__ import annotations

import json
import shutil

import common as c
import facts as F
from false_comfort import load_table

NATIVE = {"English": "English", "Urdu": "اردو", "Arabic": "العربية", "Mandarin Chinese": "普通话", "Punjabi": "ਪੰਜਾਬੀ",
          "Polish": "Polski", "Cantonese": "廣東話", "Tagalog": "Tagalog", "Spanish": "Español", "Tamil": "தமிழ்",
          "Vietnamese": "Tiếng Việt", "Hindi": "हिन्दी", "Portuguese": "Português", "Gujarati": "ગુજરાતી"}
CARD_LANGS = ["Urdu", "Punjabi", "Hindi", "Tamil", "Spanish"]
LANG_CODES = {"Urdu": "ur", "Punjabi": "pa", "Hindi": "hi", "Tamil": "ta", "Spanish": "es", "Gujarati": "gu",
              "Arabic": "ar", "Mandarin Chinese": "zh", "Tagalog": "tl"}
RES = c.ROOT / "results"
APP = c.ROOT / "app"


def jload(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def f(v):
    return None if v in ("", None) else float(v)


def ci(lo, hi):
    return [f(lo), f(hi)] if lo not in ("", None) else None


def fact_status(fact: dict, which: str, cand: str) -> str:
    present = fact[f"in_t2_{which}"]
    if present is None:
        return "unavailable"
    if present:
        return "uncertain" if fact["fuzzy"] else "survived"
    return F.change_kind(fact, cand or "")


def trace_from_row(r: dict, mode: str = "REPLAY") -> dict:
    """A verified benchmark trace in the same shape the live endpoint returns."""
    return {"mode": mode, "source": "benchmark recording (FLEURS speaker, cached run)", "clip_id": r["clip_id"],
            "language": r["language"], "expected_lang": r["iso639_1"], "detected_lang": r["asr_auto_detected_lang"],
            "lang_correct": r["asr_auto_lang_correct"], "script_ok": r["asr_auto_script_correct"],
            "asr_text": r["asr_auto_text"], "english": r["t2_auto_translation"], "english_ref": r["english_ref_raw"],
            "facts": [{"type": x["type"], "text": x["text"], "status": fact_status(x, "auto", r["t2_auto_translation"])}
                      for x in r["facts"]], "audio": f"audio/{r['clip_id']}.flac", "cer": r["cer_auto"]}


def main() -> None:
    rows = load_table()
    by_id = {r["clip_id"]: r for r in rows}
    rc, fc, pu, sc = jload("run_config.json"), jload("false_comfort.json"), jload("powerup.json"), jload("screening.json")
    ev, calls = jload("eval_summary.json"), json.loads((RES / "mississauga_311_summary.json").read_text(encoding="utf-8"))
    metrics = c.read_csv(RES / "language_metrics.csv")
    hint_by = {p["language"]: p for p in pu.get("per_language", [])}

    languages = []
    for m in sorted(metrics, key=lambda m: (m["language"] == "English", -(f(m["false_comfort_rate"]) or 0))):
        h = hint_by.get(m["language"])
        languages.append({
            "language": m["language"], "native_name": NATIVE.get(m["language"], m["language"]),
            "n_clips": int(m["n_clips"]), "n_with_facts": int(m["n_with_facts"]),
            "detect_accuracy": f(m["detect_accuracy"]), "detect_ci": ci(m["detect_lo"], m["detect_hi"]),
            "script_accuracy": f(m["script_accuracy"]), "script_ci": ci(m["script_lo"], m["script_hi"]),
            "top_confusion": m["top_confusion"],
            "cer_auto": f(m["cer_auto"]), "cer_auto_ci": ci(m["cer_lo"], m["cer_hi"]),
            "fact_survival_auto": f(m["fact_survival_auto"]), "fact_survival_ci": ci(m["fs_lo"], m["fs_hi"]),
            "false_comfort_rate": f(m["false_comfort_rate"]), "false_comfort_ci": ci(m["fc_lo"], m["fc_hi"]),
            "states": {k: int(m[f"state_{k}"]) for k in ["survived", "asr_associated_break",
                                                         "translation_baseline_break", "compensation_or_discordant"]},
            "hint": None if not h else {
                "cer": h["cer_hint"]["est"], "cer_auto": h["cer_auto"]["est"], "delta_cer_ci": [h["delta_cer"]["lo"], h["delta_cer"]["hi"]],
                "fact_survival": h["fact_survival_hint"]["est"], "fact_survival_auto": h["fact_survival_auto"]["est"],
                "detect_accuracy": h["detect_ok_hint"]["est"], "detect_accuracy_auto": h["detect_ok_auto"]["est"],
                "delta_detect_ci": [h["delta_detect_ok"]["lo"], h["delta_detect_ok"]["hi"]],
                "delta_fact_survival_ci": [h["delta_fact_survival"]["lo"], h["delta_fact_survival"]["hi"]],
                "sem_pass": h["sem_pass_hint"]["est"], "sem_pass_auto": h["sem_pass_auto"]["est"], "n": h["n_clips"]},
            "speaker_note": ("Single-sex sample (" + m["genders"] + " speakers only)") if "/" not in m["genders"] else "Mixed-sex speakers",
            "dialect_note": m["variety_note"] or "",
            "baseline": m["language"] == "English",
        })

    audio_ids = set()
    fc_examples = []
    for e in fc.get("examples", []):
        audio_ids.add(e["clip_id"])
        fc_examples.append({"clip_id": e["clip_id"], "language": e["language"], "audio": f"audio/{e['clip_id']}.flac",
                            "source_text": e["source_text"], "asr_text": e["asr_text"], "english_ref": e["english_ref"],
                            "t1_text": e["t1_text"], "t2_text": e["t2_text"], "failed_fact": e["failed_fact"],
                            "state": e["state"], "judge": e["judge"], "emb": e["emb"]})
    # examples for the other three cells, so every matrix cell can be opened
    cell_examples = {}
    from false_comfort import cell
    for name in ["reassuring_agreement", "harmless_wording", "broad_failure"]:
        picks = [r for r in rows if r["n_facts"] and cell(r, rc.get("threshold", 0.85)) == name][:6]
        cell_examples[name] = []
        for r in picks:
            audio_ids.add(r["clip_id"])
            lost = [x for x in r["facts"] if not x["in_t2_auto"]]
            cell_examples[name].append({"clip_id": r["clip_id"], "language": r["language"], "audio": f"audio/{r['clip_id']}.flac",
                                        "source_text": r["source_text"], "asr_text": r["asr_auto_text"],
                                        "english_ref": r["english_ref_raw"], "t2_text": r["t2_auto_translation"],
                                        "failed_fact": ({"type": lost[0]["type"], "text": lost[0]["text"]} if lost else None),
                                        "judge": r["judge_t2_auto"], "emb": r["emb_t2_auto"], "state": r["clip_fact_state"]})
    pu_examples = []
    for e in pu.get("examples", []):
        audio_ids.add(e["clip_id"])
        pu_examples.append({**{k: e[k] for k in ("clip_id", "language", "source_text", "english_ref", "auto", "hint",
                                                  "readback_would_fire", "readback_line")}, "audio": f"audio/{e['clip_id']}.flac"})
    rb = pu.get("readback", {})
    for e in rb.get("examples", []):
        audio_ids.add(e["clip_id"])

    # live-mode cards: one fact-bearing sentence per language; replay = the verified benchmark trace
    cards = []
    for lang in CARD_LANGS:
        g = [r for r in rows if r["language"] == lang and r["n_facts"]]
        g.sort(key=lambda r: (not (r["asr_auto_lang_correct"] and r["asr_auto_script_correct"]) is False,
                              -sum(x["type"] in ("number", "date_time") for x in r["facts"]), -r["n_facts"], int(r["fleurs_id"])))
        if not g:
            continue
        r = g[0]
        audio_ids.add(r["clip_id"])
        cards.append({"card_id": f"{LANG_CODES[lang]}_{r['fleurs_id']}", "language": lang, "iso": r["iso639_1"],
                      "native_name": NATIVE[lang], "fleurs_id": r["fleurs_id"], "source_text": r["source_text"],
                      "english_ref_raw": r["english_ref_raw"],
                      "facts": [{"type": x["type"], "text": x["text"], "norm": x["norm"]} for x in r["facts"]],
                      "replay": trace_from_row(r)})

    hc = ev.get("handcheck", {})
    evaluator = ev.get("evaluator", {})
    data = {
        "run_config": {"scribe_model": rc.get("scribe_model"), "gemini_model": rc.get("gemini_model"),
                       "embedding_model": rc.get("embedding_model"), "threshold": rc.get("threshold"),
                       "n_clips": rc.get("n_clips"), "temperature": rc.get("temperature"), "thinking": rc.get("thinking"),
                       "bootstrap": rc.get("bootstrap"), "normalisation": rc.get("normalisation"),
                       "judge_prompt_version": rc.get("judge_prompt_version"), "hint_languages": rc.get("hint_languages")},
        "languages": languages,
        "matrix": {"overall": {k: fc["matrix"][k]["n"] for k in ["reassuring_agreement", "false_comfort", "harmless_wording", "broad_failure"]},
                   "n_clips": fc["matrix"]["n_clips"],
                   "sensitivity": {t: {k: m[k]["n"] for k in ["reassuring_agreement", "false_comfort", "harmless_wording", "broad_failure"]}
                                   for t, m in fc["sensitivity"].items()},
                   "false_comfort_rate": fc["false_comfort_rate"], "fact_loss_given_semantic_pass": fc["fact_loss_given_semantic_pass"],
                   "metrics_agreed": fc["metrics_agreed"], "embedding_spread": fc.get("embedding_spread"),
                   "false_comfort_by_state": fc.get("false_comfort_by_state"),
                   "n_total": fc["n_clips"], "n_without_facts": fc["n_without_facts"]},
        "false_comfort_examples": fc_examples, "cell_examples": cell_examples,
        "powerup_examples": pu_examples, "powerup_selection": pu.get("hint_selection"),
        "readback": {k: rb.get(k) for k in ("clips_with_facts", "fires", "clips_with_wrong_fact", "potential_silent_failures_surfaced", "label")}
                    | {"examples": [{**{k: e[k] for k in ("clip_id", "language", "english_ref", "t2_text", "lost", "line")},
                                     "audio": f"audio/{e['clip_id']}.flac"} for e in rb.get("examples", [])]},
        "detection": {"home_accuracy": fc.get("detect_accuracy_home"), "home_script_accuracy": fc.get("script_accuracy_home"),
                      "stratified": fc.get("detection_stratified")},
        "neighbourhoods": [{"name": h["name"], "population": h["population"], "exposure_index": h["exposure_index"],
                            "exposure_per_1000": h["exposure_per_1000"], "coverage_pct": h["coverage_pct"],
                            "top_languages": h["top_languages"], "unmeasured_bengali": h["unmeasured_bengali"]}
                           for h in sc.get("neighbourhoods", [])],
        "screening_meta": sc.get("meta"),
        "eval": {"judge_human_agreement": hc.get("judge_human_agreement"), "human_human_agreement": hc.get("human_human_agreement"),
                 "judge_emb_corr": evaluator.get("judge_vs_embedding_spearman"),
                 "judge_facts_corr": evaluator.get("judge_vs_facts_spearman"), "n_translations": evaluator.get("n_translations")},
        "context_311": {"calls_per_business_day": round(calls["calls_offered_per_day_mean"]),
                        "avg_handle_time_sec": round(calls["handle_time_sec_call_weighted"]),
                        "business_days": calls["business_days"], "source": "Mississauga 311 2017-2018, cleaned"},
        "cards": [{k: v for k, v in cd.items()} for cd in cards],
    }
    (APP / "data").mkdir(parents=True, exist_ok=True)
    (APP / "data" / "app_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    (APP / "data" / "app_data.js").write_text("window.APP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (APP / "data" / "cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")

    audio = APP / "audio"
    audio.mkdir(exist_ok=True)
    wanted = {f"{i}.flac" for i in audio_ids}
    for p in audio.iterdir():
        if p.name not in wanted:
            p.unlink()
    for cid in audio_ids:
        dst = audio / f"{cid}.flac"
        if not dst.exists():
            shutil.copy2(c.FLEURS_DIR / by_id[cid]["audio_path"], dst)
    c.log(f"Wrote app/data/app_data.json: {len(languages)} languages, {len(fc_examples)} false-comfort examples, "
          f"{len(pu_examples)} power-up examples, {len(cards)} cards, {len(wanted)} audio clips")


if __name__ == "__main__":
    main()
