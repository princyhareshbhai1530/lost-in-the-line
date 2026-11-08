"""Build app/game_data.json for Member B's game, following the agreed data contract exactly.

Field names in the contract must not be renamed. One extra top-level key, `meta`, records
data_status (REAL / DRY_RUN_FAKE), models, sources and n for every headline number.
Also copies hero clips into app/audio/.

Usage: python src/build_game_json.py [--mock]
"""
from __future__ import annotations

import json
import shutil

import common as c
from analyze import load

NATIVE = {"en_us": "English", "ur_pk": "اردو", "ar_eg": "العربية", "cmn_hans_cn": "普通话", "pa_in": "ਪੰਜਾਬੀ",
          "pl_pl": "Polski", "yue_hant_hk": "廣東話", "fil_ph": "Tagalog", "es_419": "Español", "ta_in": "தமிழ்",
          "vi_vn": "Tiếng Việt", "hi_in": "हिन्दी", "pt_br": "Português", "gu_in": "ગુજરાતી"}
HEROES_PER_LANG = 3


def fnum(v):
    return None if v in (None, "") else float(v)


def speaker_note(r: dict) -> str:
    parts = []
    if r["single_sex"] in (True, "True"):
        parts.append(f"All {r['genders']} speakers: a language effect may be a speaker effect")
    else:
        parts.append(f"Speakers: {r['genders']}")
    if r["variety_note"]:
        parts.append(r["variety_note"])
    return ". ".join(parts)


def pick_heroes(rows: list[dict]) -> list[dict]:
    """Per language: a wrong-language/script clip if any (the hidden ear error), the clearest 'ear' break,
    the clearest 'translator' break, and a clean pass."""
    out = []
    for lang in sorted({r["lang_config"] for r in rows}):
        g = [r for r in rows if r["lang_config"] == lang]
        chosen = []
        flip = [r for r in g if not r["language_correct"] or not r["script_correct"]]
        ear = sorted([r for r in g if r["break_station"] == "ear"], key=lambda r: -(r["meaning_true"] - r["meaning_heard"]))
        trn = sorted([r for r in g if r["break_station"] == "translator"], key=lambda r: r["meaning_true"])
        ok = [r for r in g if r["break_station"] == "none"]
        for bucket in (flip[:2], ear, trn, ok):  # up to 2 wrong-language/script clips: the headline finding
            if len(chosen) >= HEROES_PER_LANG + 1:
                break
            bucket = [r for r in bucket if r not in chosen]
            for r in bucket[: 2 if bucket and bucket[0] in flip else 1]:
                chosen.append(r)
        for r in sorted(g, key=lambda r: r["total_loss"], reverse=True):
            if len(chosen) >= HEROES_PER_LANG:
                break
            if r not in chosen:
                chosen.append(r)
        out += chosen
    return out


def main() -> None:
    res = c.results_dir()
    rows = load("autodetect")
    langs = c.read_csv(res / "language_summary.csv")
    ev = json.loads((res / "eval_summary.json").read_text(encoding="utf-8"))
    hoods = json.loads((res / "neighbourhoods.json").read_text(encoding="utf-8"))
    calls = json.loads((c.ROOT / "results" / "mississauga_311_summary.json").read_text(encoding="utf-8"))
    hc_path = res / "handcheck_result.json"
    hc = json.loads(hc_path.read_text(encoding="utf-8")) if hc_path.exists() else {}
    cov = {r["language"]: r for r in c.read_csv(c.CENSUS_DIR / "derived" / "language_coverage.csv")}
    xw = {r["fleurs_config"]: r for r in c.crosswalk() if r["fleurs_config"]}
    fix = {f["lang_config"]: f for f in ev.get("fix", [])}

    languages = []
    for r in langs:
        name = xw[r["lang_config"]]["language"]
        residents = cov.get(name, {}).get("city_home_speakers") or None
        languages.append({
            "language": name, "native_name": NATIVE.get(r["lang_config"], name),
            "residents": int(residents) if residents else None,
            "hearing_loss": fnum(r["hearing_loss"]), "translation_loss": fnum(r["translation_loss"]),
            "total_loss": fnum(r["total_loss"]), "facts_survived_pct": fnum(r["facts_survived_pct"]),
            "embedding_similarity": fnum(r["embedding_similarity"]),
            "fixed_total_loss": fix.get(r["lang_config"], {}).get("fixed_total_loss"),
            "ci_total": [fnum(r["total_lo"]), fnum(r["total_hi"])], "n_clips": int(r["n_clips"]),
            "speaker_note": speaker_note(r),
        })
    languages.sort(key=lambda x: -(x["total_loss"] or 0))

    app_audio = c.ROOT / "app" / ("audio_mock" if c.is_mock() else "audio")
    app_audio.mkdir(parents=True, exist_ok=True)
    clips = []
    for r in pick_heroes(rows):
        src = c.FLEURS_DIR / r["audio_file"]
        dst = app_audio / f"{r['id']}.flac"
        if not dst.exists():
            shutil.copy2(src, dst)
        clips.append({
            "id": r["id"], "language": r["language"], "audio": f"{app_audio.name}/{dst.name}",
            "said_native": r["said_native"], "heard_native": r["heard_native"], "true_english": r["true_english"],
            "translated_from_true": r["translated_from_true"], "translated_from_heard": r["translated_from_heard"],
            "meaning_true": r["meaning_true"], "meaning_heard": r["meaning_heard"],
            "embedding_true": r["embedding_true"], "embedding_heard": r["embedding_heard"],
            "facts": [{"text": f["text"], "survived_true": f["survived_true"], "survived_heard": f["survived_heard"]}
                      for f in r["facts"]],
            "break_station": r["break_station"],
        })

    neighbourhoods = [{"name": h["name"], "residents": h["residents"], "messages_at_risk": h["messages_at_risk"],
                       "rank": h["rank"], "top_languages": h["top_languages"]} for h in hoods["neighbourhoods"]]

    res_by_name = {x["language"]: x for x in languages}
    rescued = 0
    for lc, f in fix.items():
        x = res_by_name.get(xw[lc]["language"])
        if x and x["residents"] and f["drop"] is not None:
            rescued += x["residents"] * f["drop"]

    home = [x for x in languages if x["language"] != "English"]
    worst = home[0] if home else None
    ear_led = [x["language"] for x in home if (x["hearing_loss"] or 0) > (x["translation_loss"] or 0)]
    flips = sorted([r for r in langs if r["lang_config"] != "en_us"],
                   key=lambda r: -max(fnum(r["language_flip_pct"]) or 0, fnum(r["script_flip_pct"]) or 0))
    top_flip = flips[0] if flips else None
    flip_pct = max(fnum(top_flip["language_flip_pct"]) or 0, fnum(top_flip["script_flip_pct"]) or 0) if top_flip else 0
    if top_flip and flip_pct >= 10:
        kind = "language" if (fnum(top_flip["language_flip_pct"]) or 0) >= (fnum(top_flip["script_flip_pct"]) or 0) else "script"
        lo, hi = fnum(top_flip[f"{kind}_flip_lo"]), fnum(top_flip[f"{kind}_flip_hi"])
        finding = (f"DRAFT (agree the final sentence at Freeze): the ear wrote {flip_pct:.0f}% of "
                   f"{xw[top_flip['lang_config']]['language']} callers down in the wrong {kind} "
                   f"(95% CI {lo:.0f}-{hi:.0f}%, n={top_flip['n_clips']}), "
                   f"an error the English translation hides; in {len(ear_led)} of {len(home)} home languages the ear "
                   f"loses more meaning than the translator.")
    elif worst:
        finding = (f"DRAFT (agree the final sentence at Freeze): in {len(ear_led)} of {len(home)} home languages the ear "
                   f"loses more meaning than the translator; worst is {worst['language']} with total loss "
                   f"{worst['total_loss']:.2f} (95% CI {worst['ci_total'][0]:.2f}-{worst['ci_total'][1]:.2f}, n={worst['n_clips']}).")
    else:
        finding = ""
    if c.CONFIG.get("headline_finding") and not c.is_mock():
        finding = c.CONFIG["headline_finding"]  # the sentence agreed at Freeze; numbers checked against results/
    language_flags = [{"language": xw[r["lang_config"]]["language"],
                       "language_flip_pct": fnum(r["language_flip_pct"]),
                       "language_flip_ci": [fnum(r["language_flip_lo"]), fnum(r["language_flip_hi"])],
                       "script_flip_pct": fnum(r["script_flip_pct"]),
                       "wrong_language_detected": r["wrong_language_detected"],
                       "cer": fnum(r["cer"]), "cer_excl_script_flips": fnum(r["cer_excl_script_flips"])}
                      for r in langs]
    hero_ids = {x["id"] for x in clips}
    clip_flags = {r["id"]: {"lang_config": r["lang_config"], "detected_language": r["detected_language"],
                            "language_correct": r["language_correct"], "expected_script": r["expected_script"],
                            "heard_script": r["heard_script"], "script_correct": r["script_correct"],
                            "cer": r["cer"], "gender": r["gender"]}
                  for r in rows if r["id"] in hero_ids}
    # wrong-language rate before vs after the fix (same clips), for the Power-Ups screen
    fixed_rows = load("langfirst")
    fix_flips = []
    for lc in fix:
        before = [r for r in rows if r["lang_config"] == lc]
        after = [r for r in fixed_rows if r["lang_config"] == lc]
        if before and after:
            bad = lambda g: round(100 * sum((not r["language_correct"]) or (not r["script_correct"]) for r in g) / len(g), 1)
            fix_flips.append({"language": xw[lc]["language"], "flip_pct_before": bad(before), "flip_pct_after": bad(after),
                              "total_loss_before": fix[lc]["total_loss_before"], "fixed_total_loss": fix[lc]["fixed_total_loss"],
                              "drop_ci": [fix[lc]["drop_lo"], fix[lc]["drop_hi"]], "n_clips": fix[lc]["n_clips"]})
    neighbourhood_detail = [{"name": h["name"], "pct_knows_neither": h["pct_knows_neither"],
                             "knows_neither": h["knows_neither"], "messages_at_risk_per_1000": h["messages_at_risk_per_1000"],
                             "risk_fact_loss": h["risk_fact_loss"], "unmeasured_bengali_speakers": h["unmeasured_bengali_speakers"],
                             "top_languages_detail": h["top_languages_detail"]} for h in hoods["neighbourhoods"]]
    lang_codes = {xw[r["lang_config"]]["language"]: r["lang_config"] for r in langs}
    # every scored clip, compact, for the 560-dot matrix (w = wrong language or script)
    wrong = lambda r: bool((not r["language_correct"]) or (not r["script_correct"]))
    all_clips = [{"i": r["id"], "l": r["language"], "b": r["break_station"], "w": wrong(r),
                  "m": r["meaning_heard"], "t": r["meaning_true"]} for r in rows]
    fix_clip_wrong = {r["id"]: wrong(r) for r in fixed_rows}
    fix_clip_meaning = {r["id"]: r["meaning_heard"] for r in fixed_rows}
    data = {
        "languages": languages, "clips": clips, "neighbourhoods": neighbourhoods,
        "eval": {"judge_human_agreement": hc.get("judge_human_agreement", "pending"),
                 "human_human_agreement": hc.get("human_human_agreement", "pending"),
                 "judge_vs_embedding_corr": ev["evaluator"].get("judge_vs_embedding_spearman")},
        "headline": {"finding": finding, "rescued_residents": round(rescued) if fix else None,
                     "calls_per_day": round(calls["calls_offered_per_day_mean"]),
                     "handle_time_sec": round(calls["handle_time_sec_call_weighted"])},
        "meta": {
            "data_status": c.data_status(),
            "warning": "FAKE dry-run numbers: do not show or quote." if c.is_mock() else "",
            "models": ev["models"], "n_clips_scored": ev["n_clips"],
            "judge_vs_embedding_corr_method": "Spearman, all translations except English identity",
            "rescued_residents_definition": "sum over fixed languages of residents x measured drop in total loss "
                                            "(expected messages no longer lost), not individual people",
            "fixed_languages": list(fix),
            "calls_source": "Mississauga 311 2017-2018, business days, cleaned (results/mississauga_311_summary.json)",
            "neighbourhood_sensitivity": hoods["sensitivity"],
            "language_flags": language_flags,
            "language_flags_note": "Extra fields kept in meta so the agreed contract stays unchanged. "
                                   "Flip = Scribe detected the wrong language or wrote the wrong script.",
            "judge_prompt_version": c.CONFIG["judge_prompt_version"],
            "gate": ev["gate"],
            "clip_flags": clip_flags,
            "fix_flips": fix_flips,
            "neighbourhood_detail": neighbourhood_detail,
            "lang_codes": lang_codes,
            "evaluator": ev["evaluator"],
            "audio_dir": app_audio.name,
            "all_clips": all_clips,
            "fix_clip_wrong": fix_clip_wrong,
            "fix_clip_meaning": fix_clip_meaning,
        },
    }
    out = c.ROOT / "app" / ("game_data.mock.json" if c.is_mock() else "game_data.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # Same data as a script, so the game also opens offline by double-click (file:// blocks fetch)
    js = out.with_suffix(".js")
    js.write_text("window.GAME_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    c.log(f"Wrote {out} ({len(languages)} languages, {len(clips)} clips, {len(neighbourhoods)} neighbourhoods)")


if __name__ == "__main__":
    main()
