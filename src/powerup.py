"""Power-Up Decision Lab: the same audio with the language supplied to Scribe (paired with auto-detect).

Per language with a hint run: paired change in CER, detection correctness (language and script), fact
preservation and semantic pass, each with a paired cluster bootstrap by fleurs_id.
"Would confirmation have fired?": a readback fires when the English the desk received (T2) contains a critical
fact (number, date, name, negation). Potential silent failures surfaced = clips where a fact was wrong in T2 AND
the readback would have shown a fact of that type to the caller, so the caller could catch it. Never "fixed".
Output: results/powerup.json
"""
from __future__ import annotations

import json

import common as c
import config
import facts as F
import stats as S
from false_comfort import facts_ok, load_table, sem_pass


def readback(r: dict, which: str = "auto") -> dict:
    t2 = r[f"t2_{which}_translation"] or ""
    shown = F.readback_facts(t2)
    lost = [f for f in r["facts"] if f[f"in_t2_{which}"] is False]
    shown_types = {f["type"] for f in shown}
    surfaced = [f for f in lost if f["type"] in shown_types]
    line = ""
    if shown:
        order = ["number", "date_time", "negation", "name_place"]
        ref_norms = {(f["type"], f["norm"]) for f in r["facts"]}
        # read back the value that replaced a lost fact, if there is one; else the most critical fact shown
        changed = [s for s in shown if any(s["type"] == l["type"] for l in surfaced) and (s["type"], s["norm"]) not in ref_norms]
        pick = (changed or sorted(shown, key=lambda f: order.index(f["type"])))[0]
        what = "that it was NOT so" if pick["type"] == "negation" else pick["text"]
        line = f"I heard {what}. Is that correct?"
    return {"fires": bool(shown), "shown": [{"type": f["type"], "text": f["text"]} for f in shown],
            "lost": [{"type": f["type"], "text": f["text"], "kind": F.change_kind(f, t2)} for f in lost],
            "surfaced": bool(surfaced), "line": line}


def ok_detect(r: dict, which: str) -> bool:
    if which == "auto":
        return bool(r["asr_auto_lang_correct"] and r["asr_auto_script_correct"])
    return bool(r["asr_hint_script_correct"])  # language was supplied, so only the script can still be wrong


def main() -> None:
    rows = load_table()
    thr = config.SEM_THRESHOLD
    hinted = [r for r in rows if r["asr_hint_text"] is not None]
    langs = sorted({r["language"] for r in hinted})
    per = []
    for lang in langs:
        g = [r for r in hinted if r["language"] == lang]
        gf = [r for r in g if r["n_facts"]]
        d = lambda f: S.rnd(S.cluster_boot(g, f))
        per.append({
            "language": lang, "n_clips": len(g), "n_with_facts": len(gf),
            "cer_auto": d(S.mean_of("cer_auto")), "cer_hint": d(S.mean_of("cer_hint")),
            "delta_cer": d(lambda rs: sum(r["cer_hint"] - r["cer_auto"] for r in rs) / len(rs)),
            "detect_ok_auto": d(S.share(lambda r: ok_detect(r, "auto"))),
            "detect_ok_hint": d(S.share(lambda r: ok_detect(r, "hint"))),
            "delta_detect_ok": d(lambda rs: sum(ok_detect(r, "hint") - ok_detect(r, "auto") for r in rs) / len(rs)),
            "fact_survival_auto": S.rnd(S.cluster_boot(gf, lambda rs: _fs(rs, "auto"))),
            "fact_survival_hint": S.rnd(S.cluster_boot(gf, lambda rs: _fs(rs, "hint"))),
            "delta_fact_survival": S.rnd(S.cluster_boot(gf, lambda rs: (_fs(rs, "hint") or 0) - (_fs(rs, "auto") or 0))),
            "facts_ok_auto": sum(facts_ok(r, "auto") for r in gf), "facts_ok_hint": sum(facts_ok(r, "hint") for r in gf),
            "sem_pass_auto": d(S.share(lambda r: sem_pass(r, thr, "auto"))),
            "sem_pass_hint": d(S.share(lambda r: sem_pass(r, thr, "hint"))),
        })
    # readback analysis on every clip with facts (auto condition), plus the hint condition where available
    withf = [r for r in rows if r["n_facts"]]
    rb = {r["clip_id"]: readback(r, "auto") for r in withf}
    wrong = [r for r in withf if rb[r["clip_id"]]["lost"]]
    surfaced = [r for r in wrong if rb[r["clip_id"]]["surfaced"]]
    examples = []
    for r in sorted(hinted, key=lambda r: (ok_detect(r, "auto"), r["language"], int(r["fleurs_id"]))):
        if ok_detect(r, "auto") and r["cer_auto"] - (r["cer_hint"] or 0) < 0.05:
            continue
        examples.append({
            "clip_id": r["clip_id"], "language": r["language"], "audio_path": r["audio_path"],
            "source_text": r["source_text"], "english_ref": r["english_ref_raw"],
            "auto": {"detected": r["asr_auto_detected_lang"], "detect_ok": ok_detect(r, "auto"), "text": r["asr_auto_text"],
                     "english": r["t2_auto_translation"], "cer": r["cer_auto"], "judge": r["judge_t2_auto"],
                     "facts": [{"type": f["type"], "text": f["text"], "ok": f["in_t2_auto"]} for f in r["facts"]]},
            "hint": {"detected": r["asr_hint_detected_lang"], "detect_ok": ok_detect(r, "hint"), "text": r["asr_hint_text"],
                     "english": r["t2_hint_translation"], "cer": r["cer_hint"], "judge": r["judge_t2_hint"],
                     "facts": [{"type": f["type"], "text": f["text"], "ok": f["in_t2_hint"]} for f in r["facts"]]},
            "readback_would_fire": readback(r, "auto")["fires"], "readback_line": readback(r, "auto")["line"],
        })
    rb_examples = [{"clip_id": r["clip_id"], "language": r["language"], "audio_path": r["audio_path"],
                    "english_ref": r["english_ref_raw"], "t2_text": r["t2_auto_translation"], **rb[r["clip_id"]]}
                   for r in surfaced]
    rb_examples.sort(key=lambda e: (0 if any(l["kind"] == "changed" for l in e["lost"]) else 1, e["language"]))
    out = {
        "hint_languages": langs, "hint_selection": "the 3 languages with the most wrong-language or wrong-script "
                                                   "transcripts in the auto-detect run (all others had none)",
        "per_language": per, "examples": examples[:24],
        "readback": {"clips_with_facts": len(withf), "fires": sum(v["fires"] for v in rb.values()),
                     "clips_with_wrong_fact": len(wrong), "potential_silent_failures_surfaced": len(surfaced),
                     "label": "Potential silent failures surfaced for confirmation", "examples": rb_examples[:12]},
    }
    (c.ROOT / "results" / "powerup.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in per:
        c.log(f"{p['language']}: detect ok {p['detect_ok_auto']['est']:.3f} -> {p['detect_ok_hint']['est']:.3f}; "
              f"CER {p['cer_auto']['est']:.3f} -> {p['cer_hint']['est']:.3f}; facts {p['fact_survival_auto']['est']} -> "
              f"{p['fact_survival_hint']['est']} (delta CI {p['delta_fact_survival']['lo']}, {p['delta_fact_survival']['hi']})")
    c.log(f"Readback fires on {out['readback']['fires']}/{len(withf)}; wrong fact in {len(wrong)}; surfaced {len(surfaced)}")


def _fs(rows, which):
    n = sum(len(r["facts"]) for r in rows)
    return (sum(sum(1 for f in r["facts"] if f[f"in_t2_{which}"]) for r in rows) / n) if n else None


if __name__ == "__main__":
    main()
