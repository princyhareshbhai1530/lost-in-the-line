"""False-Comfort Matrix: does the meaning signal say "fine" while a critical fact was lost?

Semantic pass (end to end, T2 from auto-detect) = judge label in {same, minor} AND embedding >= threshold (0.85,
declared in config.py; 0.80 and 0.90 as sensitivity). Facts preserved = clip has facts and none fails in T2.
Cells: reassuring_agreement (pass, facts ok), false_comfort (pass, fact lost), harmless_wording (fail, facts ok),
broad_failure (fail, fact lost). Only clips with at least one fact enter the matrix.
false_comfort_rate = false_comfort clips / clips with facts (95% paired cluster bootstrap by fleurs_id).
Also: language-detection accuracy, confusion pairs, and losses stratified by correct vs incorrect detection.
Outputs: results/false_comfort.json, results/false_comfort_by_language.csv, results/language_metrics.csv
"""
from __future__ import annotations

import json
from collections import Counter

import common as c
import config
import stats as S

CELLS = ["reassuring_agreement", "false_comfort", "harmless_wording", "broad_failure"]
FACT_RANK = {"negation": 0, "number": 1, "date_time": 2, "name_place": 3}
LANG_NAME = {"hin": "Hindi", "hi": "Hindi", "jpn": "Japanese", "bul": "Bulgarian", "zho": "Chinese", "ur": "Urdu",
             "urd": "Urdu", "pan": "Punjabi", "pa": "Punjabi", "eng": "English", "en": "English"}


def load_table() -> list[dict]:
    p = c.ROOT / "results" / "eval_table.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sem_pass(r: dict, thr: float, which: str = "auto") -> bool:
    j, e = r[f"judge_t2_{which}"], r[f"emb_t2_{which}"]
    return j in config.JUDGE_PASS and e is not None and e >= thr


def facts_ok(r: dict, which: str = "auto") -> bool:
    return all(f[f"in_t2_{which}"] for f in r["facts"])


def cell(r: dict, thr: float, which: str = "auto") -> str | None:
    if not r["n_facts"]:
        return None
    sp, fo = sem_pass(r, thr, which), facts_ok(r, which)
    return CELLS[0] if sp and fo else CELLS[1] if sp else CELLS[2] if fo else CELLS[3]


def fact_survival(rows: list[dict], which: str = "auto"):
    n = sum(len(r["facts"]) for r in rows)
    return (sum(sum(1 for f in r["facts"] if f[f"in_t2_{which}"]) for r in rows) / n) if n else None


def matrix(rows: list[dict], thr: float) -> dict:
    cnt = Counter(cell(r, thr) for r in rows if r["n_facts"])
    n = sum(cnt.values())
    return {k: {"n": cnt.get(k, 0), "pct": round(100 * cnt.get(k, 0) / n, 1) if n else None} for k in CELLS} | {"n_clips": n}


def ranked_examples(rows: list[dict], thr: float) -> list[dict]:
    ex = []
    for r in rows:
        if cell(r, thr) != "false_comfort":
            continue
        lost = [f for f in r["facts"] if not f["in_t2_auto"]]
        lost.sort(key=lambda f: FACT_RANK[f["type"]])
        ex.append((FACT_RANK[lost[0]["type"]], r, lost[0]))
    # spec ranking by fact type; within a type, ASR-associated breaks first (the ear, not the answer key)
    ex.sort(key=lambda t: (t[0], 0 if t[2]["state_auto"] == "asr_associated_break" else 1,
                           t[1]["language"], int(t[1]["fleurs_id"])))
    return [{"clip_id": r["clip_id"], "language": r["language"], "audio_path": r["audio_path"],
             "source_text": r["source_text"], "asr_text": r["asr_auto_text"], "english_ref": r["english_ref_raw"],
             "t1_text": r["t1_clean_translation"], "t2_text": r["t2_auto_translation"],
             "failed_fact": {"type": f["type"], "text": f["text"], "in_t1": f["in_t1"], "fuzzy": f["fuzzy"]},
             "state": f["state_auto"], "judge": r["judge_t2_auto"], "emb": r["emb_t2_auto"],
             "detected_lang": r["asr_auto_detected_lang"], "lang_correct": r["asr_auto_lang_correct"]}
            for _, r, f in ex]


def main() -> None:
    rows = load_table()
    thr = config.SEM_THRESHOLD
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)

    lang_rows, metric_rows = [], []
    for lang, g in sorted(by_lang.items()):
        gf = [r for r in g if r["n_facts"]]
        fc = S.cluster_boot(gf, S.share(lambda r: cell(r, thr) == "false_comfort"))
        det = S.cluster_boot(g, S.share(lambda r: r["asr_auto_lang_correct"]))
        scr = S.cluster_boot(g, S.share(lambda r: r["asr_auto_script_correct"]))
        cer = S.cluster_boot(g, S.mean_of("cer_auto"))
        fsv = S.cluster_boot(gf, lambda rs: fact_survival(rs))
        conf = Counter(r["asr_auto_detected_lang"] for r in g if not r["asr_auto_lang_correct"])
        states = Counter(r["clip_fact_state"] for r in gf)
        m = matrix(g, thr)
        lang_rows.append({"language": lang, "n_clips": len(g), "n_with_facts": len(gf),
                          **{k: m[k]["n"] for k in CELLS},
                          "false_comfort_rate": S.rnd(fc)["est"], "fc_lo": S.rnd(fc)["lo"], "fc_hi": S.rnd(fc)["hi"]})
        hint_rows = [r for r in g if r["asr_hint_text"] is not None]
        metric_rows.append({
            "language": lang, "lang_config": g[0]["lang_config"], "n_clips": len(g), "n_with_facts": len(gf),
            "detect_accuracy": det["est"], "detect_lo": det["lo"], "detect_hi": det["hi"],
            "script_accuracy": scr["est"], "script_lo": scr["lo"], "script_hi": scr["hi"],
            "top_confusion": "; ".join(f"{LANG_NAME.get(k, k)} ({v})" for k, v in conf.most_common()),
            "cer_auto": cer["est"], "cer_lo": cer["lo"], "cer_hi": cer["hi"],
            "fact_survival_auto": fsv["est"], "fs_lo": fsv["lo"], "fs_hi": fsv["hi"],
            "false_comfort_rate": fc["est"], "fc_lo": fc["lo"], "fc_hi": fc["hi"],
            **{f"state_{k}": states.get(k, 0) for k in ["survived", "asr_associated_break",
                                                         "translation_baseline_break", "compensation_or_discordant"]},
            "has_hint": bool(hint_rows),
            "genders": "/".join(sorted({r["gender"] for r in g})), "variety_note": g[0]["variety_note"],
        })
    c.write_csv(c.ROOT / "results" / "false_comfort_by_language.csv", lang_rows)
    c.write_csv(c.ROOT / "results" / "language_metrics.csv", metric_rows)

    withf = [r for r in rows if r["n_facts"]]
    overall_fc = S.cluster_boot(withf, S.share(lambda r: cell(r, thr) == "false_comfort"))
    sem_pass_rows = [r for r in withf if sem_pass(r, thr)]
    cond = S.cluster_boot(sem_pass_rows, S.share(lambda r: not facts_ok(r)))
    agree = sum(1 for r in withf if cell(r, thr) in ("reassuring_agreement", "broad_failure"))
    # stratified by detection correctness (home languages only; English has no detection errors)
    home = [r for r in rows if r["lang_config"] != "en_us"]
    strat = {}
    for name, pred in (("detect_correct", lambda r: r["asr_auto_lang_correct"] and r["asr_auto_script_correct"]),
                       ("detect_incorrect", lambda r: not (r["asr_auto_lang_correct"] and r["asr_auto_script_correct"]))):
        g = [r for r in home if pred(r)]
        gf = [r for r in g if r["n_facts"]]
        strat[name] = {"n_clips": len(g), "n_with_facts": len(gf),
                       "cer": S.rnd(S.cluster_boot(g, S.mean_of("cer_auto"))),
                       "fact_survival": S.rnd(S.cluster_boot(gf, lambda rs: fact_survival(rs))),
                       "semantic_pass": S.rnd(S.cluster_boot(g, S.share(lambda r: sem_pass(r, thr))))}
    out = {
        "threshold": thr, "n_clips": len(rows), "n_with_facts": len(withf), "n_without_facts": len(rows) - len(withf),
        "matrix": matrix(rows, thr),
        "matrix_home_languages": matrix(home, thr),
        "sensitivity": {f"{t:.2f}": matrix(rows, t) for t in config.SEM_SENSITIVITY},
        "false_comfort_rate": S.rnd(overall_fc),
        "fact_loss_given_semantic_pass": S.rnd(cond),
        "metrics_agreed": {"n": agree, "of": len(withf)},
        "detection_stratified": strat,
        "detect_accuracy_home": S.rnd(S.cluster_boot(home, S.share(lambda r: r["asr_auto_lang_correct"]))),
        "script_accuracy_home": S.rnd(S.cluster_boot(home, S.share(lambda r: r["asr_auto_script_correct"]))),
        "examples": ranked_examples(rows, thr),
        "embedding_spread": {"share_ge_0.90": round(sum(1 for r in withf if (r["emb_t2_auto"] or 0) >= 0.90) / len(withf), 4),
                             "min": min(r["emb_t2_auto"] for r in withf)},
        "false_comfort_by_state": dict(Counter(e["state"] for e in ranked_examples(rows, thr))),
        "definition": "false_comfort_rate = clips where the meaning signal passed but a critical fact was lost / clips with at least one fact",
    }
    (c.ROOT / "results" / "false_comfort.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    mx = out["matrix"]
    c.log(f"False-comfort matrix (thr {thr}, n={mx['n_clips']}): " + ", ".join(f"{k} {mx[k]['n']}" for k in CELLS))
    c.log(f"False-comfort rate {overall_fc['est']:.3f} [{overall_fc['lo']:.3f}, {overall_fc['hi']:.3f}]; "
          f"metrics agreed in {agree}/{len(withf)}; examples {len(out['examples'])}")
    for t, m in out["sensitivity"].items():
        c.log(f"  sensitivity {t}: false_comfort {m['false_comfort']['n']}, reassuring {m['reassuring_agreement']['n']}")


if __name__ == "__main__":
    main()
