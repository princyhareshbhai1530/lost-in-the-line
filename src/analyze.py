"""Per-language losses with bootstrap 95% intervals, speaker-sex and dialect slices, evaluator checks,
signal-test gate, and the measured fix (language given up front).

Usage: python src/analyze.py [--signal] [--mock]
Writes results/language_summary.csv, results/sex_check.csv, results/eval_summary.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from scipy import stats

import common as c

RNG = np.random.default_rng(c.CONFIG["random_seed"])
B = c.CONFIG["bootstrap_reps"]


def load(stage: str, signal: bool = False) -> list[dict]:
    p = c.results_dir() / f"clips_scored_{stage}{'_signal' if signal else ''}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def boot_mean(x, reps: int = B) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(x), (reps, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, reps: int = B) -> tuple[float, float, float]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(num), (reps, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def language_table(rows: list[dict]) -> list[dict]:
    out = []
    for lang in sorted({r["lang_config"] for r in rows}):
        g = [r for r in rows if r["lang_config"] == lang]
        tot, tlo, thi = boot_mean([r["total_loss"] for r in g])
        hear, hlo, hhi = boot_mean([r["hearing_loss"] for r in g])
        trn, nlo, nhi = boot_mean([r["translation_loss"] for r in g])
        fs, flo, fhi = boot_ratio([r["facts_survived_heard"] for r in g], [r["n_facts"] for r in g])
        fst, _, _ = boot_ratio([r["facts_survived_true"] for r in g], [r["n_facts"] for r in g])
        cer_, clo, chi = boot_mean([r["cer"] for r in g])
        emb, elo, ehi = boot_mean([r["embedding_heard"] for r in g])
        genders = sorted({r["gender"] for r in g})
        lflip, lflo, lfhi = boot_mean([0.0 if r["language_correct"] else 1.0 for r in g])
        sflip, sflo, sfhi = boot_mean([0.0 if r["script_correct"] else 1.0 for r in g])
        wrong = Counter(r["detected_language"] or "?" for r in g if not r["language_correct"])
        ok_script = [r["cer"] for r in g if r["script_correct"]]
        out.append({
            "lang_config": lang, "language": g[0]["language"], "n_clips": len(g),
            "hearing_loss": r4(hear), "hearing_lo": r4(hlo), "hearing_hi": r4(hhi),
            "translation_loss": r4(trn), "translation_lo": r4(nlo), "translation_hi": r4(nhi),
            "total_loss": r4(tot), "total_lo": r4(tlo), "total_hi": r4(thi),
            "facts_survived_pct": r4(fs * 100), "facts_lo": r4(flo * 100), "facts_hi": r4(fhi * 100),
            "facts_survived_true_pct": r4(fst * 100),
            "fact_loss": r4(1 - fs),
            "embedding_similarity": r4(emb), "embedding_lo": r4(elo), "embedding_hi": r4(ehi),
            "cer": r4(cer_), "cer_lo": r4(clo), "cer_hi": r4(chi),
            "wer": r4(np.mean([r["wer"] for r in g])),
            "language_detected_correct_pct": r4(100 * np.mean([r["language_correct"] for r in g])),
            "script_correct_pct": r4(100 * np.mean([r["script_correct"] for r in g])),
            "language_flip_pct": r4(100 * lflip), "language_flip_lo": r4(100 * lflo), "language_flip_hi": r4(100 * lfhi),
            "script_flip_pct": r4(100 * sflip), "script_flip_lo": r4(100 * sflo), "script_flip_hi": r4(100 * sfhi),
            "wrong_language_detected": "; ".join(f"{k}:{v}" for k, v in wrong.most_common()),
            "cer_excl_script_flips": r4(np.mean(ok_script)) if ok_script else None,
            "han_simplified_clips": sum(r.get("han_variant") == "simplified" for r in g),
            "break_ear": sum(r["break_station"] == "ear" for r in g),
            "break_translator": sum(r["break_station"] == "translator" for r in g),
            "break_none": sum(r["break_station"] == "none" for r in g),
            "genders": "/".join(genders), "single_sex": len(genders) == 1,
            "variety_note": g[0]["variety_note"],
        })
    return out


def sex_check(rows: list[dict]) -> list[dict]:
    out = []
    for lang in sorted({r["lang_config"] for r in rows}):
        g = [r for r in rows if r["lang_config"] == lang]
        for sex in sorted({r["gender"] for r in g}):
            s = [r for r in g if r["gender"] == sex]
            tot, lo, hi = boot_mean([r["total_loss"] for r in s])
            cm, clo, chi = boot_mean([r["cer"] for r in s])
            out.append({"lang_config": lang, "language": g[0]["language"], "gender": sex, "n": len(s),
                        "total_loss": r4(tot), "total_lo": r4(lo), "total_hi": r4(hi),
                        "cer": r4(cm), "cer_lo": r4(clo), "cer_hi": r4(chi),
                        "single_sex_language": len({r["gender"] for r in g}) == 1})
    return out


def pooled_sex_effect(rows: list[dict]) -> dict:
    """Within mixed-sex languages only: mean (female - male) CER, demeaned by language, bootstrap CI."""
    diffs = []
    for lang in {r["lang_config"] for r in rows}:
        g = [r for r in rows if r["lang_config"] == lang]
        f = [r["cer"] for r in g if r["gender"] == "female"]
        m = [r["cer"] for r in g if r["gender"] == "male"]
        if f and m:
            diffs.append((lang, np.mean(f) - np.mean(m), len(f), len(m)))
    if not diffs:
        return {"note": "no mixed-sex languages in this run"}
    d = np.array([x[1] for x in diffs])
    mean, lo, hi = boot_mean(d)
    return {"female_minus_male_cer": r4(mean), "ci": [r4(lo), r4(hi)], "n_mixed_languages": len(diffs),
            "per_language": {x[0]: {"diff": r4(x[1]), "n_female": x[2], "n_male": x[3]} for x in diffs}}


def evaluator_checks(rows: list[dict]) -> dict:
    """Do the three meaning measures agree? English T1 is the identity, so it is excluded."""
    judge, emb, fact_j, fact_f = [], [], [], []
    for r in rows:
        pairs = [("heard", r["meaning_heard"], r["embedding_heard"], r["facts_survived_heard"])]
        if r["lang_config"] != "en_us":
            pairs.append(("true", r["meaning_true"], r["embedding_true"], r["facts_survived_true"]))
        for _, j, e, fs in pairs:
            judge.append(j); emb.append(e)
            if r["n_facts"]:
                fact_j.append(j); fact_f.append(fs / r["n_facts"])
    res = {"n_translations": len(judge)}
    if len(judge) > 2 and np.std(judge) > 0 and np.std(emb) > 0:
        res["judge_vs_embedding_pearson"] = r4(stats.pearsonr(judge, emb)[0])
        res["judge_vs_embedding_spearman"] = r4(stats.spearmanr(judge, emb)[0])
    if len(fact_j) > 2 and np.std(fact_j) > 0 and np.std(fact_f) > 0:
        res["judge_vs_facts_spearman"] = r4(stats.spearmanr(fact_j, fact_f)[0])
        res["n_translations_with_facts"] = len(fact_j)
    if len(fact_j) > 2 and np.std(fact_f) > 0:
        e2 = [e for r in rows for e in ([r["embedding_heard"]] + ([r["embedding_true"]] if r["lang_config"] != "en_us" else []))
              if r["n_facts"]]
        if len(e2) == len(fact_f) and np.std(e2) > 0:
            res["embedding_vs_facts_spearman"] = r4(stats.spearmanr(e2, fact_f)[0])
    return res


def fix_effect(base: list[dict], fixed: list[dict]) -> list[dict]:
    """Paired bootstrap: same clips, language auto-detected vs language given up front."""
    out = []
    b = {r["id"]: r for r in base}
    for lang in sorted({r["lang_config"] for r in fixed}):
        pairs = [(b[r["id"]]["total_loss"], r["total_loss"]) for r in fixed if r["lang_config"] == lang and r["id"] in b]
        if not pairs:
            continue
        before = np.array([p[0] for p in pairs]); after = np.array([p[1] for p in pairs])
        drop, lo, hi = boot_mean(before - after)
        out.append({"lang_config": lang, "n_clips": len(pairs), "total_loss_before": r4(before.mean()),
                    "fixed_total_loss": r4(after.mean()), "drop": r4(drop), "drop_lo": r4(lo), "drop_hi": r4(hi)})
    return out


def gate(table: list[dict], ev: dict) -> dict:
    ranked = sorted(table, key=lambda r: r["total_loss"])
    best, worst = ranked[0], ranked[-1]
    differ_total = worst["total_lo"] > best["total_hi"]
    split = {r["lang_config"]: ("ear" if r["hearing_loss"] > r["translation_loss"] else "translator") for r in table}
    differ_split = len(set(v for k, v in split.items() if k != "en_us")) > 1
    fs = [r["facts_survived_pct"] for r in table if r["facts_survived_pct"] is not None]
    differ_facts = bool(fs) and (max(fs) - min(fs) >= 15)
    agree = ev.get("judge_vs_embedding_spearman")
    agree_ok = agree is not None and agree >= 0.5
    # objective ear signal that needs no AI judge: wrong language or wrong script for >=10% of a language's clips
    flips = {r["lang_config"]: max(r["language_flip_pct"] or 0, r["script_flip_pct"] or 0)
             for r in table if r["lang_config"] != "en_us"}
    flip_langs = sorted([k for k, v in flips.items() if v >= 10], key=lambda k: -flips[k])
    signal = differ_total or differ_split or differ_facts or bool(flip_langs)
    if signal and agree_ok:
        decision = "GO"
    elif signal:
        decision = "GO on objective signal; judge agreement weak - check hand-check before trusting judge-only numbers"
    else:
        decision = "ADJUST (see roadmap section 7)"
    return {"languages_differ_total_loss_ci_separated": bool(differ_total), "dominant_break_differs": differ_split,
            "fact_survival_spread_ge_15pts": differ_facts, "dominant_break_by_language": split,
            "ear_language_or_script_flips_ge_10pct": flip_langs,
            "judge_embedding_spearman": agree, "measures_broadly_agree": agree_ok, "decision": decision}


def main() -> None:
    signal = "--signal" in sys.argv
    rows = load("autodetect", signal)
    if not rows:
        sys.exit("No scored clips found. Run src/score.py first.")
    table = language_table(rows)
    out = c.results_dir()
    sfx = "_signal" if signal else ""
    c.write_csv(out / f"language_summary{sfx}.csv", table)
    c.write_csv(out / f"sex_check{sfx}.csv", sex_check(rows))
    ev = evaluator_checks(rows)
    summary = {"data_status": c.data_status(), "n_clips": len(rows), "models": {
        "scribe": c.CONFIG["scribe_model"], "gemini": c.CONFIG["gemini_model"],
        "embedding": c.CONFIG["embedding_model"], "temperature": c.CONFIG["temperature"]},
        "bootstrap_reps": B, "seed": c.CONFIG["random_seed"],
        "evaluator": ev, "sex_effect": pooled_sex_effect(rows), "gate": gate(table, ev)}
    fixed = load("langfirst", signal)
    if fixed:
        summary["fix"] = fix_effect(rows, fixed)
        c.write_csv(out / f"fix_effect{sfx}.csv", summary["fix"])
    # Fix rerun targets: the fix (language given up front) addresses language/script errors first,
    # so rank by flip rate, then by total loss.
    worst = sorted([r for r in table if r["lang_config"] != "en_us"],
                   key=lambda r: (-max(r["language_flip_pct"] or 0, r["script_flip_pct"] or 0), -r["total_loss"]))
    summary["worst_languages"] = [r["lang_config"] for r in worst[: c.CONFIG["fix_rerun_n_worst"]]]
    summary["fix_target_rule"] = "highest language/script flip rate, then highest total loss"
    (out / f"eval_summary{sfx}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    c.log(f"{'lang':12} {'n':>3} {'hear':>6} {'trans':>6} {'total [95% CI]':>22} {'facts%':>7} {'CER':>6} "
          f"{'CERok':>6} {'flip%':>6}  wrong-lang")
    for r in sorted(table, key=lambda r: -r["total_loss"]):
        cok = r["cer_excl_script_flips"]
        c.log(f"{r['lang_config']:12} {r['n_clips']:>3} {r['hearing_loss']:>6.3f} {r['translation_loss']:>6.3f} "
              f"{r['total_loss']:>6.3f} [{r['total_lo']:.3f},{r['total_hi']:.3f}] {r['facts_survived_pct'] or 0:>7.1f} {r['cer']:>6.3f} "
              f"{(cok if cok is not None else float('nan')):>6.3f} {max(r['language_flip_pct'] or 0, r['script_flip_pct'] or 0):>6.1f}"
              f"  {r['wrong_language_detected']}{'  single-sex' if r['single_sex'] else ''}")
    c.log(f"Evaluator: {ev}")
    c.log(f"GATE: {summary['gate']['decision']}  | worst for fix rerun: {summary['worst_languages']}")


if __name__ == "__main__":
    main()
