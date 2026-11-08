"""From clips to people: messages at risk per neighbourhood, with sensitivity checks.

messages_at_risk(n) = sum over measured languages of home_speakers(n, lang) x total_loss(lang)
Denominator for shares: population_language_base. Zero-population areas excluded.
Bengali has no home-language audio: reported as unmeasured speakers, never as zero risk.

Sensitivity re-rankings (Spearman vs base ranking):
  A. fact loss instead of judge total loss
  B. 2x weight on residents who know neither official language. The census does not cross
     language x knows-neither, so we assume the neighbourhood's knows-neither share applies
     equally to every language's speakers: weight = 1 + pct_knows_neither/100.
Usage: python src/unheard_index.py [--mock]
"""
from __future__ import annotations

import json

from scipy import stats

import common as c


def main() -> None:
    summ = {r["lang_config"]: r for r in c.read_csv(c.results_dir() / "language_summary.csv")}
    xw = c.crosswalk()
    lang_cols = {r["census_home_column"]: r for r in xw if r["census_home_column"]}
    hoods = [r for r in c.read_csv(c.CENSUS_DIR / "neighbourhood_language_profile_2021.csv")
             if float(r["population_language_base"] or 0) > 0]
    out = []
    for h in hoods:
        base = float(h["population_language_base"])
        kn = float(h["pct_knows_neither_english_nor_french"] or 0) / 100
        risk = risk_fact = risk_w = 0.0
        unmeasured = 0
        contrib = []
        for col, xr in lang_cols.items():
            speakers = float(h.get(col) or 0)
            s = summ.get(xr["fleurs_config"])
            if not xr["fleurs_config"] or s is None:
                unmeasured += speakers
                continue
            tl, fl = float(s["total_loss"]), float(s["fact_loss"] or 0)
            risk += speakers * tl
            risk_fact += speakers * fl
            risk_w += speakers * (1 + kn) * tl
            contrib.append((xr["language"], speakers * tl, speakers))
        contrib.sort(key=lambda x: -x[1])
        out.append({
            "name": h["neighbourhood"], "residents": int(base),
            "messages_at_risk": round(risk), "messages_at_risk_per_1000": round(1000 * risk / base, 1),
            "risk_fact_loss": round(risk_fact), "risk_knows_neither_2x": round(risk_w),
            "knows_neither": int(float(h["knows_neither_english_nor_french"] or 0)),
            "pct_knows_neither": float(h["pct_knows_neither_english_nor_french"] or 0),
            "unmeasured_bengali_speakers": int(unmeasured),
            "top_languages": [x[0] for x in contrib[:3]],
            "top_languages_detail": [{"language": x[0], "speakers": int(x[2]), "messages_at_risk": round(x[1])}
                                     for x in contrib[:3]],
        })

    def rank(key):
        order = sorted(out, key=lambda r: -r[key])
        return {r["name"]: i + 1 for i, r in enumerate(order)}

    r0, rf, rw = rank("messages_at_risk"), rank("risk_fact_loss"), rank("risk_knows_neither_2x")
    for r in out:
        r["rank"], r["rank_fact_loss"], r["rank_knows_neither_2x"] = r0[r["name"]], rf[r["name"]], rw[r["name"]]
    out.sort(key=lambda r: r["rank"])
    names = [r["name"] for r in out]
    sp_f = stats.spearmanr([r0[n] for n in names], [rf[n] for n in names])[0]
    sp_w = stats.spearmanr([r0[n] for n in names], [rw[n] for n in names])[0]
    top10 = set(names[:10])
    sens = {
        "n_neighbourhoods": len(out), "excluded_zero_population": 56 - len(out),
        "spearman_base_vs_fact_loss": round(float(sp_f), 4),
        "spearman_base_vs_knows_neither_2x": round(float(sp_w), 4),
        "top10_overlap_fact_loss": len(top10 & {r["name"] for r in out if r["rank_fact_loss"] <= 10}),
        "top10_overlap_knows_neither_2x": len(top10 & {r["name"] for r in out if r["rank_knows_neither_2x"] <= 10}),
        "city_messages_at_risk": sum(r["messages_at_risk"] for r in out),
        "city_unmeasured_bengali_speakers": sum(r["unmeasured_bengali_speakers"] for r in out),
        "assumption_knows_neither": "neighbourhood knows-neither share applied equally across languages",
        "data_status": c.data_status(),
    }
    c.write_csv(c.results_dir() / "neighbourhoods.csv", [{k: v for k, v in r.items() if k != "top_languages_detail"} | {
        "top_languages": "; ".join(r["top_languages"])} for r in out])
    (c.results_dir() / "neighbourhoods.json").write_text(json.dumps({"sensitivity": sens, "neighbourhoods": out},
                                                                    indent=1, ensure_ascii=False), encoding="utf-8")
    c.log(f"Top 5: {[(r['name'], r['messages_at_risk']) for r in out[:5]]}")
    c.log(f"Sensitivity: {sens}")


if __name__ == "__main__":
    main()
