"""Mississauga screening index: benchmark-weighted language exposure (a screening index, not a forecast).

E_n = sum over tested languages l of (P_nl x R_l) / P_n
  P_nl = residents of neighbourhood n who speak tested language l most often at home (2021 Census, 100% data)
  R_l  = measured end-to-end critical-fact loss rate for l (1 - fact survival in T2 from auto-detect)
  P_n  = population_language_base
Zero-population areas are excluded. Coverage = share of P_n who speak one of the 13 tested home languages.
Bengali has no audio in the pack: reported as unmeasured, never as zero exposure.
Census note: the legend defines these columns as "language spoken most often at home"; it does not say whether
they are single responses, and that question allows more than one answer, so counts may not be unique people.
Output: results/screening.csv, results/screening.json
"""
from __future__ import annotations

import json

import common as c


def main() -> None:
    metrics = {r["lang_config"]: r for r in c.read_csv(c.ROOT / "results" / "language_metrics.csv")}
    xw = [r for r in c.crosswalk() if r["census_home_column"]]
    hoods = [r for r in c.read_csv(c.CENSUS_DIR / "neighbourhood_language_profile_2021.csv")
             if float(r["population_language_base"] or 0) > 0]
    out = []
    for h in hoods:
        base = float(h["population_language_base"])
        exp, covered, contrib, unmeasured = 0.0, 0.0, [], 0
        for x in xw:
            speakers = float(h.get(x["census_home_column"]) or 0)
            m = metrics.get(x["fleurs_config"])
            if not x["fleurs_config"] or m is None or m["fact_survival_auto"] in ("", None):
                unmeasured += speakers
                continue
            r_l = 1 - float(m["fact_survival_auto"])
            exp += speakers * r_l
            covered += speakers
            contrib.append((x["language"], speakers, speakers * r_l))
        contrib.sort(key=lambda t: -t[2])
        out.append({"name": h["neighbourhood"], "population": int(base),
                    "exposure_index": round(exp / base, 6), "exposure_per_1000": round(1000 * exp / base, 2),
                    "coverage_pct": round(100 * covered / base, 1), "tested_speakers": int(covered),
                    "unmeasured_bengali": int(unmeasured),
                    "top_languages": [t[0] for t in contrib[:3]],
                    "top_detail": [{"language": t[0], "speakers": int(t[1])} for t in contrib[:3]]})
    out.sort(key=lambda r: -r["exposure_index"])
    for i, r in enumerate(out, 1):
        r["rank"] = i
    c.write_csv(c.ROOT / "results" / "screening.csv",
                [{k: v for k, v in r.items() if k != "top_detail"} | {"top_languages": "; ".join(r["top_languages"])} for r in out])
    meta = {"name": "benchmark-weighted language exposure (screening, not a forecast)",
            "formula": "E_n = sum_l(P_nl x R_l) / P_n; R_l = 1 - fact survival (auto-detect, end to end)",
            "n_neighbourhoods": len(out), "excluded_zero_population": 56 - len(out),
            "census_note": "Columns are 'language spoken most often at home' (100% data). The legend does not state "
                           "whether these are single responses; counts may not be unique people.",
            "bengali": "No home-language audio in the pack: unmeasured, not zero."}
    (c.ROOT / "results" / "screening.json").write_text(json.dumps({"meta": meta, "neighbourhoods": out}, indent=1,
                                                                  ensure_ascii=False), encoding="utf-8")
    c.log("Top 5 by exposure: " + ", ".join(f"{r['name']} {r['exposure_per_1000']}/1000 (coverage {r['coverage_pct']}%)" for r in out[:5]))


if __name__ == "__main__":
    main()
