"""Clean Mississauga 311 contact-centre daily call data (2017 + 2018).

Known mess (DATA_DICTIONARY.md): different headers/case per year, dd/mm/yyyy (2017) vs m/d/yyyy (2018),
'97%' stored as text, holidays written into Calls_Offered, typo dates, 44 blank rows, unsorted.
Every dropped or repaired row is logged in results/311_cleaning_log.csv. Needs no API key.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime

import numpy as np

import common as c

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def num(v: str):
    v = (v or "").strip().replace("%", "").replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def parse_date(raw: str, year: int, dayname: str, log: list, src: str):
    s = (raw or "").strip()
    fixes = []
    fixed = re.sub(r"/+", "/", s)
    if fixed != s:
        fixes.append("doubled slash")
    parts = fixed.split("/")
    try:
        if year == 2017:
            d, m, y = parts
            if m == "112":
                m = "12"; fixes.append("month 112 -> 12")
        else:
            m, d, y = parts
        if y != str(year):
            fixes.append(f"year {y} -> {year} (file year)"); y = str(year)
        cand = date(int(y), int(m), int(d))
    except (ValueError, IndexError):
        log.append({"file": src, "raw_date": raw, "action": "dropped", "reason": "unparseable date"})
        return None
    if dayname and WEEKDAYS[cand.weekday()] != dayname.strip().title():
        log.append({"file": src, "raw_date": raw, "action": "flagged",
                    "reason": f"weekday mismatch: file says {dayname}, date is {WEEKDAYS[cand.weekday()]}"})
    if fixes:
        log.append({"file": src, "raw_date": raw, "action": "repaired", "reason": "; ".join(fixes) + f" => {cand.isoformat()}"})
    return cand


def load(year: int, log: list) -> list[dict]:
    src = f"Mississauga_311_Call_Data_{year}.csv"
    rows = c.read_csv(c.CITY311_DIR / src)
    out = []
    for r in rows:
        r = {k.strip().upper(): (v or "").strip() for k, v in r.items() if k}
        if not any(r.values()) or not r.get("DATE"):
            log.append({"file": src, "raw_date": r.get("DATE", ""), "action": "dropped", "reason": "blank row"})
            continue
        offered = num(r.get("CALLS_OFFERED"))
        if offered is None:
            log.append({"file": src, "raw_date": r["DATE"], "action": "dropped",
                        "reason": f"holiday/text in Calls_Offered: {r.get('CALLS_OFFERED')}"})
            continue
        d = parse_date(r["DATE"], year, r.get("DAY_OF_WEEK", ""), log, src)
        if d is None:
            continue
        out.append({"date": d.isoformat(), "year": year, "weekday": WEEKDAYS[d.weekday()],
                    "calls_offered": int(offered), "calls_answered": int(num(r.get("CALLS_ANSWERED")) or 0),
                    "pct_answered": num(r.get("PERCENTAGE_OF_CALLS_ANSWERED")),
                    "service_level_pct": num(r.get("SERVICE_LEVEL_PERCENTAGE")),
                    "handle_time_sec": num(r.get("AVERAGE_HANDLE_TIME_SEC")),
                    "speed_of_answer_sec": num(r.get("AVERAGE_SPEED_OF_ANSWER_SEC"))})
    return out


def main() -> None:
    log: list[dict] = []
    rows = load(2017, log) + load(2018, log)
    seen: dict[str, dict] = {}
    for r in rows:
        if r["date"] in seen:
            log.append({"file": f"{r['year']}", "raw_date": r["date"], "action": "dropped", "reason": "duplicate date"})
            continue
        seen[r["date"]] = r
    rows = sorted(seen.values(), key=lambda r: r["date"])
    out = c.ROOT / "results"
    out.mkdir(exist_ok=True)
    c.write_csv(out / "mississauga_311_daily_clean.csv", rows)
    c.write_csv(out / "311_cleaning_log.csv", log, ["file", "raw_date", "action", "reason"])
    calls = np.array([r["calls_offered"] for r in rows], float)
    aht = np.array([r["handle_time_sec"] for r in rows if r["handle_time_sec"]], float)
    answered = np.array([r["calls_answered"] for r in rows], float)
    # call-weighted handle time is the honest per-call number
    w_aht = float(np.sum([r["handle_time_sec"] * r["calls_answered"] for r in rows if r["handle_time_sec"]]) /
                  np.sum([r["calls_answered"] for r in rows if r["handle_time_sec"]]))
    summary = {
        "source": "City of Mississauga Open Data, 311 Citizen Contact Centre Call Data 2017, 2018",
        "business_days": len(rows), "date_range": [rows[0]["date"], rows[-1]["date"]],
        "calls_offered_per_day_mean": round(float(calls.mean()), 1),
        "calls_offered_per_day_median": float(np.median(calls)),
        "calls_answered_per_day_mean": round(float(answered.mean()), 1),
        "handle_time_sec_mean_of_days": round(float(aht.mean()), 1),
        "handle_time_sec_call_weighted": round(w_aht, 1),
        "calls_offered_per_year_est": round(float(calls.mean()) * 250),
        "by_year": {y: {"days": int(sum(r["year"] == y for r in rows)),
                        "calls_per_day": round(float(np.mean([r["calls_offered"] for r in rows if r["year"] == y])), 1),
                        "handle_time_sec": round(float(np.mean([r["handle_time_sec"] for r in rows if r["year"] == y and r["handle_time_sec"]])), 1)}
                    for y in (2017, 2018)},
        "cleaning_actions": {a: sum(x["action"] == a for x in log) for a in ("dropped", "repaired", "flagged")},
        "note": "250 business days/yr assumed for the annual estimate. 2017-2018 data; current volumes unknown.",
    }
    (out / "mississauga_311_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
