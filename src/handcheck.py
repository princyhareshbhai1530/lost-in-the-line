"""Human spot-check of the AI judge (build spec 2.7).

export: py src/handcheck.py export
  Writes results/handcheck.csv: 20 random end-to-end pairs (seed 42), columns label_member_a and label_member_b
  left blank. The judge's labels go to results/handcheck_judge_key.csv so labelling stays blind.
  Each member fills ONLY their own column with `same` or `changed`, without looking at the other column.
  Rule: `changed` only if 311 staff would record a different fact or act differently.
score:  py src/handcheck.py score
  Human vs human agreement (N/20). Pairs where the two disagree need `label_reconciled` (agreed together);
  judge vs human agreement is reported once every pair has a final label. Judge same/minor count as same.
  Results go into results/eval_summary.json under "handcheck".
"""
from __future__ import annotations

import json
import random
import sys

import common as c
import config

COLS = ["pair_id", "clip_id", "language", "english_reference", "staff_received", "label_member_a",
        "label_member_b", "label_reconciled"]


def export() -> None:
    rows = [json.loads(x) for x in (c.ROOT / "results" / "eval_table.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = c.ROOT / "results" / "handcheck.csv"
    if out.exists() and any((r.get("label_member_a") or r.get("label_member_b") or "").strip() for r in c.read_csv(out)):
        sys.exit("results/handcheck.csv already has labels; refusing to overwrite.")
    pool = [r for r in rows if r["lang_config"] != "en_us"]
    picks = random.Random(config.SEED).sample(pool, 20)
    c.write_csv(out, [{"pair_id": i, "clip_id": r["clip_id"], "language": r["language"],
                       "english_reference": r["english_ref_raw"], "staff_received": r["t2_auto_translation"],
                       "label_member_a": "", "label_member_b": "", "label_reconciled": ""}
                      for i, r in enumerate(picks, 1)], COLS)
    c.write_csv(c.ROOT / "results" / "handcheck_judge_key.csv",
                [{"pair_id": i, "judge_label": r["judge_t2_auto"]} for i, r in enumerate(picks, 1)])
    for old in ("handcheck_A.csv", "handcheck_B.csv", "handcheck_reconcile.csv"):
        p = c.ROOT / "results" / old
        if p.exists() and not any((r.get("human_label") or "").strip() for r in c.read_csv(p)):
            p.unlink()   # empty files from the earlier format
    c.log("Wrote results/handcheck.csv (20 pairs, seed 42). Fill label_member_a / label_member_b with same or changed.")


def score() -> None:
    rows = c.read_csv(c.ROOT / "results" / "handcheck.csv")
    key = {r["pair_id"]: r["judge_label"] for r in c.read_csv(c.ROOT / "results" / "handcheck_judge_key.csv")}
    ok = {"same", "changed"}
    norm = lambda v: (v or "").strip().lower()
    n = len(rows)
    jb = lambda l: "same" if l in config.JUDGE_PASS else "changed"
    single = all(not norm(r["label_member_b"]) for r in rows)
    if single:  # one team member labelled: judge vs human only; human vs human is not measured, never invented
        missing = [r["pair_id"] for r in rows if norm(r["label_member_a"]) not in ok]
        if missing:
            sys.exit(f"Pairs {missing} are not labelled same/changed yet.")
        agree = sum(jb(key[r["pair_id"]]) == norm(r["label_member_a"]) for r in rows)
        res = {"n": n, "raters": 1, "judge_human_agreement": f"{agree}/{n}",
               "human_human_agreement": "not measured (single rater)",
               "disagreements_with_judge": [r["pair_id"] for r in rows if jb(key[r["pair_id"]]) != norm(r["label_member_a"])],
               "status": "complete (single rater)"}
        p = c.ROOT / "results" / "eval_summary.json"
        summ = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        summ["handcheck"] = res
        p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
        c.log(json.dumps(res, indent=2))
        return
    missing = [r["pair_id"] for r in rows if norm(r["label_member_a"]) not in ok or norm(r["label_member_b"]) not in ok]
    if missing:
        sys.exit(f"Pairs {missing} are not labelled same/changed by both members yet.")
    hh = sum(norm(r["label_member_a"]) == norm(r["label_member_b"]) for r in rows)
    final, pending = {}, []
    for r in rows:
        a, b, rec = norm(r["label_member_a"]), norm(r["label_member_b"]), norm(r["label_reconciled"])
        lab = a if a == b else rec if rec in ok else None
        if lab:
            final[r["pair_id"]] = lab
        else:
            pending.append(r["pair_id"])
    res = {"n": n, "human_human_agreement": f"{hh}/{n}",
           "disagreements": [r["pair_id"] for r in rows if norm(r["label_member_a"]) != norm(r["label_member_b"])]}
    if pending:
        res["status"] = f"Fill label_reconciled for pairs {pending}, then run score again."
    else:
        res["judge_human_agreement"] = f"{sum(jb(key[k]) == v for k, v in final.items())}/{n}"
        res["status"] = "complete"
    p = c.ROOT / "results" / "eval_summary.json"
    summ = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    summ["handcheck"] = res
    p.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    c.log(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"export": export, "score": score}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
