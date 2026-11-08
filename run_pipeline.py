"""One command per phase. Add --mock to any of them for the no-credit dry run.

  python run_pipeline.py dryrun      # phase 0: whole pipeline on FAKE providers (no keys, no credits)
  python run_pipeline.py signal      # phase 1: 40 clips (Urdu, Punjabi, Tagalog, English) + gate
  python run_pipeline.py full        # phase 2: all 560 clips, then analysis
  python run_pipeline.py fix         # phase 3: rerun 3 worst languages with language given up front
  python run_pipeline.py build       # phase 3: analysis + neighbourhoods (legacy summary files)
  python run_pipeline.py evidence    # build spec: eval table, facts, false comfort, power-up, screening, app data
  python run_pipeline.py hint hi_in,ta_in   # add languages to the language-supplied rerun (costs Scribe credits)
Everything is cached, so re-running costs nothing for clips already done.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def run(*args: str, mock: bool = False) -> None:
    cmd = [PY, "-X", "utf8", *args] + (["--mock"] if mock else [])
    print(">", " ".join(cmd[3:]), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def build(mock: bool) -> None:
    run("src/analyze.py", mock=mock)
    run("src/unheard_index.py", mock=mock)
    run("src/build_game_json.py", mock=mock)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    mock = "--mock" in sys.argv or stage == "dryrun"
    if stage == "dryrun":
        run("src/clean_311.py")
        run("src/score.py", mock=True)
        build(True)
        res = json.loads((ROOT / "results_mock" / "eval_summary.json").read_text(encoding="utf-8"))
        run("src/score.py", "--language-first", "--langs", ",".join(res["worst_languages"]), mock=True)
        build(True)
        run("src/handcheck.py", "export", mock=True)
    elif stage == "signal":
        run("src/score.py", "--signal", mock=mock)
        run("src/analyze.py", "--signal", mock=mock)
    elif stage == "full":
        run("src/clean_311.py")
        run("src/score.py", mock=mock)
        run("src/analyze.py", mock=mock)
    elif stage == "fix":
        res_dir = ROOT / ("results_mock" if mock else "results")
        worst = json.loads((res_dir / "eval_summary.json").read_text(encoding="utf-8"))["worst_languages"]
        run("src/score.py", "--language-first", "--langs", ",".join(worst), mock=mock)
        run("src/analyze.py", mock=mock)
    elif stage == "build":
        build(mock)
    elif stage == "evidence":   # build spec: all offline, from the cache (0 API calls)
        for s in ("src/build_eval_table.py", "src/false_comfort.py", "src/powerup.py", "src/screening.py",
                  "src/export_app_data.py"):
            run(s)
    elif stage == "hint":       # language supplied to Scribe for more languages, e.g. hint hi_in,ta_in
        langs = sys.argv[2]
        run("src/score.py", "--language-first", "--langs", langs)
        run("src/analyze.py")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
