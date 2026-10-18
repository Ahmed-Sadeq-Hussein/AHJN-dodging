# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
Draw the figures.

    python scripts/make_graphs.py                  # everything
    python scripts/make_graphs.py --list           # what each script produces
    python scripts/make_graphs.py --only mos architecture
    python scripts/make_graphs.py --open           # and show the folder

Every figure is drawn from stored results, never from a model, so this is
fast and deterministic: run it after any analysis change and the figures
follow. Output goes to GRAPHS_DIR, which is graphs/ in the repo unless .env
moves it, and graphs/INDEX.md says what each one shows.

A script that fails is reported and does not stop the others, because the
usual cause is a results file that has not been produced on this machine
rather than a broken figure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env_config import GRAPHS_DIR

# key -> (script and its arguments, what it draws)
FIGURES = {
    "mos":          (["plot_mos.py", "--split"],
                     "20a-20e  the listening test, MOS and PMOS"),
    "plan_x":       (["plot_plan_x.py"],
                     "1-5      training curves and diagnostics"),
    "distill":      (["plot_distill.py"],
                     "9-14     distillation, capacity sweep, quality cost"),
    "qc_frontier":  (["plot_qc_frontier.py"],
                     "19       the frontier after constraining the teacher"),
    "frontier":     (["plot_frontier.py"],
                     "17       soft-min against the quality budget"),
    "margin":       (["plot_margin.py"],
                     "15-16    the logit margin objective"),
    "ab_voices":    (["plot_ab_voices.py"],
                     "7-8      per-voice detector and quality"),
    "longform":     (["plot_longform.py"],
                     "18       does the edit survive longer clips"),
    "architecture": (["plot_architecture.py"],
                     "21       how the system fits together"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate the project's figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=list(FIGURES),
                    help="draw just these")
    ap.add_argument("--list", action="store_true",
                    help="list what each one produces and exit")
    ap.add_argument("--open", action="store_true",
                    help="reveal the output folder when finished")
    args = ap.parse_args()

    if args.list:
        print(f"output goes to {GRAPHS_DIR}\n")
        for key, (cmd, what) in FIGURES.items():
            print(f"  {key:<14}{what}")
            print(f"  {'':<14}{' '.join(cmd)}")
        return 0

    wanted = args.only or list(FIGURES)
    before = {p.name for p in GRAPHS_DIR.glob("*") if p.is_file()}

    failures = []
    for key in wanted:
        cmd, what = FIGURES[key]
        if not (REPO / cmd[0]).is_file():
            print(f"[skip] {key}: {cmd[0]} is not in the repo")
            continue
        print(f"\n[{key}] {what}")
        start = time.time()
        rc = subprocess.call([sys.executable, *cmd], cwd=str(REPO))
        if rc == 0:
            print(f"  done in {time.time() - start:.1f}s")
        else:
            failures.append((key, rc))
            print(f"  FAILED with exit code {rc}")

    after = {p.name for p in GRAPHS_DIR.glob("*") if p.is_file()}
    numbered = sorted(n for n in after if n[:1].isdigit())
    print("\n" + "=" * 62)
    print(f"{len(numbered)} figure files in {GRAPHS_DIR}")
    for name in sorted(after - before):
        print(f"  new: {name}")
    if failures:
        print("\nfailed:")
        for key, rc in failures:
            print(f"  {key} (exit {rc}) -- usually a results file this machine "
                  f"has not produced")
    print("\nwhat each figure shows: graphs/INDEX.md")

    if args.open:
        try:
            import os
            os.startfile(GRAPHS_DIR)       # Windows
        except Exception:
            print(f"(could not open a file browser; the folder is {GRAPHS_DIR})")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
