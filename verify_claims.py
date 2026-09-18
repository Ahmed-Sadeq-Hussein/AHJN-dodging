r"""
verify_claims.py -- re-derive the headline numbers from stored results.

    python verify_claims.py            # any env; reads JSON only, no GPU

Plan_X/RESULTS.md and README.md quote a lot of numbers. Several were wrong at
some point -- a mismatched denominator in section 5, a recovery figure measured
on the wrong subset in 7h, a "free" result whose cost had never been measured
in 7k -- and each was caught by hand, late. This checks the load-bearing ones
against the JSON the runs actually wrote, so the document can be re-verified in
a second instead of re-read.

It asserts nothing about method. It answers one question: does the number in
the write-up still match the number in the file it came from?

Exit code 0 if every claim matches, 1 otherwise. A missing results file is
reported as SKIP, not failure -- results live outside the repo under
RESULTS_ROOT and a fresh clone will not have them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import RESULTS_ROOT

RL = RESULTS_ROOT / "rl_runs"
TOL = 0.005  # numbers are quoted to 3 decimals


def _load(name):
    p = RL / name
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else None


def _mean(rows, *path):
    out = []
    for r in rows:
        v = r
        for k in path:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        if v is not None:
            out.append(float(v))
    return float(np.mean(out)) if out else None


def main() -> int:
    checks = []          # (label, claimed, actual_or_None, section)

    def claim(label, expected, actual, where, tol=TOL):
        checks.append((label, expected, actual, where, tol))

    # ---- sections 3 and 4: the two RL designs -------------------------------
    s = None
    for cand in (Path("graphs/summary.json"), RL / "summary.json"):
        if cand.is_file():
            s = json.loads(cand.read_text(encoding="utf8"))
            break
    if s:
        claim("Design A slope", -0.000727, s["design_a"]["slope"], "3", 1e-5)
        claim("Design B slope", +0.000033, s["design_b"]["slope"], "4", 1e-5)
        claim("Design B iterations", 315, s["design_b"]["iterations"], "4", 0.5)
        claim("Design B episodes", 2520, s["design_b"]["episodes"], "4", 0.5)

    # ---- section 7n / 7p: the all-15 operating points ------------------------
    fin = _load("all15_final.json")
    if fin:
        c = fin["clips"]
        claim("all-15 unedited soft-min", 0.276, _mean(c, "baseline", "softmin"), "7n")
        claim("all-15 probability x10", 0.489, _mean(c, "old", "softmin"), "7n")
        claim("all-15 margin x4", 0.463, _mean(c, "new", "softmin"), "7n")
        claim("all-15 clip count", 60, len(c), "7n", 0.5)

    big = _load("all15_eval.json")
    if big:
        c = big["clips"]
        claim("all-15 margin x14", 0.800, _mean(c, "new", "softmin"), "7n")
        n_pass = sum(1 for v in big["per_voice"] if v["new"]["softmin"] >= 0.5)
        claim("voices >= 0.5 at x14", 14, n_pass, "7n", 0.5)

    # ---- section 7r: long form under the global-file objective ---------------
    for name, length, gain, sec in (("longform_attack_medlong.json", "medium", 0.771, "7r"),
                                    ("longform_attack_medlong.json", "long", 0.405, "7r"),
                                    ("longform_xlong.json", "xlong", 0.427, "7r")):
        d = _load(name)
        if not d:
            continue
        rows = [r for r in d["rows"] if r["length"] == length]
        if rows:
            got = float(np.mean([r["attacked"]["softmin"] - r["base"]["softmin"]
                                 for r in rows]))
            claim(f"long form gain, {length}", gain, got, sec)
            # skip_frac only exists for runs made after the OOM instrumentation
            # landed. Absent means UNKNOWN, not zero -- reporting a missing
            # field as "no skips" would assert something never measured.
            skipped = [r["skip_frac"] for r in rows if "skip_frac" in r]
            claim(f"long form OOM skips, {length}", 0.0,
                  max(skipped) if len(skipped) == len(rows) else None, sec, 1e-9)

    # ---- section 7: held-out transfer ---------------------------------------
    ev = RL / "final_eval" / "final_eval.json"
    if ev.is_file():
        d = json.loads(ev.read_text(encoding="utf8"))
        held = [g for g in d["groups"] if g["label"].startswith("HELD-OUT")]
        if held:
            g = held[0]
            claim("held-out baseline soft-min", 0.1132, g["softmin"]["baseline"], "7")
            claim("held-out distilled soft-min", 0.1496, g["softmin"]["distilled"], "7")
            claim("held-out baseline mean det", 0.3400, g["meandet"]["baseline"], "7")

    # ---- report --------------------------------------------------------------
    if not checks:
        print("no results found under", RL, "-- nothing to verify")
        return 0

    width = max(len(c[0]) for c in checks)
    fails = 0
    print(f"{'claim':<{width}}  {'written':>10}  {'actual':>10}  where")
    print("-" * (width + 36))
    for label, expected, actual, where, tol in checks:
        if actual is None:
            print(f"{label:<{width}}  {expected:>10}  {'SKIP':>10}  {where}")
            continue
        ok = abs(float(actual) - float(expected)) <= tol
        fails += not ok
        mark = "" if ok else "   <-- MISMATCH"
        print(f"{label:<{width}}  {expected:>10.4g}  {actual:>10.4g}  {where}{mark}")

    n = sum(1 for c in checks if c[2] is not None)
    print("-" * (width + 36))
    print(f"{n - fails}/{n} verified" + ("" if not fails else f", {fails} MISMATCHED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
