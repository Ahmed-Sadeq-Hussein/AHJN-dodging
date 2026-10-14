# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
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

    # ---- 7n addendum: base vs speaker-conditioned, head to head -------------
    bs = _load("all15_base_vs_spk.json")
    if bs:
        c = bs["clips"]
        claim("base 21k soft-min", 0.4540, _mean(c, "new", "softmin"), "7n")
        claim("spk 31k soft-min", 0.4632, _mean(c, "old", "softmin"), "7n")
        claim("base/spk unedited", 0.2764, _mean(c, "baseline", "softmin"), "7n")
        for arm, lab in (("new", "base"), ("old", "spk")):
            n_pass = sum(1 for v in bs["per_voice"] if v[arm]["softmin"] >= 0.5)
            claim(f"{lab} voices >= 0.5", 6, n_pass, "7n", 0.5)

    # ---- 7s: the quality-constrained teacher --------------------------------
    qc = _load("all15_qc_vs_ctl.json")
    if qc:
        c = qc["clips"]
        claim("7s control soft-min", 0.4580, _mean(c, "old", "softmin"), "7s")
        claim("7s constrained soft-min", 0.4049, _mean(c, "new", "softmin"), "7s")
        bp = _mean(c, "baseline", "pesq")
        claim("7s constrained d PESQ", 0.004, _mean(c, "new", "pesq") - bp, "7s")
        claim("7s control d PESQ", -0.125, _mean(c, "old", "pesq") - bp, "7s")

    q12 = _load("all15_qc12_vs_best.json")
    if q12:
        c = q12["clips"]
        claim("7s qc x12 soft-min", 0.5881, _mean(c, "new", "softmin"), "7s")
        claim("7s qc x12 d PESQ", -0.209, _mean(c, "new", "pesq")
              - _mean(c, "baseline", "pesq"), "7s")
        n_pass = sum(1 for v in q12["per_voice"] if v["new"]["softmin"] >= 0.5)
        claim("7s qc x12 voices >= 0.5", 11, n_pass, "7s", 0.5)

    # ---- 7s: does more data help? no ---------------------------------------
    c330 = _load("all15_qc330_vs_qc180.json")
    if c330:
        c = c330["clips"]
        claim("7s 330-target soft-min", 0.5960, _mean(c, "new", "softmin"), "7s")
        claim("7s 180-target soft-min", 0.5881, _mean(c, "old", "softmin"), "7s")

    # ---- 7u: DAgger ---------------------------------------------------------
    dg = _load("all15_dagger_vs_qc330.json")
    if dg:
        c = dg["clips"]
        bp = _mean(c, "baseline", "pesq")
        claim("7u DAgger x12 soft-min", 0.5292, _mean(c, "new", "softmin"), "7u")
        claim("7u DAgger x12 d PESQ", -0.060, _mean(c, "new", "pesq") - bp, "7u")
    sw = _load("all15_dagger24_vs_18.json")
    if sw:
        c = sw["clips"]
        bp = _mean(c, "baseline", "pesq")
        claim("7u DAgger x18 soft-min", 0.6271, _mean(c, "old", "softmin"), "7u")
        claim("7u DAgger x18 d PESQ", -0.170, _mean(c, "old", "pesq") - bp, "7u")
        claim("7u DAgger x24 soft-min", 0.7262, _mean(c, "new", "softmin"), "7u")
        n = sum(1 for v in sw["per_voice"] if v["old"]["softmin"] >= 0.5)
        claim("7u DAgger x18 voices >= 0.5", 10, n, "7u", 0.5)

    # ---- 7t: the hybrid loss, a failure ------------------------------------
    hy = _load("all15_qcobj_vs_qc.json")
    if hy:
        c = hy["clips"]
        bp = _mean(c, "baseline", "pesq")
        claim("7t pure MSE x12 soft-min", 0.5881, _mean(c, "old", "softmin"), "7t")
        claim("7t hybrid x12 soft-min", 0.4126, _mean(c, "new", "softmin"), "7t")
        claim("7t hybrid x12 d PESQ", -2.147, _mean(c, "new", "pesq") - bp, "7t", 0.02)
        claim("7t hybrid x12 WER", 0.084, _mean(c, "new", "wer"), "7t")

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
