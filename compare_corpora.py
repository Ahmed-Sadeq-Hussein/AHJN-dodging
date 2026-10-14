# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
compare_corpora.py -- two PGD target sets, side by side. Any env, no GPU.

    python compare_corpora.py pgd_dataset_margin pgd_dataset_qc

Reads the stored targets and reports what the teacher achieved and what it
spent, on the (voice, sentence) pairs both corpora hold. Paired, because the
pairs are the same and per-clip variance across voices is far larger than the
effect being looked for -- baseline soft-min spans 0.24 to 0.96 in this
project, wider than any intervention in it.

This is a TEACHER-level comparison. It says nothing about what a student will
inherit; for that, distil both and run eval_all15.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import RESULTS_ROOT

RL = RESULTS_ROOT / "rl_runs"
FIELDS = [("softmin_after", "soft-min reached", 3, 1),
          ("gain", "soft-min gained", 3, 1),
          ("delta_abs", "|delta| (edit size)", 3, 1),
          ("sim_cost", "similarity cost", 4, 1),
          ("wer_cost", "WER cost (points)", 2, 100)]


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.pt")):
        try:
            t = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        rec = {
            "softmin_before": t.get("softmin_before"),
            "softmin_after": t.get("softmin_after"),
            "delta_abs": t.get("delta_abs"),
            "sim_cost": (t["sim_before"] - t["sim_after"]) if "sim_before" in t else None,
            "wer_cost": ((t["wer_after"] - t["wer_before"])
                         if t.get("wer_after") is not None and t.get("wer_before") is not None
                         else None),
            "pen_final": t.get("pen_final"),
            "penalty_kind": t.get("penalty_kind"),
            "voice": t.get("voice"),
        }
        if rec["softmin_after"] is not None and rec["softmin_before"] is not None:
            rec["gain"] = rec["softmin_after"] - rec["softmin_before"]
        out[f.name] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="first corpus (folder name under rl_runs, or a path)")
    ap.add_argument("b", help="second corpus")
    args = ap.parse_args()

    pa, pb = (Path(x) if Path(x).is_dir() else RL / x for x in (args.a, args.b))
    for p in (pa, pb):
        if not p.is_dir():
            print(f"no such corpus: {p}")
            return 1
    A, B = load(pa), load(pb)
    shared = sorted(set(A) & set(B))
    print(f"{pa.name}: {len(A)} targets    {pb.name}: {len(B)} targets"
          f"    shared: {len(shared)}")
    kinds = {r["penalty_kind"] for r in B.values()}
    print(f"perceptual constraint on {pb.name}: {kinds or 'none'}")
    if not shared:
        print("no shared (voice, sentence) pairs -- nothing to pair on")
        return 1

    print()
    w = max(len(lbl) for _, lbl, _, _ in FIELDS) + 2
    print(f"{'':<{w}}{pa.name[:18]:>19}{pb.name[:18]:>19}{'difference':>14}{'t':>8}")
    print("-" * (w + 60))
    try:
        from scipy import stats
    except Exception:
        stats = None

    for key, label, digits, scale in FIELDS:
        xa = np.array([A[k][key] for k in shared if A[k].get(key) is not None
                       and B[k].get(key) is not None], dtype=float) * scale
        xb = np.array([B[k][key] for k in shared if A[k].get(key) is not None
                       and B[k].get(key) is not None], dtype=float) * scale
        if not len(xa):
            continue
        d = xb - xa
        t = ""
        if stats is not None and len(d) > 2 and d.std() > 0:
            tv, pv = stats.ttest_rel(xb, xa)
            t = f"{tv:>8.2f}"
        print(f"{label:<{w}}{xa.mean():>19.{digits}f}{xb.mean():>19.{digits}f}"
              f"{d.mean():>+14.{digits}f}{t}")

    pens = [r["pen_final"] for r in B.values() if r.get("pen_final") is not None]
    if pens:
        print(f"\nfinal perceptual penalty on {pb.name}: mean {np.mean(pens):.3f}, "
              f"max {np.max(pens):.3f}")

    # the question the whole run is about
    ga = np.array([A[k]["gain"] for k in shared if A[k].get("gain") is not None])
    gb = np.array([B[k]["gain"] for k in shared if B[k].get("gain") is not None])
    da = np.array([A[k]["delta_abs"] for k in shared if A[k].get("delta_abs") is not None])
    db = np.array([B[k]["delta_abs"] for k in shared if B[k].get("delta_abs") is not None])
    if len(ga) and len(da):
        print(f"\nsoft-min gained per unit of edit:"
              f"  {pa.name} {ga.mean() / da.mean():.2f}"
              f"   {pb.name} {gb.mean() / db.mean():.2f}")
        print("A teacher that gains as much for a smaller edit is the point of the")
        print("constraint. Whether a student inherits that is a separate question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
