r"""
probe_teacher.py -- how high can the TEACHER go? RUN IN "voiceclone".

    python probe_teacher.py --grid

Why this is now the question
---------------------------
The distilled operator applied 10 times recovers 97.1% of PGD's gain
(section 7b). It has essentially caught its teacher. Every remaining
improvement therefore has to come from the teacher being better, not the
student learning more -- collecting more targets from a 60-step PGD run
cannot push past what 60-step PGD achieves.

PGD's configuration has never been tuned. It was set once, in the original
feasibility probe, at 60 steps / lr 0.05 / max_delta 0.6, and every target in
every dataset since has inherited those numbers. This sweeps them on stored
mels, so no generation is needed and the whole grid costs minutes.

Also swept: whether the soft-min/mean blend weight used for the objective is
the right one to attack. Weighting the mean more heavily spreads the attack
across detectors; weighting soft-min concentrates it on the worst one, which
is the number actually reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import diffscore
from collect_pgd import pgd_solve
from env_config import RESULTS_ROOT

RL_RUNS = RESULTS_ROOT / "rl_runs"
DATA = RL_RUNS / "pgd_dataset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--hardest", action="store_true",
                    help="select the LOWEST-baseline examples instead of the "
                         "first alphabetically")
    ap.add_argument("--steps", default="60,120,240")
    ap.add_argument("--lrs", default="0.02,0.05,0.1")
    ap.add_argument("--max-deltas", default="0.6")
    ap.add_argument("--softmin-weights", default="0.5")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    steps = [int(x) for x in args.steps.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    mds = [float(x) for x in args.max_deltas.split(",")]
    sws = [float(x) for x in args.softmin_weights.split(",")]

    # Selecting files alphabetically picked examples whose baseline soft-min
    # was already 0.92 -- PGD reaches 1.0 on those regardless of settings, so
    # the whole grid came out flat and said nothing. The teacher's ceiling is
    # only meaningful on the clips the detectors actually catch.
    files = sorted(DATA.glob("*.pt"))
    items = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    if args.hardest:
        items.sort(key=lambda d: d["softmin_before"])
    items = items[:args.limit]
    print(f"{len(items)} stored mels; current targets used "
          f"steps=60 lr=0.05 max_delta=0.6 softmin_weight=0.5\n", flush=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)

    base_sm = float(np.mean([d["softmin_before"] for d in items]))
    ref_sm = float(np.mean([d["softmin_after"] for d in items]))
    print(f"baseline soft-min {base_sm:.4f}   current 60-step PGD {ref_sm:.4f}\n")

    def evaluate(mel, delta):
        with torch.no_grad(), diffscore.differentiable():
            wav, _ = hift_call(mel + delta)
            sc = diffscore.detector_scores(wav.squeeze(0), device)
            return (float(diffscore.soft_min(list(sc.values()), k=args.softmin_k)),
                    {n: float(v) for n, v in sc.items()})

    rows = []
    print(f"{'steps':>6}{'lr':>7}{'max_d':>7}{'sm_w':>6}{'softmin':>10}"
          f"{'|delta|':>10}{'sec/ex':>9}")
    print("-" * 56)
    for st in steps:
        for lr in lrs:
            for md in mds:
                for sw in sws:
                    t0 = time.time()
                    sms, dms = [], []
                    for d in items:
                        mel = d["mel"].float().unsqueeze(0).to(device)
                        delta, _ = pgd_solve(mel, hift_call, device, st, md, lr,
                                             args.softmin_k, sw)
                        sm, _ = evaluate(mel, delta)
                        sms.append(sm)
                        dms.append(float(delta.abs().mean()))
                    el = (time.time() - t0) / len(items)
                    rec = {"steps": st, "lr": lr, "max_delta": md,
                           "softmin_weight": sw, "softmin": float(np.mean(sms)),
                           "delta_abs": float(np.mean(dms)), "sec_per_example": el}
                    rows.append(rec)
                    print(f"{st:>6}{lr:>7.3f}{md:>7.2f}{sw:>6.2f}"
                          f"{rec['softmin']:>10.4f}{rec['delta_abs']:>10.4f}"
                          f"{el:>9.1f}", flush=True)

    best = max(rows, key=lambda r: r["softmin"])
    print(f"\nbest: steps={best['steps']} lr={best['lr']} "
          f"max_delta={best['max_delta']} softmin_weight={best['softmin_weight']}")
    print(f"  soft-min {best['softmin']:.4f} vs current {ref_sm:.4f} "
          f"({best['softmin'] - ref_sm:+.4f}), |delta| {best['delta_abs']:.4f}, "
          f"{best['sec_per_example']:.1f} s/example")
    gain_now = ref_sm - base_sm
    gain_best = best["softmin"] - base_sm
    if gain_now > 0:
        print(f"  teacher gain improves {gain_best / gain_now:.2f}x")

    out = Path(args.out) if args.out else RL_RUNS / "teacher_sweep.json"
    out.write_text(json.dumps({"baseline_softmin": base_sm,
                               "current_pgd_softmin": ref_sm,
                               "n_examples": len(items), "grid": rows,
                               "best": best}, indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
