# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
Train Design B: SAC with a detector gradient hint.

    python scripts/train_design_b.py
    python scripts/train_design_b.py --iters 600 --resume
    python scripts/train_design_b.py --run sweep_lr3 --no-grad-hint

Design B keeps the same one-step formulation as Design A but changes what the
agent is told and how it learns. Off-policy SAC with twin Q critics and an
automatically tuned temperature, and the actor additionally receives the
detector gradient with respect to the mel as extra input channels. The hint
is the whole point: Design A has to discover the direction of improvement
from a scalar, while Design B is handed it.

Pass --no-grad-hint to train the ablation that removes those channels, which
is how the hint's contribution was measured.

Runs under the voiceclone environment. Roughly 9 GPU hours to 400 iterations.

Output goes to RESULTS_ROOT/rl_runs/designB_<run id>/, and unlike Design A a
resume also restores the replay buffer, so continuing a run is genuinely
continuous rather than a warm restart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rl import common, design_b

DEFAULT_RUN = "designB_B9h"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train Design B (SAC with a gradient hint).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run id; reusing one continues that run's directory")
    ap.add_argument("--iters", type=int, default=None,
                    help="total iterations to reach (default: the published 400)")
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--lr-actor", type=float, default=None)
    ap.add_argument("--lr-critic", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="episodes collected per iteration")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-grad-hint", action="store_true",
                    help="train the ablation without the gradient channels")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint, restoring the "
                         "replay buffer as well as the networks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    overrides = {}
    for flag, key in (("iters", "iterations"), ("voices", "voices"),
                      ("lr_actor", "lr-actor"), ("lr_critic", "lr-critic"),
                      ("batch", "batch"), ("seed", "seed")):
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = value
    if args.no_grad_hint:
        overrides["no-grad-hint"] = True

    settings = {**design_b.DEFAULTS, **overrides}
    print("Design B: SAC with a detector gradient hint")
    print(f"  run id     {args.run}")
    print(f"  resume     {'yes, buffer included' if args.resume else 'no'}")
    print(f"  grad hint  {'OFF (ablation)' if args.no_grad_hint else 'on'}")
    for key in sorted(settings):
        mark = "*" if key in overrides else " "
        print(f"  {mark} {key:<16} {settings[key]}")

    existing = common.latest_checkpoint(args.run)
    print(f"  existing   {existing.name if existing else 'none, starting fresh'}")
    if existing and not args.resume:
        print("\n  note: this run already has checkpoints. Without --resume the "
              "trainer\n        starts from iteration 0 and will overwrite them.")

    if args.dry_run:
        print(f"\ndry run, nothing trained. Directory would be "
              f"{common.run_dir(args.run)}")
        return 0

    print(flush=True)
    out = design_b.train(args.run, resume="auto" if args.resume else None,
                         **overrides)
    print(f"\nrun directory: {out}")
    print(f"inspect it with:  python ahjn.py status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
