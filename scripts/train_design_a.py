# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
Train Design A: PPO with Lagrangian constraints.

    python scripts/train_design_a.py
    python scripts/train_design_a.py --iters 400 --resume
    python scripts/train_design_a.py --run my_run --voices riven swain --lr 3e-4

Design A treats one utterance as one episode: observe the mel, apply a single
bounded edit, receive one terminal reward. That makes it a contextual bandit
rather than a full MDP, which is why there is no discount factor anywhere.
The reward is the soft-min over the four detectors, penalised by two
Lagrange multipliers that hold speaker similarity above 0.70 and word error
rate below 0.10.

Runs under the voiceclone environment. Roughly 4 GPU hours to 400 iterations
on the published configuration.

Output goes to RESULTS_ROOT/rl_runs/designA_<run id>/:
    train_log.jsonl   one row per iteration
    latest.pt         the newest checkpoint, what --resume picks up
    iter_NNNN.pt      a permanent snapshot every --snapshot-every iterations
    config.json       the settings this run was started with
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rl import common, design_a

DEFAULT_RUN = "designA_budget4h"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train Design A (PPO-Lagrangian).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run id; reusing one continues that run's directory")
    ap.add_argument("--iters", type=int, default=None,
                    help="total iterations to reach (default: the published 400)")
    ap.add_argument("--voices", nargs="*", default=None,
                    help="which voices to train on")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lambda-lr", type=float, default=None,
                    help="dual ascent rate on the constraint multipliers")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="episodes per iteration")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in the run dir, "
                         "carrying optimiser state and the multipliers")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the settings and the run directory, train nothing")
    args = ap.parse_args()

    overrides = {}
    for flag, key in (("iters", "iterations"), ("voices", "voices"),
                      ("lr", "lr"), ("lambda_lr", "lambda-lr"),
                      ("batch_size", "batch-size"), ("seed", "seed")):
        value = getattr(args, flag)
        if value is not None:
            overrides[key] = value

    settings = {**design_a.DEFAULTS, **overrides}
    print("Design A: PPO with Lagrangian constraints")
    print(f"  run id     {args.run}")
    print(f"  resume     {'yes, from the newest checkpoint' if args.resume else 'no'}")
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
    out = design_a.train(args.run, resume="auto" if args.resume else None,
                         **overrides)
    print(f"\nrun directory: {out}")
    print(f"inspect it with:  python ahjn.py status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
