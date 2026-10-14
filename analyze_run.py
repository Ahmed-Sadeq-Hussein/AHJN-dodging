# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
analyze_run.py -- read a training run's log and answer one question:
is there a real trend, or is this noise?

    python analyze_run.py                          # newest run
    python analyze_run.py --run designA_20260904   # a specific one
    python analyze_run.py --noise-floor            # sigma_ep from a frozen run

Runs in EITHER env -- it only needs numpy (and matplotlib for --plot).

The decision rule
-----------------
Ordinary least squares on mean_reward vs iteration. The standard error of
the fitted slope for n evenly spaced points is

    SE(slope) = SE_batch / sqrt( n(n^2-1)/12 ),   SE_batch = sigma_ep / sqrt(B)

so the total reward change the run can resolve at 2 sigma is
2 * SE(slope) * n. A slope is only worth believing if the total change it
implies clears that bar -- AND the constraints held. Reward climbing while
similarity or WER degrade is reward hacking, not progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import RESULTS_ROOT

RL_RUNS = RESULTS_ROOT / "rl_runs"


def load_rows(run_dir: Path) -> list[dict]:
    log = run_dir / "train_log.jsonl"
    if not log.is_file():
        raise FileNotFoundError(f"no train_log.jsonl in {run_dir}")
    return [json.loads(l) for l in log.read_text(encoding="utf8").splitlines() if l.strip()]


def newest_run() -> Path:
    runs = [p for p in RL_RUNS.glob("designA_*") if (p / "train_log.jsonl").is_file()]
    if not runs:
        raise FileNotFoundError(f"no runs with a log under {RL_RUNS}")
    return max(runs, key=lambda p: (p / "train_log.jsonl").stat().st_mtime)


def ols_slope(y: np.ndarray) -> tuple[float, float]:
    """Slope and its standard error for y measured at x = 0..n-1."""
    n = len(y)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    slope = float((xc * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    # residual standard error; needs n > 2 for a meaningful estimate
    se_slope = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1) / sxx)) if n > 2 else float("nan")
    return slope, se_slope


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir name under rl_runs/ (default: newest)")
    ap.add_argument("--noise-floor", action="store_true",
                    help="report sigma_ep only -- use on an --epochs 0 run")
    ap.add_argument("--sim-min", type=float, default=0.70)
    ap.add_argument("--wer-max", type=float, default=0.10)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    run_dir = (RL_RUNS / args.run) if args.run else newest_run()
    rows = load_rows(run_dir)
    print(f"run: {run_dir.name}   ({len(rows)} iterations)\n")

    # ---- noise floor ----
    per_ep = [r for r in rows if r.get("episode_rewards")]
    if per_ep:
        allr = np.array([v for r in per_ep for v in r["episode_rewards"]])
        sigma_ep = float(allr.std(ddof=1))
        B = int(np.mean([r["n_episodes"] for r in per_ep]))
        print(f"episodes           : {len(allr)}   batch size {B}")
        print(f"per-episode reward : mean {allr.mean():+.4f}  sigma_ep {sigma_ep:.4f}"
              f"  min {allr.min():+.4f}  max {allr.max():+.4f}")
        print(f"batch SE           : sigma_ep/sqrt(B) = {sigma_ep/np.sqrt(B):.4f}")
    else:
        sigma_ep, B = float("nan"), rows[0].get("n_episodes", 1)
        print("(no per-episode rewards logged -- rerun to get sigma_ep)")

    if args.noise_floor:
        if per_ep:
            print(f"\nUse sigma_ep = {sigma_ep:.4f} when sizing the batch.")
            for b in (4, 8, 16, 32):
                print(f"  B={b:<3} batch SE = {sigma_ep/np.sqrt(b):.4f}")
        return 0

    # ---- trend ----
    y = np.array([r["mean_reward"] for r in rows], dtype=float)
    n = len(y)
    print(f"\nmean_reward        : first {y[0]:+.4f}  last {y[-1]:+.4f}  best {y.max():+.4f}")

    if n < 3:
        print("\nneed at least 3 iterations for a trend fit")
        return 0

    slope, se = ols_slope(y)
    total = slope * n
    # resolvable change at 2 sigma, from the residual-based slope SE
    resolvable = 2 * se * n
    print(f"\nOLS slope          : {slope:+.6f} per iteration  (SE {se:.6f})")
    print(f"implied total change over {n} iterations: {total:+.4f}")
    print(f"resolvable at 2 sigma                   : {resolvable:.4f}")

    # ---- constraints ----
    sim = np.array([r["mean_similarity"] for r in rows])
    wer = np.array([r["mean_wer"] for r in rows])
    # judge a tail WINDOW, not the last point -- one noisy iteration should not
    # condemn a run. The printout names the window so the verdict cannot look
    # like it contradicts the last value.
    w = max(3, n // 4)
    sim_tail, wer_tail = sim[-w:].mean(), wer[-w:].mean()
    sim_ok = bool(sim_tail >= args.sim_min)
    wer_ok = bool(wer_tail <= args.wer_max)
    print(f"\nsimilarity         : first {sim[0]:.3f}  last {sim[-1]:.3f}  "
          f"last-{w} mean {sim_tail:.3f}  (floor {args.sim_min})  "
          f"{'OK' if sim_ok else 'VIOLATED'}")
    print(f"WER                : first {wer[0]*100:.1f}%  last {wer[-1]*100:.1f}%  "
          f"last-{w} mean {wer_tail*100:.1f}%  (cap {args.wer_max*100:.0f}%)  "
          f"{'OK' if wer_ok else 'VIOLATED'}")
    print(f"lambda_sim         : {rows[0]['lambda_sim']:.3f} -> {rows[-1]['lambda_sim']:.3f}")
    print(f"lambda_wer         : {rows[0]['lambda_wer']:.3f} -> {rows[-1]['lambda_wer']:.3f}")

    # ---- verdict ----
    # Three cases, not two: a trend can be significantly UP, significantly
    # DOWN, or absent. Folding "significantly down" into "no trend" (by
    # requiring total > 0 before calling a trend real) hides the single most
    # actionable outcome -- training that is actively making things worse
    # needs a different fix from training that is merely not learning.
    significant = abs(total) > resolvable
    improving = significant and total > 0
    degrading = significant and total < 0
    print("\n" + "=" * 58)
    if improving and sim_ok and wer_ok:
        print("VERDICT: real upward trend, constraints held -- worth more compute.")
    elif improving:
        print("VERDICT: reward rose but a constraint broke. This is reward")
        print("         hacking, not progress. Raise lambda_lr or the floors")
        print("         before spending more compute.")
    elif degrading:
        print(f"VERDICT: reward is significantly DEGRADING ({total:+.4f} over {n}")
        print(f"         iterations, resolvable {resolvable:.4f}). The policy is")
        print("         drifting away from a good starting point, not toward one.")
        print("         Identity (no edit) may be a local optimum that the noisy")
        print("         advantage estimate is walking away from -- check with")
        print("         eval_policy.py before assuming more compute helps.")
    elif sim_ok and wer_ok:
        print("VERDICT: no trend above noise. Constraints are fine, so the")
        print("         config is safe but not learning -- change something")
        print("         (lr, max_delta, softmin k) rather than adding compute.")
    else:
        print("VERDICT: no trend AND a constraint broke. Stop and reconfigure.")
    print("=" * 58)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        x = np.arange(n)
        axes[0].plot(x, y, "o-", color="#2a9d8f")
        axes[0].plot(x, np.polyval([slope, y.mean() - slope * x.mean()], x), "--",
                     color="#e76f51", label=f"slope {slope:+.5f}/iter")
        axes[0].set_ylabel("mean reward"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(x, sim, "o-", color="#264653"); axes[1].axhline(args.sim_min, ls="--", color="#e76f51")
        axes[1].set_ylabel("similarity"); axes[1].grid(alpha=.3)
        axes[2].plot(x, wer * 100, "o-", color="#264653"); axes[2].axhline(args.wer_max * 100, ls="--", color="#e76f51")
        axes[2].set_ylabel("WER %"); axes[2].set_xlabel("iteration"); axes[2].grid(alpha=.3)
        out = run_dir / "trend.png"
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
