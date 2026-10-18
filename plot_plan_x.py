# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
plot_plan_x.py -- figures for Plan_X/, comparing Design A and Design B.

    python plot_plan_x.py                      # both runs, all figures
    python plot_plan_x.py --b-run designB_B_main --a-run designA_budget4h

Runs in EITHER env (needs numpy + matplotlib only).

Every figure answers one question, and each is written to Plan_X/ alongside
the reasoning in Plan_X/RESULTS.md:

  1_learning_curves      did either design's reward actually move?
  2_per_detector         which detectors changed, and in which direction?
  3_constraints          were similarity and WER ever the binding limit?
  4_action_scale         did the policy edit at the scale where signal lives?
  5_diagnostics          SAC internals: Q loss, alpha, buffer, lambda
  6_pgd_ceiling          what a white-box per-utterance attack can reach
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import GRAPHS_DIR, RESULTS_ROOT

RL_RUNS = RESULTS_ROOT / "rl_runs"
PLAN_X = GRAPHS_DIR

# consistent palette across every figure: warm = Design A, cool = Design B
C_A, C_B, C_REF, C_BAD = "#e76f51", "#2a9d8f", "#264653", "#e9c46a"
DETC = {"AASIST": "#2a9d8f", "AASIST-L": "#8ab17d", "RawNet2": "#e76f51",
        "RawGAT-ST": "#e9c46a"}


def load(run: str) -> list[dict]:
    p = RL_RUNS / run / "train_log.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf8").splitlines() if l.strip()]


def ols(y):
    n = len(y)
    if n < 3:
        return 0.0, float("nan")
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    slope = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * (x - x.mean()))
    se = float(np.sqrt((resid ** 2).sum() / max(n - 2, 1) / sxx))
    return slope, se


def smooth(y, w=9):
    """Moving average, or the raw series when there are too few points.

    Returns (series, label) so a figure can never claim to be showing a
    9-iteration mean when it is in fact showing raw data -- which is exactly
    what happened when Design B was cut short at 7 iterations.
    """
    if len(y) < w:
        return y, "raw (too few iters to smooth)"
    return np.convolve(y, np.ones(w) / w, mode="valid"), f"{w}-iter mean"


def fig1_learning(a, b, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, rows, name, col in ((axes[0], a, "Design A (PPO, black-box)", C_A),
                                (axes[1], b, "Design B (SAC + gradient hint)", C_B)):
        if not rows:
            ax.text(.5, .5, "no data", ha="center", transform=ax.transAxes); continue
        y = np.array([r["mean_reward"] for r in rows])
        x = np.arange(len(y))
        ax.plot(x, y, "o", ms=3.5, alpha=.45, color=col)
        sm, lab = smooth(y)
        ax.plot(np.arange(len(sm)) + (len(y) - len(sm)) // 2, sm,
                color=col, lw=2, label=lab)
        s, se = ols(y)
        ax.plot(x, y.mean() + s * (x - x.mean()), "--", color="k", lw=1.2,
                label=f"slope {s:+.5f}/iter" + (f" (t={s/se:+.1f})" if se and se == se else ""))
        ax.set_title(f"{name}\n{len(y)} iterations, {sum(r['n_episodes'] for r in rows)} episodes",
                     fontsize=10)
        ax.set_xlabel("iteration"); ax.set_ylabel("mean reward")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Did reward actually move?  (NOTE: A and B optimise DIFFERENT rewards -- "
                 "A = pure soft-min, B = 0.5*soft-min + 0.5*mean. "
                 "Levels are NOT comparable; see figure 1b for the like-for-like view.)",
                 fontsize=10, y=1.06)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig1b_softmin(a, b, out):
    """The only detector metric both designs record identically."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ya = np.array([r.get("mean_detect_softmin", np.nan) for r in a], dtype=float)
    yb = np.array([r.get("mean_softmin", np.nan) for r in b], dtype=float)
    ya, yb = ya[~np.isnan(ya)], yb[~np.isnan(yb)]
    if len(ya) == 0 and len(yb) == 0:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for y, name, col in ((ya, f"Design A  (n={len(ya)})", C_A),
                         (yb, f"Design B  (n={len(yb)})", C_B)):
        if len(y) == 0:
            continue
        ax.plot(np.arange(len(y)), y, "o", ms=3.5, alpha=.35, color=col)
        sm, lab = smooth(y)
        ax.plot(np.arange(len(sm)) + (len(y) - len(sm)) // 2, sm, color=col, lw=2,
                label=f"{name}  -  {lab}")
    ax.axhline(0.0871, ls="--", color=C_REF, lw=1.3,
               label="unedited CosyVoice2 baseline (riven, 0.087)")
    ax.axhline(0.666, ls=":", color="0.35", lw=1.5,
               label="PGD ceiling, one utterance (0.666)")
    ax.set_xlabel("iteration"); ax.set_ylabel("soft-min over the 4 detectors")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("Like-for-like: soft-min, the one detector metric both designs log "
                 "(higher = better evasion)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def fig2_per_detector(b, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [r for r in b if "per_detector" in r]
    if not rows:
        return False
    names = list(rows[0]["per_detector"].keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    for n in names:
        y = np.array([r["per_detector"][n] for r in rows])
        sm, _ = smooth(y)
        ax.plot(np.arange(len(sm)), sm, lw=2, color=DETC.get(n, "k"), label=n)
    ax.axhline(.5, ls="--", color="0.4", lw=1)
    ax.text(len(rows) * .99, .515, "decision boundary", ha="right", fontsize=8, color="0.4")
    ax.set_xlabel("iteration (9-iter mean)"); ax.set_ylabel("P(human)")
    ax.set_ylim(-.02, 1.02); ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_title("Per-detector trajectory  -  Design B\n"
                 "Design A's failure was invisible here: it degraded AASIST/AASIST-L "
                 "while the soft-min reward looked flat", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def fig3_constraints(a, b, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for rows, name, col in ((a, "Design A", C_A), (b, "Design B", C_B)):
        if not rows:
            continue
        axes[0].plot(smooth(np.array([r["mean_similarity"] for r in rows]))[0],
                     color=col, lw=2, label=name)
        axes[1].plot(smooth(np.array([r["mean_wer"] for r in rows]))[0] * 100,
                     color=col, lw=2, label=name)
    axes[0].axhline(.70, ls="--", color=C_BAD); axes[0].set_ylabel("speaker similarity")
    axes[0].set_title("Similarity vs floor (0.70)", fontsize=10)
    axes[1].axhline(15, ls="--", color=C_BAD); axes[1].set_ylabel("WER %")
    axes[1].set_title("WER vs cap (15%)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("iteration (9-iter mean)"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.suptitle("Were the constraints ever the binding limit?", fontsize=12, y=1.03)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig4_action_scale(b, out, useful=0.005, design_a_scale=0.0648):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [r for r in b if "delta_abs_mean" in r]
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.array([r["delta_abs_mean"] for r in rows])
    ax.plot(y, color=C_B, lw=1.6, label="Design B |delta|")
    ax.axhline(useful, ls="--", color=C_REF, lw=1.5,
               label=f"measured useful scale ({useful})")
    ax.axhline(design_a_scale, ls="--", color=C_A, lw=1.5,
               label=f"Design A exploration scale ({design_a_scale:.3f})")
    ax.axhline(0.199, ls=":", color="0.4", lw=1.5, label="PGD solution (0.199)")
    ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("mean |delta| (log)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    ax.set_title("Edit magnitude vs where the signal actually lives\n"
                 "stepping +0.005 along the gradient moved the objective 0.458 -> 0.925; "
                 "+0.020 collapsed it to 0.143", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def fig5_diagnostics(b, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not b:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    def g(k, d=0.0):
        return np.array([r.get(k, d) for r in b], dtype=float)
    panels = [
        ("q_loss", "critic loss  (Q -> r)", axes[0, 0], C_B),
        ("alpha", "SAC temperature alpha", axes[0, 1], C_REF),
        ("buffer", "replay buffer size", axes[1, 0], C_REF),
        ("lambda_sim", "lambda_sim (floored at 0.25)", axes[1, 1], C_A),
    ]
    for key, title, ax, col in panels:
        y = g(key)
        ax.plot(y, color=col, lw=1.6)
        ax.set_title(title, fontsize=10); ax.set_xlabel("iteration"); ax.grid(alpha=.3)
    axes[1, 1].plot(g("lambda_wer"), color=C_B, lw=1.6, label="lambda_wer")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Design B internals", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def fig6_pgd(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    p = RL_RUNS / "pgd_probe" / "pgd_riven.json"
    if not p.is_file():
        return False
    d = json.loads(p.read_text(encoding="utf8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    h = d.get("history", [])
    if h:
        axes[0].plot([x["step"] for x in h], [x["softmin"] for x in h], color=C_REF, lw=2)
        axes[0].set_xlabel("PGD step"); axes[0].set_ylabel("soft-min")
        axes[0].set_title("White-box attack on one utterance", fontsize=10)
        axes[0].grid(alpha=.3)
    names = list(d["baseline"].keys())
    x = np.arange(len(names)); w = .38
    axes[1].bar(x - w/2, [d["baseline"][n] for n in names], w, label="unedited", color=C_REF)
    axes[1].bar(x + w/2, [d["adversarial"][n] for n in names], w, label="after PGD", color=C_B)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylabel("P(human)"); axes[1].axhline(.5, ls="--", color="0.4", lw=1)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.3, axis="y")
    axes[1].set_title(f"softmin {d['softmin_baseline']:.3f} -> {d['softmin_adversarial']:.3f}"
                      f"   (similarity {d['similarity_baseline']:.3f} -> "
                      f"{d['similarity_adversarial']:.3f})", fontsize=10)
    fig.suptitle("The ceiling: what a bounded mel edit can reach with full gradient access",
                 fontsize=12, y=1.03)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-run", default="designA_budget4h")
    # designB_B9h is the run section 4 documents: 396 iterations across two
    # sessions. designB_B_main is a 7-iteration stub and silently produced
    # wrong figures 1/1b/3/4/5 when it was the default.
    ap.add_argument("--b-run", default="designB_B9h")
    ap.add_argument("--out", default=str(PLAN_X))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a, b = load(args.a_run), load(args.b_run)
    print(f"Design A: {len(a)} iterations   Design B: {len(b)} iterations")

    # Refuse rather than draw blanks. With no run data these figures still
    # rendered, exited 0, and overwrote the committed versions with empty
    # axes, so a fresh clone destroyed figures 1, 1b and 3 by running the
    # graph script. An empty figure is worse than no figure.
    if not a and not b:
        print(f"\nno training rows under {RL_RUNS}.")
        print("Nothing was written: these figures would be empty and would "
              "overwrite the committed\nones. Train a run, or point "
              "RESULTS_ROOT at the tree that holds them.")
        return 1

    fig1_learning(a, b, out / "1_learning_curves.png"); print("  1_learning_curves.png")
    if fig1b_softmin(a, b, out / "1b_softmin_comparable.png"): print("  1b_softmin_comparable.png")
    if fig2_per_detector(b, out / "2_per_detector.png"): print("  2_per_detector.png")
    fig3_constraints(a, b, out / "3_constraints.png"); print("  3_constraints.png")
    if fig4_action_scale(b, out / "4_action_scale.png"): print("  4_action_scale.png")
    if fig5_diagnostics(b, out / "5_diagnostics.png"): print("  5_diagnostics.png")
    if fig6_pgd(out / "6_pgd_ceiling.png"): print("  6_pgd_ceiling.png")

    # machine-readable summary for the write-up
    summary = {}
    for name, rows in (("design_a", a), ("design_b", b)):
        if not rows:
            continue
        y = np.array([r["mean_reward"] for r in rows])
        s, se = ols(y)
        ep = np.array([v for r in rows if r.get("episode_rewards")
                       for v in r["episode_rewards"]])
        summary[name] = {
            "iterations": len(rows),
            "episodes": int(sum(r["n_episodes"] for r in rows)),
            "reward_first": float(y[0]), "reward_last": float(y[-1]),
            "reward_best": float(y.max()), "reward_mean": float(y.mean()),
            "slope": s, "slope_se": se, "slope_t": (s / se) if se and se == se else None,
            "sigma_ep": float(ep.std(ddof=1)) if len(ep) > 1 else None,
            "final_similarity": float(rows[-1]["mean_similarity"]),
            "final_wer": float(rows[-1]["mean_wer"]),
        }
        if "per_detector" in rows[-1]:
            summary[name]["per_detector_first"] = rows[0]["per_detector"]
            summary[name]["per_detector_last"] = rows[-1]["per_detector"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"\nwrote {out / 'summary.json'}")
    for k, v in summary.items():
        print(f"  {k}: slope {v['slope']:+.6f} (t={v['slope_t']:+.2f}) "
              f"reward {v['reward_first']:+.4f} -> {v['reward_last']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
