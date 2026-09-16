r"""
plot_margin.py -- figures for the margin-objective work, into Plan_X/.

    python plot_margin.py

Figure 15  the trade-off curve: soft-min against the PESQ cost of getting it,
           for the old probability model and the new margin model at several
           pass counts. The quality budget is drawn as a line, because the
           headline soft-min of 0.800 sits far outside it.
Figure 16  per voice at the two operating points, against the unedited
           baseline, with the teacher's own reach for scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import RESULTS_ROOT

RL = RESULTS_ROOT / "rl_runs"
PLAN_X = Path(__file__).resolve().parent / "Plan_X"
C_BASE, C_OLD, C_NEW, C_PGD = "#8a8f98", "#2a9d8f", "#4a5fc1", "#e07a1f"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def main() -> int:
    PLAN_X.mkdir(exist_ok=True)
    fin = load(RL / "all15_final.json")          # new x4 vs old x10
    big = load(RL / "all15_eval.json")           # new x14
    base_pesq = np.mean([c["baseline"]["pesq"] for c in fin["clips"]])

    pts = []
    for n in (2, 4, 6, 10):
        f = RL / f"cost_iters_{n}.json"
        if f.is_file():
            d = load(f)
            b = np.mean([c["baseline"]["pesq"] for c in d["clips"]])
            pts.append((n,
                        float(np.mean([c["new"]["softmin"] for c in d["clips"]])),
                        float(np.mean([c["new"]["pesq"] for c in d["clips"]]) - b),
                        float(np.mean([c["baseline"]["softmin"] for c in d["clips"]]))))

    fig, ax = plt.subplots(figsize=(11, 6))
    if pts:
        xs = [-p[2] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=C_NEW, lw=1.8, ms=6,
                label="margin model, 6 harder voices")
        for n, sm, dp, _ in pts:
            ax.annotate(f"x{n}", (-dp, sm), textcoords="offset points",
                        xytext=(6, -3), fontsize=9, color=C_NEW)
        ax.axhline(pts[0][3], color=C_BASE, lw=1.2, ls=":",
                   label=f"unedited, those voices ({pts[0][3]:.2f})")

    # the two all-15 operating points
    ax.scatter([base_pesq - np.mean([c["new"]["pesq"] for c in fin["clips"]])],
               [np.mean([c["new"]["softmin"] for c in fin["clips"]])],
               s=150, marker="s", color=C_NEW, edgecolor="black", zorder=5,
               label="margin x4, all 15 voices")
    ax.scatter([base_pesq - np.mean([c["old"]["pesq"] for c in fin["clips"]])],
               [np.mean([c["old"]["softmin"] for c in fin["clips"]])],
               s=150, marker="^", color=C_OLD, edgecolor="black", zorder=5,
               label="previous model x10, all 15")
    bp = np.mean([c["baseline"]["pesq"] for c in big["clips"]])
    ax.scatter([bp - np.mean([c["new"]["pesq"] for c in big["clips"]])],
               [np.mean([c["new"]["softmin"] for c in big["clips"]])],
               s=150, marker="*", color=C_NEW, edgecolor="black", zorder=5,
               label="margin x14, all 15 voices")

    ax.axvline(0.30, color="#b00", lw=1.4, ls="--")
    ax.text(0.31, ax.get_ylim()[0] + 0.02, "quality budget (PESQ -0.30)",
            color="#b00", fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("PESQ cost (drop from unedited)")
    ax.set_ylabel("soft-min (worst detector, 1 = human)")
    ax.set_title("More evasion is available, but only by spending audio quality",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    p1 = PLAN_X / "15_margin_tradeoff.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- per voice ----
    pv = {v["voice"]: v for v in fin["per_voice"]}
    big_pv = {v["voice"]: v for v in big["per_voice"]}
    order = sorted(pv, key=lambda v: -big_pv[v]["new"]["softmin"])
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(xs - 0.27, [pv[v]["baseline"]["softmin"] for v in order], 0.26,
           color=C_BASE, label="unedited")
    ax.bar(xs, [pv[v]["new"]["softmin"] for v in order], 0.26,
           color=C_NEW, label="margin x4 (within quality budget)")
    ax.bar(xs + 0.27, [big_pv[v]["new"]["softmin"] for v in order], 0.26,
           color=C_NEW, alpha=0.45, hatch="//",
           label="margin x14 (over budget, PESQ -0.85)")
    ax.axhline(0.5, color="#b00", lw=1.2, ls="--", label="soft-min 0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("soft-min (worst detector)")
    ax.set_title("Per voice: 14 of 15 clear 0.5 at 14 passes, 6 of 15 within the quality budget",
                 fontsize=12, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = PLAN_X / "16_margin_per_voice.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"wrote {p1}")
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
