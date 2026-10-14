# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
plot_qc_frontier.py -- figure 19: the frontier after constraining the teacher.

    python plot_qc_frontier.py

Every point is measured the same way -- all 15 voices, four unseen sentences,
three arms from one shared seed -- so unlike figure 17 (six harder voices)
these are directly comparable across sections 7n, 7s and 7t.

The question the figure answers: for a given amount of audio damage, how much
soft-min does each pipeline buy? The budget is a vertical line at -0.30 PESQ,
because a point outside it is not a usable configuration however high it
scores.
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
from env_config import GRAPHS_DIR, RESULTS_ROOT

RL = RESULTS_ROOT / "rl_runs"
C_OLD, C_QC, C_BAD, C_BASE = "#2a9d8f", "#4a5fc1", "#c2413b", "#8a8f98"

# (file, arm, label, passes, series)
POINTS = [
    ("all15_final.json", "old", "probability", 10, "old"),
    ("all15_base_vs_spk.json", "new", "margin", 4, "old"),
    ("all15_eval.json", "new", "margin", 14, "old"),
    ("all15_qc_vs_ctl.json", "new", "constrained", 4, "qc"),
    ("all15_qc12_vs_best.json", "new", "constrained", 12, "qc"),
    ("all15_qc20_vs_qc16.json", "old", "constrained", 16, "qc"),
    ("all15_qc20_vs_qc16.json", "new", "constrained", 20, "qc"),
    ("all15_qcobj_vs_qc.json", "new", "hybrid loss", 12, "bad"),
    ("all15_dagger_vs_qc330.json", "new", "DAgger", 12, "dagger"),
    ("all15_dagger24_vs_18.json", "old", "DAgger", 18, "dagger"),
    ("all15_dagger24_vs_18.json", "new", "DAgger", 24, "dagger"),
]


def read(name, arm):
    p = RL / name
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf8"))
    c = d["clips"]
    g = lambda a, k: float(np.mean([x[a][k] for x in c if a in x and k in x[a]]))
    return {"softmin": g(arm, "softmin"),
            "dpesq": g(arm, "pesq") - g("baseline", "pesq"),
            "base": g("baseline", "softmin"),
            "npass": sum(1 for v in d["per_voice"] if v[arm]["softmin"] >= 0.5)}


def main() -> int:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    rows, base = [], None
    for name, arm, label, passes, series in POINTS:
        r = read(name, arm)
        if r is None:
            print("missing", name)
            continue
        base = r["base"]
        rows.append({**r, "label": label, "passes": passes, "series": series})
    if not rows:
        print("no results to plot")
        return 1

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    style = {"old": (C_OLD, "o", "unconstrained teacher (7n)"),
             "qc": (C_QC, "s", "quality-constrained teacher (7s)"),
             "dagger": ("#b5487a", "D", "DAgger, warm-started targets (7u)"),
             "bad": (C_BAD, "X", "objective added to the loss (7t)")}

    for series, (colour, marker, legend) in style.items():
        pts = sorted([r for r in rows if r["series"] == series], key=lambda r: -r["dpesq"])
        if not pts:
            continue
        xs = [-r["dpesq"] for r in pts]
        ys = [r["softmin"] for r in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, "-", color=colour, lw=1.8, alpha=.85, zorder=2)
        ax.scatter(xs, ys, s=115, marker=marker, color=colour, edgecolor="black",
                   linewidth=.7, zorder=3, label=legend)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(f'x{r["passes"]}', (x, y), textcoords="offset points",
                        xytext=(7, -4), fontsize=9, color=colour)

    if base is not None:
        ax.axhline(base, color=C_BASE, lw=1.2, ls=":", zorder=1,
                   label=f"unedited ({base:.3f})")
    ax.axhline(0.5, color="#555", lw=1.0, ls="-.", alpha=.6, zorder=1)
    ax.text(0.02, 0.505, "soft-min 0.5", fontsize=8.5, color="#555")
    ax.axvline(0.30, color="#b00", lw=1.5, ls="--", zorder=1)
    ax.text(0.305, base + 0.02 if base else 0.3, "quality budget (PESQ -0.30)",
            color="#b00", fontsize=9, rotation=90, va="bottom")

    best = max((r for r in rows if -r["dpesq"] <= 0.30), key=lambda r: r["softmin"])
    ax.annotate(f'best inside budget\n{best["softmin"]:.3f}, {best["npass"]}/15 voices',
                (-best["dpesq"], best["softmin"]),
                textcoords="offset points", xytext=(-14, 26), fontsize=9,
                ha="right", color=C_QC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_QC, lw=1.2))

    ax.set_xlabel("PESQ cost (drop from unedited)")
    ax.set_ylabel("soft-min (worst detector, 1 = human)")
    ax.set_title("Constraining the teacher moves the frontier\n"
                 "all 15 voices, unseen sentences; a point right of the red line is not usable",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = GRAPHS_DIR / "19_qc_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"wrote {out}")
    print(f"{'series':<14}{'passes':>7}{'soft-min':>10}{'dPESQ':>9}{'>=0.5':>8}")
    for r in sorted(rows, key=lambda r: (r["series"], r["passes"])):
        print(f'{r["label"]:<14}{r["passes"]:>7}{r["softmin"]:>10.3f}'
              f'{r["dpesq"]:>+9.3f}{r["npass"]:>6}/15')
    return 0


if __name__ == "__main__":
    sys.exit(main())
