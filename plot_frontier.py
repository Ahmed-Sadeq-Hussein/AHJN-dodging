# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
plot_frontier.py -- the evasion/quality frontier across every method tried.

    python plot_frontier.py

One figure, one question: for a given amount of audio damage, how much
soft-min can each method buy? All points are measured on the same six harder
voices (unedited soft-min 0.131, PESQ 3.78) with the same two unseen
sentences, so they are directly comparable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import GRAPHS_DIR, RESULTS_ROOT

RL = RESULTS_ROOT / "rl_runs"
PLAN_X = GRAPHS_DIR

# (glob, label, colour, marker) -- each file is one (model, pass count) point
FAMILIES = [
    ("cost_iters_*.json", "distilled from PGD", "#4a5fc1", "o"),
    ("dcost_*.json", "direct, mean(delta^2)", "#b0413e", "x"),
    ("pc_p_mask8_*.json", "direct, masking penalty", "#2a9d8f", "^"),
    ("pc_p_both2_*.json", "direct, STFT+mask", "#e07a1f", "s"),
    ("pc4_p_both2.json", "direct, STFT+mask", "#e07a1f", "s"),
    ("lc_long_big_*.json", "direct, long + big net", "#7a4fc1", "v"),
    ("c35_*.json", "direct, quality CONSTRAINT", "#111111", "D"),
    ("cc_c0.35_*.json", "direct, quality CONSTRAINT", "#111111", "D"),
]


def point(path):
    d = json.loads(Path(path).read_text(encoding="utf8"))
    cl = d["clips"]
    if not cl or "new" not in cl[0]:
        return None
    sm = float(np.mean([c["new"]["softmin"] for c in cl]))
    dp = float(np.mean([c["baseline"]["pesq"] - c["new"]["pesq"] for c in cl]))
    base = float(np.mean([c["baseline"]["softmin"] for c in cl]))
    n = re.search(r"_(\d+)\.json$", Path(path).name)
    return sm, dp, base, (n.group(1) if n else "?")


def main() -> int:
    PLAN_X.mkdir(exist_ok=True)

    # same reason as plot_plan_x: with no evaluation files this drew an empty
    # frontier over the committed one
    # glob() returns a generator, which is truthy even when it yields
    # nothing, so this has to consume it
    if not any(True for pattern, *_ in FAMILIES for _ in RL.glob(pattern)):
        print(f"no evaluation files under {RL}. Nothing written, so the "
              f"committed figure survives.")
        return 1

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    seen, base_sm = set(), None

    for pattern, label, colour, marker in FAMILIES:
        pts = []
        for f in sorted(RL.glob(pattern)):
            p = point(f)
            if p:
                pts.append(p)
                base_sm = p[2]
        if not pts:
            continue
        pts.sort(key=lambda t: t[1])
        lbl = label if label not in seen else None
        seen.add(label)
        ax.plot([p[1] for p in pts], [p[0] for p in pts], marker=marker,
                color=colour, lw=1.6, ms=7, label=lbl, alpha=0.9)
        for sm, dp, _, n in pts:
            ax.annotate(f"x{n}", (dp, sm), textcoords="offset points",
                        xytext=(5, -4), fontsize=8, color=colour)

    if base_sm is not None:
        ax.axhline(base_sm, color="#8a8f98", lw=1.3, ls=":",
                   label=f"unedited ({base_sm:.2f})")
    ax.axvline(0.30, color="#b00", lw=1.5, ls="--")
    ax.text(0.315, 0.16, "quality budget\nPESQ -0.30", color="#b00", fontsize=9,
            va="bottom")

    ax.set_xlabel("audio damage: PESQ drop from unedited  (lower is better)")
    ax.set_ylabel("soft-min, worst detector  (higher is better)")
    ax.set_title("Evasion bought per unit of audio damage, six harder voices\n"
                 "labels are the number of passes applied at inference",
                 fontsize=12, loc="left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=-0.05)
    fig.tight_layout()
    out = PLAN_X / "17_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # the in-budget ranking, printed so the figure is not the only record
    print(f"\n{'config':<34}{'soft-min':>10}{'d PESQ':>9}{'sm per PESQ pt':>16}")
    rows = []
    for pattern, label, _, _ in FAMILIES:
        for f in sorted(RL.glob(pattern)):
            p = point(f)
            if p and p[1] <= 0.30:
                rows.append((f"{label} x{p[3]}", p[0], p[1],
                             (p[0] - p[2]) / max(p[1], 1e-6)))
    for name, sm, dp, eff in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<34}{sm:>10.3f}{-dp:>9.3f}{eff:>16.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
