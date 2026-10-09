r"""
plot_mos.py -- figure 20: what human listeners said about the two RL designs.

    python plot_mos.py
    python plot_mos.py --ratings "MOS TEST RL/mos_ratings.json"

The listening test is the only evidence in the project that does not come from
the detectors. Thirty items, fifteen voices in two registers, three arms from
one shared seed, every item rated on all three arms by the same listener. The
pairing is what makes it readable: within an item the mel edit is the only
difference, so a paired test asks exactly the right question.

Panel C is the one that settles the project. Both axes point the same way --
up and right is better -- so a design that helps would sit above and right of
the unedited cluster. Neither does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import GRAPHS_DIR

# The repository's figure palette, kept so this sits beside figures 1-19.
# Grey is the untouched control, red the design that hurt most, blue the
# gentler one. Every series is also direct-labelled, so identity never rests
# on colour alone.
C_BASE, C_A, C_B = "#8a8f98", "#c2413b", "#4a5fc1"
ARMS = ["baseline", "design_a", "design_b"]
NAME = {"baseline": "Unedited", "design_a": "Design A", "design_b": "Design B"}
COLOR = {"baseline": C_BASE, "design_a": C_A, "design_b": C_B}
MARK = {"baseline": "o", "design_a": "s", "design_b": "^"}


def load(path: Path):
    d = json.loads(path.read_text(encoding="utf8"))
    sub, obj = defaultdict(dict), defaultdict(dict)
    for r in d["ratings"]:
        sub[r["voice"]][r["arm"]] = r
    for c in d["clips"]:
        if c["arm"] != "reference":
            obj[c["voice"]][c["arm"]] = c
    items = sorted(i for i in sub if len(sub[i]) == len(ARMS))
    return d, sub, obj, items


def ci95(v):
    """Half-width of the 95% interval on the mean."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="MOS TEST RL/mos_ratings.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.ratings)
    if not path.is_file():
        print(f"no {path} -- export the ratings from the listening test first")
        return 1

    data, sub, obj, items = load(path)
    if not items:
        print("no item was rated on all three arms")
        return 1

    def col(arm, field, src):
        return np.array([src[i][arm][field] for i in items], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    axA, axB, axC, axD = axes.ravel()

    # ---- A: the two subjective scales, mean with a 95% interval ----------
    width, xs = 0.26, np.arange(2)
    for k, arm in enumerate(ARMS):
        m = [col(arm, "mos", sub).mean(), col(arm, "pmos", sub).mean()]
        e = [ci95(col(arm, "mos", sub)), ci95(col(arm, "pmos", sub))]
        off = (k - 1) * (width + 0.02)
        axA.bar(xs + off, m, width, color=COLOR[arm], label=NAME[arm],
                zorder=3, edgecolor="white", linewidth=2)
        axA.errorbar(xs + off, m, yerr=e, fmt="none", ecolor="#2b2f38",
                     elinewidth=1.3, capsize=4, zorder=4)
        for x, v, ee in zip(xs + off, m, e):
            axA.text(x, v + ee + 0.10, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=9, color="#2b2f38", zorder=5)
    axA.set_xticks(xs)
    axA.set_xticklabels(["MOS\nhow human it sounds",
                         "PMOS\nhow much it is still them"])
    axA.set_ylabel("rating, 1 to 5")
    axA.set_ylim(0, 5.9)
    axA.set_title("A. Both designs sound worse to a listener\n"
                  f"{len(items)} items, mean with a 95% interval",
                  fontsize=11, loc="left")
    axA.grid(axis="y", alpha=0.25)
    axA.set_axisbelow(True)
    axA.legend(fontsize=9, loc="upper right", ncol=3, columnspacing=1.0)

    # ---- B: paired MOS difference against the unedited clone -------------
    base = col("baseline", "mos", sub)
    rng = np.random.default_rng(7)
    for k, arm in enumerate(["design_a", "design_b"]):
        d = col(arm, "mos", sub) - base
        y = k + rng.uniform(-0.17, 0.17, len(d))
        axB.scatter(d, y, s=34, color=COLOR[arm], alpha=0.75, zorder=3,
                    marker=MARK[arm], edgecolor="white", linewidth=0.8)
        axB.scatter([d.mean()], [k], s=190, color=COLOR[arm], zorder=5,
                    marker="D", edgecolor="#2b2f38", linewidth=1.4)
        t, p = stats.ttest_rel(col(arm, "mos", sub), base)
        axB.text(-3.15, k + 0.30,
                 f"{NAME[arm]}   mean {d.mean():+.2f}   "
                 f"{int((d < 0).sum())} of {len(d)} rated worse   p = {p:.1e}",
                 ha="left", va="bottom", fontsize=9.5, color="#2b2f38")
    axB.axvline(0, color="#2b2f38", lw=1.4, zorder=2)
    axB.text(0.05, 1.66, "no change", fontsize=8.5, color="#5a6070")
    axB.set_yticks([])
    axB.set_xlim(-3.25, 1.35)
    axB.set_ylim(-0.6, 1.75)
    axB.set_xlabel("change in MOS against the same item unedited")
    axB.set_title("B. Paired, item by item\n"
                  "one dot is one voice and line; diamond is the mean",
                  fontsize=11, loc="left")
    axB.grid(axis="x", alpha=0.25)
    axB.set_axisbelow(True)

    # ---- C: the dominance picture ----------------------------------------
    # the three centroids sit on a diagonal, so each label is pushed clear of
    # the next marker down the line
    LABEL_OFF = {"baseline": (16, 6), "design_b": (18, -4), "design_a": (18, -30)}
    for arm in ARMS:
        x, y = col(arm, "softmin", obj), col(arm, "mos", sub)
        axC.scatter(x, y, s=38, color=COLOR[arm], alpha=0.40, zorder=3,
                    marker=MARK[arm], edgecolor="white", linewidth=0.7)
        axC.scatter([x.mean()], [y.mean()], s=320, color=COLOR[arm], zorder=6,
                    marker=MARK[arm], edgecolor="#2b2f38", linewidth=1.8)
        axC.annotate(f"{NAME[arm]}\n{x.mean():.3f}, {y.mean():.2f}",
                     (x.mean(), y.mean()), textcoords="offset points",
                     xytext=LABEL_OFF[arm], fontsize=9.5, color="#2b2f38",
                     fontweight="bold", zorder=7)
    bx = col("baseline", "softmin", obj).mean()
    by = col("baseline", "mos", sub).mean()
    axC.axvline(bx, color=C_BASE, lw=1.0, ls=":", zorder=2)
    axC.axhline(by, color=C_BASE, lw=1.0, ls=":", zorder=2)
    axC.fill_between([bx, 1.0], by, 5.6, color="#2a9d8f", alpha=0.10, zorder=1)
    axC.text(bx + 0.004, 5.42, "better on both axes: nothing is here",
             fontsize=9, color="#2a7f74", va="top")
    axC.set_xlabel("soft-min detector score (1 = human to every detector)")
    axC.set_ylabel("MOS from the listener")
    axC.set_ylim(0.6, 5.6)
    axC.set_title("C. Neither design trades quality for evasion\n"
                  "both move down and left: more detectable and worse sounding",
                  fontsize=11, loc="left")
    axC.grid(alpha=0.25)
    axC.set_axisbelow(True)

    # ---- D: listener against machine --------------------------------------
    allm = np.concatenate([col(a, "mos", sub) for a in ARMS])
    alls = np.concatenate([col(a, "softmin", obj) for a in ARMS])
    for arm in ARMS:
        axD.scatter(col(arm, "softmin", obj), col(arm, "mos", sub), s=36,
                    color=COLOR[arm], alpha=0.55, marker=MARK[arm], zorder=3,
                    edgecolor="white", linewidth=0.7, label=NAME[arm])
    b1, b0 = np.polyfit(alls, allm, 1)
    xx = np.linspace(alls.min(), alls.max(), 50)
    axD.plot(xx, b1 * xx + b0, color="#2b2f38", lw=2, zorder=5)
    r, pr = stats.pearsonr(allm, alls)
    rho, prho = stats.spearmanr(allm, alls)
    axD.text(0.03, 0.96, f"r = {r:+.2f}  (p = {pr:.3f})\n"
                         f"rho = {rho:+.2f}  (p = {prho:.3f})",
             transform=axD.transAxes, va="top", fontsize=10, color="#2b2f38",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#d6d9de"))
    axD.set_xlabel("soft-min detector score")
    axD.set_ylabel("MOS from the listener")
    axD.set_title("D. The detectors and the ear mostly agree\n"
                  f"all {len(items) * len(ARMS)} clips pooled",
                  fontsize=11, loc="left")
    axD.grid(alpha=0.25)
    axD.set_axisbelow(True)
    axD.legend(fontsize=9, loc="lower right")

    fig.suptitle("Figure 20. The listening test: both reinforcement-learning "
                 "designs are worse than doing nothing",
                 fontsize=13.5, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = Path(args.out) if args.out else GRAPHS_DIR / "20_mos_rl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # ---- the same numbers as text, so the figure can be checked ----------
    print(f"\n{len(items)} items rated on all three arms "
          f"(exported {data.get('exportedAt', '?')})")
    print(f"\n{'arm':<10}{'MOS':>7}{'PMOS':>7}{'soft-min':>11}{'PESQ':>7}"
          f"{'sim':>7}")
    for a in ARMS:
        print(f"{NAME[a]:<10}{col(a,'mos',sub).mean():>7.2f}"
              f"{col(a,'pmos',sub).mean():>7.2f}"
              f"{col(a,'softmin',obj).mean():>11.3f}"
              f"{col(a,'pesq',obj).mean():>7.2f}{col(a,'sim',obj).mean():>7.3f}")

    print(f"\n{'paired vs unedited':<24}{'diff':>8}{'t':>8}{'p':>11}{'dz':>7}"
          f"{'worse':>8}")
    for a in ("design_a", "design_b"):
        for field, src, lab in (("mos", sub, "MOS"), ("pmos", sub, "PMOS"),
                                ("softmin", obj, "soft-min"),
                                ("pesq", obj, "PESQ")):
            bb, xx2 = col("baseline", field, src), col(a, field, src)
            d = xx2 - bb
            t, p = stats.ttest_rel(xx2, bb)
            dz = d.mean() / d.std(ddof=1)
            print(f"{NAME[a] + ' ' + lab:<24}{d.mean():>+8.3f}{t:>8.2f}"
                  f"{p:>11.2e}{dz:>+7.2f}{int((d < 0).sum()):>6}/{len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
