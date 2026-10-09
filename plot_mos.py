r"""
plot_mos.py -- figure 20: what human listeners said about the two RL designs.

    python plot_mos.py                 # the combined four-panel figure
    python plot_mos.py --split         # and each panel as its own figure
    python plot_mos.py --ratings "MOS TEST RL/mos_ratings.json"

The listening test is the only evidence in the project that does not come from
a network. Thirty items, fifteen voices in two registers, three arms from one
shared seed, every item rated on all three arms by the same listener. The
pairing is what makes it readable: within an item the Mel edit is the only
difference, so a paired test asks exactly the right question.

Panel C is the one that settles the project. Both axes point the same way, up
and right is better, so a design that helped would sit above and right of the
unedited cluster. Neither does.

Panel D is a caution rather than a result. The pooled correlation between MOS
and the soft-min looks encouraging until you remove the between-arm variance,
at which point it disappears: the two measures agree on the ranking of the
three arms and on nothing finer. See `--split` output 20d.
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
# gentler one. Every series is also direct-labelled or marker-coded, so
# identity never rests on colour alone.
C_BASE, C_A, C_B = "#8a8f98", "#c2413b", "#4a5fc1"
INK, INK_2 = "#2b2f38", "#5a6070"
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
    return 0.0 if len(v) < 2 else (
        stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def boot_ci(d, n=20000, seed=11):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    b = d[idx].mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


# --------------------------------------------------------------------------
# the panels, each able to stand on its own
# --------------------------------------------------------------------------

def panel_scale(ax, col, items, field, label, question):
    """One subjective scale: arm means with a 95% interval."""
    xs = np.arange(len(ARMS))
    vals = [col(a, field).mean() for a in ARMS]
    errs = [ci95(col(a, field)) for a in ARMS]
    for k, a in enumerate(ARMS):
        ax.bar(k, vals[k], 0.6, color=COLOR[a], zorder=3,
               edgecolor="white", linewidth=2)
        ax.errorbar(k, vals[k], yerr=errs[k], fmt="none", ecolor=INK,
                    elinewidth=1.3, capsize=5, zorder=4)
        ax.text(k, vals[k] + errs[k] + 0.10, f"{vals[k]:.2f}", ha="center",
                va="bottom", fontsize=11, color=INK, zorder=5)
        if a != "baseline":
            d = col(a, field) - col("baseline", field)
            t, p = stats.ttest_rel(col(a, field), col("baseline", field))
            ax.text(k, 0.22, f"{d.mean():+.2f}\np = {p:.1e}", ha="center",
                    va="bottom", fontsize=9, color="white", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([NAME[a] for a in ARMS])
    ax.set_ylabel(f"{label}, 1 to 5")
    ax.set_ylim(0, 5.9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return vals


def panel_paired(ax, col, items, field, label):
    """Per-item change against the unedited clone."""
    base = col("baseline", field)
    rng = np.random.default_rng(7)
    for k, a in enumerate(["design_a", "design_b"]):
        d = col(a, field) - base
        y = k + rng.uniform(-0.17, 0.17, len(d))
        ax.scatter(d, y, s=38, color=COLOR[a], alpha=0.75, zorder=3,
                   marker=MARK[a], edgecolor="white", linewidth=0.8)
        ax.scatter([d.mean()], [k], s=200, color=COLOR[a], zorder=5,
                   marker="D", edgecolor=INK, linewidth=1.4)
        lo, hi = boot_ci(d)
        t, p = stats.ttest_rel(col(a, field), base)
        ax.plot([lo, hi], [k, k], color=INK, lw=2.4, zorder=4,
                solid_capstyle="butt", alpha=0.65)
        ax.text(-3.15, k + 0.30,
                f"{NAME[a]}   mean {d.mean():+.2f}   95% CI "
                f"[{lo:+.2f}, {hi:+.2f}]   {int((d < 0).sum())} of {len(d)} "
                f"worse   p = {p:.1e}",
                ha="left", va="bottom", fontsize=9.5, color=INK)
    ax.axvline(0, color=INK, lw=1.4, zorder=2)
    ax.text(0.05, 1.66, "no change", fontsize=8.5, color=INK_2)
    ax.set_yticks([])
    ax.set_xlim(-3.25, 1.35)
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel(f"change in {label} against the same item unedited")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def panel_dominance(ax, col, colo):
    """Listener quality against detector evasion: the whole argument."""
    off = {"baseline": (16, 6), "design_b": (18, -4), "design_a": (18, -30)}
    for a in ARMS:
        x, y = colo(a, "softmin"), col(a, "mos")
        ax.scatter(x, y, s=38, color=COLOR[a], alpha=0.40, zorder=3,
                   marker=MARK[a], edgecolor="white", linewidth=0.7)
        ax.scatter([x.mean()], [y.mean()], s=320, color=COLOR[a], zorder=6,
                   marker=MARK[a], edgecolor=INK, linewidth=1.8)
        ax.annotate(f"{NAME[a]}\n{x.mean():.3f}, {y.mean():.2f}",
                    (x.mean(), y.mean()), textcoords="offset points",
                    xytext=off[a], fontsize=9.5, color=INK,
                    fontweight="bold", zorder=7)
    bx, by = colo("baseline", "softmin").mean(), col("baseline", "mos").mean()
    ax.axvline(bx, color=C_BASE, lw=1.0, ls=":", zorder=2)
    ax.axhline(by, color=C_BASE, lw=1.0, ls=":", zorder=2)
    ax.fill_between([bx, 1.0], by, 5.6, color="#2a9d8f", alpha=0.10, zorder=1)
    ax.text(bx + 0.004, 5.42, "better on both axes: nothing is here",
            fontsize=9, color="#2a7f74", va="top")
    ax.set_xlabel("soft-min detector score (1 = human to every detector)")
    ax.set_ylabel("MOS from the listener")
    ax.set_ylim(0.6, 5.6)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def panel_correlation(ax, col, colo, items):
    """The pooled correlation, and why it does not mean what it looks like."""
    allm = np.concatenate([col(a, "mos") for a in ARMS])
    alls = np.concatenate([colo(a, "softmin") for a in ARMS])
    for a in ARMS:
        x, y = colo(a, "softmin"), col(a, "mos")
        ax.scatter(x, y, s=36, color=COLOR[a], alpha=0.45, marker=MARK[a],
                   zorder=3, edgecolor="white", linewidth=0.7, label=NAME[a])
        # each arm's own fit: flat, which is the point
        if x.std() > 0:
            s1, s0 = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 20)
            ax.plot(xx, s1 * xx + s0, color=COLOR[a], lw=1.6, ls="--",
                    zorder=4, alpha=0.9)
    # the line through the three arm means: this is all the pooled fit sees
    mx = [colo(a, "softmin").mean() for a in ARMS]
    my = [col(a, "mos").mean() for a in ARMS]
    order = np.argsort(mx)
    ax.plot(np.array(mx)[order], np.array(my)[order], color=INK, lw=2.2,
            zorder=5, marker="o", ms=9, mfc="white", mew=2)

    r, pr = stats.pearsonr(allm, alls)
    cm = np.concatenate([col(a, "mos") - col(a, "mos").mean() for a in ARMS])
    cs = np.concatenate([colo(a, "softmin") - colo(a, "softmin").mean()
                         for a in ARMS])
    rw, pw = stats.pearsonr(cm, cs)
    within = "   ".join(
        f"{NAME[a]} r = {stats.pearsonr(col(a,'mos'), colo(a,'softmin'))[0]:+.2f}"
        for a in ARMS)
    ax.text(0.03, 0.97,
            f"pooled over all {len(items) * 3} clips:  r = {r:+.2f} "
            f"(p = {pr:.3f})\n"
            f"within arm, between-arm variance removed:  r = {rw:+.2f} "
            f"(p = {pw:.2f}), not significant\n{within}",
            transform=ax.transAxes, va="top", fontsize=8.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#d6d9de"))
    ax.set_xlabel("soft-min detector score")
    ax.set_ylabel("MOS from the listener")
    ax.set_ylim(1.2, 6.2)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="MOS TEST RL/mos_ratings.json")
    ap.add_argument("--split", action="store_true",
                    help="also write each panel as its own figure")
    args = ap.parse_args()

    path = Path(args.ratings)
    if not path.is_file():
        print(f"no {path} -- export the ratings from the listening test first")
        return 1

    data, sub, obj, items = load(path)
    if not items:
        print("no item was rated on all three arms")
        return 1

    def col(a, f):
        return np.array([sub[i][a][f] for i in items], dtype=float)

    def colo(a, f):
        return np.array([obj[i][a][f] for i in items], dtype=float)

    n = len(items)
    written = []

    # ---- combined -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    a1, a2, a3, a4 = axes.ravel()
    panel_scale(a1, col, items, "mos", "MOS", "how human it sounds")
    a1.set_title(f"A. MOS: both designs sound less human\n"
                 f"{n} items, mean with a 95% interval",
                 fontsize=11, loc="left")
    panel_paired(a2, col, items, "mos", "MOS")
    a2.set_title("B. Paired, item by item\n"
                 "one dot is one voice and line; diamond is the mean",
                 fontsize=11, loc="left")
    panel_dominance(a3, col, colo)
    a3.set_title("C. Neither design trades quality for evasion\n"
                 "both move down and left: more detectable and worse sounding",
                 fontsize=11, loc="left")
    panel_correlation(a4, col, colo, items)
    a4.set_title("D. The two measures agree on the arms, not within them\n"
                 "dashed per-arm fits are flat; the slope is between arms",
                 fontsize=11, loc="left")
    fig.suptitle("Figure 20. The listening test: both reinforcement-learning "
                 "designs are worse than doing nothing",
                 fontsize=13.5, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = GRAPHS_DIR / "20_mos_rl.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    written.append(out)

    # ---- the panels on their own ----------------------------------------
    if args.split:
        f, ax = plt.subplots(figsize=(7.2, 5.6))
        panel_scale(ax, col, items, "mos", "MOS", "")
        ax.set_title("Figure 20a. MOS: how human the clip sounds\n"
                     f"{n} paired items, mean with a 95% interval",
                     fontsize=11, loc="left")
        f.tight_layout()
        o = GRAPHS_DIR / "20a_mos.png"
        f.savefig(o, dpi=150)
        plt.close(f)
        written.append(o)

        f, ax = plt.subplots(figsize=(9.5, 4.2))
        panel_paired(ax, col, items, "mos", "MOS")
        ax.set_title("Figure 20b. MOS item by item against the same clip "
                     "unedited\nbar is the bootstrap 95% interval on the mean",
                     fontsize=11, loc="left")
        f.tight_layout()
        o = GRAPHS_DIR / "20b_mos_paired.png"
        f.savefig(o, dpi=150)
        plt.close(f)
        written.append(o)

        f, ax = plt.subplots(figsize=(7.2, 5.6))
        panel_scale(ax, col, items, "pmos", "PMOS", "")
        ax.set_title("Figure 20c. PMOS: how much it is still the character\n"
                     f"{n} paired items, mean with a 95% interval",
                     fontsize=11, loc="left")
        f.tight_layout()
        o = GRAPHS_DIR / "20c_pmos.png"
        f.savefig(o, dpi=150)
        plt.close(f)
        written.append(o)

        f, ax = plt.subplots(figsize=(7.6, 6.0))
        panel_dominance(ax, col, colo)
        ax.set_title("Figure 20d. Neither design trades quality for evasion\n"
                     "both move down and left, so nothing is exchanged",
                     fontsize=11, loc="left")
        f.tight_layout()
        o = GRAPHS_DIR / "20d_dominance.png"
        f.savefig(o, dpi=150)
        plt.close(f)
        written.append(o)

        f, ax = plt.subplots(figsize=(7.6, 6.0))
        panel_correlation(ax, col, colo, items)
        ax.set_title("Figure 20e. The listener and the detectors agree on the "
                     "ranking of\nthe arms and on nothing finer",
                     fontsize=11, loc="left")
        f.tight_layout()
        o = GRAPHS_DIR / "20e_agreement.png"
        f.savefig(o, dpi=150)
        plt.close(f)
        written.append(o)

    for o in written:
        print(f"wrote {o}")

    # ---- the same numbers as text ---------------------------------------
    print(f"\n{n} items rated on all three arms "
          f"(exported {data.get('exportedAt', '?')})")
    print(f"\n{'arm':<10}{'MOS':>7}{'PMOS':>7}{'soft-min':>11}{'PESQ':>7}"
          f"{'sim':>7}")
    for a in ARMS:
        print(f"{NAME[a]:<10}{col(a,'mos').mean():>7.2f}"
              f"{col(a,'pmos').mean():>7.2f}{colo(a,'softmin').mean():>11.3f}"
              f"{colo(a,'pesq').mean():>7.2f}{colo(a,'sim').mean():>7.3f}")

    print(f"\n{'paired vs unedited':<22}{'diff':>8}{'95% CI':>18}{'t':>8}"
          f"{'p':>10}{'dz':>7}{'worse':>8}")
    for a in ("design_a", "design_b"):
        for field, lab, src in (("mos", "MOS", col), ("pmos", "PMOS", col),
                                ("softmin", "soft-min", colo),
                                ("pesq", "PESQ", colo)):
            b, x = src("baseline", field), src(a, field)
            d = x - b
            t, p = stats.ttest_rel(x, b)
            lo, hi = boot_ci(d)
            print(f"{NAME[a] + ' ' + lab:<22}{d.mean():>+8.3f}"
                  f"   [{lo:+.3f}, {hi:+.3f}]{t:>8.2f}{p:>10.1e}"
                  f"{d.mean()/d.std(ddof=1):>+7.2f}"
                  f"{int((d < 0).sum()):>6}/{len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
