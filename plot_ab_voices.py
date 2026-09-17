r"""
plot_ab_voices.py -- turn eval_ab_voices.py's JSON into the two requested
figures, written into Plan_X/.

    python plot_ab_voices.py

Figure 7  detector scores per voice: mean over the four detectors (top) and
          the soft-min worst case the training objective actually optimises
          (bottom), baseline vs Design A vs Design B, with group averages.
Figure 8  the two constraints per voice: similarity to the reference speaker
          and WER, same three arms, same group averages.

Voices are split into the five randomly-drawn evaluation voices and the four
the policies were trained on, because "no gain on unseen voices" and "no gain
anywhere" are different findings and only the trained group separates them.
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

PLAN_X = GRAPHS_DIR
JSON = RESULTS_ROOT / "rl_runs" / "ab_voice_eval" / "ab_voice_eval.json"

C_BASE, C_A, C_B = "#8a8f98", "#d1495b", "#2a9d8f"
ARMS = (("base", "baseline (no edit)", C_BASE),
        ("A", "Design A (PPO)", C_A),
        ("B", "Design B (SAC)", C_B))


def layout(per_voice):
    """x positions, tick labels, and the index where the trained block starts."""
    held = [r for r in per_voice if not r["trained_B"]]
    trained = [r for r in per_voice if r["trained_B"]]
    return held, trained


def grouped(ax, held, trained, key, scale=1.0, pct=False):
    """One panel: per-voice grouped bars, then the two group averages."""
    labels, blocks = [], []
    for r in held:
        labels.append(r["voice"])
        blocks.append(r)
    for r in trained:
        labels.append(r["voice"] + " *")
        blocks.append(r)

    def mean_of(rs, arm):
        return float(np.mean([r[f"{arm}_{key}"] for r in rs])) if rs else np.nan

    avg_cols = []
    if held:
        avg_cols.append(("AVG\nevaluation", {a: mean_of(held, a) for a, _, _ in ARMS}))
    if trained:
        avg_cols.append(("AVG\ntrained *", {a: mean_of(trained, a) for a, _, _ in ARMS}))

    n = len(blocks)
    gap = 0.8  # visual break before the averages
    xs = np.arange(n, dtype=float)
    xa = np.arange(len(avg_cols), dtype=float) + n + gap
    width = 0.26

    for i, (arm, label, colour) in enumerate(ARMS):
        off = (i - 1) * width
        v = [r[f"{arm}_{key}"] * scale for r in blocks]
        ax.bar(xs + off, v, width, color=colour, label=label,
               edgecolor="white", linewidth=0.5)
        va = [c[1][arm] * scale for c in avg_cols]
        ax.bar(xa + off, va, width, color=colour, edgecolor="black", linewidth=1.1)

    if trained:
        ax.axvspan(len(held) - 0.5, n - 0.5, color="#f2c14e", alpha=0.13, zorder=0)
    ax.axvline(n - 0.5 + gap / 2, color="#333", lw=0.9, ls=":")

    ax.set_xticks(list(xs) + list(xa))
    ax.set_xticklabels(labels + [c[0] for c in avg_cols], fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    return avg_cols


def main() -> int:
    if not JSON.is_file():
        print(f"no results at {JSON} -- run eval_ab_voices.py first")
        return 1
    d = json.loads(JSON.read_text(encoding="utf8"))
    pv = d["per_voice"]
    held, trained = layout(pv)
    PLAN_X.mkdir(parents=True, exist_ok=True)

    a_it, b_it = d["arms"]["A"]["iteration"], d["arms"]["B"]["iteration"]
    sub = (f"{d['pairs_per_voice']} utterances per voice, identical seeds across all "
           f"three arms  |  A = {d['arms']['A']['run']} iter {a_it}, "
           f"B = {d['arms']['B']['run']} iter {b_it}  |  * = voice the policies trained on")

    # ---------------- figure 7: detector scores ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Detector score per voice, best Design A vs best Design B\n"
                 "higher = scored as more human (1.0 = human, 0.0 = AI)",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.925, sub, ha="center", fontsize=8.5, color="#444")

    avg = grouped(axes[0], held, trained, "meandet")
    axes[0].set_ylabel("mean over the 4 detectors")
    axes[0].set_title("plain mean across AASIST, AASIST-L, RawNet2, RawGAT-ST",
                      fontsize=10, loc="left")
    axes[0].legend(fontsize=9, ncol=3, loc="upper left", framealpha=0.9)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="#b00", lw=0.8, ls="--", alpha=0.6)
    axes[0].text(-0.45, 0.52, "detector decision boundary", fontsize=7.5, color="#b00")

    grouped(axes[1], held, trained, "softmin")
    axes[1].set_ylabel("soft-min (worst detector)")
    axes[1].set_title("soft-min -- the worst-case term Design B optimises; "
                      "this is the number that has to move", fontsize=10, loc="left")
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout(rect=(0, 0, 1, 0.915))
    p7 = PLAN_X / "7_per_voice_detector.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)

    # ---------------- figure 8: similarity and WER ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Voice quality per voice -- what the edit costs\n"
                 "similarity to the target speaker, and word error rate",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.925, sub, ha="center", fontsize=8.5, color="#444")

    grouped(axes[0], held, trained, "sim")
    axes[0].set_ylabel("cosine similarity to reference")
    axes[0].set_title("similarity to the target voice -- higher is better",
                      fontsize=10, loc="left")
    axes[0].axhline(0.70, color="#b00", lw=1.0, ls="--")
    axes[0].text(-0.45, 0.715, "constraint floor 0.70", fontsize=7.5, color="#b00")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=9, ncol=3, loc="lower left", framealpha=0.9)

    grouped(axes[1], held, trained, "wer", scale=100.0, pct=True)
    axes[1].set_ylabel("word error rate")
    axes[1].set_title("WER -- lower is better", fontsize=10, loc="left")
    axes[1].axhline(15.0, color="#b00", lw=1.0, ls="--")
    axes[1].text(-0.45, 15.6, "constraint ceiling 15%", fontsize=7.5, color="#b00")

    fig.tight_layout(rect=(0, 0, 1, 0.915))
    p8 = PLAN_X / "8_per_voice_quality.png"
    fig.savefig(p8, dpi=150)
    plt.close(fig)

    # ---------------- console summary ----------------
    def blk(rs, name):
        if not rs:
            return
        print(f"\n{name} ({len(rs)} voices)")
        for key, lab, sc, u in (("meandet", "mean detector", 1, ""),
                                ("softmin", "soft-min     ", 1, ""),
                                ("sim", "similarity   ", 1, ""),
                                ("wer", "WER          ", 100, "%")):
            b = np.mean([r[f"base_{key}"] for r in rs]) * sc
            a = np.mean([r[f"A_{key}"] for r in rs]) * sc
            bb = np.mean([r[f"B_{key}"] for r in rs]) * sc
            print(f"  {lab}  base {b:7.3f}{u}   A {a:7.3f}{u} ({a - b:+.3f})"
                  f"   B {bb:7.3f}{u} ({bb - b:+.3f})")

    blk(held, "EVALUATION VOICES (randomly drawn, never trained on)")
    blk(trained, "TRAINED VOICES (reference group)")
    print(f"\nwrote {p7}\nwrote {p8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
