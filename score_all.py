r"""
score_all.py -- phase 2 of the benchmark. RUN IN THE "voicefake" ENV.

Scores everything generate_all.py produced with all four anti-spoofing models
and writes the results + two figures to D:\results and benchmarks\discriminator\.

    conda activate voicefake
    python score_all.py --voice riven

Every run also scores the ORIGINAL human reference.wav as a control. Without
it the numbers are uninterpretable: a low score on cloned audio only means
"detected" if the same detector gives the genuine recording a high score.

Two figures:
  1. HEATMAP     every clip x every detector, colour = P(human). Answers
                 "which detector catches which clone?" at a glance.
  2. BAR CHART   human reference vs each cloned model, grouped by detector.
                 Answers "is a given detector actually distinguishing real
                 from generated speech, and which model evades it best?" --
                 the core question this repo (AHJN-dodging) exists to ask.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import detectors

from env_config import GENERATORS_DIR as GENERATORS, DISCRIMINATOR_DIR as DISCRIMINATOR


def collect(voice: str) -> list[dict]:
    """Human reference first (as control), then every generated clip."""
    items: list[dict] = []
    man_path = GENERATORS / f"manifest_{voice}.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf8"))
        ref = Path(man["reference_wav"])
        if ref.is_file():
            items.append({"label": "HUMAN (reference)", "source": "reference",
                          "truth": "human", "path": ref})
        for o in man["outputs"]:
            items.append({"label": o["model"], "source": o["model"],
                          "truth": "ai", "path": Path(o["path"])})
    else:
        print(f"no manifest at {man_path}; falling back to a directory scan")
        for wav in sorted(GENERATORS.rglob(f"{voice}.wav")):
            items.append({"label": wav.parent.name, "source": wav.parent.name,
                          "truth": "ai", "path": wav})
    return [i for i in items if i["path"].is_file()]


# --------------------------------------------------------------------------
# 1. clip x detector heatmap
# --------------------------------------------------------------------------

def plot_score_heatmap(rows: list[dict], names: list[str], out_png: Path) -> None:
    """Every clip x every detector, one glance at who catches what.

    Diverging red/green rather than Tester.py's magma -- these values carry
    a verdict (human vs AI), not a plain magnitude, and red-for-AI /
    green-for-human reads instantly without checking the colourbar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["clip"] for r in rows]
    mat = np.array([[r[n] for n in names] for r in rows], dtype=float)

    n_rows, n_cols = mat.shape
    fig_w = max(6.5, 1.15 * n_cols + 2.6)
    fig_h = max(3.2, 0.55 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_cols)); ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Detector confidence per clip\n1.00 = called human, 0.00 = called AI",
                 fontsize=11, pad=12)

    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="black" if 0.25 < v < 0.75 else "white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(human)")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="0.3", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. human vs AI, grouped by detector
# --------------------------------------------------------------------------

def plot_human_vs_ai_bars(rows: list[dict], names: list[str], out_png: Path) -> None:
    """Grouped bars: one group per detector, one bar per clip source.

    The 0.5 line is the natural read-off point for which side a detector
    landed on -- these models are binary classifiers, so it doubles as
    their decision boundary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted({r["source"] for r in rows}, key=lambda s: (s != "reference", s))
    source_labels = ["Human (reference)" if s == "reference" else s for s in sources]

    n_det, n_src = len(names), len(sources)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * n_det + 1.5), 5.0))

    width = 0.8 / n_src
    x = np.arange(n_det)
    palette = ["#2a9d8f", "#e76f51", "#e9c46a", "#264653", "#9b5de5", "#f4a261"]

    for si, (src, lab) in enumerate(zip(sources, source_labels)):
        vals = []
        for n in names:
            v = [r[n] for r in rows if r["source"] == src]
            vals.append(float(np.nanmean(v)) if v else np.nan)
        offset = (si - (n_src - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=lab, color=palette[si % len(palette)])

    ax.axhline(0.5, color="0.35", linewidth=1, linestyle="--", zorder=0)
    ax.text(n_det - 0.5, 0.505, "decision boundary (0.5)", fontsize=8,
            color="0.35", ha="right", va="bottom")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("P(human)  --  1.0 = human, 0.0 = AI")
    ax.set_title("Human reference vs each cloned model, per detector", fontsize=11, pad=12)
    ax.legend(fontsize=9, ncols=min(n_src, 3), loc="upper center",
             bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="riven")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "min", "median"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    items = collect(args.voice)
    if not items:
        print(f"nothing to score for {args.voice!r} -- run generate_all.py first")
        return 1

    DISCRIMINATOR.mkdir(parents=True, exist_ok=True)
    names = list(detectors.MODEL_ORDER)
    rows = []

    print(f"scoring {len(items)} clip(s); 1.0 = human, 0.0 = AI\n")
    header = f"{'clip':<22}{'truth':<8}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))

    for it in items:
        scores = detectors.score_file(it["path"], aggregate=args.aggregate,
                                      device=args.device)
        rows.append({
            "voice": args.voice, "clip": it["label"], "source": it["source"],
            "truth": it["truth"], "path": str(it["path"]),
            **{n: round(float(s), 6) for n, s in zip(names, scores)},
            "mean_across_detectors": round(float(np.nanmean(scores)), 6),
        })
        print(f"{it['label']:<22}{it['truth']:<8}"
              + "".join(f"{s:>12.4f}" for s in scores))

    # per-detector separation between the human control and the clones
    human = [r for r in rows if r["truth"] == "human"]
    ai = [r for r in rows if r["truth"] == "ai"]
    summary = {}
    if human and ai:
        print(f"\n{'detector':<14}{'human':>10}{'ai (mean)':>12}{'margin':>10}   verdict")
        print("-" * 60)
        for n in names:
            h = float(np.nanmean([r[n] for r in human]))
            a = float(np.nanmean([r[n] for r in ai]))
            margin = h - a
            verdict = ("separates" if margin > 0.20 else
                       "weak" if margin > 0.05 else
                       "no separation" if margin > -0.05 else "INVERTED")
            summary[n] = {"human": h, "ai_mean": a, "margin": margin,
                          "verdict": verdict}
            print(f"{n:<14}{h:>10.4f}{a:>12.4f}{margin:>10.4f}   {verdict}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = DISCRIMINATOR / f"scores_{args.voice}_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path = DISCRIMINATOR / f"scores_{args.voice}_{stamp}.json"
    json_path.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "voice": args.voice,
        "aggregate": args.aggregate,
        "convention": "1.0 = human (bonafide), 0.0 = AI (spoofed)",
        "rows": rows,
        "separation": summary,
    }, indent=2), encoding="utf8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")

    if not args.skip_plots:
        heatmap_png = DISCRIMINATOR / f"heatmap_{args.voice}_{stamp}.png"
        plot_score_heatmap(rows, names, heatmap_png)
        print(f"wrote {heatmap_png}")

        if human and ai:
            bars_png = DISCRIMINATOR / f"human_vs_ai_{args.voice}_{stamp}.png"
            plot_human_vs_ai_bars(rows, names, bars_png)
            print(f"wrote {bars_png}")
        else:
            print("skipped human-vs-AI bar chart: need both a human reference "
                 "and at least one generated clip")

    return 0


if __name__ == "__main__":
    sys.exit(main())
