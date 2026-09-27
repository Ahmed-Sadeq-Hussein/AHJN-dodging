r"""
plot_longform.py -- figure 18: long audio under the global-file objective.

    python plot_longform.py

Section 7e measured the learned operator's gain vanishing at ~72 s and
recorded it as a scope limit. Section 7r re-tests the same lengths with each
chunk optimised against the WHOLE FILE's soft-min (longform.py). This draws
the two against each other.

Left    per clip, soft-min before and after, grouped by audio length.
Right   gain against length: the old per-chunk numbers from 7e beside the new
        global-objective ones. The question the figure answers is whether the
        failure at length was in the audio or in what was being optimised.

Reads whatever runs exist, so it is safe to draw mid-run: longform_attack_
medlong.json (13 s, 50 s) and longform_xlong.json (~72 s), which longform.py
now rewrites after every clip.
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
C_BASE, C_NEW, C_OLD = "#8a8f98", "#4a5fc1", "#2a9d8f"

# section 7e, per-chunk operator at each length (mean over the voices tested)
SEVEN_E = {"medium": (13.6, 0.164), "xlong": (72.0, 0.004)}


def load(name):
    p = RL / name
    if not p.is_file():
        return []
    return json.loads(p.read_text(encoding="utf8")).get("rows", [])


def main() -> int:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load("longform_attack_medlong.json") + load("longform_xlong.json")
    if not rows:
        print("no long-form results yet")
        return 1

    order = ["medium", "long", "xlong"]
    rows.sort(key=lambda r: (order.index(r["length"]) if r["length"] in order
                             else 9, -r["attacked"]["softmin"]))
    skipped = [r for r in rows if r.get("skip_frac")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    xs = np.arange(len(rows))
    ax1.bar(xs - 0.2, [r["base"]["softmin"] for r in rows], 0.38,
            color=C_BASE, label="unedited")
    ax1.bar(xs + 0.2, [r["attacked"]["softmin"] for r in rows], 0.38,
            color=C_NEW, label="attacked against the whole-file score")
    ax1.axhline(0.5, color="#b00", lw=1.2, ls="--", label="soft-min 0.5")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{r['voice']}\n{r['seconds']:.0f}s" for r in rows],
                        fontsize=8)
    ax1.set_ylabel("soft-min (worst detector, 1 = human)")
    ax1.set_title("Per clip: every length moves once the objective is the file's",
                  fontsize=11, loc="left")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9)

    # right: gain against length, old vs new
    by_len = {}
    for r in rows:
        by_len.setdefault(r["length"], []).append(
            (r["seconds"], r["attacked"]["softmin"] - r["base"]["softmin"]))
    nl = sorted(((float(np.mean([s for s, _ in v])),
                  float(np.mean([g for _, g in v])), k)
                 for k, v in by_len.items()))
    ax2.plot([x[0] for x in nl], [x[1] for x in nl], "-o", color=C_NEW, lw=2,
             ms=8, label="global file objective (7r, teacher)")
    ol = sorted(SEVEN_E.values())
    ax2.plot([x[0] for x in ol], [x[1] for x in ol], "-s", color=C_OLD, lw=2,
             ms=8, label="per-chunk learned operator (7e)")
    ax2.axhline(0, color="#444", lw=1)
    ax2.set_xlabel("audio length (seconds)")
    ax2.set_ylabel("soft-min gain")
    ax2.set_title("The failure at length was the objective, not the audio",
                  fontsize=11, loc="left")
    ax2.grid(alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    if skipped:
        fig.text(0.01, 0.01, f"warning: {len(skipped)} clip(s) hit OOM skips "
                 f"(max {max(r['skip_frac'] for r in skipped):.0%} of chunk "
                 f"steps) and are optimised on a fraction of the file",
                 fontsize=8, color="#b00")

    fig.tight_layout()
    out = GRAPHS_DIR / "18_longform.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} clips, lengths {sorted(by_len)})")
    for s, g, k in nl:
        print(f"  {k:<8}{s:5.0f}s   mean gain {g:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
