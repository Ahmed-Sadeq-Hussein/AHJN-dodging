# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
collect_graphs.py -- gather every figure into graphs/ and write its index.

    python collect_graphs.py

Figures are produced by half a dozen scripts and some land outside the repo
(Tester.py and score_all.py write under RESULTS_ROOT, analyze_run.py writes a
trend.png inside each training run). This copies those in under graphs/runs/
and regenerates graphs/INDEX.md so the whole set is browsable in one place
with a caption saying what each one shows and which section of
Plan_X/RESULTS.md it belongs to.

Safe to re-run: it copies only when the source is newer, and never deletes.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import GRAPHS_DIR, RESULTS_ROOT

# figure number -> (what it shows, RESULTS.md section)
CAPTIONS = {
    "1_learning_curves": ("Design A and B training curves: reward, soft-min and "
                          "constraint terms per iteration", "3, 4"),
    "1b_softmin_comparable": ("Design A vs B on the same soft-min metric, since "
                              "the two optimised different rewards", "4"),
    "2_per_detector": ("Per-detector movement under Design A -- the aggregator "
                       "left AASIST undefended and it degraded", "3"),
    "3_constraints": ("Similarity and WER against their limits; neither ever "
                      "bound", "4"),
    "4_action_scale": ("Edit magnitude over training: the max_delta cap was "
                       "never approached", "4"),
    "5_diagnostics": ("SAC internals -- the entropy coefficient rising for nine "
                      "hours, the signature of a critic that cannot rank", "4"),
    "6_pgd_ceiling": ("What white-box PGD reaches, establishing the intervention "
                      "point works", "2"),
    "7_per_voice_detector": ("Design A vs B per voice on detector scores", "5"),
    "8_per_voice_quality": ("What A and B cost in similarity and WER", "5"),
    "9_distill_curves": ("Distillation: regression loss, direction agreement, "
                         "and the fraction of PGD's gain recovered", "7"),
    "10_distill_vs_rl": ("Distilled model vs RL on unseen voices and sentences",
                         "7, 7h"),
    "11_distill_cost": ("Similarity, WER, PESQ and STOI for the distilled edit",
                        "7"),
    "12_distill_sweep": ("Capacity sweep -- smaller networks won when imitating "
                         "fixed targets", "7"),
    "13_amplify": ("Applying the edit repeatedly compounds; scaling it up does "
                   "not. The step-operator result", "7b, 7h"),
    "14_per_voice_analysis": ("Which voices the edit helps, against reference "
                              "acoustics that mostly fail to predict it", "7l"),
    "15_margin_tradeoff": ("Evasion against PESQ cost for the margin objective, "
                           "with the quality budget marked", "7m, 7n"),
    "16_margin_per_voice": ("Per voice at both operating points: 14 of 15 clear "
                            "soft-min 0.5 when quality is not constrained", "7n"),
    "17_frontier": ("Every method on one axis: soft-min bought per unit of "
                    "audio damage", "7q"),
    "18_longform": ("Long audio attacked against the whole-file score: the "
                    "failure at length was the objective, not the audio", "7r"),
    "19_qc_frontier": ("Constraining the teacher moves the evasion/quality "
                       "frontier; all 15 voices on one basis", "7s, 7t"),
}


def main() -> int:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = GRAPHS_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)

    copied = 0
    for src in sorted(RESULTS_ROOT.rglob("*.png")):
        rel = src.relative_to(RESULTS_ROOT)
        dst = runs_dir / "_".join(rel.parts)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied += 1
    print(f"copied {copied} figure(s) from {RESULTS_ROOT} into {runs_dir}")

    main_figs = sorted(
        (p for p in GRAPHS_DIR.glob("*.png")),
        key=lambda p: (not p.stem[0].isdigit(),
                       int("".join(c for c in p.stem.split("_")[0] if c.isdigit())
                           or 0), p.stem))

    lines = ["# Figures", "",
             "Every graph this project produced. The numbered set is the main "
             "sequence, cited by section in `Plan_X/RESULTS.md`; `runs/` holds "
             "the incidental ones written by `Tester.py`, `score_all.py` and "
             "`analyze_run.py` during individual runs.", "",
             "Regenerate the collection with `python collect_graphs.py`.", "",
             "## Main sequence", "",
             "| figure | shows | RESULTS.md |", "|---|---|---|"]
    for p in main_figs:
        cap, sec = CAPTIONS.get(p.stem, ("--", "--"))
        lines.append(f"| [`{p.name}`]({p.name}) | {cap} | {sec} |")

    extra = sorted(runs_dir.glob("*.png"))
    if extra:
        lines += ["", "## From individual runs", "",
                  "| figure | source |", "|---|---|"]
        for p in extra:
            lines.append(f"| [`runs/{p.name}`](runs/{p.name}) | "
                         f"`{p.name.replace('_', '/', p.name.count('_') - 1)}` |")

    (GRAPHS_DIR / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"wrote {GRAPHS_DIR / 'INDEX.md'}  ({len(main_figs)} main, {len(extra)} from runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
