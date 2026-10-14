# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
plot_distill.py -- figures for the distillation run, into graphs/.

    python plot_distill.py            # defaults are the documented invocation

Figure  9  distillation learning curves: regression loss, the cosine between
           predicted and target edit (direction agreement, which MSE hides),
           and the fraction of PGD's objective gain the network actually
           recovers on held-out sentences -- the number the run is judged on.
Figure 10  final head-to-head on unseen voices and unseen sentences:
           baseline, Design B (RL), distilled, with PGD's per-utterance
           result drawn as the reachable ceiling.
Figure 11  what the edit costs: similarity, WER, and the PESQ/STOI estimates
           this project only started measuring after the PGD audit.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import GRAPHS_DIR, RESULTS_ROOT

PLAN_X = GRAPHS_DIR
RL_RUNS = RESULTS_ROOT / "rl_runs"

C_BASE, C_B, C_D, C_PGD = "#8a8f98", "#2a9d8f", "#4a5fc1", "#e07a1f"
ARM_STYLE = {"baseline": ("baseline (no edit)", C_BASE),
             "designB": ("Design B (SAC/RL)", C_B),
             "distilled": ("distilled from PGD", C_D)}


def fig_curves(run_dir: Path):
    log = run_dir / "train_log.jsonl"
    if not log.is_file():
        return None
    rows = [json.loads(l) for l in log.read_text(encoding="utf8").splitlines() if l.strip()]
    if not rows:
        return None
    ep = np.array([r["epoch"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Distillation: learning to predict PGD's edit by regression",
                 fontsize=13, y=0.98)

    axes[0].plot(ep, [r["train_mse"] for r in rows], color=C_D, lw=1.4, label="train")
    axes[0].plot(ep, [r["val_mse"] for r in rows], color="#b00", lw=1.4,
                 label="val (unseen sentences)")
    axes[0].set_ylabel("MSE on the edit")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title("regression loss -- a gap opening here is memorisation",
                      fontsize=10, loc="left")

    axes[1].plot(ep, [r["train_cos"] for r in rows], color=C_D, lw=1.4, label="train")
    axes[1].plot(ep, [r["val_cos"] for r in rows], color="#b00", lw=1.4, label="val")
    axes[1].axhline(0, color="#333", lw=0.8, ls=":")
    axes[1].set_ylabel("cosine(pred, PGD target)")
    axes[1].legend(fontsize=9)
    axes[1].set_title("direction agreement -- MSE can fall while this stays at zero, "
                      "which would mean the net is only shrinking its output",
                      fontsize=10, loc="left")

    rec = [(r["epoch"], r["recovered"]) for r in rows if r.get("recovered") is not None]
    if rec:
        x, y = zip(*rec)
        axes[2].plot(x, np.array(y) * 100, color="#111", lw=1.6, marker="o", ms=3)
    axes[2].axhline(100, color=C_PGD, lw=1.2, ls="--", label="PGD (per-utterance)")
    axes[2].axhline(0, color=C_BASE, lw=1.2, ls="--", label="no edit")
    axes[2].set_ylabel("% of PGD's gain recovered")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=9)
    axes[2].set_title("the decisive metric: predicted edit pushed through HiFT "
                      "and the detectors, on held-out sentences",
                      fontsize=10, loc="left")

    for a in axes:
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = PLAN_X / "9_distill_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _bars(ax, groups, arms, key, pgd_line=None, scale=1.0, pct=False):
    n = len(groups)
    xs = np.arange(n, dtype=float)
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        off = (i - (len(arms) - 1) / 2) * width
        v = [g[key][arm] * scale for g in groups]
        label, colour = ARM_STYLE.get(arm, (arm, "#777"))
        ax.bar(xs + off, v, width, color=colour, label=label,
               edgecolor="white", linewidth=0.6)
    if pgd_line is not None:
        ax.axhline(pgd_line * scale, color=C_PGD, lw=1.6, ls="--",
                   label="PGD ceiling (per-utterance)")
    ax.set_xticks(xs)
    # The group labels are long ("REFERENCE voices (seen in collection)") and
    # at two groups per axis they were drawing on top of each other, which made
    # figure 11 unreadable. Wrap rather than truncate: which group is which is
    # the whole point of the comparison.
    ax.set_xticklabels(["\n".join(textwrap.wrap(g["label"].split(",")[0], 18))
                        for g in groups], fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def fig_final(payload, pgd_stats):
    groups = payload["groups"]
    arms = payload["arms"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Detection evasion on unseen voices and unseen sentences",
                 fontsize=13, y=0.97)
    _bars(axes[0], groups, arms, "softmin",
          pgd_line=pgd_stats.get("softmin_after"))
    axes[0].set_ylabel("soft-min (worst detector)")
    axes[0].set_title("soft-min -- the objective", fontsize=10, loc="left")
    axes[0].legend(fontsize=8.5, loc="upper left")

    _bars(axes[1], groups, arms, "meandet")
    axes[1].set_ylabel("mean over the 4 detectors")
    axes[1].set_title("mean detector score (1 = human)", fontsize=10, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = PLAN_X / "10_distill_vs_rl.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_cost(payload):
    groups = payload["groups"]
    arms = payload["arms"]
    have_q = "pesq" in groups[0]
    ncol = 4 if have_q else 2

    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 5.2))
    fig.suptitle("What the edit costs -- voice, intelligibility, perceptual quality",
                 fontsize=13, y=0.97)

    _bars(axes[0], groups, arms, "sim")
    axes[0].set_ylabel("similarity to target speaker")
    axes[0].axhline(0.70, color="#b00", lw=1.0, ls="--")
    axes[0].set_title("similarity (higher better)", fontsize=9.5, loc="left")
    axes[0].legend(fontsize=7.5, loc="lower left")

    _bars(axes[1], groups, arms, "wer", scale=100.0, pct=True)
    axes[1].set_ylabel("word error rate")
    axes[1].axhline(15.0, color="#b00", lw=1.0, ls="--")
    axes[1].set_title("WER (lower better)", fontsize=9.5, loc="left")

    if have_q:
        _bars(axes[2], groups, arms, "pesq")
        axes[2].set_ylabel("SQUIM PESQ estimate")
        axes[2].set_title("PESQ (higher better)", fontsize=9.5, loc="left")
        _bars(axes[3], groups, arms, "stoi")
        axes[3].set_ylabel("SQUIM STOI estimate")
        axes[3].set_ylim(0.8, 1.02)
        axes[3].set_title("STOI (higher better)", fontsize=9.5, loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = PLAN_X / "11_distill_cost.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_sweep(control_rec=None):
    """Recovered fraction against model capacity, over every sweep run.

    The ordering here is the finding: every step toward a SMALLER, more
    regularised network improved held-out recovery, which is what a dataset
    of 221 examples in an 80xT output space should do.
    """
    runs = sorted(RL_RUNS.glob("distill_*"))
    pts = []
    for r in runs:
        cfg_p, ck = r / "config.json", r / "best.pt"
        if not (cfg_p.is_file() and ck.is_file()):
            continue
        cfg = json.loads(cfg_p.read_text(encoding="utf8"))
        # This figure's claim is that every point trained on the SAME 221
        # targets, so only runs on the default (probability-objective) dataset
        # belong. Later sections added distill runs on pgd_dataset_margin,
        # _breath and _lodo; those have a different corpus and a different
        # denominator for "recovered", and silently mixing them in made the
        # figure contradict its own title.
        if cfg.get("data"):
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        if "recovered" not in blob:
            continue
        n = sum(v.numel() for v in blob["actor"].values())
        pts.append({"name": r.name.replace("distill_", ""), "params": n,
                    "recovered": blob["recovered"],
                    "wd": cfg.get("weight_decay", 0.0),
                    "noise": cfg.get("input_noise", 0.0)})
    if not pts:
        return None
    pts.sort(key=lambda p: p["params"])

    fig, ax = plt.subplots(figsize=(11, 6))
    for p in pts:
        reg = p["wd"] + p["noise"]
        ax.scatter(p["params"], p["recovered"] * 100, s=90,
                   c=[[0.15, 0.35 + min(reg, 1.0) * 0.5, 0.6]],
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(p["name"], (p["params"], p["recovered"] * 100),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("actor parameters (log scale)")
    ax.set_ylabel("% of PGD's gain recovered on held-out sentences")
    ax.axhline(0, color=C_BASE, lw=1.2, ls="--", label="no edit")
    ax.axhline(100, color=C_PGD, lw=1.2, ls="--", label="PGD (per-utterance)")
    if control_rec is not None:
        ax.axhline(control_rec * 100, color="#b00", lw=1.2, ls=":",
                   label=f"one fixed average edit ({control_rec * 100:+.1f}%)")
    ax.set_title("Distillation capacity sweep -- smaller and more regularised wins\n"
                 "every point is a network trained on the same 221 PGD solutions",
                 fontsize=12, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = PLAN_X / "12_distill_sweep.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_amplify(run):
    """Recovered gain against iteration count, one line per edit scale.

    The shape is the argument: iterating compounds while scaling overshoots,
    which is how a learned descent direction behaves and how a memorised
    displacement does not.
    """
    j = RL_RUNS / run / "amplify_scan.json"
    if not j.is_file():
        return None
    d = json.loads(j.read_text(encoding="utf8"))
    rows = d["results"]
    scales = sorted({r["scale"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(scales):
        rs = sorted([r for r in rows if r["scale"] == s], key=lambda r: r["iters"])
        c = cmap(i / max(len(scales) - 1, 1))
        axes[0].plot([r["iters"] for r in rs], [r["recovered"] * 100 for r in rs],
                     marker="o", ms=5, lw=1.8, color=c, label=f"scale {s:g}")
        axes[1].plot([r["iters"] for r in rs], [r["softmin"] for r in rs],
                     marker="o", ms=5, lw=1.8, color=c, label=f"scale {s:g}")

    axes[0].axhline(0, color=C_BASE, lw=1.2, ls="--", label="no edit")
    axes[0].axhline(100, color=C_PGD, lw=1.2, ls="--", label="PGD (60 grad steps)")
    axes[0].set_ylabel("% of PGD's gain recovered")
    axes[0].set_xlabel("forward passes (each re-hints on the edited mel)")
    axes[0].set_title("iterating compounds; scaling does not", fontsize=10, loc="left")
    axes[0].legend(fontsize=8.5, ncol=2)

    axes[1].axhline(d["baseline_softmin"], color=C_BASE, lw=1.2, ls="--",
                    label=f"baseline {d['baseline_softmin']:.3f}")
    axes[1].axhline(d["pgd_softmin"], color=C_PGD, lw=1.2, ls="--",
                    label=f"PGD {d['pgd_softmin']:.3f}")
    axes[1].set_ylabel("soft-min (worst detector)")
    axes[1].set_xlabel("forward passes")
    axes[1].set_title("absolute soft-min reached", fontsize=10, loc="left")
    axes[1].legend(fontsize=8.5)

    for a in axes:
        a.grid(alpha=0.25, lw=0.6)
        a.set_axisbelow(True)
        a.set_xticks(sorted({r["iters"] for r in rows}))
    fig.suptitle("The distilled network is a step operator, not a one-shot edit",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = PLAN_X / "13_amplify.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    # These defaults are the documented invocation (Plan_X/RESULTS.md,
    # "Reproducing"). distill_distill1 was an early sweep run, and leaving it
    # as the default meant a bare `python plot_distill.py` silently redrew
    # figures 9-12 from the wrong model and dropped the control line.
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--control", type=float, default=-0.155,
                    help="recovered fraction of the fixed-average-edit control")
    ap.add_argument("--eval", default=None)
    args = ap.parse_args()

    PLAN_X.mkdir(parents=True, exist_ok=True)
    written = []

    p = fig_curves(RL_RUNS / args.run)
    if p:
        written.append(p)

    p = fig_sweep(args.control)
    if p:
        written.append(p)

    p = fig_amplify(args.run)
    if p:
        written.append(p)

    # PGD's own numbers, straight from the collection log, as the ceiling
    pgd_stats = {}
    clog = RL_RUNS / "pgd_dataset" / "collect_log.jsonl"
    if clog.is_file():
        rows = [json.loads(l) for l in clog.read_text(encoding="utf8").splitlines()
                if l.strip()]
        if rows:
            pgd_stats = {
                "softmin_before": float(np.mean([r["softmin_before"] for r in rows])),
                "softmin_after": float(np.mean([r["softmin_after"] for r in rows])),
                "sim_before": float(np.mean([r["sim_before"] for r in rows])),
                "sim_after": float(np.mean([r["sim_after"] for r in rows])),
                "n": len(rows)}
            print(f"PGD dataset: {pgd_stats['n']} examples, softmin "
                  f"{pgd_stats['softmin_before']:.4f} -> {pgd_stats['softmin_after']:.4f}, "
                  f"similarity {pgd_stats['sim_before']:.4f} -> {pgd_stats['sim_after']:.4f}")

    ev = Path(args.eval) if args.eval else RL_RUNS / "final_eval" / "final_eval.json"
    if ev.is_file():
        payload = json.loads(ev.read_text(encoding="utf8"))
        written.append(fig_final(payload, pgd_stats))
        written.append(fig_cost(payload))
    else:
        print(f"(no final eval at {ev} yet -- run eval_final.py for figures 10 and 11)")

    for w in written:
        print(f"wrote {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
