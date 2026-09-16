r"""
probe_objective.py -- which objective maximises SOFT-MIN? RUN IN "voiceclone".

    python probe_objective.py --hard 6 --median 6

Section 7l found that after the edit AASIST and AASIST-L sit at ~1.00 on
nearly every voice while 13 of 15 are blocked by RawNet2 or RawGAT-ST. The
objective every target was collected with (0.5 soft-min + 0.5 mean) still
sends about a quarter of its gradient to those saturated detectors, and PGD
drives AASIST to 0.9997 -- budget spent where soft-min cannot benefit.

This runs PGD with each candidate objective on the same stored mels and
reports what actually matters: the resulting SOFT-MIN, plus similarity and
PESQ, since an objective that wins by spending a much larger edit is not a
win. The teacher sets the ceiling, so this has to be settled before any
re-collection or retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import diffscore
import quality
import Tester
from env_config import RESULTS_ROOT
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"

MODES = [
    ("blend (current)", dict(mode="blend", k=8.0, softmin_weight=0.5)),
    ("softmin k=8", dict(mode="softmin", k=8.0)),
    ("softmin k=32", dict(mode="softmin", k=32.0)),
    ("capped tau=0.90", dict(mode="capped", tau=0.90)),
    ("capped tau=0.70", dict(mode="capped", tau=0.70)),
    ("hinge tau=0.90", dict(mode="hinge", tau=0.90)),
    # margin-space: the probability is saturated on exactly the clips that
    # matter, so these optimise the pre-softmax logit difference instead
    ("MARGIN capped +4", dict(space="margin", mode="capped", cap=4.0)),
    ("MARGIN capped +8", dict(space="margin", mode="capped", cap=8.0)),
    ("MARGIN softmin", dict(space="margin", mode="softmin", k=1.0)),
]


def pgd(mel, hift, device, steps, max_delta, lr, obj_kw):
    raw = torch.zeros_like(mel, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    with diffscore.differentiable():
        for _ in range(steps):
            opt.zero_grad()
            delta = max_delta * torch.tanh(raw)
            wav, _ = hift(mel + delta)
            kw = dict(obj_kw)
            if kw.pop("space", None) == "margin":
                obj = diffscore.margin_objective(
                    diffscore.detector_margins(wav.squeeze(0), device), **kw)
            else:
                obj = diffscore.detector_objective_v2(
                    diffscore.detector_scores(wav.squeeze(0), device), **kw)
            (-obj).backward()
            opt.step()
    with torch.no_grad():
        return (max_delta * torch.tanh(raw)).detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", type=int, default=6)
    ap.add_argument("--median", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--modes", default=None,
                    help="comma-separated substrings; run only objectives whose "
                         "label contains one of them")
    args = ap.parse_args()
    modes = MODES
    if args.modes:
        keys = [k.strip().lower() for k in args.modes.split(",")]
        modes = [m for m in MODES if any(k in m[0].lower() for k in keys)]
    print("objectives:", [m[0] for m in modes], flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = [torch.load(f, map_location="cpu", weights_only=False)
             for f in sorted((RL_RUNS / "pgd_dataset").glob("*.pt"))]
    items.sort(key=lambda d: d["softmin_before"])
    mid = len(items) // 2
    sel = ([("hard", d) for d in items[:args.hard]]
           + [("median", d) for d in items[mid:mid + args.median]])

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    names = None
    tmp = RL_RUNS / "_obj_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    def measure(mel, ref):
        with torch.no_grad(), diffscore.differentiable():
            w, _ = hift(mel)
            sc = diffscore.detector_scores(w.squeeze(0), device)
        p = tmp / "x.wav"
        sf.write(str(p), w.squeeze(0).squeeze(0).cpu().numpy(), cosy.sample_rate)
        q = quality.measure(p, ref, device)
        vals = list(sc.values())
        return {"softmin": float(diffscore.soft_min(vals, k=8.0)),
                "meandet": float(torch.stack(vals).mean()),
                "det": {n: float(v) for n, v in sc.items()},
                "sim": float(np.dot(Tester.speaker_embedding(p),
                                    Tester.speaker_embedding(ref))),
                "pesq": q["pesq"]}

    rows = []
    for i, (group, d) in enumerate(sel):
        mel = d["mel"].float().unsqueeze(0).to(device)
        ref, _ = load_voice(d["voice"])
        base = measure(mel, ref)
        names = names or list(base["det"])
        for label, kw in modes:
            delta = pgd(mel, hift, device, args.steps, args.max_delta, args.lr, kw)
            m = measure(mel + delta, ref)
            rows.append({"group": group, "voice": d["voice"],
                         "key": f"{d['voice']}_t{d['text_idx']:02d}",
                         "objective": label, "base": base, "after": m,
                         "delta_abs": float(delta.abs().mean())})
            # two earlier sweeps were killed mid-run with no traceback; append
            # each result so a dead process still leaves usable data behind
            with (RL_RUNS / "objective_sweep_partial.jsonl").open("a", encoding="utf8") as fh:
                fh.write(json.dumps(rows[-1]) + chr(10))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"  [{i + 1}/{len(sel)}] {group:<6} {d['voice']}_t{d['text_idx']:02d} "
              f"base soft-min {base['softmin']:.3f}", flush=True)

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()

    w = 11
    print("\n" + "=" * 118)
    print(f"{'group':<8}{'objective':<18}{'softmin':>9}{'vs blend':>10}{'meandet':>9}"
          f"{'|delta|':>9}{'sim':>8}{'PESQ':>8}   " + "  ".join(f"{n[:9]:>9}" for n in names))
    print("-" * 118)
    summary = []
    for group in ("hard", "median", "all"):
        ref_sm = np.mean([r["after"]["softmin"] for r in rows
                          if r["objective"] == "blend (current)"
                          and (group == "all" or r["group"] == group)])
        for label, _ in modes:
            rs = [r for r in rows if r["objective"] == label
                  and (group == "all" or r["group"] == group)]
            if not rs:
                continue
            rec = {"group": group, "objective": label, "n": len(rs),
                   "softmin": float(np.mean([r["after"]["softmin"] for r in rs])),
                   "meandet": float(np.mean([r["after"]["meandet"] for r in rs])),
                   "delta_abs": float(np.mean([r["delta_abs"] for r in rs])),
                   "sim": float(np.mean([r["after"]["sim"] for r in rs])),
                   "pesq": float(np.mean([r["after"]["pesq"] for r in rs])),
                   "per_det": {n: float(np.mean([r["after"]["det"][n] for r in rs]))
                               for n in names}}
            summary.append(rec)
            print(f"{group:<8}{label:<18}{rec['softmin']:>9.4f}"
                  f"{rec['softmin'] - ref_sm:>+10.4f}{rec['meandet']:>9.4f}"
                  f"{rec['delta_abs']:>9.3f}{rec['sim']:>8.3f}{rec['pesq']:>8.2f}   "
                  + "  ".join(f"{rec['per_det'][n]:>9.3f}" for n in names))
        print("-" * 118)

    best = max((r for r in summary if r["group"] == "all"), key=lambda r: r["softmin"])
    cur = next(r for r in summary if r["group"] == "all"
               and r["objective"] == "blend (current)")
    print(f"best overall: {best['objective']}  soft-min {best['softmin']:.4f} vs "
          f"{cur['softmin']:.4f} ({best['softmin'] - cur['softmin']:+.4f}), "
          f"sim {best['sim']:.3f} vs {cur['sim']:.3f}, "
          f"PESQ {best['pesq']:.2f} vs {cur['pesq']:.2f}")

    out = Path(args.out) if args.out else RL_RUNS / "objective_sweep.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
