r"""
probe_amplify.py -- is the distilled edit simply too small? RUN IN "voiceclone".

    python probe_amplify.py --run distill_final

The distilled network predicts edits at |delta| ~ 0.08x PGD's magnitude while
recovering ~35% of PGD's objective gain. Two readings of that:

  * it found a genuinely different, cheaper direction, and scaling it up will
    overshoot and help nothing;
  * it found the right direction but regression-to-the-mean shrank the
    magnitude -- MSE against noisy targets always does this -- in which case a
    scalar multiplier recovers more gain for free.

Iterating is the other cheap axis. One forward pass is one step of an
amortised attack; re-computing the gradient hint on the edited mel and
predicting again is a second step. If the network has learned a descent
DIRECTION rather than a fixed displacement, iterating should compound.

Both are scanned here on stored mels from the held-out SENTENCES of the
distillation dataset, so nothing is regenerated and the whole sweep costs
minutes. Whatever wins is then confirmed on held-out VOICES with real
generation, which is the only claim that counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import diffscore
from env_config import RESULTS_ROOT
from eval_generalization import build_policy

DATA_ROOT = RESULTS_ROOT / "rl_runs" / "pgd_dataset"
RL_RUNS = RESULTS_ROOT / "rl_runs"


def apply_policy(net, mel, hift_call, device, k, w, scale=1.0, iters=1,
                 space="prob", cap=4.0):
    """Apply the distilled edit `iters` times, each scaled by `scale`.

    The gradient hint is recomputed on the current (already edited) mel at
    every step, which is what makes iteration meaningful rather than just
    multiplying the first prediction by `iters`.
    """
    cur = mel
    total = torch.zeros_like(mel)
    for _ in range(iters):
        hint = diffscore.mel_gradient_hint(cur, hift_call, device, k=k,
                                           softmin_weight=w, space=space, cap=cap)
        with torch.no_grad():
            d = scale * net.max_delta * torch.tanh(net(cur, hint))
        total = total + d
        cur = mel + total
    return cur, total


def score(mel, hift_call, device, k, w):
    with torch.no_grad(), diffscore.differentiable():
        wav, _ = hift_call(mel)
        sc = diffscore.detector_scores(wav.squeeze(0), device)
        return (float(diffscore.detector_objective(sc, k=k, softmin_weight=w)),
                float(diffscore.soft_min(list(sc.values()), k=k)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--val-texts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=18)
    ap.add_argument("--scales", default="1,2,3,4,6,8,12")
    ap.add_argument("--iters", default="1,2,3")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--data", default=None,
                    help="dataset directory (defaults to the original one)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scales = [float(x) for x in args.scales.split(",")]
    iters = [int(x) for x in args.iters.split(",")]

    blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                      map_location="cpu", weights_only=False)
    net, kind, _ = build_policy(blob, device, args)
    net.eval()
    ck = blob.get("args", {})
    space = ck.get("objective", "prob")
    cap = ck.get("margin_cap", 4.0)
    print(f"{args.run}/{args.checkpoint}  iteration {blob.get('iteration')}  {kind}"
          f"  [{space} hint]")

    data_dir = Path(args.data) if args.data else DATA_ROOT
    items = []
    for f in sorted(data_dir.glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        items.append(d)
    max_ti = max(d["text_idx"] for d in items)
    val = [d for d in items if d["text_idx"] >= max_ti - args.val_texts + 1][:args.limit]
    print(f"scanning on {len(val)} held-out-sentence examples\n")

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)

    base_obj, base_sm, pgd_obj, pgd_sm = [], [], [], []
    mels = []
    for d in val:
        mel = d["mel"].float().unsqueeze(0).to(device)
        mels.append(mel)
        base_obj.append(d["obj_before"]); pgd_obj.append(d["obj_after"])
        base_sm.append(d["softmin_before"]); pgd_sm.append(d["softmin_after"])
    base_obj = np.array(base_obj); pgd_obj = np.array(pgd_obj)
    print(f"baseline softmin {np.mean(base_sm):.4f}   PGD {np.mean(pgd_sm):.4f}\n")

    w = 11
    print(f"{'scale':>7}" + "".join(f"{'iters=' + str(n):>{w + 3}}" for n in iters))
    print(f"{'':>7}" + "".join(f"{'softmin':>{w}}{'rec':>4}" for _ in iters))
    print("-" * (7 + len(iters) * (w + 3)))

    results = []
    best = None
    for s in scales:
        line = f"{s:>7.1f}"
        for n in iters:
            objs, sms, mags = [], [], []
            for mel, d in zip(mels, val):
                if net.spk_dim:
                    net.set_speaker(d["spk_emb"])
                edited, total = apply_policy(net, mel, hift_call, device,
                                             args.softmin_k, args.softmin_weight,
                                             scale=s, iters=n, space=space, cap=cap)
                o, sm = score(edited, hift_call, device,
                              args.softmin_k, args.softmin_weight)
                objs.append(o); sms.append(sm)
                mags.append(float(total.abs().mean()))
            objs = np.array(objs)
            rec = float(np.mean((objs - base_obj) / np.maximum(pgd_obj - base_obj, 1e-6)))
            sm_mean = float(np.mean(sms))
            line += f"{sm_mean:>{w}.4f}{rec * 100:>+4.0f}"
            results.append({"scale": s, "iters": n, "softmin": sm_mean,
                            "recovered": rec, "delta_abs": float(np.mean(mags))})
            if best is None or rec > best["recovered"]:
                best = results[-1]
        print(line, flush=True)

    print(f"\nbest: scale {best['scale']:.1f}, iters {best['iters']}  -> "
          f"softmin {best['softmin']:.4f}, {best['recovered'] * 100:+.1f}% recovered, "
          f"|delta| {best['delta_abs']:.4f}")
    print(f"(baseline softmin {np.mean(base_sm):.4f}, PGD {np.mean(pgd_sm):.4f}, "
          f"PGD |delta| ~0.161)")

    out = RL_RUNS / args.run / "amplify_scan.json"
    out.write_text(json.dumps({"run": args.run, "n_examples": len(val),
                               "baseline_softmin": float(np.mean(base_sm)),
                               "pgd_softmin": float(np.mean(pgd_sm)),
                               "results": results, "best": best}, indent=2),
                   encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
