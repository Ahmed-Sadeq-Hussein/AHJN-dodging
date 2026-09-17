r"""
train_direct.py -- train the edit network on the OBJECTIVE, not on PGD's answer.
RUN IN "voiceclone", alone on the GPU.

    python train_direct.py --init distill_margin_spk --max-hours 1.5

Why this exists
---------------
Distillation minimises MSE between the network's edit and PGD's edit. That is
not the quantity anyone cares about, and the gap shows: on the margin dataset
a single fixed average edit recovers 34.5% of PGD's objective gain while the
trained network recovers 25% (section 7n). Initialising the network *at* that
fixed edit and learning only the residual changed nothing (24.6%, 24.9%) --
training pulls it back to the same place. MSE against noisy per-utterance
targets shrinks predictions toward their mean; the shrunken edit has lower MSE
and a worse objective.

The path mel -> HiFT -> detectors is differentiable end to end (that is what
makes PGD possible at all), so the network can simply be trained on the
objective itself:

    loss = -margin_objective(detectors(hift(mel + delta)))  +  beta * ||delta||^2

No targets, no imitation, no exploration -- the gradient of the true objective
flows straight into the network weights. Design A and B tried to reach this
through reward sampling and failed; distillation reached it through a proxy.
This is the direct route, and it was available the whole time.

The magnitude penalty is what keeps similarity and PESQ intact: section 7n
showed the quality cost tracks |delta| almost perfectly, so beta is the dial
between evasion and audio quality.

Starting from a distilled checkpoint rather than from scratch, because that
already encodes the shared component and costs nothing to reuse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import diffscore
import perceptual as P
from env_config import RESULTS_ROOT
from eval_generalization import build_policy
from sac_agent import GradInformedActor

RL = RESULTS_ROOT / "rl_runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RL / "pgd_dataset_margin"))
    ap.add_argument("--init", default="distill_margin_spk",
                    help="distilled run to start from ('none' to start fresh)")
    ap.add_argument("--init-ckpt", default="best.pt")
    ap.add_argument("--run-id", default="direct")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=2.0,
                    help="weight on the quality penalty; the evasion/quality dial")
    ap.add_argument("--penalty", default="l2",
                    choices=("l2", "mask", "mrstft", "mrstft+mask"),
                    help="l2 = mean(delta^2), the version that failed in 7o. "
                         "mask = delta^2 weighted toward quiet regions. "
                         "mrstft = multi-resolution STFT distance between the "
                         "edited and unedited waveform, i.e. measured on what "
                         "is actually heard")
    ap.add_argument("--mask-floor", type=float, default=0.05)
    ap.add_argument("--pen-target", type=float, default=None,
                    help="treat quality as a CONSTRAINT, not a price: pay "
                         "nothing while the penalty is below this, then "
                         "beta * (pen - target)^2. A soft penalty is always "
                         "outbid by the objective given enough steps -- beta=2 "
                         "held |delta| for 30 min and lost it over 3 hours")
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--dilations", default="1")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--val-texts", type=int, default=3)
    ap.add_argument("--margin-cap", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-limit", type=int, default=10)
    ap.add_argument("--max-hours", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items = [torch.load(f, map_location="cpu", weights_only=False)
             for f in sorted(Path(args.data).glob("*.pt"))]
    max_ti = max(d["text_idx"] for d in items)
    train = [d for d in items if d["text_idx"] < max_ti - args.val_texts + 1]
    val = [d for d in items if d["text_idx"] >= max_ti - args.val_texts + 1]
    # a fixed, evenly spread subset for the running eval. Taking the first N
    # gave a subset whose unedited soft-min was 0.72 against 0.31 across all
    # val mels -- the progress numbers would have tracked the easiest clips.
    val_idx = np.linspace(0, len(val) - 1, min(args.eval_limit, len(val))).astype(int)
    val_sample = [val[i] for i in dict.fromkeys(val_idx.tolist())]
    print(f"{len(train)} train / {len(val)} val mels from {args.data}"
          f"  (running eval on {len(val_sample)} spread across the val set)")

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)

    if args.init and args.init != "none":
        blob = torch.load(RL / args.init / args.init_ckpt, map_location="cpu",
                          weights_only=False)
        net, kind, targs = build_policy(blob, device, args)
        print(f"initialised from {args.init}/{args.init_ckpt} ({kind})")
    else:
        dil = tuple(int(x) for x in args.dilations.split(","))
        net = GradInformedActor(hidden=args.hidden, max_delta=0.6, dilations=dil,
                                use_grad_hint=True).to(device)
        targs = {"hidden": args.hidden, "dilations": list(dil), "max_delta": 0.6}
        rf = 1 + sum(4 * d for d in dil)
        print(f"initialised fresh: {sum(p.numel() for p in net.parameters()):,} "
              f"params, receptive field {rf} frames ({rf * 256 / 24000:.2f} s)")
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def score(mel, delta=None):
        """soft-min and the margin objective for a (possibly edited) mel."""
        m = mel if delta is None else mel + delta
        with torch.no_grad(), diffscore.differentiable():
            wav, _ = hift(m)
            sc = diffscore.detector_scores(wav.squeeze(0), device)
            mg = diffscore.detector_margins(wav.squeeze(0), device)
        return (float(diffscore.soft_min(list(sc.values()), k=8.0)),
                float(diffscore.margin_objective(mg, cap=args.margin_cap,
                                                 mode="softmin", k=1.0)))

    def evaluate(limit):
        net.eval()
        sms, mags = [], []
        for d in (val_sample if limit <= len(val_sample) else val[:limit]):
            mel = d["mel"].float().unsqueeze(0).to(device)
            if net.spk_dim:
                net.set_speaker(d["spk_emb"])
            hint = diffscore.mel_gradient_hint(mel, hift, device, space="margin",
                                               cap=args.margin_cap)
            with torch.no_grad():
                delta = net.max_delta * torch.tanh(net(mel, hint))
            sm, _ = score(mel, delta)
            sms.append(sm)
            mags.append(float(delta.abs().mean()))
        net.train()
        return float(np.mean(sms)), float(np.mean(mags))

    base_sm = float(np.mean([d["softmin_before"] for d in val_sample]))
    pgd_sm = float(np.mean([d["softmin_after"] for d in val_sample]))
    sm0, mag0 = evaluate(args.eval_limit)
    print(f"\nval soft-min: unedited {base_sm:.4f}   at init {sm0:.4f}   "
          f"PGD {pgd_sm:.4f}   |delta| {mag0:.4f}")
    print(f"loss = -margin_objective + {args.beta} * {args.penalty}\n", flush=True)

    run_dir = RL / f"direct_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2),
                                         encoding="utf8")
    log = run_dir / "train_log.jsonl"
    ck_args = {**targs, "objective": "margin", "margin_cap": args.margin_cap,
               "spk_dim": getattr(net, "spk_dim", 0),
               "profile_bias": getattr(net, "profile_bias", False)}

    best = sm0
    torch.save({"actor": net.state_dict(), "args": ck_args, "distilled": True,
                "iteration": 0, "val_softmin": sm0}, run_dir / "best.pt")

    t0 = time.time()
    step = 0
    while step < args.max_steps and (time.time() - t0) / 3600 < args.max_hours:
        idx = rng.choice(len(train), size=args.batch, replace=False)
        loss_terms = []
        opt.zero_grad()
        for i in idx:
            d = train[int(i)]
            mel = d["mel"].float().unsqueeze(0).to(device)
            T = mel.shape[-1]
            if T > args.window:
                s = int(rng.integers(0, T - args.window + 1))
                mel = mel[:, :, s:s + args.window]
            if net.spk_dim:
                net.set_speaker(d["spk_emb"])
            # the hint is an input, not part of the graph we differentiate
            hint = diffscore.mel_gradient_hint(mel, hift, device, space="margin",
                                               cap=args.margin_cap)
            # HiFT injects noise, so an unseeded clean/edited pair differs by
            # 0.025 peak and 0.059 of STFT distance on its own -- more than the
            # penalty being measured. Reseeding makes the two passes identical,
            # leaving the edit as the only difference.
            hseed = int(rng.integers(0, 2 ** 31 - 1))
            clean = None
            if "mrstft" in args.penalty:
                with torch.no_grad(), diffscore.differentiable():
                    torch.manual_seed(hseed)
                    clean, _ = hift(mel)
            with diffscore.differentiable():
                delta = net.max_delta * torch.tanh(net(mel, hint))
                torch.manual_seed(hseed)
                wav, _ = hift(mel + delta)
                obj = diffscore.margin_objective(
                    diffscore.detector_margins(wav.squeeze(0), device),
                    cap=args.margin_cap, mode="softmin", k=1.0)
                pen = P.penalty(args.penalty, delta, mel, edited_wav=wav,
                                clean_wav=clean, floor=args.mask_floor)
                if args.pen_target is None:
                    loss = -obj + args.beta * pen
                else:
                    loss = -obj + args.beta * torch.clamp(
                        pen - args.pen_target, min=0.0).pow(2)
                (loss / len(idx)).backward()
            loss_terms.append((float(obj), float(delta.abs().mean()), float(pen)))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1

        if step % args.eval_every == 0:
            sm, mag = evaluate(args.eval_limit)
            el = (time.time() - t0) / 60
            flag = ""
            if sm > best:
                best = sm
                torch.save({"actor": net.state_dict(), "args": ck_args,
                            "distilled": True, "iteration": step,
                            "val_softmin": sm}, run_dir / "best.pt")
                flag = "  <- best"
            print(f"step {step:>5}  obj {np.mean([t[0] for t in loss_terms]):>7.3f}  "
                  f"pen {np.mean([t[2] for t in loss_terms]):>7.4f}  "
                  f"|delta| {np.mean([t[1] for t in loss_terms]):.4f}  ||  "
                  f"val soft-min {sm:.4f}  |delta| {mag:.4f}  ({el:.0f} min){flag}",
                  flush=True)
            with log.open("a", encoding="utf8") as fh:
                fh.write(json.dumps({"step": step, "val_softmin": sm,
                                     "val_delta": mag, "minutes": el}) + chr(10))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sm, mag = evaluate(len(val))
    print(f"\nfinal val soft-min {sm:.4f} (best checkpoint {best:.4f}) "
          f"vs unedited {base_sm:.4f}, PGD {pgd_sm:.4f}; |delta| {mag:.4f}")
    torch.save({"actor": net.state_dict(), "args": ck_args, "distilled": True,
                "iteration": step, "val_softmin": sm}, run_dir / "final.pt")
    print(f"checkpoints in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
