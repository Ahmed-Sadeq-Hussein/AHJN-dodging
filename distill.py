# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
distill.py -- train the actor to predict PGD's edit by regression.
RUN IN "voiceclone", and NOT while collect_pgd.py is still running: two
CosyVoice2 processes on one 10 GB card thrash, and that has already cost this
project an entire training window.

    python distill.py --max-hours 2.0

The argument for doing it this way
----------------------------------
PPO (Design A) degraded the objective significantly. SAC with a gradient hint
(Design B) came out statistically indistinguishable from doing nothing across
396 iterations, and its entropy coefficient rose monotonically for nine hours
-- the signature of a critic that predicts reward well but cannot RANK two
candidate edits. Meanwhile PGD reaches softmin 0.666 in sixty steps on the
same mel.

The failure is therefore not the intervention point, not the action bound
(run B3h sat at |delta| 0.12-0.20, exactly PGD's magnitude, and did no
better), and not the constraints (they never bound: similarity 0.82 against a
0.70 floor, WER 5% against a 15% ceiling). It is that finding a good direction
in 80xT dimensions by sampling is hopeless, while gradient descent does it
directly.

So the target is supplied instead of discovered. The network never has to
explore; it only has to interpolate between solved instances.

What this run actually tests
----------------------------
Whether a single edit FUNCTION exists at all. PGD re-solves per utterance and
generalises to nothing by construction; the only thing a learned policy adds
is amortisation. Two outcomes, both worth having:

  * validation objective approaches PGD's  -> the function exists, and we have
    it; evasion becomes a forward pass instead of sixty gradient steps.
  * regression fits training data but not held-out -> the edit is
    utterance-specific, no such function exists, and every RL variant was
    doomed from the start. That closes the question rather than leaving it
    open for another nine-hour run.

Reported per epoch: MSE, the cosine between predicted and target edit
(direction agreement, which MSE hides), the magnitude ratio, and periodically
the real objective obtained by pushing the PREDICTED edit through HiFT and the
detectors -- reported as the fraction of PGD's gain recovered, which is the
number that decides the outcome above.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diffscore
from env_config import RESULTS_ROOT
from sac_agent import GradInformedActor

DATA_ROOT = RESULTS_ROOT / "rl_runs" / "pgd_dataset"
RUN_ROOT = RESULTS_ROOT / "rl_runs"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class PGDDataset:
    """(mel, hint, delta*) triples, cropped to a fixed window for batching."""

    def __init__(self, files, window: int):
        self.items = []
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            self.items.append(d)
        self.window = window

    def __len__(self):
        return len(self.items)

    def batch(self, idx, device, rng):
        mels, hints, deltas, masks, spks = [], [], [], [], []
        W = self.window
        for i in idx:
            d = self.items[i]
            mel, hint, dl = d["mel"].float(), d["hint"].float(), d["delta"].float()
            T = mel.shape[-1]
            if T >= W:
                s = rng.integers(0, T - W + 1)
                mel, hint, dl = mel[:, s:s + W], hint[:, s:s + W], dl[:, s:s + W]
                m = torch.ones(W)
            else:
                pad = W - T
                m = torch.cat([torch.ones(T), torch.zeros(pad)])
                mel = F.pad(mel, (0, pad))
                hint = F.pad(hint, (0, pad))
                dl = F.pad(dl, (0, pad))
            mels.append(mel); hints.append(hint); deltas.append(dl); masks.append(m)
            spks.append(d.get("spk_emb", torch.zeros(192)).float())
        return (torch.stack(mels).to(device), torch.stack(hints).to(device),
                torch.stack(deltas).to(device), torch.stack(masks).to(device),
                torch.stack(spks).to(device))


def masked_stats(pred, targ, mask):
    """MSE, direction cosine and magnitude ratio, all ignoring padding."""
    m = mask.unsqueeze(1)
    n = m.sum() * pred.shape[1]
    mse = (((pred - targ) ** 2) * m).sum() / n
    p = (pred * m).flatten(1)
    t = (targ * m).flatten(1)
    cos = F.cosine_similarity(p, t, dim=1).mean()
    ratio = (p.norm(dim=1) / (t.norm(dim=1) + 1e-8)).mean()
    return mse, cos, ratio


# --------------------------------------------------------------------------
# the metric that actually decides the question
# --------------------------------------------------------------------------

def _obj_value(wav24, device, names, k, w, space, cap):
    if space == "margin":
        return float(diffscore.margin_objective(
            diffscore.detector_margins(wav24, device, names=names), cap=cap))
    return float(diffscore.detector_objective(
        diffscore.detector_scores(wav24, device, names=names), k=k,
        softmin_weight=w))


def objective_eval(net, items, hift_call, device, k, w, limit=8, names=None,
                   space="prob", cap=4.0):
    """Push the PREDICTED edit through HiFT + detectors on held-out mels.

    Returns the mean recovered fraction of PGD's own gain:
        (obj(mel + delta_pred) - obj(mel)) / (obj(mel + delta_pgd) - obj(mel))
    1.0 means the network matched PGD without optimising; 0.0 means it
    achieved nothing. Uses the stored mels, so no CosyVoice2 generation is
    needed here.
    """
    net.eval()
    recs, absolutes = [], []
    with torch.no_grad():
        for d in items[:limit]:
            mel = d["mel"].float().unsqueeze(0).to(device)
            hint = d["hint"].float().unsqueeze(0).to(device)
            if net.spk_dim:
                net.set_speaker(d["spk_emb"])
            delta = net.max_delta * torch.tanh(net(mel, hint))
            with diffscore.differentiable():
                wav, _ = hift_call(mel + delta)
                sc = diffscore.detector_scores(wav.squeeze(0), device, names=names)
                obj = _obj_value(wav.squeeze(0), device, names, k, w, space, cap)
                sm = float(diffscore.soft_min(list(sc.values()), k=k))
            gain_pgd = d["obj_after"] - d["obj_before"]
            gain_net = obj - d["obj_before"]
            if abs(gain_pgd) > 1e-6:
                recs.append(gain_net / gain_pgd)
            absolutes.append({"softmin_base": d["softmin_before"],
                              "softmin_net": sm,
                              "softmin_pgd": d["softmin_after"]})
    net.train()
    return (float(np.mean(recs)) if recs else 0.0), absolutes


def constant_edit_eval(profile, items, hift_call, device, k, w, limit=8,
                       names=None):
    """Control: apply ONE fixed per-channel edit to every utterance.

    This separates two very different worlds. If the average of PGD's
    solutions, applied blind, recovers a decent share of the gain, then a
    universal component exists and any network only has to learn the
    residual on top of it. If it recovers nothing, PGD's edit is
    utterance-specific, and no input-independent function -- learned or
    otherwise -- can substitute for per-utterance optimisation.

    Cheap to run and it bounds the whole approach from below, so it is worth
    knowing before reading any learning curve.
    """
    recs = []
    with torch.no_grad():
        for d in items[:limit]:
            mel = d["mel"].float().unsqueeze(0).to(device)
            delta = profile.view(1, -1, 1).expand_as(mel)
            with diffscore.differentiable():
                wav, _ = hift_call(mel + delta)
                sc = diffscore.detector_scores(wav.squeeze(0), device, names=names)
                obj = float(diffscore.detector_objective(sc, k=k, softmin_weight=w))
            gain_pgd = d["obj_after"] - d["obj_before"]
            if abs(gain_pgd) > 1e-6:
                recs.append((obj - d["obj_before"]) / gain_pgd)
    return float(np.mean(recs)) if recs else 0.0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--run-id", default="distill1")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="gaussian noise added to the input mel during training")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop if the objective eval has not improved in N evals (0 = off)")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dilations", default="1,2,4,8,16",
                    help="comma-separated; '1,1,1' reproduces the Design B trunk")
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--val-texts", type=int, default=3,
                    help="highest N text indices held out for validation")
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="drop targets whose PGD objective gain is below this")
    ap.add_argument("--max-sim-cost", type=float, default=0.10,
                    help="drop targets that cost more similarity than this")
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-limit", type=int, default=8)
    ap.add_argument("--objective", choices=("prob", "margin"), default=None,
                    help="scoring space for the objective eval; defaults to "
                         "whatever the dataset was collected with")
    ap.add_argument("--margin-cap", type=float, default=4.0)
    ap.add_argument("--init-profile", action="store_true",
                    help="initialise the actor at the mean PGD edit and learn "
                         "only the residual; that fixed edit alone already "
                         "recovers 34.5%% on the margin dataset")
    ap.add_argument("--spk-cond", action="store_true",
                    help="condition the actor on the CAM++ speaker embedding "
                         "(FiLM). Recovery ranges 15-101%% by voice while the "
                         "network cannot currently tell voices apart")
    ap.add_argument("--match", default=None,
                    help="restrict to the target files another corpus has, "
                         "for a like-for-like control")
    ap.add_argument("--obj-weight", type=float, default=0.0,
                    help="add the real margin objective to the MSE loss "
                         "(0 = pure imitation, the original behaviour)")
    ap.add_argument("--obj-clips", type=int, default=2,
                    help="clips per batch pushed through HiFT for that term; "
                         "each one costs a full backward through the vocoder "
                         "and all four detectors, so keep it small")
    ap.add_argument("--max-wer-cost", type=float, default=0.05,
                    help="drop targets whose WER rose by more than this "
                         "(needs a dataset collected with --measure-wer)")
    ap.add_argument("--detectors", default=None,
                    help="subset the objective eval scores against; must match "
                         "what the targets were collected with")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dilations = tuple(int(x) for x in args.dilations.split(","))
    spk_dim = 192 if args.spk_cond else 0
    opt_names = ([n.strip() for n in args.detectors.split(",")]
                 if args.detectors else None)
    data_dir = Path(args.data) if args.data else DATA_ROOT
    files = sorted(data_dir.glob("*.pt"))
    if args.match:
        # Train the control on exactly the (voice, sentence) pairs another
        # corpus holds, so a comparison between two corpora is about how the
        # targets were made and not about how many there are.
        keep = {f.name for f in Path(args.match).glob("*.pt")}
        before = len(files)
        files = [f for f in files if f.name in keep]
        print(f"matched against {args.match}: {len(files)} of {before} kept")
        if not files:
            print("nothing matched -- are the two corpora named the same way?")
            return 1
    if not files:
        print(f"no dataset at {data_dir} -- run collect_pgd.py first")
        return 1

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"loading {len(files)} examples from {data_dir}", flush=True)
    ds = PGDDataset(files, args.window)

    # PGD is unconstrained: it maximises the detector objective and nothing
    # else. On a voice that already passes (ahmed baselines at soft-min 0.99)
    # there is no gain available, so it wanders -- one collected target spends
    # |delta| 0.17 to move the objective by nothing while dropping similarity
    # 0.824 -> 0.664. Regressing onto that teaches the network to make a large
    # damaging edit on exactly the inputs that needed no edit. Filter such
    # targets out rather than letting them define the identity case.
    keep, dropped = [], {"no_gain": 0, "sim_cost": 0, "wer_cost": 0}
    for d in ds.items:
        wer_cost = ((d["wer_after"] - d["wer_before"])
                    if d.get("wer_after") is not None
                    and d.get("wer_before") is not None else 0.0)
        if d["obj_after"] - d["obj_before"] < args.min_gain:
            dropped["no_gain"] += 1
        elif d["sim_before"] - d["sim_after"] > args.max_sim_cost:
            dropped["sim_cost"] += 1
        elif wer_cost > args.max_wer_cost:
            dropped["wer_cost"] += 1
        else:
            keep.append(d)
    print(f"kept {len(keep)}/{len(ds.items)} targets "
          f"(dropped {dropped['no_gain']} with gain < {args.min_gain}, "
          f"{dropped['sim_cost']} costing > {args.max_sim_cost} similarity, "
          f"{dropped['wer_cost']} costing > {args.max_wer_cost} WER)")
    if not keep:
        print("nothing usable left after filtering")
        return 1
    ds.items = keep

    # hold out the highest text indices: unseen SENTENCES, seen voices.
    # unseen voices are handled by the separate final evaluation.
    max_ti = max(d["text_idx"] for d in ds.items)
    val_cut = max_ti - args.val_texts + 1
    train_idx = [i for i, d in enumerate(ds.items) if d["text_idx"] < val_cut]
    val_idx = [i for i, d in enumerate(ds.items) if d["text_idx"] >= val_cut]
    val_items = [ds.items[i] for i in val_idx]
    print(f"train {len(train_idx)} examples / val {len(val_idx)} "
          f"(texts >= {val_cut} held out)")

    gains = np.array([d["obj_after"] - d["obj_before"] for d in ds.items])
    sms = np.array([d["softmin_after"] - d["softmin_before"] for d in ds.items])
    print(f"PGD targets: mean objective gain {gains.mean():+.4f}, "
          f"mean softmin gain {sms.mean():+.4f}")
    print(f"            |delta| mean {np.mean([d['delta_abs'] for d in ds.items]):.4f}")

    # the objective space must match how the targets were collected: a
    # margin-collected obj_before/obj_after cannot be compared against a
    # probability-space evaluation
    space = args.objective or ds.items[0].get("objective", "prob")
    cap = ds.items[0].get("margin_cap", args.margin_cap)
    print(f"objective space: {space}" + (f" (cap {cap})" if space == "margin" else "")
          + (f"   voice-conditioned (spk_dim {spk_dim})" if spk_dim else ""))

    net = GradInformedActor(hidden=args.hidden, max_delta=args.max_delta,
                            use_grad_hint=True, dilations=dilations,
                            spk_dim=spk_dim,
                            profile_bias=args.init_profile).to(device)
    rf = 1 + sum(4 * d for d in dilations)
    print(f"actor: {sum(p.numel() for p in net.parameters()):,} params, "
          f"receptive field {rf} frames ({rf * 256 / 24000:.2f} s)")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # HiFT is needed only for the objective evaluation, but that is the
    # metric the whole run is judged on, so it is not optional.
    import cloners
    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)

    # control, before any training: what does the AVERAGE PGD edit achieve?
    prof = torch.stack([d["delta"].float().mean(dim=1)
                        for d in (ds.items[i] for i in train_idx)]).mean(0).to(device)
    const_rec = constant_edit_eval(prof, val_items, hift_call, device,
                                   args.softmin_k, args.softmin_weight,
                                   args.eval_limit, names=opt_names)
    print(f"control -- one fixed average edit applied to every utterance "
          f"recovers {const_rec * 100:+.1f}% of PGD's gain "
          f"(|profile| mean {float(prof.abs().mean()):.4f})", flush=True)
    if args.init_profile:
        net.set_profile(prof)
        print("actor initialised at that edit; it now learns the residual",
              flush=True)
    print(flush=True)

    run_dir = RUN_ROOT / f"distill_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")
    log_path = run_dir / "train_log.jsonl"

    ckpt_args = {"detectors": opt_names, "spk_dim": spk_dim,
                 "profile_bias": bool(args.init_profile),
                 "objective": space, "margin_cap": cap,
                 "max_delta": args.max_delta, "hidden": args.hidden,
                 "dilations": list(dilations), "no_grad_hint": False,
                 "init_log_std": -4.5, "min_log_std": -6.0,
                 "voices": sorted({d["voice"] for d in ds.items}),
                 "window": args.window, "run_id": args.run_id}

    def save(name, epoch, extra=None):
        blob = {"actor": net.state_dict(), "args": ckpt_args,
                "iteration": epoch, "distilled": True}
        if extra:
            blob.update(extra)
        torch.save(blob, run_dir / name)

    print(f"\ntraining, budget {args.max_hours} h\n", flush=True)
    t0 = time.time()
    best_rec = -1e9
    stale = 0
    steps_per_epoch = max(1, len(train_idx) // args.batch)

    for epoch in range(args.epochs):
        if (time.time() - t0) / 3600.0 >= args.max_hours:
            print(f"\nhit the {args.max_hours} h budget")
            break

        perm = rng.permutation(train_idx)
        tot = {"mse": 0.0, "cos": 0.0, "ratio": 0.0}
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            if len(idx) == 0:
                continue
            mel, hint, targ, mask, spk = ds.batch(idx, device, rng)
            if args.input_noise > 0:
                mel = mel + args.input_noise * torch.randn_like(mel)
            pred = net.max_delta * torch.tanh(net(mel, hint, spk))
            mse, cos, ratio = masked_stats(pred, targ, mask)
            loss = mse
            if args.obj_weight > 0:
                # MSE against noisy per-utterance targets shrinks predictions
                # toward their mean: a smaller edit has lower MSE and a WORSE
                # objective, which is why the magnitude ratio settles near
                # 0.08 and the fixed-average-edit control beat the trained
                # network (34.5% against 25%). Adding the real objective makes
                # that shrinkage cost something. MSE stays for stability --
                # pure objective training converges lower (0.40 against
                # 0.465), so this is meant to be a hybrid, not a replacement.
                k = min(args.obj_clips, mel.shape[0])
                obj = 0.0
                for b in range(k):
                    frames = int(mask[b].sum().item())
                    if frames < 8:
                        continue
                    m1 = (mel[b:b + 1, :, :frames] + pred[b:b + 1, :, :frames])
                    with diffscore.differentiable():
                        wav, _ = hift_call(m1)
                        mar = diffscore.detector_margins(wav.squeeze(0), device,
                                                         max_windows=2)
                        obj = obj + diffscore.margin_objective(
                            mar, cap=args.margin_cap, mode="softmin", k=1.0)
                if k:
                    loss = loss - args.obj_weight * (obj / k)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot["mse"] += float(mse); tot["cos"] += float(cos); tot["ratio"] += float(ratio)
        for k_ in tot:
            tot[k_] /= steps_per_epoch

        with torch.no_grad():
            vidx = np.array(val_idx)
            mel, hint, targ, mask, spk = ds.batch(vidx, device, rng)
            pred = net.max_delta * torch.tanh(net(mel, hint, spk))
            vmse, vcos, vratio = masked_stats(pred, targ, mask)

        rec = None
        if epoch % args.eval_every == 0:
            rec, absolutes = objective_eval(net, val_items, hift_call, device,
                                            args.softmin_k, args.softmin_weight,
                                            args.eval_limit, names=opt_names,
                                            space=space, cap=cap)
            sm_net = float(np.mean([a["softmin_net"] for a in absolutes]))
            sm_base = float(np.mean([a["softmin_base"] for a in absolutes]))
            sm_pgd = float(np.mean([a["softmin_pgd"] for a in absolutes]))
            print(f"epoch {epoch:>4}  train mse {tot['mse']:.5f} cos {tot['cos']:+.3f} "
                  f"| val mse {float(vmse):.5f} cos {float(vcos):+.3f} "
                  f"mag {float(vratio):.2f} | softmin base {sm_base:.3f} "
                  f"net {sm_net:.3f} pgd {sm_pgd:.3f}  recovered {rec * 100:+.1f}%",
                  flush=True)
            if rec > best_rec:
                best_rec = rec
                stale = 0
                save("best.pt", epoch, {"recovered": rec, "val_cos": float(vcos)})
            else:
                stale += 1
                if args.patience and stale >= args.patience:
                    print(f"\nno improvement in {stale} evaluations; stopping early")
                    break
        elif epoch % 5 == 0:
            print(f"epoch {epoch:>4}  train mse {tot['mse']:.5f} cos {tot['cos']:+.3f} "
                  f"| val mse {float(vmse):.5f} cos {float(vcos):+.3f} "
                  f"mag {float(vratio):.2f}", flush=True)

        with log_path.open("a", encoding="utf8") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_mse": tot["mse"], "train_cos": tot["cos"],
                "train_ratio": tot["ratio"], "val_mse": float(vmse),
                "val_cos": float(vcos), "val_ratio": float(vratio),
                "recovered": rec, "seconds": time.time() - t0}) + "\n")
        save("latest.pt", epoch)

    rec, absolutes = objective_eval(net, val_items, hift_call, device,
                                    args.softmin_k, args.softmin_weight,
                                    len(val_items), names=opt_names,
                                    space=space, cap=cap)
    print("\n" + "=" * 78)
    print("FINAL, on held-out SENTENCES (voices were seen during distillation)")
    print("=" * 78)
    sm_base = float(np.mean([a["softmin_base"] for a in absolutes]))
    sm_net = float(np.mean([a["softmin_net"] for a in absolutes]))
    sm_pgd = float(np.mean([a["softmin_pgd"] for a in absolutes]))
    print(f"  softmin  baseline {sm_base:.4f}   distilled {sm_net:.4f}   PGD {sm_pgd:.4f}")
    print(f"  fraction of PGD's gain recovered: {rec * 100:+.1f}%")
    print(f"  best during training:             {best_rec * 100:+.1f}%")
    save("final.pt", -1, {"recovered": rec})
    print(f"\ncheckpoints in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
