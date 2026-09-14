r"""
collect_pgd.py -- build a supervised dataset of PGD solutions. RUN IN "voiceclone".

    python collect_pgd.py --max-hours 3.5

What this is for
----------------
Three RL sessions (~3,600 episodes, PPO and SAC) produced a policy that is
statistically indistinguishable from doing nothing, while PGD moves the same
objective from 0.085 to 0.666 in sixty gradient steps on the same mel. The
diagnosis is not action scale -- run B3h sat at |delta| 0.12-0.20, squarely in
PGD's solution range, and did no better; at that magnitude corr(|delta|,
softmin) was -0.554, i.e. large edits in the wrong direction actively hurt.
The problem is direction-finding in an 80xT space, which gradient descent
solves directly and sampling does not.

So: stop exploring, and copy the answer. For each (voice, text) pair this
script captures the real mel CosyVoice2 produces, runs PGD to convergence,
and stores (mel, gradient hint, delta*) so a network can be trained to
predict delta* by regression. That converts an exploration problem into a
supervised one.

The gradient hint is stored rather than recomputed at training time because
it costs a full backward pass through HiFT and all four detectors -- roughly
as expensive as a PGD step, which would dominate every training epoch.

Resumability: each example is a separate file keyed by voice and text index,
and existing files are skipped, so the run can be stopped and restarted
without losing work or repeating it.
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
import detectors as det
import diffscore
import Tester
from env_config import RESULTS_ROOT
from mel_policy import MelResidualPolicy, mel_edit_hook
from textbank import TRAIN_TEXTS, TRAIN_VOICES
from train_ppo_mel import load_voice

DATA_ROOT = RESULTS_ROOT / "rl_runs" / "pgd_dataset"


def pgd_solve(mel, hift_call, device, steps, max_delta, lr, k, softmin_weight,
              names=None):
    """Ascend the detector objective w.r.t. a bounded mel residual.

    The bound is structural rather than a projection: delta = max_delta *
    tanh(raw), so every iterate is feasible and Adam never has to be clipped
    back onto the constraint set.
    """
    raw = torch.zeros_like(mel, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    traj = []
    with diffscore.differentiable():
        for step in range(steps):
            opt.zero_grad()
            delta = max_delta * torch.tanh(raw)
            wav, _ = hift_call(mel + delta)
            sc = diffscore.detector_scores(wav.squeeze(0), device, names=names)
            obj = diffscore.detector_objective(sc, k=k, softmin_weight=softmin_weight)
            (-obj).backward()
            opt.step()
            traj.append(float(obj))
    with torch.no_grad():
        delta = max_delta * torch.tanh(raw)
    return delta.detach(), traj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--texts", type=int, default=None,
                    help="how many texts from TRAIN_TEXTS to use (default: all)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--breathify", action="store_true",
                    help="insert [breath]/[quick_breath] at clause boundaries, "
                         "building a dramatised training corpus")
    ap.add_argument("--detectors", default=None,
                    help="comma-separated subset to optimise against; the rest "
                         "are still scored but never influence the target "
                         "(leave-one-detector-out)")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--max-hours", type=float, default=3.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    all_names = list(det.MODEL_ORDER)
    opt_names = ([n.strip() for n in args.detectors.split(",")]
                 if args.detectors else all_names)
    held_names = [n for n in all_names if n not in opt_names]
    bad = [n for n in opt_names if n not in all_names]
    if bad:
        print(f"unknown detector(s): {bad}; expected {all_names}")
        return 1
    voices = args.voices or TRAIN_VOICES
    texts = TRAIN_TEXTS[:args.texts] if args.texts else TRAIN_TEXTS
    if args.breathify:
        import prosody
        texts = [prosody.breathify(t) for t in texts]
    out_dir = Path(args.out) if args.out else DATA_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "collect_log.jsonl"

    print(f"voices : {len(voices)}  {voices}")
    print(f"texts  : {len(texts)}")
    print(f"targets: {len(voices) * len(texts)} examples, PGD {args.steps} steps each")
    if args.breathify:
        print(f"style  : breathified -- e.g. {texts[0]!r}")
    print(f"budget : {args.max_hours} h")
    print(f"optimise against : {opt_names}")
    if held_names:
        print(f"HELD OUT         : {held_names}  (scored for reporting, never "
              f"influences the target or the hint)")
    print(flush=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    identity = MelResidualPolicy(max_delta=args.max_delta).to(device)   # zero-init: no-op

    # Text-major, not voice-major. Voice-major would spend the whole budget on
    # the first few voices if it runs out of time, leaving the rest with zero
    # examples -- fatal for a dataset whose entire purpose is generalisation
    # across voices. Round-robin means an early stop truncates the text axis
    # evenly instead, and every voice keeps the same coverage.
    order = [(ti, text, vi, voice)
             for ti, text in enumerate(texts)
             for vi, voice in enumerate(voices)]

    t0 = time.time()
    done = skipped = 0
    enrolled: set[str] = set()
    refs = {v: load_voice(v) for v in voices}

    for ti, text, vi, voice in order:
            ref, prompt = refs[voice]
            key = f"{voice}_t{ti:02d}"
            path = out_dir / f"{key}.pt"
            if path.is_file():
                skipped += 1
                continue
            if (time.time() - t0) / 3600.0 >= args.max_hours:
                print(f"\nhit the {args.max_hours} h budget; stopping cleanly")
                print(f"collected {done} this session, {skipped} already present")
                return 0

            if voice not in enrolled:
                cloners.enroll(ref, "CosyVoice2", prompt_text=prompt,
                               alias=voice, device=device)
                enrolled.add(voice)

            torch.manual_seed(args.seed + vi * 1000 + ti)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + vi * 1000 + ti)

            # capture the real mel by generating through a zero-init hook
            tmp = out_dir / f"_{key}_base.wav"
            with mel_edit_hook(cosy.model, identity, deterministic=True,
                               device=device) as rec:
                cloners.generate(text, ref, model="CosyVoice2",
                                 out_path=tmp, device=device)
            if not rec.mels:
                print(f"  {key}: no mel captured, skipping")
                tmp.unlink(missing_ok=True)
                continue
            mel = rec.mels[0].clone().to(device)

            hint = diffscore.mel_gradient_hint(mel, hift_call, device,
                                               k=args.softmin_k,
                                               softmin_weight=args.softmin_weight,
                                               names=opt_names)

            with torch.no_grad(), diffscore.differentiable():
                w0, _ = hift_call(mel)
                s0_all = diffscore.detector_scores(w0.squeeze(0), device)
                s0 = {n: s0_all[n] for n in opt_names}
                obj0 = float(diffscore.detector_objective(
                    s0, k=args.softmin_k, softmin_weight=args.softmin_weight))
                sm0 = float(diffscore.soft_min(list(s0.values()), k=args.softmin_k))

            delta, traj = pgd_solve(mel, hift_call, device, args.steps,
                                    args.max_delta, args.lr,
                                    args.softmin_k, args.softmin_weight,
                                    names=opt_names)

            with torch.no_grad(), diffscore.differentiable():
                w1, _ = hift_call(mel + delta)
                s1_all = diffscore.detector_scores(w1.squeeze(0), device)
                s1 = {n: s1_all[n] for n in opt_names}
                obj1 = float(diffscore.detector_objective(
                    s1, k=args.softmin_k, softmin_weight=args.softmin_weight))
                sm1 = float(diffscore.soft_min(list(s1.values()), k=args.softmin_k))

            # similarity of the PGD result, so bad targets can be filtered
            import soundfile as sf
            adv = out_dir / f"_{key}_adv.wav"
            sf.write(str(adv), w1.squeeze(0).squeeze(0).cpu().numpy(), cosy.sample_rate)
            ref_emb = Tester.speaker_embedding(ref)
            sim0 = float(np.dot(Tester.speaker_embedding(tmp), ref_emb))
            sim1 = float(np.dot(Tester.speaker_embedding(adv), ref_emb))
            tmp.unlink(missing_ok=True)
            adv.unlink(missing_ok=True)

            torch.save({"voice": voice, "text_idx": ti, "text": text,
                        "mel": mel.squeeze(0).half().cpu(),
                        "hint": hint.squeeze(0).half().cpu(),
                        "delta": delta.squeeze(0).half().cpu(),
                        "obj_before": obj0, "obj_after": obj1,
                        "detectors": opt_names, "held_out": held_names,
                        "per_det_before": {n: float(v) for n, v in s0_all.items()},
                        "per_det_after": {n: float(v) for n, v in s1_all.items()},
                        "softmin_before": sm0, "softmin_after": sm1,
                        "sim_before": sim0, "sim_after": sim1,
                        "delta_abs": float(delta.abs().mean()),
                        "steps": args.steps, "max_delta": args.max_delta},
                       path)

            done += 1
            el = time.time() - t0
            rate = el / max(done, 1)
            remaining = len(voices) * len(texts) - skipped - done
            with log_path.open("a", encoding="utf8") as f:
                f.write(json.dumps({
                    "key": key, "voice": voice, "text_idx": ti,
                    "softmin_before": sm0, "softmin_after": sm1,
                    "obj_before": obj0, "obj_after": obj1,
                    "per_det_before": {n: float(v) for n, v in s0_all.items()},
                    "per_det_after": {n: float(v) for n, v in s1_all.items()},
                    "sim_before": sim0, "sim_after": sim1,
                    "delta_abs": float(delta.abs().mean()),
                    "frames": int(mel.shape[-1]), "seconds": el}) + "\n")
            print(f"  [{done:>3}] {key:<20} softmin {sm0:.3f} -> {sm1:.3f}  "
                  f"sim {sim0:.3f} -> {sim1:.3f}  |d| {float(delta.abs().mean()):.3f}  "
                  f"({rate:.0f}s/ex, ~{remaining * rate / 60:.0f} min left)", flush=True)

    print(f"\ncomplete: {done} collected this session, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
