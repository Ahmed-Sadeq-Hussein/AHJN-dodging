# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
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
import perceptual as P
import Tester
from env_config import RESULTS_ROOT
from mel_policy import MelResidualPolicy, mel_edit_hook
from textbank import TRAIN_TEXTS, TRAIN_VOICES
from train_ppo_mel import load_voice

DATA_ROOT = RESULTS_ROOT / "rl_runs" / "pgd_dataset"


def objective_value(wav24, device, space, names, k, softmin_weight, cap,
                    margin_mode="softmin", margin_k=1.0):
    """The quantity PGD ascends, in probability or logit-margin space."""
    if space == "margin":
        return diffscore.margin_objective(
            diffscore.detector_margins(wav24, device, names=names), cap=cap,
            mode=margin_mode, k=margin_k)
    return diffscore.detector_objective(
        diffscore.detector_scores(wav24, device, names=names),
        k=k, softmin_weight=softmin_weight)


def pgd_solve(mel, hift_call, device, steps, max_delta, lr, k, softmin_weight,
              names=None, space="prob", cap=4.0, margin_mode="softmin",
              margin_k=1.0, penalty_kind=None, pen_target=0.35, beta=50.0):
    """Ascend the detector objective w.r.t. a bounded mel residual.

    The bound is structural rather than a projection: delta = max_delta *
    tanh(raw), so every iterate is feasible and Adam never has to be clipped
    back onto the constraint set.

    space="margin" ascends capped logit margins instead of probabilities --
    on hard clips the blocking detectors are saturated at P ~ 1e-6 and the
    probability gradient has effectively vanished (see diffscore).

    penalty_kind turns on a PERCEPTUAL CONSTRAINT on the teacher itself
    (perceptual.py: "mask", "mrstft", "mrstft+mask"). Without it PGD maximises
    evasion and nothing else, and section 7j measured what that costs: the
    teacher's own solutions run -0.64 PESQ, already outside the -0.30 budget
    the students are held to. Targets were filtered afterwards for similarity
    and WER but never for audio quality, so the corpus taught the student to
    make expensive edits and the filter never caught it.

    It is a hinge, not a price: nothing is charged until the penalty passes
    pen_target, then steeply. Section 7q is the reason -- a weighted price on
    edit size held for thirty minutes and was then outbid, |delta| drifting
    0.023 -> 0.19 over three hours, while the same run with a hinge held at
    0.028 throughout.

    HiFT injects noise, so the clean pass is rendered under the SAME seed as
    the edited one. Two separate calls differ by 0.059 of STFT distance on
    their own, which is larger than the penalty being measured.
    """
    raw = torch.zeros_like(mel, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    traj, pens = [], []
    with diffscore.differentiable():
        for step in range(steps):
            opt.zero_grad()
            delta = max_delta * torch.tanh(raw)
            seed = torch.randint(0, 2 ** 31 - 1, (1,)).item()
            torch.manual_seed(seed)
            wav, _ = hift_call(mel + delta)
            obj = objective_value(wav.squeeze(0), device, space, names,
                                  k, softmin_weight, cap, margin_mode, margin_k)
            loss = -obj
            if penalty_kind:
                with torch.no_grad():
                    torch.manual_seed(seed)
                    clean, _ = hift_call(mel)
                pen = P.penalty(penalty_kind, delta, mel,
                                edited_wav=wav, clean_wav=clean)
                loss = loss + beta * torch.clamp(pen - pen_target, min=0.0).pow(2)
                pens.append(float(pen))
            loss.backward()
            opt.step()
            traj.append(float(obj))
    with torch.no_grad():
        delta = max_delta * torch.tanh(raw)
    return delta.detach(), traj, (pens[-1] if pens else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--texts", type=int, default=None,
                    help="how many texts from TRAIN_TEXTS to use (default: all)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--warm-start", default=None,
                    help="distill run whose operator pre-edits the mel before "
                         "PGD solves (DAgger: collect on the states the "
                         "operator actually reaches)")
    ap.add_argument("--warm-ckpt", default="best.pt")
    ap.add_argument("--warm-iters", default="2,4,8",
                    help="comma-separated pass counts, cycled across examples")
    ap.add_argument("--penalty", default=None,
                    choices=(None, "mask", "mrstft", "mrstft+mask"),
                    help="constrain the TEACHER perceptually while it solves")
    ap.add_argument("--pen-target", type=float, default=0.35,
                    help="the hinge: nothing is charged below this")
    ap.add_argument("--beta", type=float, default=50.0)
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
    ap.add_argument("--objective", choices=("prob", "margin"), default="prob",
                    help="prob: 0.5 soft-min + 0.5 mean of P(human), the space "
                         "every earlier dataset used. margin: capped logit "
                         "margins, which still carry gradient where a detector "
                         "is saturated")
    ap.add_argument("--margin-cap", type=float, default=4.0)
    ap.add_argument("--margin-mode", choices=("softmin", "capped"),
                    default="softmin",
                    help="softmin drives the WORST margin up and reached "
                         "soft-min 0.9998 on median clips; capped stops each "
                         "detector at the cap")
    ap.add_argument("--measure-wer", action="store_true",
                    help="transcribe the unedited and edited audio so targets "
                         "that hurt intelligibility can be filtered at training")
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
    if voices == ["all"]:
        from env_config import VOICE_DATA_ROOT
        voices = sorted(p.name for p in VOICE_DATA_ROOT.iterdir()
                        if p.is_dir() and (p / "reference.wav").is_file())
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
    print(f"objective: {args.objective}"
          + (f" (cap {args.margin_cap})" if args.objective == "margin" else "")
          + (f"   perceptual constraint: {args.penalty} "
             f"(hinge at {args.pen_target}, beta {args.beta})"
             if args.penalty else "   perceptual constraint: none"))
    print(f"optimise against : {opt_names}")
    if held_names:
        print(f"HELD OUT         : {held_names}  (scored for reporting, never "
              f"influences the target or the hint)")
    print(flush=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    identity = MelResidualPolicy(max_delta=args.max_delta).to(device)   # zero-init: no-op

    WARM_SCHEDULE = [int(x) for x in str(args.warm_iters).split(",") if x.strip()]
    warm_net = None
    if args.warm_start:
        from eval_final import AmplifiedAdapter
        from eval_generalization import build_policy
        ck = RESULTS_ROOT / "rl_runs" / args.warm_start / args.warm_ckpt
        if not ck.is_file():
            print(f"no warm-start checkpoint at {ck}")
            return 1
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        net, kind, _ = build_policy(blob, device, None)
        net.eval()
        targs = blob.get("args", {})
        warm_net = AmplifiedAdapter(net, hift_call, device, args.softmin_k,
                                    args.softmin_weight, scale=1.0, iters=1,
                                    names=targs.get("detectors"),
                                    space=targs.get("objective", "prob"),
                                    cap=targs.get("margin_cap", 4.0))
        print(f"warm start: {args.warm_start}/{args.warm_ckpt} ({kind}), "
              f"passes cycled over {WARM_SCHEDULE}")

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
            key = f"{voice}_t{ti:02d}" + (f"_w{args.warm_iters}" if args.warm_start else "")
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

            # DAgger: move the state to where the OPERATOR actually puts it.
            # Every target so far was solved on an unedited mel, but the
            # operator is applied 10-12 times at inference, so passes 2
            # onward run on mels no target ever covered -- textbook covariate
            # shift. Warm-starting applies the trained student first and lets
            # PGD solve the correction from there, so the corpus contains the
            # states the student will really meet.
            warm_k = 0
            if warm_net is not None:
                warm_k = WARM_SCHEDULE[(vi * len(texts) + ti) % len(WARM_SCHEDULE)]
                if warm_k:
                    warm_net.iters = warm_k
                    # NOT under no_grad: mel_gradient_hint runs its own
                    # backward pass internally, so suppressing autograd here
                    # makes it fail with "does not require grad". Detach the
                    # result instead -- PGD below starts from a fresh leaf.
                    d_warm, _, _ = warm_net.sample(mel, deterministic=True)
                    mel = (mel + d_warm).detach()

            hint = diffscore.mel_gradient_hint(mel, hift_call, device,
                                               k=args.softmin_k,
                                               softmin_weight=args.softmin_weight,
                                               names=opt_names,
                                               space=args.objective,
                                               cap=args.margin_cap,
                                               margin_mode=args.margin_mode)

            with torch.no_grad(), diffscore.differentiable():
                w0, _ = hift_call(mel)
                s0_all = diffscore.detector_scores(w0.squeeze(0), device)
                m0_all = diffscore.detector_margins(w0.squeeze(0), device)
                s0 = {n: s0_all[n] for n in opt_names}
                obj0 = float(objective_value(w0.squeeze(0), device, args.objective,
                                             opt_names, args.softmin_k,
                                             args.softmin_weight, args.margin_cap,
                                             args.margin_mode))
                sm0 = float(diffscore.soft_min(list(s0.values()), k=args.softmin_k))

            delta, traj, pen_final = pgd_solve(mel, hift_call, device, args.steps,
                                    args.max_delta, args.lr,
                                    args.softmin_k, args.softmin_weight,
                                    names=opt_names, space=args.objective,
                                    cap=args.margin_cap,
                                    margin_mode=args.margin_mode,
                                    penalty_kind=args.penalty,
                                    pen_target=args.pen_target, beta=args.beta)

            with torch.no_grad(), diffscore.differentiable():
                w1, _ = hift_call(mel + delta)
                s1_all = diffscore.detector_scores(w1.squeeze(0), device)
                m1_all = diffscore.detector_margins(w1.squeeze(0), device)
                s1 = {n: s1_all[n] for n in opt_names}
                obj1 = float(objective_value(w1.squeeze(0), device, args.objective,
                                             opt_names, args.softmin_k,
                                             args.softmin_weight, args.margin_cap,
                                             args.margin_mode))
                sm1 = float(diffscore.soft_min(list(s1.values()), k=args.softmin_k))

            # similarity of the PGD result, so bad targets can be filtered
            import soundfile as sf
            adv = out_dir / f"_{key}_adv.wav"
            sf.write(str(adv), w1.squeeze(0).squeeze(0).cpu().numpy(), cosy.sample_rate)
            ref_emb = Tester.speaker_embedding(ref)
            sim0 = float(np.dot(Tester.speaker_embedding(tmp), ref_emb))
            sim1 = float(np.dot(Tester.speaker_embedding(adv), ref_emb))

            # WER is the third target the brief asks for, and until now it was
            # only ever checked at evaluation. Measuring it per target lets
            # distillation drop edits that hurt intelligibility instead of
            # discovering the damage afterwards.
            wer0 = wer1 = None
            if args.measure_wer:
                spoken = cloners.strip_markup(text)
                wer0 = Tester.wer(spoken, Tester.transcribe(tmp, "base", device))[0]
                wer1 = Tester.wer(spoken, Tester.transcribe(adv, "base", device))[0]

            tmp.unlink(missing_ok=True)
            adv.unlink(missing_ok=True)

            torch.save({"voice": voice, "text_idx": ti, "text": text,
                        "mel": mel.squeeze(0).half().cpu(),
                        "hint": hint.squeeze(0).half().cpu(),
                        "delta": delta.squeeze(0).half().cpu(),
                        "spk_emb": torch.tensor(ref_emb, dtype=torch.float32),
                        "objective": args.objective, "margin_cap": args.margin_cap,
                        "margin_mode": args.margin_mode,
                        "wer_before": wer0, "wer_after": wer1,
                        "obj_before": obj0, "obj_after": obj1,
                        "detectors": opt_names, "held_out": held_names,
                        "per_det_before": {n: float(v) for n, v in s0_all.items()},
                        "per_det_after": {n: float(v) for n, v in s1_all.items()},
                        "margin_before": {n: float(v) for n, v in m0_all.items()},
                        "margin_after": {n: float(v) for n, v in m1_all.items()},
                        "softmin_before": sm0, "softmin_after": sm1,
                        "sim_before": sim0, "sim_after": sim1,
                        "delta_abs": float(delta.abs().mean()),
                        "penalty_kind": args.penalty, "pen_final": pen_final,
                        "pen_target": args.pen_target, "warm_k": warm_k,
                        "warm_start": args.warm_start,
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
                    "wer_before": wer0, "wer_after": wer1,
                    "objective": args.objective,
                    "delta_abs": float(delta.abs().mean()),
                    "penalty_kind": args.penalty, "pen_final": pen_final,
                    "frames": int(mel.shape[-1]), "seconds": el}) + "\n")
            print(f"  [{done:>3}] {key:<20} softmin {sm0:.3f} -> {sm1:.3f}  "
                  f"sim {sim0:.3f} -> {sim1:.3f}  |d| {float(delta.abs().mean()):.3f}  "
                  f"({rate:.0f}s/ex, ~{remaining * rate / 60:.0f} min left)", flush=True)

    print(f"\ncomplete: {done} collected this session, {skipped} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
