r"""
longform.py -- can the attack win on 70-second audio? RUN IN "voiceclone".

    python longform.py --voices evelynn thresh riven --steps 40

The problem (Plan_X section 7e)
-------------------------------
The learned edit gains +0.26 soft-min at 13.6 s and nothing at ~72 s. The
cause is an aggregation mismatch, not a capability limit: CosyVoice2 splits
long text into chunks, so `mel_edit_hook` fires once per chunk on a ~20 s mel,
while the detectors window the *concatenated* file into ~4 s slices and
average ~19 of them. Each chunk is optimised against its own local score; the
file is judged on a global average the optimiser never sees.

The naive fix -- put every chunk in one graph and differentiate the file
score -- exhausts a 10 GB card (section 7e records the OOM at four passes on
72 s).

What this does instead
----------------------
soft-min over detectors is a smooth function of four per-detector means, and
each mean decomposes over windows. So:

  1. no-grad pass over the WHOLE file: render every chunk, concatenate, and
     compute each detector's mean margin. From those four numbers take
     w_i = d(soft-min)/d(margin_i), the softmax weights.
  2. per chunk, with grad: maximise sum_i w_i * margin_i(chunk) -- the local
     contribution to the GLOBAL objective, weighted by how much the file's
     worst detector currently cares about each one.

Only one chunk is ever in the graph, so memory is bounded by chunk length
rather than file length, while the direction being optimised is the file's.
The weights are refreshed every round, so this is a fixed-point iteration on
the global soft-min rather than a one-off linearisation.

This measures the TEACHER's reach at length. If PGD cannot win here, no
learned operator will either, and long form is a ceiling rather than a
training problem.
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
import detectors as det
import diffscore
import perceptual as P
import quality
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from mel_policy import MelResidualPolicy, mel_edit_hook
from textbank import EVAL_TEXTS
from train_ppo_mel import load_voice

RL = RESULTS_ROOT / "rl_runs"
LENGTHS = {"short": " ".join(EVAL_TEXTS[:1]),
           "medium": " ".join(EVAL_TEXTS[:3]),
           "long": " ".join(EVAL_TEXTS * 2)}


def global_weights(mels, deltas, hift, device, k=1.0):
    """Per-detector softmin weights from the WHOLE concatenated file."""
    with torch.no_grad(), diffscore.differentiable():
        wavs = [hift(m + d)[0].squeeze(0).squeeze(0) for m, d in zip(mels, deltas)]
        full = torch.cat(wavs, dim=-1)
        margins = diffscore.detector_margins(full, device)
        names = list(margins)
        vals = torch.stack([margins[n] for n in names])
        w = torch.softmax(-k * vals, dim=0)      # d(softmin)/d(margin_i)
        sc = diffscore.detector_scores(full, device)
        sm = float(diffscore.soft_min(list(sc.values()), k=8.0))
    return names, w, sm, full


def attack(mels, hift, device, steps, max_delta, lr, beta, pen_target,
           refresh=5):
    raws = [torch.zeros_like(m, requires_grad=True) for m in mels]
    opt = torch.optim.Adam(raws, lr=lr)
    names, w, sm0, _ = global_weights(
        mels, [max_delta * torch.tanh(r).detach() for r in raws], hift, device)
    traj = [sm0]

    for step in range(steps):
        if step % refresh == 0 and step > 0:
            with torch.no_grad():
                cur = [max_delta * torch.tanh(r) for r in raws]
            names, w, sm, _ = global_weights(mels, cur, hift, device)
            traj.append(sm)
        opt.zero_grad()
        for mel, raw in zip(mels, raws):
            with diffscore.differentiable():
                delta = max_delta * torch.tanh(raw)
                seed = torch.randint(0, 2 ** 31 - 1, (1,)).item()
                torch.manual_seed(seed)
                wav, _ = hift(mel + delta)
                with torch.no_grad():
                    torch.manual_seed(seed)
                    clean, _ = hift(mel)
                mg = diffscore.detector_margins(wav.squeeze(0), device)
                # the chunk's contribution to the FILE's soft-min
                obj = sum(w[i] * mg[n] for i, n in enumerate(names))
                pen = P.penalty("mrstft+mask", delta, mel,
                                edited_wav=wav, clean_wav=clean)
                loss = -obj + beta * torch.clamp(pen - pen_target, min=0).pow(2)
                loss.backward()
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        return [(max_delta * torch.tanh(r)).detach() for r in raws], traj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+", default=["evelynn", "thresh", "riven"])
    ap.add_argument("--lengths", default="medium,long")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=50.0)
    ap.add_argument("--pen-target", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--keep-wavs", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    identity = MelResidualPolicy(max_delta=args.max_delta).to(device)
    out_dir = GENERATORS_DIR / "longform_attack"
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = [x.strip() for x in args.lengths.split(",")]

    rows = []
    for vi, voice in enumerate(args.voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        ref_emb = Tester.speaker_embedding(ref)

        for li, length in enumerate(lengths):
            text = LENGTHS[length]
            torch.manual_seed(args.seed + vi * 10 + li)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + vi * 10 + li)
            base_p = out_dir / f"{voice}_{length}_base.wav"
            with mel_edit_hook(cosy.model, identity, deterministic=True,
                               device=device) as rec:
                cloners.generate(text, ref, model="CosyVoice2",
                                 out_path=base_p, device=device)
            mels = [m.clone().to(device) for m in rec.mels]
            if not mels:
                continue

            deltas, traj = attack(mels, hift, device, args.steps, args.max_delta,
                                  args.lr, args.beta, args.pen_target)
            with torch.no_grad(), diffscore.differentiable():
                wavs = [hift(m + d)[0].squeeze(0).squeeze(0)
                        for m, d in zip(mels, deltas)]
                full = torch.cat(wavs, dim=-1)
            adv_p = out_dir / f"{voice}_{length}_attacked.wav"
            sf.write(str(adv_p), full.cpu().numpy(), cosy.sample_rate)

            def sc(p):
                arr = det.score_file(p, device=device)
                t = torch.tensor([float(v) for v in arr], dtype=torch.float64)
                q = quality.measure(p, ref, device)
                return {"det": {n: float(v) for n, v in zip(det.MODEL_ORDER, arr)},
                        "softmin": float(diffscore.soft_min(list(t), k=8.0)),
                        "sim": float(np.dot(Tester.speaker_embedding(p), ref_emb)),
                        "wer": Tester.wer(text, Tester.transcribe(p, "base",
                                                                  device))[0],
                        "pesq": q["pesq"]}

            b, a = sc(base_p), sc(adv_p)
            rows.append({"voice": voice, "length": length, "chunks": len(mels),
                         "seconds": float(full.shape[-1] / cosy.sample_rate),
                         "base": b, "attacked": a, "traj": traj,
                         "delta_abs": float(np.mean([d.abs().mean().item()
                                                     for d in deltas]))})
            print(f"  {voice:<12}{length:<8}{len(mels)} chunks "
                  f"{rows[-1]['seconds']:.0f}s   soft-min {b['softmin']:.3f} -> "
                  f"{a['softmin']:.3f}   PESQ {b['pesq']:.2f} -> {a['pesq']:.2f}   "
                  f"sim {b['sim']:.3f} -> {a['sim']:.3f}", flush=True)
            if not args.keep_wavs:
                base_p.unlink(missing_ok=True)
                adv_p.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 92)
    print(f"{'length':<10}{'n':>3}{'secs':>7}{'chunks':>8}{'base sm':>10}"
          f"{'attacked':>10}{'gain':>8}{'d PESQ':>9}{'d sim':>8}")
    print("-" * 92)
    summary = []
    for length in lengths:
        rs = [r for r in rows if r["length"] == length]
        if not rs:
            continue
        rec = {"length": length, "n": len(rs),
               "seconds": float(np.mean([r["seconds"] for r in rs])),
               "chunks": float(np.mean([r["chunks"] for r in rs])),
               "base": float(np.mean([r["base"]["softmin"] for r in rs])),
               "attacked": float(np.mean([r["attacked"]["softmin"] for r in rs])),
               "d_pesq": float(np.mean([r["attacked"]["pesq"] - r["base"]["pesq"]
                                        for r in rs])),
               "d_sim": float(np.mean([r["attacked"]["sim"] - r["base"]["sim"]
                                       for r in rs]))}
        summary.append(rec)
        print(f"{length:<10}{rec['n']:>3}{rec['seconds']:>7.0f}{rec['chunks']:>8.1f}"
              f"{rec['base']:>10.3f}{rec['attacked']:>10.3f}"
              f"{rec['attacked'] - rec['base']:>+8.3f}{rec['d_pesq']:>+9.2f}"
              f"{rec['d_sim']:>+8.3f}")

    out = RL / "longform_attack.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
