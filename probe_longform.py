r"""
probe_longform.py -- does any of this survive at 50 seconds? RUN IN "voiceclone".

    python probe_longform.py --voices evelynn thresh riven --iters 4

Why
---
Section 6b of Plan_X/RESULTS.md found that baseline soft-min collapses ~7x
between a 4.6 s clip and a 50 s one (0.356 -> 0.050), with RawNet2 at 0.0007
and RawGAT-ST at 0.0001 on the long clip. Every training run, every PGD
target and every evaluation in this project used 5-15 s utterances, i.e. the
easiest regime. So the headline result -- distilled x4 lifting held-out
soft-min to 2.1x baseline -- is currently a claim about short-form only.

There is also a structural reason to expect trouble, visible in the
generation logs: CosyVoice2 splits long text into chunks and yields speech
per chunk, so `mel_edit_hook` fires once per chunk on a ~20 s mel. The
detectors, meanwhile, window the *concatenated* file into 64600-sample
(~4 s) windows and average. The policy therefore optimises a local objective
while detection aggregates globally over many more windows, and averaging
punishes an edit that helps some windows and not others.

This probe measures the length curve for three arms at matched seeds:
    baseline        unedited
    distilled xN    the working operator, applied per chunk
    pgd (optional)  per-chunk PGD, as a ceiling at this length

Reported per length: every detector, soft-min, similarity, WER, and SQUIM
PESQ/STOI, so a gain that only arrives by wrecking the audio is visible.
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
import quality
import Tester
from collect_pgd import pgd_solve
from env_config import GENERATORS_DIR, RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import build_policy
from mel_policy import MelResidualPolicy, mel_edit_hook
from textbank import EVAL_TEXTS

RL_RUNS = RESULTS_ROOT / "rl_runs"

# One sentence, then increasingly long stacks of unseen sentences. Built from
# EVAL_TEXTS so nothing here appeared in collection or distillation.
LENGTHS = {
    "short": " ".join(EVAL_TEXTS[:1]),
    "medium": " ".join(EVAL_TEXTS[:3]),
    "long": " ".join(EVAL_TEXTS * 2),
}


class PGDChunkPolicy(torch.nn.Module):
    """Runs PGD on each mel chunk as it passes through the hook.

    This is the honest per-length ceiling: the same white-box attack the
    distilled operator was trained to imitate, given the same chunked view of
    a long utterance that the operator gets. It is NOT a global optimum over
    the whole file -- no arm here has one, because CosyVoice2 never exposes
    the whole file as a single mel.
    """

    def __init__(self, hift_call, device, steps, max_delta, lr, k, w):
        super().__init__()
        self.max_delta = max_delta
        self.min_log_std = -3.0
        self.log_std = torch.nn.Parameter(torch.full((80,), -4.5))
        self.hift_call, self.device = hift_call, device
        self.steps, self.lr, self.k, self.w = steps, lr, k, w

    def sample(self, mel, deterministic=False):
        delta, _ = pgd_solve(mel, self.hift_call, self.device, self.steps,
                             self.max_delta, self.lr, self.k, self.w)
        return delta, torch.zeros(mel.shape[0], device=mel.device), delta

    def to(self, *a, **kw):
        return self


def score_clip(path, ref, text, device, k, whisper, names):
    arr = det.score_file(path, device=device)
    s = {n: float(v) for n, v in zip(names, arr)}
    t = torch.tensor([s[n] for n in names], dtype=torch.float64)
    q = quality.measure(path, ref, device)
    return {"det": s,
            "softmin": float(diffscore.soft_min(list(t), k=k)),
            "meandet": float(t.mean()),
            "sim": float(np.dot(Tester.speaker_embedding(path),
                                Tester.speaker_embedding(ref))),
            "wer": Tester.wer(text, Tester.transcribe(path, whisper, device))[0],
            "pesq": q["pesq"], "stoi": q["stoi"],
            "seconds": _duration(path)}


def _duration(path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+", default=["evelynn", "thresh", "riven"])
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pgd", action="store_true",
                    help="also run per-chunk PGD as a ceiling (slow)")
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--keep-wavs", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)
    lengths = [x.strip() for x in args.lengths.split(",")]

    blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                      map_location="cpu", weights_only=False)
    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    net, kind, _ = build_policy(blob, device, args)
    net.eval()
    print(f"{args.run}/{args.checkpoint}  {kind}  applied x{args.iters} "
          f"at scale {args.scale}")

    arms = {"baseline": None,
            f"distilled_x{args.iters}": AmplifiedAdapter(
                net, hift_call, device, args.softmin_k, args.softmin_weight,
                scale=args.scale, iters=args.iters)}
    if args.pgd:
        arms["pgd_chunk"] = PGDChunkPolicy(hift_call, device, args.pgd_steps,
                                           0.6, 0.05, args.softmin_k,
                                           args.softmin_weight)

    out_dir = GENERATORS_DIR / "longform_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vi, voice in enumerate(args.voices):
        from train_ppo_mel import load_voice
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        for li, length in enumerate(lengths):
            text = LENGTHS[length]
            seed = args.seed + vi * 100 + li
            row = {"voice": voice, "length": length, "words": len(text.split())}
            for arm, pol in arms.items():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                p = out_dir / f"{voice}_{length}_{arm}.wav"
                t0 = time.time()
                if pol is None:
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
                    row[f"{arm}_delta"] = 0.0
                else:
                    with mel_edit_hook(cosy.model, pol, deterministic=True,
                                       device=device) as rec:
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=p, device=device)
                    row[f"{arm}_delta"] = (
                        float(np.mean([d.abs().mean().item() for d in rec.deltas]))
                        if rec.deltas else 0.0)
                    row[f"{arm}_chunks"] = len(rec.deltas)
                row[f"{arm}_gen_s"] = time.time() - t0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row[arm] = score_clip(p, ref, text, device, args.softmin_k,
                                      args.whisper, names)
                if not args.keep_wavs:
                    p.unlink(missing_ok=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
            arm_names = list(arms)
            print(f"  {voice:<12} {length:<7} {row['words']:>3}w  "
                  + "  ".join(f"{a}: sm {row[a]['softmin']:.4f}" for a in arm_names),
                  flush=True)

    w = 13
    arm_names = list(arms)
    print("\n" + "=" * 100)
    print("SOFT-MIN BY LENGTH (mean over voices)")
    print(f"{'length':<10}{'words':>7}{'chunks':>8}" + "".join(f"{a:>{w}}" for a in arm_names))
    print("-" * 100)
    summary = []
    key_arm = arm_names[1]
    for length in lengths:
        rs = [r for r in rows if r["length"] == length]
        rec = {"length": length, "words": rs[0]["words"],
               "chunks": float(np.mean([r.get(f"{key_arm}_chunks", 1) for r in rs]))}
        for a in arm_names:
            for m in ("softmin", "meandet", "sim", "wer", "pesq", "stoi"):
                rec[f"{a}_{m}"] = float(np.mean([r[a][m] for r in rs]))
            rec[f"{a}_perdet"] = {n: float(np.mean([r[a]["det"][n] for r in rs]))
                                  for n in names}
        summary.append(rec)
        print(f"{length:<10}{rec['words']:>7}{rec['chunks']:>8.1f}"
              + "".join(f"{rec[a + '_softmin']:>{w}.4f}" for a in arm_names))

    print("\nGAIN OVER BASELINE, and what it costs")
    print(f"{'length':<10}" + "".join(
        f"{a + ' d_sm':>{w}}{a + ' d_sim':>{w}}" for a in arm_names[1:]))
    print("-" * 100)
    for rec in summary:
        line = f"{rec['length']:<10}"
        for a in arm_names[1:]:
            line += (f"{rec[a + '_softmin'] - rec['baseline_softmin']:>+{w}.4f}"
                     f"{rec[a + '_sim'] - rec['baseline_sim']:>+{w}.4f}")
        print(line)

    print("\nPER-DETECTOR at each length (baseline -> best arm)")
    for rec in summary:
        print(f"  {rec['length']:<8}" + "  ".join(
            f"{n}: {rec['baseline_perdet'][n]:.4f}->{rec[key_arm + '_perdet'][n]:.4f}"
            for n in names))

    out = RL_RUNS / "longform_probe.json"
    out.write_text(json.dumps({"run": args.run, "iters": args.iters,
                               "scale": args.scale, "voices": args.voices,
                               "arms": arm_names, "detector_order": names,
                               "summary": summary,
                               "clips": rows}, indent=2, default=float),
                   encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
