# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
demo_best.py -- render arbitrary text with the best configuration, and keep
the wavs. RUN IN "voiceclone".

    python demo_best.py --text "I sentence you to death, for your transgressions." \
        --voices Tywin_2 Tywin Joffrey --iters 4

Best configuration, as established in Plan_X/RESULTS.md section 7:
    distill_final/best.pt   21,352 params, receptive field 5 mel frames
    scale 1.0, applied 4x   each pass re-hints on the already-edited mel

Both arms share a seed. A deterministic policy consumes no RNG, so baseline
and edited replay the same speech tokens and the same HiFT vocoder noise --
the mel edit is the only difference, which is what makes them comparable by
ear rather than two different takes.

Scores every clip on all four detectors plus similarity, WER and SQUIM
PESQ/STOI, because an example is only worth listening to alongside what it
cost.
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
import detectors as det
import diffscore
import quality
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import build_policy
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--extended", default=None,
                    help="a longer passage containing the line, to show the "
                         "~14 s regime where the operator is strongest")
    ap.add_argument("--voices", nargs="+", default=["Tywin_2"])
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=8888)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)

    blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                      map_location="cpu", weights_only=False)
    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    net, kind, _ = build_policy(blob, device, args)
    net.eval()
    ck_names = blob.get("args", {}).get("detectors")
    adapter = AmplifiedAdapter(net, hift_call, device, args.softmin_k,
                               args.softmin_weight, scale=args.scale,
                               iters=args.iters, names=ck_names)

    out_dir = Path(args.out_dir) if args.out_dir else GENERATORS_DIR / "best_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.run}/{args.checkpoint}  {kind}  x{args.iters} @ scale {args.scale}")
    print(f"writing to {out_dir}\n")

    variants = [("line", args.text)]
    if args.extended:
        variants.append(("extended", args.extended))

    rows = []
    for vi, voice in enumerate(args.voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        ref_emb = Tester.speaker_embedding(ref)

        for ki, (kind_name, text) in enumerate(variants):
            seed = args.seed + vi * 100 + ki
            row = {"voice": voice, "variant": kind_name, "text": text}
            for arm in ("baseline", f"best_x{args.iters}"):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                p = out_dir / f"{voice}_{kind_name}_{arm}.wav"
                if arm == "baseline":
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
                else:
                    with mel_edit_hook(cosy.model, adapter, deterministic=True,
                                       device=device) as rec:
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=p, device=device)
                    row["delta_abs"] = (
                        float(np.mean([d.abs().mean().item() for d in rec.deltas]))
                        if rec.deltas else 0.0)
                arr = det.score_file(p, device=device)
                s = {n: float(v) for n, v in zip(names, arr)}
                t = torch.tensor([s[n] for n in names], dtype=torch.float64)
                q = quality.measure(p, ref, device)
                row[arm] = {
                    "path": str(p), "det": s,
                    "softmin": float(diffscore.soft_min(list(t), k=args.softmin_k)),
                    "meandet": float(t.mean()),
                    "sim": float(np.dot(Tester.speaker_embedding(p), ref_emb)),
                    # WER must be scored against the SPOKEN words. Comparing a
                    # transcript to text containing '<strong>' or '[breath]'
                    # counts each tag as a missed word -- it inflated the
                    # breathy variants here to 59-65% before this was fixed.
                    "wer": Tester.wer(cloners.strip_markup(text),
                                      Tester.transcribe(p, args.whisper,
                                                        device))[0],
                    "pesq": q["pesq"], "stoi": q["stoi"]}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
            b, e = row["baseline"], row[f"best_x{args.iters}"]
            print(f"  {voice:<12}{kind_name:<10} softmin {b['softmin']:.4f} -> "
                  f"{e['softmin']:.4f}   meandet {b['meandet']:.4f} -> "
                  f"{e['meandet']:.4f}   sim {b['sim']:.3f} -> {e['sim']:.3f}   "
                  f"wer {b['wer']*100:.1f}% -> {e['wer']*100:.1f}%", flush=True)

    arm = f"best_x{args.iters}"
    w = 12
    print("\n" + "=" * 96)
    print(f"{'voice':<12}{'variant':<10}{'arm':<11}"
          + "".join(f"{n:>12}" for n in names) + f"{'softmin':>10}")
    print("-" * 96)
    for row in rows:
        for a in ("baseline", arm):
            print(f"{row['voice']:<12}{row['variant']:<10}{a:<11}"
                  + "".join(f"{row[a]['det'][n]:>12.4f}" for n in names)
                  + f"{row[a]['softmin']:>10.4f}")

    print("\n" + "=" * 96)
    print(f"{'voice':<12}{'variant':<10}{'arm':<11}{'PESQ':>10}{'STOI':>10}"
          f"{'sim':>10}{'WER':>10}")
    print("-" * 96)
    for row in rows:
        for a in ("baseline", arm):
            print(f"{row['voice']:<12}{row['variant']:<10}{a:<11}"
                  f"{row[a]['pesq']:>10.3f}{row[a]['stoi']:>10.4f}"
                  f"{row[a]['sim']:>10.3f}{row[a]['wer']*100:>9.1f}%")

    (out_dir / "best_demo.json").write_text(
        json.dumps({"run": args.run, "iters": args.iters, "scale": args.scale,
                    "detector_order": names, "clips": rows}, indent=2),
        encoding="utf8")
    print(f"\nwrote {out_dir / 'best_demo.json'}")
    for row in rows:
        for a in ("baseline", arm):
            print(f"  {row[a]['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
