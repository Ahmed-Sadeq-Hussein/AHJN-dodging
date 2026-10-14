# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
demo_listen.py -- produce listenable examples for one voice at three script
lengths, baseline and policy-edited, and score each one. RUN IN "voiceclone".

    python demo_listen.py --voice evelynn --run designB_B9h

Unlike the evaluation scripts this one KEEPS the wav files -- they are the
point. Each length is generated twice from the same seed: once untouched and
once with the learned mel edit applied. A deterministic policy consumes no
RNG, so the two clips share the same speech tokens and the same vocoder noise
and differ only by the edit, which is what makes them worth comparing by ear.
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
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from eval_generalization import build_policy, _EvalAdapter
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"

SCRIPTS = {
    "short": (
        "Every shadow keeps a secret, and I keep them all."
    ),
    "medium": (
        "The train arrives at platform nine just before midnight, and the "
        "station is almost empty. Rain falls steadily on the tin roof while "
        "the whole building smells of woodsmoke and cold coffee. Nobody is "
        "waiting, and nobody is expected."
    ),
    "long": (
        "There is nothing more deceptive than an obvious fact, especially "
        "when nobody has bothered to look at it closely. The quick brown fox "
        "jumps over the lazy dog while the morning light spills across the "
        "valley and the river turns the colour of old brass. Down in the "
        "town the baker has already opened, and the smell of bread carries "
        "further than it has any right to on a morning this cold. She sold "
        "seashells by the seashore every summer for as long as anyone could "
        "remember, and the tourists never once asked where they came from. "
        "By the time the clock strikes seven the streets will be full again, "
        "and none of this quiet will have left a trace behind it. The train "
        "arrives at platform nine just before midnight, and the station is "
        "almost empty. Rain fell steadily on the tin roof, and the whole "
        "house smelled of woodsmoke and coffee for three days afterwards. "
        "Nothing here is ever quite as simple as it first appears to be."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="evelynn")
    ap.add_argument("--run", default="designB_B9h")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=7171)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)

    ck = RL_RUNS / args.run / args.checkpoint
    if not ck.is_file():
        print(f"no checkpoint at {ck}")
        return 1
    blob = torch.load(ck, map_location="cpu", weights_only=False)

    out_dir = Path(args.out_dir) if args.out_dir else GENERATORS_DIR / f"{args.voice}_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    net, kind, targs = build_policy(blob, device, args)
    adapter = _EvalAdapter(net, hift_call, device, args.softmin_k, args.softmin_weight)

    ref, prompt = load_voice(args.voice)
    cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=args.voice, device=device)

    print(f"voice      : {args.voice}   reference {ref}")
    print(f"checkpoint : {args.run}/{args.checkpoint}  iteration {blob.get('iteration')}")
    print(f"design     : {kind}")
    print(f"writing to : {out_dir}\n")

    results = []
    for li, (length, text) in enumerate(SCRIPTS.items()):
        seed = args.seed + li * 10
        words = len(text.split())
        print(f"[{length}] {words} words", flush=True)
        row = {"length": length, "words": words, "text": text, "seed": seed}

        for arm in ("baseline", "designB"):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            path = out_dir / f"{args.voice}_{length}_{arm}.wav"
            if arm == "baseline":
                cloners.generate(text, ref, model="CosyVoice2",
                                 out_path=path, device=device)
                dmag = 0.0
            else:
                with mel_edit_hook(cosy.model, adapter,
                                   deterministic=True, device=device) as rec:
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=path, device=device)
                dmag = (float(np.mean([d.abs().mean().item() for d in rec.deltas]))
                        if rec.deltas else 0.0)
            row[f"{arm}_path"] = str(path)
            row[f"{arm}_delta_abs"] = dmag
        results.append(row)

    print("\nscoring...", flush=True)
    ref_emb = Tester.speaker_embedding(ref)
    for row in results:
        for arm in ("baseline", "designB"):
            p = Path(row[f"{arm}_path"])
            arr = det.score_file(p, device=device)
            s = {n: float(v) for n, v in zip(names, arr)}
            t = torch.tensor([s[n] for n in names], dtype=torch.float64)
            row[f"{arm}_det"] = s
            row[f"{arm}_softmin"] = float(diffscore.soft_min(list(t), k=args.softmin_k))
            row[f"{arm}_meandet"] = float(t.mean())
            row[f"{arm}_sim"] = float(np.dot(Tester.speaker_embedding(p), ref_emb))
            row[f"{arm}_wer"] = Tester.wer(
                row["text"], Tester.transcribe(p, args.whisper, device))[0]

    w = 12
    print("\n" + "=" * 100)
    print(f"{'script':<9}{'words':>6}{'arm':<11}"
          f"{'meandet':>{w}}{'softmin':>{w}}{'sim':>{w}}{'WER':>{w}}{'|delta|':>{w}}")
    print("-" * 100)
    for row in results:
        for arm in ("baseline", "designB"):
            print(f"{row['length']:<9}{row['words']:>6}  {arm:<9}"
                  f"{row[f'{arm}_meandet']:>{w}.4f}{row[f'{arm}_softmin']:>{w}.4f}"
                  f"{row[f'{arm}_sim']:>{w}.4f}{row[f'{arm}_wer'] * 100:>{w - 1}.1f}%"
                  f"{row[f'{arm}_delta_abs']:>{w}.4f}")
        print("-" * 100)

    print("\nper-detector scores (1 = human, 0 = AI)")
    print(f"{'script':<9}{'arm':<11}" + "".join(f"{n:>13}" for n in names))
    for row in results:
        for arm in ("baseline", "designB"):
            print(f"{row['length']:<9}{arm:<11}"
                  + "".join(f"{row[f'{arm}_det'][n]:>13.4f}" for n in names))

    meta = out_dir / "demo.json"
    meta.write_text(json.dumps(
        {"voice": args.voice, "run": args.run, "checkpoint": args.checkpoint,
         "iteration": blob.get("iteration"), "design": kind,
         "reference": str(ref), "detector_order": names, "clips": results},
        indent=2), encoding="utf8")
    print(f"\nwrote {meta}")
    for row in results:
        for arm in ("baseline", "designB"):
            print(f"  {row[f'{arm}_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
