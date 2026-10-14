# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
mos_test.py -- build the listening-test corpus. RUN IN "voiceclone".

    python mos_test.py                       # all 15 voices, all 4 arms
    python mos_test.py --voices evelynn jhin --out-dir "MOS TEST"

Renders one fixed line in every voice under every operating point this project
established, scores each clip on the full metric set, and writes the folder
layout the rating app reads:

    MOS TEST/
      references/<voice>.wav      the clip that was zero-shot cloned from
      baseline/<voice>.wav        unedited CosyVoice2
      probability_x10/<voice>.wav distill_final, 10 passes
      margin_x4/<voice>.wav       distill_margin_base, 4 passes  (recommended)
      margin_x14/<voice>.wav      distill_margin_base, 14 passes (over budget)
      scores.json                 every metric for every clip

**All four arms share one seed per voice.** A deterministic policy consumes no
RNG, so the arms replay the same speech tokens and the same HiFT vocoder noise
and the mel edit is the ONLY difference between them. Without that they would
be four different takes and a listener would be rating delivery, not the edit.

Scores per clip: the four detectors and their soft-min (1 = human), speaker
similarity against the reference, WER (Whisper against the target text), and
SQUIM PESQ/STOI. Reference clips are scored on detectors only -- similarity to
themselves is 1.0 by construction and they have no target text.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
from env_config import RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import all_voices, build_policy
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"

TEXT = ("The sky shun bright with a beautifull velvet blue. "
        "So long have I waited for such a sight")

# label -> (run, checkpoint, passes, what it is)
ARMS = {
    "baseline": (None, None, 0,
                 "unedited CosyVoice2 clone"),
    "probability_x10": ("distill_final", "best.pt", 10,
                        "probability-objective operator, 10 passes "
                        "(soft-min 0.489, PESQ -0.28)"),
    "margin_x4": ("distill_margin_base", "best.pt", 4,
                  "margin-objective operator, 4 passes -- the recommended "
                  "configuration (soft-min 0.454, PESQ -0.17)"),
    "margin_x14": ("distill_margin_base", "best.pt", 14,
                   "margin-objective operator, 14 passes -- outside the "
                   "quality budget (soft-min 0.800, PESQ -0.85)"),
}


def adapter_for(run, ckpt, iters, device, hift):
    p = RL_RUNS / run / ckpt
    if not p.is_file():
        return None, f"MISSING {p}"
    blob = torch.load(p, map_location="cpu", weights_only=False)
    net, kind, _ = build_policy(blob, device, None)
    net.eval()
    ck = blob.get("args", {})
    ad = AmplifiedAdapter(net, hift, device, 8.0, 0.5, scale=1.0, iters=iters,
                          names=ck.get("detectors"),
                          space=ck.get("objective", "prob"),
                          cap=ck.get("margin_cap", 4.0))
    n = sum(v.numel() for v in blob["actor"].values()) if "actor" in blob else 0
    return ad, f"{kind} x{iters} [{ck.get('objective', 'prob')} hint, {n:,} params]"


def score(path, ref, ref_emb, text, device, names, with_quality=True):
    arr = det.score_file(path, device=device)
    t = torch.tensor([float(v) for v in arr], dtype=torch.float64)
    out = {"det": {n: float(v) for n, v in zip(names, arr)},
           "softmin": float(diffscore.soft_min(list(t), k=8.0)),
           "meandet": float(t.mean())}
    if with_quality:
        q = quality.measure(path, ref, device)
        out["sim"] = float(np.dot(Tester.speaker_embedding(path), ref_emb))
        out["wer"] = Tester.wer(cloners.strip_markup(text),
                                Tester.transcribe(path, "base", device))[0]
        out["pesq"] = q["pesq"]
        out["stoi"] = q["stoi"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="MOS TEST")
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None,
                    help=f"subset of {list(ARMS)}")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    voices = args.voices or all_voices()
    arms = {k: v for k, v in ARMS.items()
            if args.arms is None or k in args.arms}
    root = Path(args.out_dir)
    (root / "references").mkdir(parents=True, exist_ok=True)
    for a in arms:
        (root / a).mkdir(parents=True, exist_ok=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    names = list(det.MODEL_ORDER)

    built = {}
    for label, (run, ckpt, iters, _) in arms.items():
        if run is None:
            built[label] = None
            print(f"  {label:<18} unedited")
            continue
        ad, desc = adapter_for(run, ckpt, iters, device, hift)
        if ad is None:
            print(f"  {label:<18} {desc}")
            return 1
        built[label] = ad
        print(f"  {label:<18} {desc}")

    print(f"\n{len(voices)} voices x {len(arms)} arms = "
          f"{len(voices) * len(arms)} clips")
    print(f'text: "{args.text}"\n', flush=True)

    clips = []
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        ref_emb = Tester.speaker_embedding(ref)
        for ad in built.values():
            if ad is not None and hasattr(ad, "set_speaker"):
                ad.set_speaker(ref_emb)

        rp = root / "references" / f"{voice}.wav"
        shutil.copy2(ref, rp)
        clips.append({"voice": voice, "arm": "reference",
                      "file": f"references/{voice}.wav",
                      "prompt_text": prompt,
                      **score(rp, ref, ref_emb, "", device, names,
                              with_quality=False)})

        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)

        # one seed for the whole voice: the arms differ only by the mel edit
        seed = args.seed + vi
        line = f"  [{vi + 1}/{len(voices)}] {voice:<13}"
        for label, ad in built.items():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            p = root / label / f"{voice}.wav"
            if ad is None:
                cloners.generate(args.text, ref, model="CosyVoice2",
                                 out_path=p, device=device)
            else:
                with mel_edit_hook(cosy.model, ad, deterministic=True,
                                   device=device):
                    cloners.generate(args.text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
            s = score(p, ref, ref_emb, args.text, device, names)
            clips.append({"voice": voice, "arm": label,
                          "file": f"{label}/{voice}.wav", **s})
            line += f" {label.split('_')[0][:4]} {s['softmin']:.2f}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(line, flush=True)

    payload = {
        "text": args.text,
        "seed": args.seed,
        "detector_order": names,
        "note": ("All arms for a voice share one seed, so the mel edit is the "
                 "only difference between them. 1 = human on every detector; "
                 "soft-min is the worst of the four."),
        "arms": {k: {"run": v[0], "checkpoint": v[1], "passes": v[2],
                     "what": v[3]} for k, v in arms.items()},
        "clips": clips,
    }
    (root / "scores.json").write_text(json.dumps(payload, indent=2),
                                      encoding="utf8")

    print("\n" + "=" * 78)
    print(f"{'arm':<18}{'soft-min':>10}{'sim':>8}{'WER':>8}{'PESQ':>8}{'STOI':>8}")
    print("-" * 78)
    for label in arms:
        rs = [c for c in clips if c["arm"] == label]
        if not rs:
            continue
        g = lambda k: float(np.mean([r[k] for r in rs if k in r]))
        print(f"{label:<18}{g('softmin'):>10.3f}{g('sim'):>8.3f}"
              f"{g('wer') * 100:>7.1f}%{g('pesq'):>8.2f}{g('stoi'):>8.3f}")
    rr = [c for c in clips if c["arm"] == "reference"]
    print(f"{'(references)':<18}{float(np.mean([r['softmin'] for r in rr])):>10.3f}"
          f"{'--':>8}{'--':>8}{'--':>8}{'--':>8}")
    print("-" * 78)
    print(f"\nwrote {root / 'scores.json'}  ({len(clips)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
