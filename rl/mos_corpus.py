r"""
rl.mos_corpus -- render the listening-test corpus from the RL policies only.

    python -m rl.mos_corpus
    python -m rl.mos_corpus --voices riven thresh --out-dir "MOS TEST RL"

Three arms, nothing else:

    baseline   the unedited CosyVoice2 clone
    design_a   PPO with Lagrangian constraints
    design_b   SAC with a gradient hint

ALL THREE SHARE ONE SEED PER VOICE. Both policies are driven deterministically
and consume no RNG, so the arms replay the same speech tokens and the same
HiFT vocoder noise, and the Mel edit is the only audible difference. The
rendered files come out identical in length to the sample, which is the check
that it held. Without this a listener is rating three different takes rather
than the thing under test.

Every clip is scored on the full metric set through scorecard.score_clip: the
four detectors and their soft-min, speaker similarity, WER and SQUIM
PESQ/STOI. Reference recordings are scored on detectors only, since their
similarity to themselves is 1.0 by construction and they have no target text.

Output is the layout the rating page reads, plus scores.json. Build the page
with:  python mos_build_app.py --dir "MOS TEST RL"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cloners
import Tester
from eval_generalization import all_voices
from rl.common import load_policy, load_voice
from scorecard import score_clip

# Two lines per voice, in different registers. The first is held out from
# every training set (textbank.EVAL_TEXTS), so neither policy has seen it; the
# second is declamatory, which stresses a different part of the synthesiser.
# A listener who hears only one register is rating that register, not the edit.
TEXTS = [
    "Whatever happens tomorrow, remember that the tide comes in twice "
    "and asks nobody first.",
    "Beyond heaven and earth, He alone was the honored one. "
    "The one to stand above all the other men.",
]

# label -> (run id, checkpoint, what it is)
ARMS = {
    "baseline": (None, None,
                 "unedited CosyVoice2 clone, no policy"),
    "design_a": ("designA_budget4h", "latest.pt",
                 "PPO-Lagrangian; made detection significantly easier "
                 "(slope -0.000727, t = -7.26)"),
    "design_b": ("designB_B9h", "latest.pt",
                 "SAC with a gradient hint; flat "
                 "(slope +0.000033, t = +0.46)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="MOS TEST RL")
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--texts", nargs="+", default=TEXTS)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-quality", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    voices = args.voices or all_voices()
    arms = {k: v for k, v in ARMS.items() if args.arms is None or k in args.arms}

    root = Path(args.out_dir)
    (root / "references").mkdir(parents=True, exist_ok=True)
    for a in arms:
        (root / a).mkdir(parents=True, exist_ok=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    adapters = {}
    for label, (run, ckpt, _) in arms.items():
        if run is None:
            adapters[label] = None
            print(f"  {label:<10} unedited")
            continue
        adapter, kind, cosy = load_policy(run, ckpt, device=device, cosy=cosy)
        adapters[label] = adapter
        print(f"  {label:<10} {run}/{ckpt}  [{kind}]")

    print(f"\n{len(voices)} voices x {len(args.texts)} lines x {len(arms)} arms"
          f" = {len(voices) * len(args.texts) * len(arms)} clips")
    for i, t in enumerate(args.texts, 1):
        print(f'  line {i}: "{t}"')
    print(flush=True)

    clips = []
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        rp = root / "references" / f"{voice}.wav"
        shutil.copy2(ref, rp)
        ref_scores = score_clip(rp, device=device, quality=not args.no_quality)
        for ti in range(len(args.texts)):
            # our keys after the splat: score_clip returns a "file" of its own
            # (an OS path) and it must not win over the page-relative one
            clips.append({**ref_scores,
                          "voice": f"{voice} line {ti + 1}", "speaker": voice,
                          "line": ti + 1, "arm": "reference",
                          "file": f"references/{voice}.wav",
                          "prompt_text": prompt})

        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        ref_emb = Tester.speaker_embedding(ref)
        from mel_policy import mel_edit_hook

        for ti, text in enumerate(args.texts):
            # one seed per (voice, line): the three arms of a line replay the
            # same tokens and the same vocoder noise, so the mel edit is the
            # only difference between them
            seed = args.seed + vi * 10 + ti
            item = f"{voice} line {ti + 1}"
            line = f"  [{vi + 1}/{len(voices)}] {item:<22}"

            for label, adapter in adapters.items():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                rel = f"{label}/{voice}_L{ti + 1}.wav"
                pth = root / rel
                if adapter is None:
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=pth, device=device)
                else:
                    with mel_edit_hook(cosy.model, adapter, deterministic=True,
                                       device=device):
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=pth, device=device)
                s = score_clip(pth, reference=ref, text=text, device=device,
                               ref_embedding=ref_emb,
                               quality=not args.no_quality)
                clips.append({**s, "voice": item, "speaker": voice,
                              "line": ti + 1, "arm": label, "file": rel})
                line += f" {label.split('_')[-1]} {s.get('softmin', float('nan')):.2f}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(line, flush=True)

    payload = {
        "texts": list(args.texts), "text": args.texts[0], "seed": args.seed,
        "note": ("Reinforcement-learning arms only. All arms for a voice share "
                 "one seed, so the mel edit is the only difference between "
                 "them. 1 = human on every detector; soft-min is the worst "
                 "of the four."),
        "arms": {k: {"run": v[0], "checkpoint": v[1], "what": v[2]}
                 for k, v in arms.items()},
        "clips": clips,
    }
    (root / "scores.json").write_text(json.dumps(payload, indent=2),
                                      encoding="utf8")

    print("\n" + "=" * 70)
    print(f"{'arm':<12}{'soft-min':>10}{'sim':>8}{'WER':>8}{'PESQ':>8}")
    print("-" * 70)
    for label in arms:
        rs = [c for c in clips if c["arm"] == label]
        g = lambda k: float(np.mean([r[k] for r in rs if k in r])) if rs else float("nan")
        print(f"{label:<12}{g('softmin'):>10.3f}{g('sim'):>8.3f}"
              f"{g('wer') * 100:>7.1f}%{g('pesq'):>8.2f}")
    print(f"\nwrote {root / 'scores.json'}  ({len(clips)} clips)")
    print(f'next: python mos_build_app.py --dir "{root}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
