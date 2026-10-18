# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
Generate audio from a trained policy (or from the plain clone).

    python scripts/generate_audio.py --voice riven --text "hello there"
    python scripts/generate_audio.py --design b --voice jhin --text "..." --out b.wav
    python scripts/generate_audio.py --design a --voice riven --text "..." --score
    python scripts/generate_audio.py --all-arms --voice riven --text "..."

--all-arms is the useful one. It writes the same line three times, unedited
and through both designs, under ONE seed. Both policies are deterministic at
inference and consume no RNG, so all three replay the same speech tokens and
the same vocoder noise, and the mel edit is the only audible difference. The
files come out identical in length, which is the check that it held. Compare
two clips rendered under different seeds and you are comparing takes, not
edits.

Add --score to run every clip through the full metric set: the four
detectors and their soft-min, speaker similarity against the reference, WER
against the text you asked for, and SQUIM PESQ / STOI.

Runs under the voiceclone environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rl import common

ARMS = {
    "baseline": (None, "unedited CosyVoice2 clone"),
    "a": ("designA_budget4h", "Design A, PPO-Lagrangian"),
    "b": ("designB_B9h", "Design B, SAC with a gradient hint"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Speak a line with a trained policy, or without one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--voice", required=True,
                    help="a folder name under VOICE_DATA_ROOT, e.g. riven")
    ap.add_argument("--text", required=True, help="what to say")
    ap.add_argument("--design", choices=["a", "b"], default=None,
                    help="omit for the unedited clone")
    ap.add_argument("--run", default=None,
                    help="override the run id for the chosen design")
    ap.add_argument("--checkpoint", default="latest.pt")
    ap.add_argument("--out", default=None,
                    help="output wav (default: <voice>_<arm>.wav)")
    ap.add_argument("--out-dir", default=".",
                    help="where --all-arms writes its files")
    ap.add_argument("--all-arms", action="store_true",
                    help="render unedited, Design A and Design B on one seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true",
                    help="score every clip produced")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    wanted = ["baseline", "a", "b"] if args.all_arms else [args.design or "baseline"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'voice "{args.voice}"  seed {args.seed}')
    print(f'text  "{args.text}"')
    print()

    # load CosyVoice2 once and hand it to every arm; generate() returns the
    # path it wrote, not the model, so the model has to be kept here
    import cloners
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cosy = cloners.load_model("CosyVoice2", device=device)

    produced: list[tuple[str, Path]] = []
    for arm in wanted:
        run_id, label = ARMS[arm]
        if args.run and arm == args.design:
            run_id = args.run

        if args.out and not args.all_arms:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = out_dir / f"{args.voice}_{arm}.wav"

        common.generate(
            run_id, args.voice, args.text, path,
            checkpoint=args.checkpoint, device=device, seed=args.seed,
            cosy=cosy, no_edit=run_id is None)
        print(f"  {label:<38} -> {path}")
        produced.append((label, path))

    if not args.score:
        print("\npass --score to measure these.")
        return 0

    from scorecard import score_clip
    ref, _ = common.load_voice(args.voice)
    print("\n" + "=" * 74)
    print(f"{'arm':<24}{'soft-min':>10}{'sim':>8}{'WER':>9}{'PESQ':>8}{'STOI':>8}")
    print("-" * 74)
    for label, path in produced:
        row = score_clip(path, reference=ref, text=args.text, device=device)
        get = lambda k, d=float("nan"): row.get(k, d)
        print(f"{label[:23]:<24}{get('softmin'):>10.3f}{get('sim'):>8.3f}"
              f"{get('wer') * 100:>8.1f}%{get('pesq'):>8.2f}{get('stoi'):>8.3f}")
        for problem in row.get("problems", []):
            print(f"{'':<24}note: {problem}")
    print("\n1 = human on soft-min. Higher similarity, lower WER, higher PESQ "
          "are all better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
