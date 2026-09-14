r"""
apply_best.py -- speak text in an enrolled voice using the best configuration
found in Plan_X. RUN IN THE "voiceclone" ENV.

    python apply_best.py --voice evelynn --text "I sentence you to death." --out out.wav
    python apply_best.py --voice evelynn --text-file lines.txt --out-dir clips\ --score

Best configuration (Plan_X/RESULTS.md sections 7 and 7h):

    checkpoint   rl_runs/distill_final/best.pt   21,352 params, 5-frame receptive field
    passes       10, each re-computing the gradient hint on the edited mel
    scale        1.0

On held-out voices with unseen sentences this took mean detector score
0.34 -> 0.69 (t = 12.26) and soft-min 0.11 -> 0.37, with similarity and WER
unchanged but PESQ down 0.22. --iters 4 keeps PESQ intact for about half the
gain. It is strongest on ~5-15 s utterances; on ~70 s clips the gain
disappears (section 7e), and ten passes on a long clip can exhaust a 10 GB GPU.

--no-edit renders the same seed without the edit, for an A/B comparison.
--score prints the four detector scores, similarity and WER for each clip.

The edit needs gradients through the vocoder and all four detectors, so every
call loads the detectors as well as CosyVoice2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import diffscore
from env_config import RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import build_policy
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

BEST_RUN = "distill_final"
BEST_CKPT = "best.pt"
BEST_ITERS = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file", help="one line per clip")
    ap.add_argument("--voice", required=True,
                    help="a folder name under VOICE_DATA_ROOT with reference.wav")
    ap.add_argument("--out", default=None, help="output path (single --text)")
    ap.add_argument("--out-dir", default=None, help="output folder (--text-file)")
    ap.add_argument("--iters", type=int, default=BEST_ITERS)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--run", default=BEST_RUN)
    ap.add_argument("--checkpoint", default=BEST_CKPT)
    ap.add_argument("--no-edit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.text is not None:
        lines = [args.text]
        outs = [Path(args.out or f"{args.voice}.wav")]
    else:
        lines = [l.strip() for l in Path(args.text_file).read_text(encoding="utf8")
                 .splitlines() if l.strip()]
        od = Path(args.out_dir or f"{args.voice}_clips")
        od.mkdir(parents=True, exist_ok=True)
        outs = [od / f"{args.voice}_{i:03d}.wav" for i in range(len(lines))]

    cosy = cloners.load_model("CosyVoice2", device=device)
    ref, prompt = load_voice(args.voice)
    cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=args.voice,
                   device=device)

    adapter = None
    if not args.no_edit:
        ck = RESULTS_ROOT / "rl_runs" / args.run / args.checkpoint
        if not ck.is_file():
            print(f"no checkpoint at {ck}")
            return 1
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        net, kind, _ = build_policy(blob, device, args)
        net.eval()
        adapter = AmplifiedAdapter(
            net, diffscore.hift_forward_fn(cosy.model.hift), device, 8.0, 0.5,
            scale=args.scale, iters=args.iters,
            names=blob.get("args", {}).get("detectors"))
        print(f"edit: {args.run}/{args.checkpoint} ({kind}) x{args.iters}")
    else:
        print("edit: none (--no-edit)")

    if args.score:
        import detectors as det
        import Tester
        ref_emb = Tester.speaker_embedding(ref)

    for i, (text, out) in enumerate(zip(lines, outs)):
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)
        out.parent.mkdir(parents=True, exist_ok=True)
        if adapter is None:
            cloners.generate(text, ref, model="CosyVoice2", out_path=out,
                             device=device)
        else:
            with mel_edit_hook(cosy.model, adapter, deterministic=True,
                               device=device):
                cloners.generate(text, ref, model="CosyVoice2", out_path=out,
                                 device=device)
        line = f"  {out}"
        if args.score:
            arr = det.score_file(out, device=device)
            t = torch.tensor([float(v) for v in arr], dtype=torch.float64)
            sim = float(np.dot(Tester.speaker_embedding(out), ref_emb))
            wer = Tester.wer(cloners.strip_markup(text),
                             Tester.transcribe(out, "base", device))[0]
            line += ("  " + "  ".join(f"{n}={float(v):.3f}"
                                      for n, v in zip(det.MODEL_ORDER, arr))
                     + f"  softmin={float(diffscore.soft_min(list(t))):.3f}"
                     + f"  sim={sim:.3f}  wer={wer * 100:.1f}%")
        print(line, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
