r"""
probe_teacher_quality.py -- what does a stronger PGD teacher cost? RUN IN "voiceclone".

    python probe_teacher_quality.py

probe_teacher.py found that 150 steps at lr 0.12 gives 1.53x the soft-min gain
of the current 60-step / lr 0.05 teacher on the hardest clips -- but with an
edit twice as large (|delta| 0.32 vs 0.16). A stronger teacher is only worth
re-collecting targets for if that larger edit keeps the voice and the audio
intact, and the distilled student would inherit whatever it costs.

This measures, on the same stored mels for every configuration:

    soft-min      the objective
    similarity    CAM++ cosine to the ORIGINAL reference.wav, and its change
                  relative to the unedited vocoder output of the same mel
    PESQ / STOI   SQUIM estimates, same comparison
    WER           whisper against the utterance text

Sampled from both ends of the difficulty range: the hardest clips (where the
stronger teacher's gain showed up) and median clips (where it may simply cost
quality for nothing).
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
import diffscore
import quality
import Tester
from collect_pgd import pgd_solve
from env_config import RESULTS_ROOT
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"
CONFIGS = [("current 60/0.05", 60, 0.05),
           ("150/0.05", 150, 0.05),
           ("strong 150/0.12", 150, 0.12)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", type=int, default=6)
    ap.add_argument("--median", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    items = [torch.load(f, map_location="cpu", weights_only=False)
             for f in sorted((RL_RUNS / "pgd_dataset").glob("*.pt"))]
    items.sort(key=lambda d: d["softmin_before"])
    mid = len(items) // 2
    sel = ([("hard", d) for d in items[:args.hard]]
           + [("median", d) for d in items[mid:mid + args.median]])

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    tmp = RL_RUNS / "_teacher_quality_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    def render(mel, path):
        with torch.no_grad(), diffscore.differentiable():
            w, _ = hift(mel)
            sc = diffscore.detector_scores(w.squeeze(0), device)
            sm = float(diffscore.soft_min(list(sc.values())))
        sf.write(str(path), w.squeeze(0).squeeze(0).cpu().numpy(), cosy.sample_rate)
        return sm

    rows = []
    for i, (group, d) in enumerate(sel):
        mel = d["mel"].float().unsqueeze(0).to(device)
        ref, _ = load_voice(d["voice"])
        ref_emb = Tester.speaker_embedding(ref)
        text = cloners.strip_markup(d["text"])

        p0 = tmp / "base.wav"
        sm0 = render(mel, p0)
        q0 = quality.measure(p0, ref, device)
        base = {"softmin": sm0,
                "sim": float(np.dot(Tester.speaker_embedding(p0), ref_emb)),
                "pesq": q0["pesq"], "stoi": q0["stoi"],
                "wer": Tester.wer(text, Tester.transcribe(p0, "base", device))[0]}

        for label, steps, lr in CONFIGS:
            delta, _ = pgd_solve(mel, hift, device, steps, 0.6, lr, 8.0, 0.5)
            p1 = tmp / "edit.wav"
            sm1 = render(mel + delta, p1)
            q1 = quality.measure(p1, ref, device)
            rows.append({
                "group": group, "key": f"{d['voice']}_t{d['text_idx']:02d}",
                "config": label, "delta_abs": float(delta.abs().mean()),
                "softmin": sm1, "base_softmin": base["softmin"],
                "sim": float(np.dot(Tester.speaker_embedding(p1), ref_emb)),
                "base_sim": base["sim"],
                "pesq": q1["pesq"], "base_pesq": base["pesq"],
                "stoi": q1["stoi"], "base_stoi": base["stoi"],
                "wer": Tester.wer(text, Tester.transcribe(p1, "base", device))[0],
                "base_wer": base["wer"]})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"  [{i + 1}/{len(sel)}] {group:<6} {d['voice']}_t{d['text_idx']:02d} "
              f"base soft-min {sm0:.3f}", flush=True)

    for f in tmp.glob("*.wav"):
        f.unlink()
    tmp.rmdir()

    def t_of(x):
        x = np.asarray(x)
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        return x.mean(), (x.mean() / se if se and se > 0 else float("nan"))

    print("\n" + "=" * 100)
    print(f"{'group':<8}{'config':<18}{'|delta|':>8}{'softmin':>9}{'d sm':>8}"
          f"{'d sim':>9}{'t':>7}{'d PESQ':>9}{'t':>7}{'d STOI':>9}{'d WER':>8}")
    print("-" * 100)
    summary = []
    for group in ("hard", "median", "all"):
        for label, _, _ in CONFIGS:
            rs = [r for r in rows if r["config"] == label
                  and (group == "all" or r["group"] == group)]
            dsim, tsim = t_of([r["sim"] - r["base_sim"] for r in rs])
            dpq, tpq = t_of([r["pesq"] - r["base_pesq"] for r in rs])
            rec = {"group": group, "config": label, "n": len(rs),
                   "delta_abs": float(np.mean([r["delta_abs"] for r in rs])),
                   "softmin": float(np.mean([r["softmin"] for r in rs])),
                   "d_softmin": float(np.mean([r["softmin"] - r["base_softmin"] for r in rs])),
                   "d_sim": float(dsim), "t_sim": float(tsim),
                   "d_pesq": float(dpq), "t_pesq": float(tpq),
                   "d_stoi": float(np.mean([r["stoi"] - r["base_stoi"] for r in rs])),
                   "d_wer": float(np.mean([r["wer"] - r["base_wer"] for r in rs]))}
            summary.append(rec)
            print(f"{group:<8}{label:<18}{rec['delta_abs']:>8.3f}{rec['softmin']:>9.4f}"
                  f"{rec['d_softmin']:>+8.3f}{rec['d_sim']:>+9.4f}{rec['t_sim']:>7.2f}"
                  f"{rec['d_pesq']:>+9.3f}{rec['t_pesq']:>7.2f}{rec['d_stoi']:>+9.4f}"
                  f"{rec['d_wer'] * 100:>+7.1f}%")
        print("-" * 100)

    out = Path(args.out) if args.out else RL_RUNS / "teacher_quality.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
