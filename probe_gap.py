r"""
probe_gap.py -- how close is the student to its teacher, voice by voice?
RUN IN "voiceclone".

    python probe_gap.py                      # every voice with a reference.wav
    python probe_gap.py --voices evelynn thresh --texts 2

The 97% recovery figure (Plan_X section 7h) was measured on unseen sentences
from the eleven TRAINING voices only. Whether more training data would help
depends on whether that holds for voices the student never saw -- a large
teacher-student gap on held-out voices means the student is not generalising
and more voices in the dataset would pay off; a small gap means the student is
capped by the teacher and more data buys nothing.

For every (voice, unseen sentence) the SAME mel is scored three ways, through
the same differentiable vocoder + detector path:

    base      unedited
    student   distill_final/best.pt applied 10x, re-hinting each pass
    teacher   60-step PGD with the settings every training target used

Recovery per voice is (student gain) / (teacher gain), computed from the
voice's mean gains rather than averaging per-clip ratios, which explode when
the teacher's gain on a clip is near zero. Voices already near 1.0 at
baseline (ahmed) have no headroom and their ratio is flagged, not trusted.
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
import diffscore
from collect_pgd import pgd_solve
from env_config import RESULTS_ROOT, VOICE_DATA_ROOT
from eval_generalization import build_policy
from mel_policy import MelResidualPolicy, mel_edit_hook
from probe_amplify import apply_policy
from textbank import EVAL_TEXTS, TRAIN_VOICES
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"
K, W = 8.0, 0.5


def all_voices() -> list[str]:
    return sorted(p.name for p in VOICE_DATA_ROOT.iterdir()
                  if p.is_dir() and (p / "reference.wav").is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--texts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pgd-steps", type=int, default=60)
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1357)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    voices = args.voices or all_voices()
    texts = EVAL_TEXTS[:args.texts]
    work = RL_RUNS / "_gap_tmp"
    work.mkdir(parents=True, exist_ok=True)

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                      map_location="cpu", weights_only=False)
    net, kind, _ = build_policy(blob, device, args)
    net.eval()
    identity = MelResidualPolicy().to(device)

    def score(mel):
        with torch.no_grad(), diffscore.differentiable():
            w, _ = hift(mel)
            sc = diffscore.detector_scores(w.squeeze(0), device)
            vals = list(sc.values())
            return (float(diffscore.soft_min(vals, k=K)),
                    float(torch.stack(vals).mean()),
                    float(diffscore.detector_objective(sc, k=K, softmin_weight=W)),
                    {n: float(v) for n, v in sc.items()})

    print(f"student: {args.run}/{args.checkpoint} x{args.iters}   "
          f"teacher: PGD {args.pgd_steps} steps")
    print(f"{len(voices)} voices x {len(texts)} unseen sentences\n", flush=True)

    rows = []
    t0 = time.time()
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        for ti, text in enumerate(texts):
            torch.manual_seed(args.seed + vi * 100 + ti)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + vi * 100 + ti)
            with mel_edit_hook(cosy.model, identity, deterministic=True,
                               device=device) as rec:
                cloners.generate(text, ref, model="CosyVoice2",
                                 out_path=work / "tmp.wav", device=device)
            if not rec.mels:
                continue
            mel = rec.mels[0].clone().to(device)

            b = score(mel)
            student_mel, sdelta = apply_policy(net, mel, hift, device, K, W,
                                               scale=1.0, iters=args.iters)
            s = score(student_mel)
            tdelta, _ = pgd_solve(mel, hift, device, args.pgd_steps, 0.6, 0.05, K, W)
            t = score(mel + tdelta)
            rows.append({"voice": voice, "trained": voice in TRAIN_VOICES,
                         "text_idx": ti,
                         "base": {"softmin": b[0], "meandet": b[1], "obj": b[2], "det": b[3]},
                         "student": {"softmin": s[0], "meandet": s[1], "obj": s[2], "det": s[3],
                                     "delta_abs": float(sdelta.abs().mean())},
                         "teacher": {"softmin": t[0], "meandet": t[1], "obj": t[2], "det": t[3],
                                     "delta_abs": float(tdelta.abs().mean())}})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"  [{vi + 1}/{len(voices)}] {voice:<12} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)

    (work / "tmp.wav").unlink(missing_ok=True)
    work.rmdir()

    def recovery(rs, key):
        sg = np.mean([r["student"][key] - r["base"][key] for r in rs])
        tg = np.mean([r["teacher"][key] - r["base"][key] for r in rs])
        return float(sg), float(tg), (float(sg / tg) if tg > 0.02 else float("nan"))

    print("\n" + "=" * 104)
    print(f"{'voice':<13}{'set':<10}{'base sm':>9}{'student':>9}{'teacher':>9}"
          f"{'sm recov':>10}{'obj recov':>11}{'base md':>9}{'student':>9}{'teacher':>9}")
    print("-" * 104)
    per_voice = []
    for voice in voices:
        rs = [r for r in rows if r["voice"] == voice]
        if not rs:
            continue
        _, _, rsm = recovery(rs, "softmin")
        _, _, rob = recovery(rs, "obj")
        rec = {"voice": voice, "trained": rs[0]["trained"], "n": len(rs),
               "recovery_softmin": rsm, "recovery_obj": rob}
        for arm in ("base", "student", "teacher"):
            for key in ("softmin", "meandet"):
                rec[f"{arm}_{key}"] = float(np.mean([r[arm][key] for r in rs]))
        per_voice.append(rec)
        flag = "  (no headroom)" if rec["base_softmin"] > 0.85 else ""
        fmt = lambda x: f"{x * 100:>8.0f}%" if x == x else f"{'n/a':>9}"
        print(f"{voice:<13}{'trained' if rec['trained'] else 'HELD-OUT':<10}"
              f"{rec['base_softmin']:>9.3f}{rec['student_softmin']:>9.3f}"
              f"{rec['teacher_softmin']:>9.3f}{fmt(rsm):>10}{fmt(rob):>11}"
              f"{rec['base_meandet']:>9.3f}{rec['student_meandet']:>9.3f}"
              f"{rec['teacher_meandet']:>9.3f}{flag}")

    print("\n" + "=" * 104)
    print("POOLED (voices with headroom only)")
    for label, want in (("trained voices", True), ("held-out voices", False)):
        rs = [r for r in rows if r["trained"] == want and r["base"]["softmin"] <= 0.85]
        if not rs:
            continue
        sg, tg, rr = recovery(rs, "obj")
        sgs, tgs, rrs = recovery(rs, "softmin")
        print(f"  {label:<16} n={len(rs):>3}  objective: student +{sg:.3f} / teacher "
              f"+{tg:.3f} = {rr * 100:.0f}%    soft-min: +{sgs:.3f} / +{tgs:.3f} "
              f"= {rrs * 100:.0f}%")

    out = RL_RUNS / "gap_probe.json"
    out.write_text(json.dumps({"run": args.run, "iters": args.iters,
                               "pgd_steps": args.pgd_steps, "texts": texts,
                               "per_voice": per_voice, "clips": rows},
                              indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
