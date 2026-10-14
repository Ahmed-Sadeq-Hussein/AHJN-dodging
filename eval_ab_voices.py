# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
eval_ab_voices.py -- head-to-head evaluation of the BEST Design A and the BEST
Design B checkpoint on a set of voices. RUN IN THE "voiceclone" ENV.

    python eval_ab_voices.py --voices Tywin_2 ahmed evelynn mordekaiser thresh --pairs 4

Why this exists rather than running eval_generalization.py twice
---------------------------------------------------------------
Running the single-checkpoint script once per design regenerates the baseline
each time, so A and B would be compared against two different (if seed-matched)
baseline sets, and a third of the compute would be spent producing a clip that
already exists.  Here each (voice, text, seed) triple is generated exactly three
times -- baseline, A-edited, B-edited -- from the identical seed.  A
deterministic policy consumes no RNG, so all three arms replay the same LLM
speech tokens and the same HiFT vocoder noise; the mel edit is the only
difference between them.  That makes A vs B a paired comparison on a shared
control, which is the only way the difference between two near-flat policies is
measurable at this sample size.

Reported per voice, then averaged over voices:
  * mean detector score  -- plain mean over the four detectors (1 = human)
  * soft-min             -- the training objective's worst-case term
  * similarity           -- cosine to the reference speaker embedding
  * WER                  -- whisper transcript vs the prompt text
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
from env_config import RESULTS_ROOT
from eval_generalization import build_policy, _EvalAdapter
from mel_policy import mel_edit_hook
from train_ppo_mel import DEFAULT_TEXTS, load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"


def agg(d: np.ndarray):
    """mean, standard error, t for a paired difference vector."""
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else 0.0), float("nan"), float("nan")
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return m, se, (m / se if se > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+", required=True)
    ap.add_argument("--pairs", type=int, default=4, help="utterances per voice")
    ap.add_argument("--a-run", default="designA_budget4h")
    ap.add_argument("--a-ckpt", default="best.pt")
    ap.add_argument("--b-run", default="designB_B9h")
    ap.add_argument("--b-ckpt", default="best.pt")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=90906)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)

    arms = {}
    for tag, run, ckpt in (("A", args.a_run, args.a_ckpt), ("B", args.b_run, args.b_ckpt)):
        path = RL_RUNS / run / ckpt
        if not path.is_file():
            print(f"no checkpoint at {path}")
            return 1
        blob = torch.load(path, map_location="cpu", weights_only=False)
        arms[tag] = {"blob": blob, "run": run, "ckpt": ckpt,
                     "iteration": blob.get("iteration"),
                     "trained_on": sorted(blob.get("args", {}).get("voices", []))}
        print(f"{tag}: {run}/{ckpt}  iteration {blob.get('iteration')}  "
              f"trained on {arms[tag]['trained_on']}")

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)

    for tag in ("A", "B"):
        net, kind, targs = build_policy(arms[tag]["blob"], device, args)
        arms[tag]["kind"] = kind
        arms[tag]["adapter"] = _EvalAdapter(net, hift_call, device,
                                            args.softmin_k, args.softmin_weight)
        print(f"{tag}: design {kind}")

    out_dir = RESULTS_ROOT / "rl_runs" / "ab_voice_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\ngenerating: {len(args.voices)} voices x {args.pairs} utterances x 3 arms "
          f"= {len(args.voices) * args.pairs * 3} clips\n")

    rows = []
    for vi, voice in enumerate(args.voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice, device=device)

        for pi in range(args.pairs):
            text = DEFAULT_TEXTS[pi % len(DEFAULT_TEXTS)]
            seed = args.seed + vi * 100 + pi
            row = {"voice": voice, "pair": pi, "text": text, "ref": str(ref)}

            def reseed():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            reseed()
            base = out_dir / f"{voice}_{pi}_base.wav"
            cloners.generate(text, ref, model="CosyVoice2", out_path=base, device=device)
            row["base_path"] = base

            for tag in ("A", "B"):
                reseed()
                p = out_dir / f"{voice}_{pi}_{tag}.wav"
                with mel_edit_hook(cosy.model, arms[tag]["adapter"],
                                   deterministic=True, device=device) as rec:
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
                row[f"{tag}_path"] = p
                row[f"{tag}_delta_abs"] = (
                    float(np.mean([d.abs().mean().item() for d in rec.deltas]))
                    if rec.deltas else 0.0)

            rows.append(row)
        print(f"  [{vi + 1}/{len(args.voices)}] {voice:<13} {args.pairs} utterances", flush=True)

    print("\nscoring every clip (4 detectors + whisper + speaker embedding)...", flush=True)
    for ri, r in enumerate(rows):
        ref_emb = Tester.speaker_embedding(Path(r["ref"]))
        for arm in ("base", "A", "B"):
            p = Path(r[f"{arm}_path"])
            arr = det.score_file(p, device=device)
            s = {n: float(v) for n, v in zip(names, arr)}
            t = torch.tensor([s[n] for n in names], dtype=torch.float64)
            r[f"{arm}_det"] = s
            r[f"{arm}_softmin"] = float(diffscore.soft_min(list(t), k=args.softmin_k))
            r[f"{arm}_meandet"] = float(t.mean())
            r[f"{arm}_sim"] = float(np.dot(Tester.speaker_embedding(p), ref_emb))
            r[f"{arm}_wer"] = Tester.wer(
                r["text"], Tester.transcribe(p, args.whisper, device))[0]
            p.unlink(missing_ok=True)
        print(f"  scored {ri + 1}/{len(rows)}", flush=True)

    # ------------------------------------------------------------------
    # per-voice aggregation
    # ------------------------------------------------------------------
    per_voice = []
    for voice in args.voices:
        rs = [r for r in rows if r["voice"] == voice]
        rec = {"voice": voice, "n": len(rs),
               "trained_A": voice in arms["A"]["trained_on"],
               "trained_B": voice in arms["B"]["trained_on"]}
        for arm in ("base", "A", "B"):
            for key in ("meandet", "softmin", "sim", "wer"):
                rec[f"{arm}_{key}"] = float(np.mean([r[f"{arm}_{key}"] for r in rs]))
            rec[f"{arm}_perdet"] = {
                n: float(np.mean([r[f"{arm}_det"][n] for r in rs])) for n in names}
        for tag in ("A", "B"):
            rec[f"{tag}_delta_abs"] = float(np.mean([r[f"{tag}_delta_abs"] for r in rs]))
            for key in ("meandet", "softmin", "sim", "wer"):
                d = np.array([r[f"{tag}_{key}"] - r[f"base_{key}"] for r in rs])
                m, se, t = agg(d)
                rec[f"d_{tag}_{key}"] = m
                rec[f"t_{tag}_{key}"] = t
        per_voice.append(rec)

    w = 14
    print("\n" + "=" * 104)
    print(f"{'voice':<13}{'mean detector score':>{w * 3}}{'soft-min':>{w * 3}}")
    print(f"{'':<13}{'base':>{w}}{'A':>{w}}{'B':>{w}}{'base':>{w}}{'A':>{w}}{'B':>{w}}")
    print("-" * 104)
    for r in per_voice:
        print(f"{r['voice']:<13}{r['base_meandet']:>{w}.4f}{r['A_meandet']:>{w}.4f}"
              f"{r['B_meandet']:>{w}.4f}{r['base_softmin']:>{w}.4f}"
              f"{r['A_softmin']:>{w}.4f}{r['B_softmin']:>{w}.4f}")
    print("-" * 104)
    print(f"{'AVERAGE':<13}"
          + "".join(f"{np.mean([r[k] for r in per_voice]):>{w}.4f}" for k in
                    ("base_meandet", "A_meandet", "B_meandet",
                     "base_softmin", "A_softmin", "B_softmin")))

    print("\n" + "=" * 104)
    print(f"{'voice':<13}{'similarity to reference':>{w * 3}}{'WER':>{w * 3}}")
    print(f"{'':<13}{'base':>{w}}{'A':>{w}}{'B':>{w}}{'base':>{w}}{'A':>{w}}{'B':>{w}}")
    print("-" * 104)
    for r in per_voice:
        print(f"{r['voice']:<13}{r['base_sim']:>{w}.4f}{r['A_sim']:>{w}.4f}"
              f"{r['B_sim']:>{w}.4f}{r['base_wer'] * 100:>{w - 1}.1f}%"
              f"{r['A_wer'] * 100:>{w - 1}.1f}%{r['B_wer'] * 100:>{w - 1}.1f}%")
    print("-" * 104)
    print(f"{'AVERAGE':<13}"
          + "".join(f"{np.mean([r[k] for r in per_voice]):>{w}.4f}" for k in
                    ("base_sim", "A_sim", "B_sim"))
          + "".join(f"{np.mean([r[k] for r in per_voice]) * 100:>{w - 1}.1f}%" for k in
                    ("base_wer", "A_wer", "B_wer")))

    print("\n" + "=" * 104)
    print("PAIRED DIFFERENCES vs the shared baseline (pooled over every clip)")
    print("=" * 104)
    for tag in ("A", "B"):
        print(f"  design {tag}  ({arms[tag]['kind']}, |delta| "
              f"{np.mean([r[f'{tag}_delta_abs'] for r in rows]):.4f})")
        for key, label, scale in (("meandet", "mean detector", 1.0),
                                  ("softmin", "soft-min     ", 1.0),
                                  ("sim", "similarity   ", 1.0),
                                  ("wer", "WER          ", 100.0)):
            d = np.array([r[f"{tag}_{key}"] - r[f"base_{key}"] for r in rows])
            m, se, t = agg(d)
            sig = "significant" if abs(t) > 2 else "not distinguishable from 0"
            unit = "%" if scale != 1.0 else " "
            print(f"    {label}  {m * scale:+8.4f}{unit} SE {se * scale:.4f}  "
                  f"t={t:+6.2f}   {sig}")

    out = Path(args.out) if args.out else out_dir / "ab_voice_eval.json"
    payload = {
        "seed": args.seed, "pairs_per_voice": args.pairs, "voices": args.voices,
        "arms": {t: {k: arms[t][k] for k in ("run", "ckpt", "iteration",
                                             "kind", "trained_on")}
                 for t in ("A", "B")},
        "detector_order": names,
        "per_voice": per_voice,
        "clips": [{k: (str(v) if isinstance(v, Path) else v)
                   for k, v in r.items() if not k.endswith("_path")} for r in rows],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
