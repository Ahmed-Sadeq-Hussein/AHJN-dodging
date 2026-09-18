r"""
eval_all15.py -- the scorecard: every voice, every target. RUN IN "voiceclone".

    python eval_all15.py --new-run distill_margin_base --new-iters 4

The brief is a single model that is good on all 15 voices at three things at
once -- detection, similarity and WER -- so this reports all three per voice,
plus PESQ, and marks each voice pass/fail against explicit thresholds rather
than leaving "good" undefined:

    soft-min      >= 0.50   every detector leans human
    similarity    within 0.02 of the unedited clone
    WER           within 2 points of the unedited clone
    PESQ          within 0.30 of the unedited clone

Three arms per utterance from one shared seed, so the edit is the only
difference: unedited, the previous best model, and the new one. Results are
appended per clip, so a killed run still leaves usable data.
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
from env_config import RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import all_voices, build_policy
from mel_policy import mel_edit_hook
from textbank import EVAL_TEXTS, TRAIN_VOICES
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"
THRESHOLDS = {"softmin": 0.50, "sim_drop": 0.02, "wer_rise": 0.02, "pesq_drop": 0.30}


def adapter_for(run, ckpt, device, hift, iters, args):
    p = RL_RUNS / run / ckpt
    if not p.is_file():
        return None, None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    net, kind, _ = build_policy(blob, device, args)
    net.eval()
    ck = blob.get("args", {})
    ad = AmplifiedAdapter(net, hift, device, 8.0, 0.5, scale=1.0, iters=iters,
                          names=ck.get("detectors"),
                          space=ck.get("objective", "prob"),
                          cap=ck.get("margin_cap", 4.0))
    return ad, f"{kind} x{iters} [{ck.get('objective', 'prob')} hint]"


def main() -> int:
    ap = argparse.ArgumentParser()
    # "distill_margin" never existed as a directory -- the margin runs are
    # distill_margin_base / _spk / _spk32 -- so the old default could not load.
    # base is the one to prefer: section 7n found speaker conditioning does
    # nothing (26.5% recovered unconditioned against 26.3% conditioned), so the
    # smaller unconditioned model is the honest recommendation.
    ap.add_argument("--new-run", default="distill_margin_base")
    ap.add_argument("--new-ckpt", default="best.pt")
    ap.add_argument("--new-iters", type=int, default=4)
    ap.add_argument("--old-run", default="distill_final")
    ap.add_argument("--old-ckpt", default="best.pt")
    ap.add_argument("--old-iters", type=int, default=10)
    ap.add_argument("--texts", type=int, default=4)
    ap.add_argument("--voices", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)
    voices = args.voices or all_voices()
    texts = EVAL_TEXTS[:args.texts]

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)

    arms = {"baseline": None}
    old, old_kind = adapter_for(args.old_run, args.old_ckpt, device, hift,
                                args.old_iters, args)
    if old:
        arms["old"] = old
        print(f"old: {args.old_run} -- {old_kind}")
    new, new_kind = adapter_for(args.new_run, args.new_ckpt, device, hift,
                                args.new_iters, args)
    if new is None:
        print(f"no checkpoint at {RL_RUNS / args.new_run / args.new_ckpt}")
        return 1
    arms["new"] = new
    print(f"new: {args.new_run} -- {new_kind}")

    work = RL_RUNS / "_all15_tmp"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RL_RUNS / "all15_eval.json"
    stream = out.with_suffix(".jsonl")
    print(f"\n{len(voices)} voices x {len(texts)} unseen sentences x {len(arms)} arms"
          f" = {len(voices) * len(texts) * len(arms)} clips\n", flush=True)

    rows = []
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        ref_emb = Tester.speaker_embedding(ref)
        for ad in (arms.get("old"), arms.get("new")):
            if ad is not None:
                ad.set_speaker(ref_emb)

        for ti, text in enumerate(texts):
            seed = args.seed + vi * 100 + ti
            row = {"voice": voice, "text_idx": ti,
                   "trained": voice in TRAIN_VOICES}
            for arm, ad in arms.items():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                p = work / f"{voice}_{ti}_{arm}.wav"
                if ad is None:
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
                else:
                    with mel_edit_hook(cosy.model, ad, deterministic=True,
                                       device=device):
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=p, device=device)
                arr = det.score_file(p, device=device)
                t = torch.tensor([float(v) for v in arr], dtype=torch.float64)
                q = quality.measure(p, ref, device)
                row[arm] = {
                    "det": {n: float(v) for n, v in zip(names, arr)},
                    "softmin": float(diffscore.soft_min(list(t), k=8.0)),
                    "meandet": float(t.mean()),
                    "sim": float(np.dot(Tester.speaker_embedding(p), ref_emb)),
                    "wer": Tester.wer(cloners.strip_markup(text),
                                      Tester.transcribe(p, "base", device))[0],
                    "pesq": q["pesq"], "stoi": q["stoi"]}
                p.unlink(missing_ok=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
            with stream.open("a", encoding="utf8") as fh:
                fh.write(json.dumps(row) + chr(10))
        print(f"  [{vi + 1}/{len(voices)}] {voice:<13} "
              f"soft-min {np.mean([r['baseline']['softmin'] for r in rows if r['voice'] == voice]):.3f}"
              f" -> {np.mean([r['new']['softmin'] for r in rows if r['voice'] == voice]):.3f}",
              flush=True)

    for f in work.glob("*"):
        f.unlink()
    work.rmdir()

    arm_names = list(arms)
    w = 9
    print("\n" + "=" * 112)
    print("PER VOICE (mean over sentences)")
    print(f"{'voice':<13}{'set':<7}" + "".join(f"{'sm ' + a:>{w}}" for a in arm_names)
          + "".join(f"{'sim ' + a:>{w}}" for a in arm_names)
          + f"{'WER base':>10}{'WER new':>9}{'PESQ b':>8}{'PESQ n':>8}{'verdict':>9}")
    print("-" * 112)
    per_voice, passed = [], 0
    for voice in voices:
        rs = [r for r in rows if r["voice"] == voice]
        if not rs:
            continue
        m = {a: {k: float(np.mean([r[a][k] for r in rs]))
                 for k in ("softmin", "meandet", "sim", "wer", "pesq", "stoi")}
             for a in arm_names}
        ok = (m["new"]["softmin"] >= THRESHOLDS["softmin"]
              and m["baseline"]["sim"] - m["new"]["sim"] <= THRESHOLDS["sim_drop"]
              and m["new"]["wer"] - m["baseline"]["wer"] <= THRESHOLDS["wer_rise"]
              and m["baseline"]["pesq"] - m["new"]["pesq"] <= THRESHOLDS["pesq_drop"])
        passed += bool(ok)
        rec = {"voice": voice, "trained": rs[0]["trained"], "pass": bool(ok),
               "n": len(rs), **{a: m[a] for a in arm_names}}
        rec["per_det_new"] = {n: float(np.mean([r["new"]["det"][n] for r in rs]))
                              for n in names}
        per_voice.append(rec)
        print(f"{voice:<13}{'train' if rec['trained'] else 'other':<7}"
              + "".join(f"{m[a]['softmin']:>{w}.3f}" for a in arm_names)
              + "".join(f"{m[a]['sim']:>{w}.3f}" for a in arm_names)
              + f"{m['baseline']['wer'] * 100:>9.1f}%{m['new']['wer'] * 100:>8.1f}%"
              + f"{m['baseline']['pesq']:>8.2f}{m['new']['pesq']:>8.2f}"
              + f"{'PASS' if ok else 'fail':>9}")

    print("-" * 112)
    print(f"{'ALL':<13}{'':<7}"
          + "".join(f"{np.mean([v[a]['softmin'] for v in per_voice]):>{w}.3f}" for a in arm_names)
          + "".join(f"{np.mean([v[a]['sim'] for v in per_voice]):>{w}.3f}" for a in arm_names)
          + f"{np.mean([v['baseline']['wer'] for v in per_voice]) * 100:>9.1f}%"
          + f"{np.mean([v['new']['wer'] for v in per_voice]) * 100:>8.1f}%"
          + f"{np.mean([v['baseline']['pesq'] for v in per_voice]):>8.2f}"
          + f"{np.mean([v['new']['pesq'] for v in per_voice]):>8.2f}"
          + f"{passed}/{len(per_voice):>8}")

    print("\nPAIRED CHANGE, new model vs unedited (pooled over every clip)")
    for key, label, scale in (("softmin", "soft-min", 1), ("meandet", "mean detector", 1),
                              ("sim", "similarity", 1), ("wer", "WER", 100),
                              ("pesq", "PESQ", 1)):
        d = np.array([r["new"][key] - r["baseline"][key] for r in rows])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {label:<15}{d.mean() * scale:>+9.4f}   t = {d.mean() / se:>+6.2f}")

    print("\nPER-DETECTOR after the new edit (mean over voices)")
    for n in names:
        print(f"  {n:<12}{np.mean([v['per_det_new'][n] for v in per_voice]):.4f}")

    out.write_text(json.dumps({"thresholds": THRESHOLDS, "arms": arm_names,
                               "new_run": args.new_run, "old_run": args.old_run,
                               "per_voice": per_voice, "passed": passed,
                               "clips": rows}, indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
