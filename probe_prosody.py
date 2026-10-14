# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
probe_prosody.py -- does dramatising the delivery change detector scores?
RUN IN "voiceclone".

    python probe_prosody.py --voices Tywin_2 evelynn thresh riven --variants 6

Six renditions of identical content per voice (see prosody.py): neutral,
punctuation pauses, <strong> emphasis, emphasis plus [breath], and two
inference_instruct2 style prompts (dramatic and calm, the latter as a
control for the instruct pathway itself).

Everything is generated with a shared seed per (voice, variant) so the only
difference between styles is the delivery. Optionally each style is also run
through the distilled edit, which answers the second question: whether
expressive prosody and the learned edit stack, cancel, or interfere.

WER is scored against the markup-stripped text -- comparing a transcript to a
string containing '<strong>' would count the tag as a missed word and make
every expressive variant look like it damaged intelligibility.
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
import detectors as det
import diffscore
import prosody
import quality
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import build_policy
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"


def agg(d):
    n = len(d)
    if n < 2:
        return (float(np.mean(d)) if n else 0.0), float("nan"), float("nan")
    m = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    return m, se, (m / se if se > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+",
                    default=["Tywin_2", "evelynn", "thresh", "riven"])
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--styles", default=None,
                    help="comma-separated subset of prosody.ALL_STYLES")
    ap.add_argument("--with-edit", action="store_true",
                    help="also run each style through the distilled edit")
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--keep-wavs", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)
    styles = ([s.strip() for s in args.styles.split(",")] if args.styles
              else list(prosody.ALL_STYLES))
    variants = prosody.VARIANTS[:args.variants]

    cosy = cloners.load_model("CosyVoice2", device=device)
    adapter = None
    if args.with_edit:
        blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                          map_location="cpu", weights_only=False)
        net, kind, _ = build_policy(blob, device, args)
        net.eval()
        hift_call = diffscore.hift_forward_fn(cosy.model.hift)
        adapter = AmplifiedAdapter(net, hift_call, device, args.softmin_k,
                                   args.softmin_weight, scale=1.0,
                                   iters=args.iters,
                                   names=blob.get("args", {}).get("detectors"))
        print(f"edit arm: {args.run}/{args.checkpoint} {kind} x{args.iters}")

    arms = ["plain"] + (["edited"] if adapter else [])
    out_dir = GENERATORS_DIR / "prosody_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.voices) * len(variants) * len(styles) * len(arms)
    print(f"{len(args.voices)} voices x {len(variants)} lines x {len(styles)} "
          f"styles x {len(arms)} arms = {total} clips\n", flush=True)

    rows = []
    for vi, voice in enumerate(args.voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        ref_emb = Tester.speaker_embedding(ref)

        for ci, variant in enumerate(variants):
            for style in styles:
                text, instruct = prosody.render(variant, style)
                spoken = cloners.strip_markup(text)
                seed = args.seed + vi * 1000 + ci * 10
                row = {"voice": voice, "line": ci, "style": style,
                       "text": text, "instruct": instruct, "spoken": spoken}
                ok = True
                for arm in arms:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    p = out_dir / f"{voice}_{ci}_{style}_{arm}.wav"
                    try:
                        if arm == "plain":
                            cloners.generate(text, ref, model="CosyVoice2",
                                             out_path=p, device=device,
                                             instruct=instruct)
                        else:
                            with mel_edit_hook(cosy.model, adapter,
                                               deterministic=True,
                                               device=device):
                                cloners.generate(text, ref, model="CosyVoice2",
                                                 out_path=p, device=device,
                                                 instruct=instruct)
                    except Exception as e:
                        print(f"  {voice} line{ci} {style} {arm}: FAILED "
                              f"{type(e).__name__}: {e}", flush=True)
                        ok = False
                        break

                    arr = det.score_file(p, device=device)
                    s = {n: float(v) for n, v in zip(names, arr)}
                    t = torch.tensor([s[n] for n in names], dtype=torch.float64)
                    rec = {"det": s,
                           "softmin": float(diffscore.soft_min(list(t), k=args.softmin_k)),
                           "meandet": float(t.mean()),
                           "sim": float(np.dot(Tester.speaker_embedding(p), ref_emb)),
                           "wer": Tester.wer(spoken, Tester.transcribe(
                               p, args.whisper, device))[0],
                           "seconds": _dur(p)}
                    if not args.no_quality:
                        q = quality.measure(p, ref, device)
                        rec["pesq"], rec["stoi"] = q["pesq"], q["stoi"]
                    row[arm] = rec
                    if not args.keep_wavs:
                        p.unlink(missing_ok=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if ok:
                    rows.append(row)
            print(f"  [{vi+1}/{len(args.voices)}] {voice:<12} line {ci+1}/"
                  f"{len(variants)} done", flush=True)

    metrics = ["softmin", "meandet", "sim", "wer", "seconds"] + (
        [] if args.no_quality else ["pesq", "stoi"])

    print("\n" + "=" * 104)
    print("BY STYLE, pooled over voices and lines (arm: plain)")
    print(f"{'style':<20}{'n':>4}" + "".join(f"{m:>12}" for m in metrics))
    print("-" * 104)
    base = [r for r in rows if r["style"] == "neutral"]
    summary = []
    for style in styles:
        rs = [r for r in rows if r["style"] == style]
        if not rs:
            continue
        rec = {"style": style, "n": len(rs)}
        for m in metrics:
            rec[m] = float(np.mean([r["plain"][m] for r in rs]))
        rec["perdet"] = {n: float(np.mean([r["plain"]["det"][n] for r in rs]))
                         for n in names}
        summary.append(rec)
        print(f"{style:<20}{len(rs):>4}"
              + "".join(f"{rec[m]:>12.4f}" for m in metrics))

    # paired against neutral on the same (voice, line)
    print("\n" + "=" * 104)
    print("PAIRED vs neutral, same voice and same line")
    print(f"{'style':<20}{'d softmin':>12}{'t':>8}{'d meandet':>12}{'t':>8}"
          f"{'d sim':>10}{'d WER':>10}")
    print("-" * 104)
    key = {(r["voice"], r["line"]): r for r in base}
    for style in styles:
        if style == "neutral":
            continue
        pairs = [(r, key[(r["voice"], r["line"])]) for r in rows
                 if r["style"] == style and (r["voice"], r["line"]) in key]
        if not pairs:
            continue
        dsm = [a["plain"]["softmin"] - b["plain"]["softmin"] for a, b in pairs]
        dmd = [a["plain"]["meandet"] - b["plain"]["meandet"] for a, b in pairs]
        dsi = [a["plain"]["sim"] - b["plain"]["sim"] for a, b in pairs]
        dwe = [a["plain"]["wer"] - b["plain"]["wer"] for a, b in pairs]
        m1, _, t1 = agg(dsm)
        m2, _, t2 = agg(dmd)
        print(f"{style:<20}{m1:>+12.4f}{t1:>8.2f}{m2:>+12.4f}{t2:>8.2f}"
              f"{float(np.mean(dsi)):>+10.4f}{float(np.mean(dwe))*100:>+9.1f}%")
        for rec in summary:
            if rec["style"] == style:
                rec.update({"d_softmin": m1, "t_softmin": t1,
                            "d_meandet": m2, "t_meandet": t2,
                            "d_sim": float(np.mean(dsi)),
                            "d_wer": float(np.mean(dwe))})

    if adapter:
        print("\n" + "=" * 104)
        print("DOES THE LEARNED EDIT STILL WORK ON EXPRESSIVE SPEECH?")
        print(f"{'style':<20}{'plain sm':>12}{'edited sm':>12}{'gain':>10}{'t':>8}")
        print("-" * 104)
        for style in styles:
            rs = [r for r in rows if r["style"] == style and "edited" in r]
            if not rs:
                continue
            d = [r["edited"]["softmin"] - r["plain"]["softmin"] for r in rs]
            m, _, t = agg(d)
            print(f"{style:<20}"
                  f"{float(np.mean([r['plain']['softmin'] for r in rs])):>12.4f}"
                  f"{float(np.mean([r['edited']['softmin'] for r in rs])):>12.4f}"
                  f"{m:>+10.4f}{t:>8.2f}")
            for rec in summary:
                if rec["style"] == style:
                    rec["edit_gain"] = m
                    rec["edit_t"] = t

    print("\nPER-DETECTOR by style (arm: plain)")
    print(f"{'style':<20}" + "".join(f"{n:>13}" for n in names))
    for rec in summary:
        print(f"{rec['style']:<20}"
              + "".join(f"{rec['perdet'][n]:>13.4f}" for n in names))

    out = Path(args.out) if args.out else RL_RUNS / "prosody_probe.json"
    out.write_text(json.dumps({"voices": args.voices, "styles": styles,
                               "n_lines": len(variants), "arms": arms,
                               "detector_order": names,
                               "summary": summary, "clips": rows},
                              indent=2, default=float), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


def _dur(path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


if __name__ == "__main__":
    sys.exit(main())
