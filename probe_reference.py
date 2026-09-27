r"""
probe_reference.py -- is the detectable artefact coming from the REFERENCE?
RUN IN "voiceclone".

    python probe_reference.py --voices evelynn thresh mordekaiser riven swain jhin

The observation this tests
--------------------------
Section 0 of Plan_X/RESULTS.md records the largest single effect in the
project, and it is not one we caused: baseline soft-min runs 0.24 (evelynn)
to 0.96 (ahmed) across voices, and the split tracks recording provenance.
The low scorers are studio-processed game voice-over whose *reference files*
already fail RawNet2 and RawGAT-ST before any cloning happens. If the clone
inherits detectable structure from the reference rather than generating it,
then cleaning the reference before enrolment should move the clone's score --
and it would do so at zero inference cost, unlike every method in sections
3-7.

Treatments applied to reference.wav before enrolment:

    raw          untouched (control)
    denoise      spectral-gate noise reduction (noisereduce)
    highpass     80 Hz high-pass, removing rumble and DC offset
    normalise    peak normalisation to -3 dBFS
    combined     denoise -> highpass -> normalise

Reported at two levels, because they answer different questions:

  * the REFERENCE's own detector scores, before and after treatment -- does
    the treatment actually remove whatever the detectors are reacting to?
  * the CLONE's scores when enrolled from the treated reference -- does that
    removal propagate through synthesis?

Similarity is always measured against the ORIGINAL untreated reference: the
target is the real speaker, and a treatment that improves detector scores by
drifting away from the voice has not solved the problem.
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
import detectors as det
import diffscore
import quality
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from textbank import EVAL_TEXTS
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"
TREATMENTS = ("raw", "denoise", "highpass", "normalise", "combined")


def _denoise(wav, sr):
    import noisereduce as nr
    return nr.reduce_noise(y=wav, sr=sr, stationary=False, prop_decrease=0.8)


def _highpass(wav, sr, cutoff=80.0):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff / (sr / 2), btype="highpass", output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def _normalise(wav, target_dbfs=-3.0):
    peak = float(np.abs(wav).max())
    if peak < 1e-8:
        return wav
    return (wav * (10 ** (target_dbfs / 20.0) / peak)).astype(np.float32)


def treat(src: Path, dst: Path, how: str) -> Path:
    wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if how == "denoise":
        wav = _denoise(wav, sr)
    elif how == "highpass":
        wav = _highpass(wav, sr)
    elif how == "normalise":
        wav = _normalise(wav)
    elif how == "combined":
        wav = _normalise(_highpass(_denoise(wav, sr), sr))
    sf.write(str(dst), np.asarray(wav, dtype=np.float32), sr)
    return dst


def score_audio(path, device, k, names):
    arr = det.score_file(path, device=device)
    s = {n: float(v) for n, v in zip(names, arr)}
    t = torch.tensor([s[n] for n in names], dtype=torch.float64)
    return s, float(diffscore.soft_min(list(t), k=k)), float(t.mean())


def agg(d):
    d = np.asarray(d)
    if len(d) < 2:
        return float(d.mean()) if len(d) else 0.0, float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+",
                    default=["evelynn", "thresh", "mordekaiser", "riven",
                             "swain", "jhin"])
    ap.add_argument("--texts", type=int, default=4)
    ap.add_argument("--treatments", default=",".join(TREATMENTS))
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--keep-wavs", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)
    treatments = [t.strip() for t in args.treatments.split(",")]
    texts = EVAL_TEXTS[:args.texts]

    work = GENERATORS_DIR / "reference_probe"
    work.mkdir(parents=True, exist_ok=True)
    cloners.load_model("CosyVoice2", device=device)

    print(f"{len(args.voices)} voices x {len(treatments)} treatments x "
          f"{len(texts)} sentences = "
          f"{len(args.voices)*len(treatments)*len(texts)} clones\n", flush=True)

    ref_rows, clone_rows = [], []
    for vi, voice in enumerate(args.voices):
        src, prompt = load_voice(voice)
        orig_emb = Tester.speaker_embedding(src)

        for how in treatments:
            tref = work / f"{voice}_ref_{how}.wav"
            treat(src, tref, how)

            s, sm, md = score_audio(tref, device, args.softmin_k, names)
            rsim = float(np.dot(Tester.speaker_embedding(tref), orig_emb))
            ref_rows.append({"voice": voice, "treatment": how, "det": s,
                             "softmin": sm, "meandet": md, "sim_to_orig": rsim})
            print(f"  {voice:<12} ref {how:<10} softmin {sm:.4f}  "
                  f"meandet {md:.4f}  sim {rsim:.3f}", flush=True)

            cloners.enroll(tref, "CosyVoice2", prompt_text=prompt,
                           alias=f"{voice}_{how}", device=device)
            for ti, text in enumerate(texts):
                seed = args.seed + vi * 100 + ti
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                p = work / f"{voice}_{how}_{ti}.wav"
                cloners.generate(text, tref, model="CosyVoice2",
                                 out_path=p, device=device)
                s, sm, md = score_audio(p, device, args.softmin_k, names)
                q = quality.measure(p, src, device)
                clone_rows.append({
                    "voice": voice, "treatment": how, "text_idx": ti,
                    "det": s, "softmin": sm, "meandet": md,
                    "sim": float(np.dot(Tester.speaker_embedding(p), orig_emb)),
                    "wer": Tester.wer(text, Tester.transcribe(
                        p, args.whisper, device))[0],
                    "pesq": q["pesq"], "stoi": q["stoi"]})
                if not args.keep_wavs:
                    p.unlink(missing_ok=True)
            if not args.keep_wavs and how != "raw":
                tref.unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    w = 12
    print("\n" + "=" * 96)
    print("THE REFERENCE FILES THEMSELVES (mean over voices)")
    print(f"{'treatment':<14}{'softmin':>{w}}{'meandet':>{w}}"
          + "".join(f"{n:>12}" for n in names))
    print("-" * 96)
    for how in treatments:
        rs = [r for r in ref_rows if r["treatment"] == how]
        print(f"{how:<14}{np.mean([r['softmin'] for r in rs]):>{w}.4f}"
              f"{np.mean([r['meandet'] for r in rs]):>{w}.4f}"
              + "".join(f"{np.mean([r['det'][n] for r in rs]):>12.4f}"
                        for n in names))

    print("\n" + "=" * 96)
    print("THE CLONES MADE FROM THEM (mean over voices and sentences)")
    print(f"{'treatment':<14}{'softmin':>{w}}{'meandet':>{w}}{'sim':>{w}}"
          f"{'WER':>{w}}{'PESQ':>{w}}")
    print("-" * 96)
    base = {(r["voice"], r["text_idx"]): r
            for r in clone_rows if r["treatment"] == "raw"}
    summary = []
    for how in treatments:
        rs = [r for r in clone_rows if r["treatment"] == how]
        rec = {"treatment": how, "n": len(rs)}
        for m in ("softmin", "meandet", "sim", "wer", "pesq", "stoi"):
            rec[m] = float(np.mean([r[m] for r in rs]))
        rec["perdet"] = {n: float(np.mean([r["det"][n] for r in rs]))
                         for n in names}
        summary.append(rec)
        print(f"{how:<14}{rec['softmin']:>{w}.4f}{rec['meandet']:>{w}.4f}"
              f"{rec['sim']:>{w}.4f}{rec['wer']*100:>{w-1}.1f}%"
              f"{rec['pesq']:>{w}.3f}")

    print("\nPAIRED vs raw reference, same voice and sentence")
    print(f"{'treatment':<14}{'d softmin':>12}{'t':>8}{'d meandet':>12}{'t':>8}"
          f"{'d sim':>10}{'d WER':>10}")
    print("-" * 96)
    for how in treatments:
        if how == "raw":
            continue
        pairs = [(r, base[(r["voice"], r["text_idx"])]) for r in clone_rows
                 if r["treatment"] == how and (r["voice"], r["text_idx"]) in base]
        if not pairs:
            continue
        m1, t1 = agg([a["softmin"] - b["softmin"] for a, b in pairs])
        m2, t2 = agg([a["meandet"] - b["meandet"] for a, b in pairs])
        ds = float(np.mean([a["sim"] - b["sim"] for a, b in pairs]))
        dw = float(np.mean([a["wer"] - b["wer"] for a, b in pairs]))
        print(f"{how:<14}{m1:>+12.4f}{t1:>8.2f}{m2:>+12.4f}{t2:>8.2f}"
              f"{ds:>+10.4f}{dw*100:>+9.1f}%")
        for rec in summary:
            if rec["treatment"] == how:
                rec.update({"d_softmin": m1, "t_softmin": t1,
                            "d_meandet": m2, "t_meandet": t2,
                            "d_sim": ds, "d_wer": dw})

    print("\nPER-VOICE clone soft-min")
    print(f"{'voice':<14}" + "".join(f"{h:>12}" for h in treatments))
    for voice in args.voices:
        print(f"{voice:<14}" + "".join(
            f"{np.mean([r['softmin'] for r in clone_rows if r['voice']==voice and r['treatment']==h]):>12.4f}"
            for h in treatments))

    out = Path(args.out) if args.out else RL_RUNS / "reference_probe.json"
    out.write_text(json.dumps({"voices": args.voices, "treatments": treatments,
                               "texts": texts, "detector_order": names,
                               "references": ref_rows, "clones": clone_rows,
                               "summary": summary}, indent=2, default=float),
                   encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
