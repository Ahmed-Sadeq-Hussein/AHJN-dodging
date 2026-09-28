r"""
quality.py -- perceptual quality metrics, the thing this project has been
missing. RUN IN THE "voiceclone" ENV.

    python quality.py --pairs base.wav edited.wav [more pairs...]
    python quality.py --dir "D:\results and benchmarks\generators\evelynn_demo"

Why this exists
---------------
The README's goal is to dodge detection "while maintaining MOS and WER".
Until now only WER and speaker similarity were tracked, and neither can see
the failure mode that matters most for an adversarial mel edit: buzz,
roughness, metallic ringing, phase smear. A clip can hold 0.86 cosine
similarity to the target speaker and 0% WER while sounding obviously broken,
because the speaker embedding is trained to be invariant to exactly the kind
of channel distortion an attack introduces.

torchaudio's SQUIM models fill the gap without a clean reference recording:

  SQUIM_OBJECTIVE   estimates PESQ, STOI and SI-SDR from the degraded signal
                    alone (no reference needed)
  SQUIM_SUBJECTIVE  estimates MOS given the signal and any non-matching
                    reference of clean speech

Both run at 16 kHz. Reported per clip, and as the delta between an edited
clip and its baseline, which is the number that says what the edit cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

SR = 16000
_OBJ = None
_SUBJ = None


def _models(device):
    global _OBJ, _SUBJ
    if _OBJ is None:
        _OBJ = SQUIM_OBJECTIVE.get_model().to(device).eval()
    if _SUBJ is None:
        _SUBJ = SQUIM_SUBJECTIVE.get_model().to(device).eval()
    return _OBJ, _SUBJ


def load16(path: Path, device) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    return wav.to(device)


def measure(path: Path, ref_path: Path | None = None, device="cuda") -> dict:
    """PESQ/STOI/SI-SDR estimates, plus a MOS estimate when a reference is given.

    PESQ, STOI and SI-SDR come from SQUIM_OBJECTIVE, which needs NO reference
    -- it estimates them from the degraded signal alone. Only the MOS estimate
    needs one, and only as a non-matching sample of clean speech for
    SQUIM_SUBJECTIVE; the natural choice here is the voice's own
    reference.wav, which is real recorded speech.

    Passing ref_path is therefore optional. Without it you still get the three
    objective numbers and "mos" is omitted.
    """
    obj_m, subj_m = _models(device)
    wav = load16(Path(path), device)
    ref = load16(Path(ref_path), device) if ref_path is not None else None

    # SQUIM is trained on utterance-length inputs; very long clips are
    # chunked and averaged rather than truncated, so a 60 s script is not
    # silently scored on its first few seconds.
    chunk = SR * 10
    n = wav.shape[-1]
    starts = list(range(0, max(n - chunk + 1, 1), chunk)) or [0]
    pesq = stoi = sisdr = mos = 0.0
    with torch.no_grad():
        for s in starts:
            w = wav[:, s:s + chunk]
            if w.shape[-1] < SR:                      # too short to score
                continue
            # SQUIM_OBJECTIVE returns (stoi, pesq, si_sdr) in that order --
            # getting it wrong reads a 4.03 PESQ as a 4.03 STOI, which is not
            # even in STOI's range, so the ordering is worth pinning here.
            st, p, sd = obj_m(w)
            pesq += float(p[0]); stoi += float(st[0]); sisdr += float(sd[0])
            if ref is not None:
                r = ref[:, :w.shape[-1]]
                if r.shape[-1] < w.shape[-1]:         # tile a short reference
                    reps = int(w.shape[-1] / r.shape[-1]) + 1
                    r = ref.repeat(1, reps)[:, :w.shape[-1]]
                mos += float(subj_m(w, r)[0])
    k = max(len(starts), 1)
    out = {"pesq": pesq / k, "stoi": stoi / k, "si_sdr": sisdr / k}
    if ref is not None:
        out["mos"] = mos / k
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="baseline.wav edited.wav [baseline2.wav edited2.wav ...]")
    ap.add_argument("--dir", default=None,
                    help="a demo dir: pairs up *_baseline.wav with *_designB.wav")
    ap.add_argument("--reference", default=None,
                    help="clean speech for the MOS estimate; defaults to the baseline clip")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pairs: list[tuple[str, Path, Path]] = []
    if args.dir:
        d = Path(args.dir)
        for b in sorted(d.glob("*_baseline.wav")):
            for suffix in ("designB", "pgd", "edited", "distill"):
                e = b.with_name(b.name.replace("_baseline", f"_{suffix}"))
                if e.is_file():
                    pairs.append((b.stem.replace("_baseline", ""), b, e))
                    break
    if args.pairs:
        it = iter(args.pairs)
        for b, e in zip(it, it):
            pairs.append((Path(b).stem, Path(b), Path(e)))
    if not pairs:
        print("nothing to measure")
        return 1

    ref = Path(args.reference) if args.reference else None
    w = 10
    print(f"{'clip':<22}{'arm':<10}{'MOS':>{w}}{'PESQ':>{w}}{'STOI':>{w}}{'SI-SDR':>{w}}")
    print("-" * 72)
    results = []
    for name, base, edit in pairs:
        r = ref or base
        mb = measure(base, r, device)
        me = measure(edit, r, device)
        print(f"{name:<22}{'baseline':<10}{mb['mos']:>{w}.3f}{mb['pesq']:>{w}.3f}"
              f"{mb['stoi']:>{w}.3f}{mb['si_sdr']:>{w}.2f}")
        print(f"{'':<22}{'edited':<10}{me['mos']:>{w}.3f}{me['pesq']:>{w}.3f}"
              f"{me['stoi']:>{w}.3f}{me['si_sdr']:>{w}.2f}")
        print(f"{'':<22}{'delta':<10}{me['mos'] - mb['mos']:>+{w}.3f}"
              f"{me['pesq'] - mb['pesq']:>+{w}.3f}{me['stoi'] - mb['stoi']:>+{w}.3f}"
              f"{me['si_sdr'] - mb['si_sdr']:>+{w}.2f}")
        print("-" * 72)
        results.append({"clip": name, "baseline": mb, "edited": me,
                        "d_mos": me["mos"] - mb["mos"],
                        "d_pesq": me["pesq"] - mb["pesq"],
                        "d_stoi": me["stoi"] - mb["stoi"]})

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
