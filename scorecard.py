r"""
scorecard.py -- every metric for one clip, from one call. RUN IN "voiceclone".

    from scorecard import score_clip
    s = score_clip("clip.wav", reference="reference.wav", text="what it says")
    print(s["softmin"], s["sim"], s["wer"], s["pesq"])

    python scorecard.py clip.wav --reference reference.wav --text "what it says"
    python scorecard.py out/*.wav --reference reference.wav --text-file lines.txt --json out.json

A clip's score touches four separate model stacks, and until now every caller
wired them up itself -- mos_test.py, eval_all15.py, longform.py and half the
probes each repeated the same block, which is how the WER-against-markup bug
in demo_best.py survived as long as it did. This is that block, once.

    detection     detectors.py          AASIST, AASIST-L, RawNet2, RawGAT-ST
                                        P(human) per model, 1 = human
    similarity    Tester.py             CAM++ (campplus.onnx) cosine
    WER           Tester.py             whisper transcript vs the target text
    perceptual    quality.py            torchaudio SQUIM: PESQ, STOI, SI-SDR

What you get back depends on what you pass in. `text` unlocks WER.
`reference` unlocks similarity, and a MOS estimate on top of the perceptual
metrics. PESQ, STOI and SI-SDR need NO reference -- SQUIM_OBJECTIVE estimates
them from the degraded signal alone -- so they are computed for every clip,
including a reference recording being scored on its own.

Nothing here raises if a metric cannot be produced. A missing model, a clip
too short for a detector, a whisper failure -- each leaves its keys absent and
records why under "problems", because a run that scores 200 clips should not
die on the one that is 0.4 s long.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detectors as det
import diffscore

__all__ = ["score_clip", "score_many", "SOFTMIN_K"]

SOFTMIN_K = 8.0     # the k every evaluation in this repo reports soft-min at


def _softmin(values, k: float = SOFTMIN_K) -> float:
    t = torch.tensor([float(v) for v in values], dtype=torch.float64)
    return float(diffscore.soft_min(list(t), k=k))


def score_clip(path, reference=None, text=None, *, device=None,
               whisper_model: str = "base", models=None, quality: bool = True,
               softmin_k: float = SOFTMIN_K, ref_embedding=None) -> dict:
    """Every available metric for one audio file.

    path          the clip to score
    reference     the speaker's reference recording; enables sim + PESQ/STOI
    text          what the clip is meant to say; enables WER
    models        detector subset, default all four (see detectors.MODEL_ORDER)
    quality       set False to skip SQUIM, which is the slowest part
    ref_embedding a precomputed CAM++ embedding for `reference`; pass it when
                  scoring many clips against the same speaker and you save one
                  onnx pass per clip

    Returns a flat dict. Keys that could not be computed are absent, and the
    reason is in `problems`.
    """
    path = Path(path)
    out: dict = {"file": str(path), "problems": []}

    # ---- detection -------------------------------------------------------
    try:
        scores = det.score_file_dict(path, models, device=device)
        if scores:
            out["det"] = scores
            out["softmin"] = _softmin(scores.values(), softmin_k)
            out["meandet"] = float(np.mean(list(scores.values())))
            out["worst"] = min(scores, key=scores.get)
        else:
            out["problems"].append("no detector returned a score")
    except Exception as e:
        out["problems"].append(f"detectors: {e}")

    # ---- similarity ------------------------------------------------------
    if reference is not None:
        try:
            import Tester
            ref_emb = ref_embedding if ref_embedding is not None \
                else Tester.speaker_embedding(Path(reference))
            out["sim"] = float(np.dot(Tester.speaker_embedding(path), ref_emb))
        except Exception as e:
            out["problems"].append(f"similarity: {e}")

    # ---- WER -------------------------------------------------------------
    if text is not None:
        try:
            import cloners
            import Tester
            # Score against the SPOKEN words. CosyVoice2 prosody tags
            # (<strong>, [breath], ...) are not spoken, and comparing whisper's
            # transcript against the raw markup inflated WER from ~20% to ~60%
            # in an earlier version of demo_best.py.
            target = cloners.strip_markup(text)
            hyp = Tester.transcribe(path, whisper_model, device)
            rate, sub, dele, ins, n_words = Tester.wer(target, hyp)
            out.update({"wer": float(rate), "wer_sub": sub, "wer_del": dele,
                        "wer_ins": ins, "wer_words": n_words, "transcript": hyp})
        except Exception as e:
            out["problems"].append(f"wer: {e}")

    # ---- perceptual quality ---------------------------------------------
    # No reference needed: SQUIM_OBJECTIVE is reference-free. A reference adds
    # only the MOS estimate, which SQUIM_SUBJECTIVE wants a non-matching
    # sample of clean speech for.
    if quality:
        try:
            import quality as Q
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            out.update(Q.measure(path, Path(reference) if reference else None, dev))
        except Exception as e:
            out["problems"].append(f"quality: {e}")

    if not out["problems"]:
        out.pop("problems")
    return out


def score_many(paths, reference=None, texts=None, *, device=None, **kw) -> list:
    """score_clip over several files, reusing the speaker embedding.

    texts: one string for all clips, a list parallel to `paths`, or None.
    """
    paths = [Path(p) for p in paths]
    if isinstance(texts, str) or texts is None:
        texts = [texts] * len(paths)
    if len(texts) != len(paths):
        raise ValueError(f"{len(texts)} texts for {len(paths)} clips")

    ref_emb = None
    if reference is not None:
        try:
            import Tester
            ref_emb = Tester.speaker_embedding(Path(reference))
        except Exception:
            ref_emb = None      # score_clip will report it per clip

    rows = []
    for p, t in zip(paths, texts):
        rows.append(score_clip(p, reference, t, device=device,
                               ref_embedding=ref_emb, **kw))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _fmt(row: dict) -> str:
    bits = [Path(row["file"]).name]
    if "softmin" in row:
        bits.append(f'soft-min {row["softmin"]:.3f}')
        bits.append("[" + " ".join(f"{n} {v:.2f}" for n, v in row["det"].items()) + "]")
    if "sim" in row:
        bits.append(f'sim {row["sim"]:.3f}')
    if "wer" in row:
        bits.append(f'WER {row["wer"] * 100:.1f}%')
    if "pesq" in row:
        bits.append(f'PESQ {row["pesq"]:.2f}')
    if "stoi" in row:
        bits.append(f'STOI {row["stoi"]:.3f}')
    if row.get("problems"):
        bits.append("| " + "; ".join(row["problems"]))
    return "  ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio", nargs="+", help="clip(s) to score")
    ap.add_argument("--reference", default=None, help="enables similarity and PESQ/STOI")
    ap.add_argument("--text", default=None, help="enables WER")
    ap.add_argument("--text-file", default=None, help="one target line per clip")
    ap.add_argument("--no-quality", action="store_true", help="skip SQUIM, the slow part")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--json", default=None, help="write every row here")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    texts = args.text
    if args.text_file:
        texts = [l.strip() for l in Path(args.text_file).read_text(encoding="utf8")
                 .splitlines() if l.strip()]

    rows = score_many(args.audio, args.reference, texts, device=args.device,
                      whisper_model=args.whisper, models=args.models,
                      quality=not args.no_quality)
    for r in rows:
        print(_fmt(r))

    if len(rows) > 1:
        got = lambda k: [r[k] for r in rows if k in r]
        print("-" * 60)
        for k, label, scale in [("softmin", "soft-min", 1), ("sim", "similarity", 1),
                                ("wer", "WER", 100), ("pesq", "PESQ", 1)]:
            v = got(k)
            if v:
                unit = "%" if scale == 100 else ""
                print(f"  mean {label:<11}{np.mean(v) * scale:.3f}{unit}  (n={len(v)})")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf8")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
