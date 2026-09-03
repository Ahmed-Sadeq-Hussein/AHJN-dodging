r"""
Tester.py -- generation-quality metrics for the cloning side of the testbed.
RUN IN THE "voiceclone" ENV (it needs onnxruntime, whisper and matplotlib).

Two metrics, two figures:

  1. SIMILARITY  cosine similarity between speaker embeddings, over an array
                 of voices, rendered as an N x N heatmap. Answers "does the
                 clone actually sound like its reference, and is it further
                 from everyone else?"

  2. WER         word error rate of each clip against its known target text,
                 as a ranked table plus a bar chart ordered best-first.
                 Answers "is the generated speech actually intelligible?"

    conda activate voiceclone
    python Tester.py --voice riven                # clones + their references
    python Tester.py --voices riven ahmed david   # several speakers
    python Tester.py --refs-only                  # just the human references

Embeddings come from campplus.onnx, the CAM++ speaker-verification model that
already ships inside the CosyVoice2 checkpoint -- the very same embedding
CosyVoice2 conditions on. No extra model download, and similarity is measured
in the space the generator itself uses.

Transcription uses openai-whisper, which downloads its weights on first run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from env_config import (
    VOICE_DATA_ROOT as VOICE_DATA,
    GENERATORS_DIR as GENERATORS,
    METRICS_DIR,
    CAMPPLUS_ONNX,
)

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------
# clip collection
# --------------------------------------------------------------------------

@dataclass
class Clip:
    label: str          # what shows on the axes / in the table
    path: Path
    speaker: str        # voice folder name, e.g. "riven"
    source: str         # "reference" or a model name
    target_text: str | None = None   # ground truth, for WER


def collect_clips(voices: list[str], include_generated: bool = True) -> list[Clip]:
    clips: list[Clip] = []
    for voice in voices:
        ref = VOICE_DATA / voice / "reference.wav"
        txt = VOICE_DATA / voice / "reference.txt"
        if ref.is_file():
            clips.append(Clip(
                label=f"{voice} (ref)", path=ref, speaker=voice, source="reference",
                target_text=txt.read_text(encoding="utf8").strip() if txt.is_file() else None,
            ))
        if not include_generated:
            continue
        man = GENERATORS / f"manifest_{voice}.json"
        text = None
        if man.is_file():
            text = json.loads(man.read_text(encoding="utf8")).get("text")
        for model_dir in sorted(p for p in GENERATORS.iterdir() if p.is_dir()):
            if model_dir.name == METRICS_DIR.name:
                continue
            wav = model_dir / f"{voice}.wav"
            if wav.is_file():
                clips.append(Clip(
                    label=f"{voice} ({model_dir.name})", path=wav,
                    speaker=voice, source=model_dir.name, target_text=text,
                ))
    return clips


def discover_voices() -> list[str]:
    if not VOICE_DATA.is_dir():
        return []
    return sorted(
        p.name for p in VOICE_DATA.iterdir()
        if p.is_dir() and (p / "reference.wav").is_file()
    )


# --------------------------------------------------------------------------
# 1. speaker similarity
# --------------------------------------------------------------------------

_SESSION = None


def _campplus_session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime
        if not CAMPPLUS_ONNX.is_file():
            raise FileNotFoundError(f"campplus.onnx not found at {CAMPPLUS_ONNX}")
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        _SESSION = onnxruntime.InferenceSession(
            str(CAMPPLUS_ONNX), sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _SESSION


def speaker_embedding(path: Path) -> np.ndarray:
    """192-d CAM++ speaker embedding, L2-normalised.

    Mirrors CosyVoiceFrontEnd._extract_spk_embedding: 80-bin kaldi fbank,
    dither off, mean-subtracted over time.
    """
    import librosa
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if wav.size < SAMPLE_RATE // 10:
        raise ValueError(f"{path.name} is too short to embed ({wav.size} samples)")
    speech = torch.from_numpy(wav).unsqueeze(0)

    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=SAMPLE_RATE)
    feat = feat - feat.mean(dim=0, keepdim=True)

    sess = _campplus_session()
    emb = sess.run(None, {sess.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0]
    emb = np.asarray(emb).flatten().astype(np.float64)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def similarity_matrix(clips: list[Clip]) -> tuple[list[str], np.ndarray]:
    embs = np.stack([speaker_embedding(c.path) for c in clips])
    # embeddings are L2-normalised, so the dot product IS cosine similarity
    return [c.label for c in clips], embs @ embs.T


def plot_similarity_heatmap(labels: list[str], mat: np.ndarray, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    size = max(6.0, 0.62 * n + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))

    # cosine similarity of speaker embeddings is effectively in [0, 1] here;
    # fixing the scale keeps runs comparable instead of auto-stretching
    im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Speaker similarity (CAM++ cosine)\n1.00 = same voice, ~0 = unrelated",
                 fontsize=11, pad=12)

    if n <= 24:                     # annotations stop being legible past this
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v < 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.3", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. word error rate
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalise_text(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def wer(reference: str, hypothesis: str) -> tuple[float, int, int, int, int]:
    """Word error rate by Levenshtein alignment.

    Returns (wer, substitutions, deletions, insertions, n_reference_words).
    Implemented directly rather than pulling in jiwer -- it is a short DP and
    one less dependency to pin.
    """
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return (float(m > 0), 0, 0, m, 0)

    # d[i][j] = edit distance between ref[:i] and hyp[:j]; back[i][j] = op
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])

    # walk back to split the distance into S / D / I
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            S += 1; i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return ((S + D + I) / n, S, D, I, n)


_ASR = None


def transcribe(path: Path, model_size: str = "base", device: str | None = None) -> str:
    """Transcribe one clip with whisper.

    Audio is decoded here with librosa and handed to whisper as a float32
    array. Passing a PATH instead would send whisper down its own load_audio(),
    which shells out to the ffmpeg binary -- not present on this machine, and
    an unnecessary system dependency when librosa already reads these files.
    """
    global _ASR
    import librosa
    import whisper
    if _ASR is None:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  loading whisper '{model_size}' on {dev} (downloads on first use)...")
        _ASR = whisper.load_model(model_size, device=dev)
    # whisper expects mono float32 at exactly 16 kHz
    wav, _ = librosa.load(str(path), sr=16_000, mono=True)
    return _ASR.transcribe(wav.astype(np.float32), fp16=False, language="en")["text"].strip()


def plot_wer_ranking(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["wer"])          # best (lowest) first
    labels = [r["label"] for r in rows]
    vals = [r["wer"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.45 * len(rows) + 1.6)))
    ypos = np.arange(len(rows))
    # best at the top
    colors = ["#2a9d8f" if v <= 10 else "#e9c46a" if v <= 25 else "#e76f51" for v in vals]
    ax.barh(ypos, vals, color=colors)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("word error rate (%)  -- lower is better")
    ax.set_title("Intelligibility ranking (whisper transcript vs target text)",
                 fontsize=11, pad=10)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.015 + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(max(vals) * 1.18, 5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Similarity heatmap + WER ranking")
    ap.add_argument("--voice", default=None, help="single voice (shorthand for --voices)")
    ap.add_argument("--voices", nargs="*", default=None,
                    help="voices to include; default = every folder with a reference.wav")
    ap.add_argument("--refs-only", action="store_true", help="skip generated clips")
    ap.add_argument("--whisper", default="base",
                    choices=["tiny", "base", "small", "medium", "large"])
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-similarity", action="store_true")
    ap.add_argument("--out", default=str(METRICS_DIR))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    voices = args.voices or ([args.voice] if args.voice else discover_voices())
    if not voices:
        print(f"no voices found under {VOICE_DATA}")
        return 1

    clips = collect_clips(voices, include_generated=not args.refs_only)
    if not clips:
        print("no clips found")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = voices[0] if len(voices) == 1 else f"{len(voices)}voices"

    print(f"{len(clips)} clip(s):")
    for c in clips:
        print(f"  {c.label:<28} {c.path}")

    # ---- similarity ----
    if not args.skip_similarity:
        if len(clips) < 2:
            print("\nneed at least 2 clips for a similarity matrix; skipping")
        else:
            print("\n=== speaker similarity (CAM++ cosine) ===")
            labels, mat = similarity_matrix(clips)
            w = max(len(l) for l in labels) + 1
            print(" " * w + "".join(f"{i:>7}" for i in range(len(labels))))
            for i, lab in enumerate(labels):
                print(f"{lab:<{w}}" + "".join(f"{mat[i, j]:>7.3f}" for j in range(len(labels))))

            png = out_dir / f"similarity_{tag}_{stamp}.png"
            plot_similarity_heatmap(labels, mat, png)
            csv_path = out_dir / f"similarity_{tag}_{stamp}.csv"
            with csv_path.open("w", newline="", encoding="utf8") as fh:
                w_ = csv.writer(fh)
                w_.writerow([""] + labels)
                for lab, row in zip(labels, mat):
                    w_.writerow([lab] + [round(float(v), 6) for v in row])
            print(f"\nwrote {png}\nwrote {csv_path}")

            # the number that actually matters: clone vs its own reference
            pairs = []
            for i, ci in enumerate(clips):
                if ci.source == "reference":
                    continue
                for j, cj in enumerate(clips):
                    if cj.source == "reference" and cj.speaker == ci.speaker:
                        pairs.append((ci.label, cj.label, float(mat[i, j])))
            if pairs:
                print("\nclone vs its own reference:")
                for a, b, v in pairs:
                    verdict = ("same speaker" if v >= 0.70 else
                               "borderline" if v >= 0.50 else "different speaker")
                    print(f"  {a:<28} vs {b:<20} {v:.3f}   {verdict}")

    # ---- WER ----
    if not args.skip_wer:
        scored = [c for c in clips if c.target_text]
        if not scored:
            print("\nno target text available for any clip; skipping WER")
        else:
            print(f"\n=== word error rate ({len(scored)} clip(s)) ===")
            rows = []
            for c in scored:
                hyp = transcribe(c.path, args.whisper, args.device)
                rate, S, D, I, N = wer(c.target_text, hyp)
                rows.append({
                    "label": c.label, "speaker": c.speaker, "source": c.source,
                    "wer": round(rate, 6), "sub": S, "del": D, "ins": I, "ref_words": N,
                    "target": c.target_text, "transcript": hyp, "path": str(c.path),
                })
                print(f"  {c.label:<28} WER {rate*100:6.2f}%   S{S} D{D} I{I} / {N} words")
                print(f"      heard: {hyp!r}")

            rows.sort(key=lambda r: r["wer"])
            print("\nranking (best first):")
            for i, r in enumerate(rows, 1):
                print(f"  {i:>2}. {r['label']:<28} {r['wer']*100:6.2f}%")

            png = out_dir / f"wer_{tag}_{stamp}.png"
            plot_wer_ranking(rows, png)
            csv_path = out_dir / f"wer_{tag}_{stamp}.csv"
            with csv_path.open("w", newline="", encoding="utf8") as fh:
                w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w_.writeheader(); w_.writerows(rows)
            json_path = out_dir / f"wer_{tag}_{stamp}.json"
            json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
            print(f"\nwrote {png}\nwrote {csv_path}\nwrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
