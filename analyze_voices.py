r"""
analyze_voices.py -- why does the edit work on some voices and not others?
RUN IN "voiceclone".

    python analyze_voices.py --collect     # GPU: reference scores + edit spectra
    python analyze_voices.py               # analysis + figures from cached data

Across voices the same edit ranges from decisive to useless, and section 0
already suspected the cause is the reference recording rather than anything
about synthesis. This tests that directly: it measures acoustic properties of
each reference.wav and correlates them with how much the voice gains.

Measured per reference:
    duration, RMS level, crest factor  loudness and dynamics; heavy studio
                                       compression shows up as a low crest
    spectral centroid / rolloff85      overall brightness
    hf_fraction (>8 kHz)               energy above 8 kHz -- near zero means
                                       the recording was band-limited or
                                       codec-compressed, the clearest marker
                                       of processed game voice-over
    spectral flatness                  noisiness vs tonality
    est. SNR                           from the quietest vs loudest frames
    f0 median / spread                 pitch, as an acoustic property only
    clipping fraction                  samples at full scale
    the reference's OWN detector scores

Correlations use Spearman rho (monotone, robust to outliers) over at most 15
voices, so nothing here is conclusive on its own -- rho needs |rho| > ~0.51
for p < 0.05 at n = 15. Treat this as a map of what to test next, not proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import GRAPHS_DIR, RESULTS_ROOT, VOICE_DATA_ROOT

RL_RUNS = RESULTS_ROOT / "rl_runs"
PLAN_X = GRAPHS_DIR
CACHE = RL_RUNS / "voice_features.json"
SR_TARGET = 24000


# --------------------------------------------------------------------------
# reference acoustics
# --------------------------------------------------------------------------

def reference_features(path: Path) -> dict:
    import librosa

    y, sr = librosa.load(str(path), sr=None, mono=True)
    dur = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    peak = float(np.abs(y).max()) or 1e-9
    rms_mean = float(rms.mean()) or 1e-9

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    power = S ** 2
    total = float(power.sum()) or 1e-9

    # voiced frames only, so silence does not dominate the pitch statistics
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=2048)
    voiced = f0[(f0 > 60) & (f0 < 400)]

    # crude SNR: median energy of the quietest 10% of frames vs the loudest 10%
    q = np.quantile(rms, [0.1, 0.9])
    snr = 20 * np.log10((q[1] + 1e-9) / (q[0] + 1e-9))

    return {
        "sample_rate": int(sr),
        "duration_s": float(dur),
        "rms_dbfs": float(20 * np.log10(rms_mean)),
        "peak_dbfs": float(20 * np.log10(peak)),
        "crest_db": float(20 * np.log10(peak / rms_mean)),
        "centroid_hz": float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        "rolloff85_hz": float(librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.85).mean()),
        "hf_fraction_8k": float(power[freqs >= 8000].sum() / total),
        "hf_fraction_4k": float(power[freqs >= 4000].sum() / total),
        "flatness": float(librosa.feature.spectral_flatness(S=S).mean()),
        "est_snr_db": float(snr),
        "f0_median": float(np.median(voiced)) if len(voiced) else float("nan"),
        "f0_iqr": float(np.subtract(*np.percentile(voiced, [75, 25])))
        if len(voiced) else float("nan"),
        "clip_fraction": float(np.mean(np.abs(y) > 0.999)),
    }


def collect(voices, device):
    """GPU pass: reference detector scores, plus where the edit lands in frequency."""
    import torch
    import cloners, detectors as det, diffscore
    from eval_generalization import build_policy
    from mel_policy import MelResidualPolicy, mel_edit_hook
    from probe_amplify import apply_policy
    from textbank import EVAL_TEXTS
    from train_ppo_mel import load_voice

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift = diffscore.hift_forward_fn(cosy.model.hift)
    blob = torch.load(RL_RUNS / "distill_final" / "best.pt", map_location="cpu",
                      weights_only=False)
    net, _, _ = build_policy(blob, device, argparse.Namespace())
    net.eval()
    identity = MelResidualPolicy().to(device)
    work = RL_RUNS / "_voice_tmp"
    work.mkdir(parents=True, exist_ok=True)

    out = {}
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        rec = {"reference": reference_features(ref)}
        arr = det.score_file(ref, device=device)
        rec["reference_det"] = {n: float(v) for n, v in zip(det.MODEL_ORDER, arr)}

        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                       device=device)
        spectra, mags = [], []
        for ti, text in enumerate(EVAL_TEXTS[:2]):
            torch.manual_seed(97 + vi * 10 + ti)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(97 + vi * 10 + ti)
            with mel_edit_hook(cosy.model, identity, deterministic=True,
                               device=device) as r:
                cloners.generate(text, ref, model="CosyVoice2",
                                 out_path=work / "t.wav", device=device)
            if not r.mels:
                continue
            mel = r.mels[0].clone().to(device)
            _, delta = apply_policy(net, mel, hift, device, 8.0, 0.5,
                                    scale=1.0, iters=10)
            # mean |delta| per mel channel: where in frequency the edit acts
            spectra.append(delta.abs().mean(dim=(0, 2)).cpu().numpy())
            mags.append(float(delta.abs().mean()))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if spectra:
            rec["delta_spectrum"] = np.mean(spectra, axis=0).tolist()
            rec["delta_abs"] = float(np.mean(mags))
        out[voice] = rec
        print(f"  [{vi + 1}/{len(voices)}] {voice}", flush=True)

    for f in work.glob("*"):
        f.unlink()
    work.rmdir()
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"wrote {CACHE}")
    return out


# --------------------------------------------------------------------------

def spearman(x, y):
    """Spearman rho with a p-value, via scipy.

    An earlier hand-rolled version ranked with argsort(argsort(.)), which
    assigns distinct ranks to tied values. `clip_fraction` is identically zero
    for all 15 voices -- a constant with no information -- and that tie
    handling turned it into a perfect rho = -1.00. Constants are now dropped
    explicitly and ties get average ranks.
    """
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), len(x)
    r, p = stats.spearmanr(x, y)
    return float(r), float(p), len(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    gap_p = RL_RUNS / "gap_probe.json"
    if not gap_p.is_file():
        print(f"run probe_gap.py first (no {gap_p})")
        return 1
    gap = json.loads(gap_p.read_text(encoding="utf8"))
    per_voice = {r["voice"]: r for r in gap["per_voice"]}
    voices = list(per_voice)

    if args.collect or not CACHE.is_file():
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        feats = collect(voices, device)
    else:
        feats = json.loads(CACHE.read_text(encoding="utf8"))

    rows = []
    for v in voices:
        if v not in feats:
            continue
        g = per_voice[v]
        f = feats[v]["reference"]
        rows.append({
            "voice": v, "trained": g["trained"],
            "base": g["base_softmin"], "student": g["student_softmin"],
            "teacher": g["teacher_softmin"],
            "gain": g["student_softmin"] - g["base_softmin"],
            "teacher_gain": g["teacher_softmin"] - g["base_softmin"],
            "recovery": g.get("recovery_softmin", float("nan")),
            "base_md": g["base_meandet"], "student_md": g["student_meandet"],
            "gain_md": g["student_meandet"] - g["base_meandet"],
            "ref_softmin": float(np.min(list(feats[v]["reference_det"].values()))),
            "delta_abs": feats[v].get("delta_abs", float("nan")),
            **{k: f[k] for k in ("duration_s", "rms_dbfs", "crest_db",
                                 "centroid_hz", "rolloff85_hz", "hf_fraction_8k",
                                 "hf_fraction_4k", "flatness", "est_snr_db",
                                 "f0_median", "f0_iqr", "clip_fraction")}})
    rows.sort(key=lambda r: -r["gain"])

    w = 9
    print("\n" + "=" * 116)
    print("PER VOICE, best first by soft-min gain")
    print(f"{'voice':<13}{'set':<10}{'base':>7}{'stud':>7}{'teach':>7}{'gain':>8}"
          f"{'recov':>8}{'refSM':>7}{'hf8k':>8}{'crest':>7}{'SNR':>7}{'roll85':>8}{'f0':>6}")
    print("-" * 116)
    for r in rows:
        rec = f"{r['recovery'] * 100:>7.0f}%" if r["recovery"] == r["recovery"] else f"{'n/a':>8}"
        print(f"{r['voice']:<13}{'train' if r['trained'] else 'HELD-OUT':<10}"
              f"{r['base']:>7.3f}{r['student']:>7.3f}{r['teacher']:>7.3f}"
              f"{r['gain']:>+8.3f}{rec}{r['ref_softmin']:>7.3f}"
              f"{r['hf_fraction_8k']:>8.4f}{r['crest_db']:>7.1f}"
              f"{r['est_snr_db']:>7.1f}{r['rolloff85_hz']:>8.0f}{r['f0_median']:>6.0f}")

    print("\n" + "=" * 116)
    print("WHAT PREDICTS THE GAIN?  Spearman rho over voices "
          f"(n={len(rows)}; |rho| > 0.51 is p < 0.05 at n=15)")
    print("-" * 116)
    targets = [("soft-min gain", "gain"), ("mean-detector gain", "gain_md"),
               ("recovery vs teacher", "recovery")]
    predictors = ["base", "ref_softmin", "hf_fraction_8k", "hf_fraction_4k",
                  "rolloff85_hz", "centroid_hz", "crest_db", "est_snr_db",
                  "flatness", "rms_dbfs", "duration_s", "f0_median", "f0_iqr",
                  "clip_fraction", "delta_abs", "teacher_gain"]
    print(f"{'predictor':<20}" + "".join(f"{t[0]:>22}" for t in targets))
    print(f"{'':<20}" + "".join(f"{'(* = p < 0.05)':>22}" for _ in targets))
    corr = {}
    for p in predictors:
        line = f"{p:<20}"
        for _, key in targets:
            r, pv, n = spearman([x[p] for x in rows], [x[key] for x in rows])
            corr[f"{p}|{key}"] = {"rho": r, "p": pv, "n": n}
            if r != r:
                line += f"{'constant':>22}"
            else:
                mark = "*" if pv < 0.05 else " "
                line += f"{r:>+16.2f} p={pv:<4.2f}{mark}"
        print(line)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLAN_X.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Which voices the edit helps, and what explains it", fontsize=13)
        names = [r["voice"] for r in rows]
        xs = np.arange(len(rows))
        axes[0].barh(xs, [r["base"] for r in rows], color="#8a8f98", label="no edit")
        axes[0].barh(xs, [r["gain"] for r in rows], left=[r["base"] for r in rows],
                     color="#4a5fc1", label="added by the edit")
        axes[0].plot([r["teacher"] for r in rows], xs, "o", ms=5, color="#e07a1f",
                     label="teacher (PGD)")
        axes[0].set_yticks(xs); axes[0].set_yticklabels(names, fontsize=8)
        axes[0].invert_yaxis(); axes[0].set_xlabel("soft-min (1 = human)")
        axes[0].legend(fontsize=8); axes[0].grid(axis="x", alpha=0.25)
        for ax, key, lbl in ((axes[1], "hf_fraction_8k", "energy above 8 kHz (fraction)"),
                             (axes[2], "ref_softmin", "reference's own worst detector score")):
            ax.scatter([r[key] for r in rows], [r["gain"] for r in rows],
                       c=["#4a5fc1" if r["trained"] else "#c1444a" for r in rows], s=55)
            for r in rows:
                ax.annotate(r["voice"], (r[key], r["gain"]), fontsize=7,
                            textcoords="offset points", xytext=(4, 3))
            rho, pv, _ = spearman([r[key] for r in rows], [r["gain"] for r in rows])
            ax.set_xlabel(lbl); ax.set_ylabel("soft-min gain from the edit")
            ax.set_title(f"rho = {rho:+.2f}  (p = {pv:.2f})", fontsize=10, loc="left")
            ax.grid(alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = PLAN_X / "14_per_voice_analysis.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__}: {e})")

    (RL_RUNS / "voice_analysis.json").write_text(
        json.dumps({"rows": rows, "spearman": corr}, indent=2), encoding="utf8")
    print(f"wrote {RL_RUNS / 'voice_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
