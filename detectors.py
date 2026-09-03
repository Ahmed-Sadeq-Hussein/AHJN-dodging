r"""
detectors.py -- one entry point for scoring an audio file with the four
ASVspoof countermeasure models stored wherever VOICE_DETECTION_ROOT in .env points.

    from detectors import score_file, MODEL_ORDER

    scores = score_file(r"C:\clip.wav", [True, True, False, True])
    # -> array([0.98, 0.95,  nan, 0.87])

Score convention (as requested):

    1.0  ->  confidently HUMAN   (bonafide)
    0.0  ->  confidently AI      (spoofed / synthetic)

The value is P(bonafide) from the model's 2-class output. Index 1 is bonafide
in all four repos -- confirmed from aasist/data_utils.py, which encodes
``1 if label == "bonafide" else 0``, and from every repo's
``batch_score = batch_out[:, 1]``. Getting this backwards silently inverts
every result, so it is pinned per-model in _SPECS below rather than assumed.

Deselected models come back as nan so column indices stay stable across
calls -- convenient when accumulating a results table.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

# torch is imported lazily inside the functions that need it, so that
# check_install() and the selection/windowing helpers work before you have
# built the environment. (All annotations here are strings, courtesy of
# `from __future__ import annotations`, so they never touch torch either.)

# Where the detector repos and weights live -- read from env_config, which
# loads .env next to this file. No hardcoded machine-specific path here.
from env_config import VOICE_DETECTION_ROOT as ROOT

SAMPLE_RATE = 16_000
NB_SAMP = 64_600          # ~4.04 s -- the input length all four were trained on

MODEL_ORDER: tuple[str, ...] = ("AASIST", "AASIST-L", "RawNet2", "RawGAT-ST")


# --------------------------------------------------------------------------
# repo loading
# --------------------------------------------------------------------------

def _load_py(path: Path, mod_name: str):
    """Import a .py file under an explicit module name.

    Necessary because the repos collide: RawNet2 and RawGAT-ST each ship a
    top-level ``model.py``, and all three RawGAT variants define a class
    called ``RawGAT_ST`` in different files. Importing by filename would let
    them shadow each other in sys.modules.
    """
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if not path.is_file():
        raise FileNotFoundError(f"expected repo file missing: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_state_dict(net: torch.nn.Module, ckpt: Path, device: torch.device) -> None:
    """Load a checkpoint into ``net``, tolerating DataParallel prefixes.

    All four shipped checkpoints are bare state_dicts with no 'module.'
    prefix and no wrapper key (verified by unpickling their key lists), but
    anything you fine-tune yourself under nn.DataParallel will have one, so
    both shapes are handled.
    """
    import torch
    if not ckpt.is_file():
        raise FileNotFoundError(f"weights not found: {ckpt}")
    try:
        blob = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:                      # torch < 1.13 has no weights_only
        blob = torch.load(ckpt, map_location=device)

    if isinstance(blob, dict):
        for wrapper in ("state_dict", "model_state_dict", "model"):
            inner = blob.get(wrapper)
            if isinstance(inner, dict):
                blob = inner
                break
    if any(str(k).startswith("module.") for k in blob):
        blob = {str(k)[len("module."):]: v for k, v in blob.items()}

    missing, unexpected = net.load_state_dict(blob, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt.name}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"first missing={list(missing)[:3]} "
            f"first unexpected={list(unexpected)[:3]}"
        )


# --------------------------------------------------------------------------
# model builders
# --------------------------------------------------------------------------

def _build_aasist(device: torch.device, variant: str = "AASIST") -> torch.nn.Module:
    conf = json.loads((ROOT / "aasist" / "config" / f"{variant}.conf").read_text())
    mod = _load_py(ROOT / "aasist" / "models" / "AASIST.py", "_cm_aasist")
    net = mod.Model(conf["model_config"])
    _load_state_dict(net, ROOT / "aasist" / "models" / "weights" / f"{variant}.pth", device)
    return net


def _build_rawnet2(device: torch.device) -> torch.nn.Module:
    import yaml
    base = ROOT / "asvspoof2021-baselines" / "LA" / "Baseline-RawNet2"
    d_args = yaml.safe_load((base / "model_config_RawNet.yaml").read_text())["model"]
    mod = _load_py(base / "model.py", "_cm_rawnet2")
    net = mod.RawNet(d_args, device)
    # NB: the only pretrained RawNet2 is the DF-track model; there is no
    # official LA-track checkpoint.
    _load_state_dict(
        net, ROOT / "models" / "RawNet2-DF" / "pre_trained_DF_RawNet2.pth", device
    )
    return net


def _build_rawgat(device: torch.device, variant: str = "mul") -> torch.nn.Module:
    import yaml
    base = ROOT / "RawGAT-ST-antispoofing"
    d_args = yaml.safe_load((base / "model_config_RawGAT_ST.yaml").read_text())["model"]
    # the repo's top-level model.py *is* the 'mul' variant; add/concat live
    # in RawGAT_models/ and share the same class name.
    src = {
        "mul": base / "model.py",
        "add": base / "RawGAT_models" / "model_RawGAT_ST_add.py",
        "cat": base / "RawGAT_models" / "model_RawGAT_ST_concat.py",
    }[variant]
    mod = _load_py(src, f"_cm_rawgat_{variant}")
    net = mod.RawGAT_ST(d_args, device)
    ckpt_dir = {"mul": "RawGAT_ST_mul", "add": "RawGAT_ST_add", "cat": "RawGAT_ST_cat"}[variant]
    _load_state_dict(net, base / "Pre_trained_models" / ckpt_dir / "Best_epoch.pth", device)
    return net


@dataclass(frozen=True)
class _Spec:
    name: str
    build: Callable[[torch.device], torch.nn.Module]
    # forward() returns different shapes: AASIST gives (hidden, logits)
    unpack: Callable[[Any], torch.Tensor] = lambda out: out
    # 'logits'      -> needs softmax   (AASIST, AASIST-L, RawGAT-ST)
    # 'log_softmax' -> needs exp       (RawNet2 forward ends in self.logsoftmax)
    activation: str = "logits"
    forward_kwargs: Mapping[str, Any] = field(default_factory=dict)


_SPECS: dict[str, _Spec] = {
    "AASIST": _Spec(
        "AASIST",
        lambda d: _build_aasist(d, "AASIST"),
        unpack=lambda out: out[1],
        activation="logits",
    ),
    "AASIST-L": _Spec(
        "AASIST-L",
        lambda d: _build_aasist(d, "AASIST-L"),
        unpack=lambda out: out[1],
        activation="logits",
    ),
    "RawNet2": _Spec(
        "RawNet2",
        _build_rawnet2,
        activation="log_softmax",
    ),
    "RawGAT-ST": _Spec(
        "RawGAT-ST",
        _build_rawgat,
        activation="logits",
        forward_kwargs={"Freq_aug": False},
    ),
}

_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def get_model(name: str, device: torch.device) -> torch.nn.Module:
    """Build + load a model once, then reuse it. Loading is the slow part."""
    key = (name, str(device))
    if key not in _CACHE:
        net = _SPECS[name].build(device)
        net.to(device).eval()
        _CACHE[key] = net
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


# every file this module needs, relative to ROOT
_REQUIRED: dict[str, tuple[str, ...]] = {
    "AASIST": (
        r"aasist\config\AASIST.conf",
        r"aasist\models\AASIST.py",
        r"aasist\models\weights\AASIST.pth",
    ),
    "AASIST-L": (
        r"aasist\config\AASIST-L.conf",
        r"aasist\models\AASIST.py",
        r"aasist\models\weights\AASIST-L.pth",
    ),
    "RawNet2": (
        r"asvspoof2021-baselines\LA\Baseline-RawNet2\model.py",
        r"asvspoof2021-baselines\LA\Baseline-RawNet2\model_config_RawNet.yaml",
        r"models\RawNet2-DF\pre_trained_DF_RawNet2.pth",
    ),
    "RawGAT-ST": (
        r"RawGAT-ST-antispoofing\model.py",
        r"RawGAT-ST-antispoofing\model_config_RawGAT_ST.yaml",
        r"RawGAT-ST-antispoofing\Pre_trained_models\RawGAT_ST_mul\Best_epoch.pth",
    ),
}


def check_install(verbose: bool = True) -> dict[str, list[str]]:
    """Report which models ROOT can actually serve.

    Returns {model_name: [missing paths]}; an empty list means that model is
    ready. Worth calling first on a new machine -- otherwise a wrong ROOT
    surfaces much later as a FileNotFoundError mid-run.
    """
    missing = {
        name: [p for p in paths if not (ROOT / p).is_file()]
        for name, paths in _REQUIRED.items()
    }
    if verbose:
        print(f"ROOT = {ROOT}" + ("" if ROOT.is_dir() else "   <-- does not exist!"))
        for name in MODEL_ORDER:
            gaps = missing[name]
            if not gaps:
                print(f"  [ok]      {name}")
            else:
                print(f"  [MISSING] {name}")
                for p in gaps:
                    print(f"              {p}")
        if any(missing.values()):
            print("\nset VOICE_DETECTION_ROOT if the models live elsewhere.")
    return missing


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _read_any(path: Path) -> tuple[np.ndarray, int]:
    errors = []
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), always_2d=False)
        return np.asarray(x), int(sr)
    except Exception as exc:
        errors.append(f"soundfile: {exc}")
    try:
        import librosa
        x, sr = librosa.load(str(path), sr=None, mono=False)
        x = np.asarray(x)
        return (x.T if x.ndim > 1 else x), int(sr)
    except Exception as exc:
        errors.append(f"librosa: {exc}")
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        return wav.numpy().T, int(sr)
    except Exception as exc:
        errors.append(f"torchaudio: {exc}")
    raise RuntimeError(
        f"could not read {path}\n  " + "\n  ".join(errors)
        + "\ninstall soundfile (wav/flac) or librosa (mp3/m4a)"
    )


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    try:
        import soxr
        return soxr.resample(x, src, dst)
    except Exception:
        pass
    try:
        import librosa
        return librosa.resample(x, orig_sr=src, target_sr=dst)
    except Exception:
        pass
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any audio file as mono float32 at ``sr`` Hz."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    x, native = _read_any(path)
    if x.ndim > 1:
        x = x.mean(axis=1)                      # downmix to mono
    x = np.asarray(x, dtype=np.float32)
    if native != sr:
        x = np.asarray(_resample(x, native, sr), dtype=np.float32)
    if x.size == 0:
        raise ValueError(f"{path} decoded to zero samples")
    return x


def _windows(x: np.ndarray, nb_samp: int = NB_SAMP, hop: int | None = None) -> np.ndarray:
    """Slice into (n, nb_samp) windows.

    Short clips are tile-repeated up to nb_samp, matching the repos' own
    ``pad``. Long clips are windowed rather than truncated to the first 4 s,
    so the whole file contributes to the score.
    """
    hop = nb_samp if hop is None else hop
    if hop < 1:
        raise ValueError("hop must be >= 1")
    if x.shape[0] < nb_samp:
        reps = int(nb_samp / x.shape[0]) + 1
        return np.tile(x, reps)[:nb_samp][None, :]
    starts = list(range(0, x.shape[0] - nb_samp + 1, hop))
    if starts[-1] + nb_samp < x.shape[0]:
        starts.append(x.shape[0] - nb_samp)      # keep the tail
    return np.stack([x[s:s + nb_samp] for s in starts])


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def _flatten(seq: Iterable) -> list:
    out: list = []
    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _normalise_selection(models) -> list[bool]:
    """Accept a bool matrix/list, a dict, a name list, or None (= all)."""
    if models is None:
        return [True] * len(MODEL_ORDER)
    if isinstance(models, Mapping):
        unknown = set(models) - set(MODEL_ORDER)
        if unknown:
            raise KeyError(f"unknown model(s) {sorted(unknown)}; expected {list(MODEL_ORDER)}")
        return [bool(models.get(n, False)) for n in MODEL_ORDER]
    if isinstance(models, str):
        models = [models]
    flat = _flatten(models)
    if flat and all(isinstance(m, str) for m in flat):
        unknown = set(flat) - set(MODEL_ORDER)
        if unknown:
            raise KeyError(f"unknown model(s) {sorted(unknown)}; expected {list(MODEL_ORDER)}")
        return [n in set(flat) for n in MODEL_ORDER]
    if len(flat) != len(MODEL_ORDER):
        raise ValueError(
            f"expected {len(MODEL_ORDER)} flags for {list(MODEL_ORDER)}, got {len(flat)}"
        )
    return [bool(v) for v in flat]


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def score_file(
    audio_path: str | Path,
    models=None,
    *,
    device: str | torch.device | None = None,
    hop: int | None = None,
    max_windows: int | None = None,
    aggregate: str = "mean",
    batch_size: int = 8,
) -> np.ndarray:
    """Score one audio file with the selected detectors.

    Parameters
    ----------
    audio_path : any format your soundfile/librosa install can read.
    models     : which detectors to run. Accepts
                   [True, False, True, True]        aligned to MODEL_ORDER
                   [[True, False], [True, True]]    nested, gets flattened
                   {"AASIST": True, "RawNet2": True}
                   ["AASIST", "RawNet2"]
                   None                             all four
    hop        : window stride in samples. Defaults to NB_SAMP
                 (non-overlapping). Use NB_SAMP // 2 for denser coverage.
    max_windows: cap windows per file; None = use the whole file.
    aggregate  : 'mean' | 'min' | 'median' over windows. 'min' is the
                 paranoid setting -- one convincingly synthetic 4 s stretch
                 drags the whole file's score down.

    Returns
    -------
    np.ndarray, shape (4,), float64, aligned to MODEL_ORDER.
    Each entry is P(human) in [0, 1]; 1.0 = human, 0.0 = AI.
    Unselected models are nan.
    """
    import torch
    flags = _normalise_selection(models)
    if not any(flags):
        raise ValueError("no models selected")
    if aggregate not in ("mean", "min", "median"):
        raise ValueError(f"aggregate must be mean/min/median, got {aggregate!r}")

    device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    wav = load_audio(audio_path)
    wins = _windows(wav, NB_SAMP, hop)
    if max_windows is not None:
        wins = wins[:max_windows]
    batch_all = torch.from_numpy(wins).float()

    out = np.full(len(MODEL_ORDER), np.nan, dtype=np.float64)
    reducer = {"mean": np.mean, "min": np.min, "median": np.median}[aggregate]

    for idx, (name, want) in enumerate(zip(MODEL_ORDER, flags)):
        if not want:
            continue
        spec = _SPECS[name]
        net = get_model(name, device)

        probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, batch_all.shape[0], batch_size):
                chunk = batch_all[start:start + batch_size].to(device)
                raw = spec.unpack(net(chunk, **spec.forward_kwargs))
                if spec.activation == "logits":
                    p = torch.softmax(raw, dim=1)
                else:                       # already log-probabilities
                    p = torch.exp(raw)
                probs.append(p[:, 1].detach().cpu().numpy())   # index 1 = bonafide

        out[idx] = reducer(np.concatenate(probs))

    return np.clip(out, 0.0, 1.0)


def score_file_dict(audio_path, models=None, **kw) -> dict[str, float]:
    """Same as score_file but keyed by name, with unselected models omitted."""
    arr = score_file(audio_path, models, **kw)
    return {n: float(v) for n, v in zip(MODEL_ORDER, arr) if not np.isnan(v)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Score audio for AI-generated speech (1=human, 0=AI)"
    )
    ap.add_argument("audio", nargs="?", help="audio file to score")
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"subset of {list(MODEL_ORDER)}")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "min", "median"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify ROOT can find every model, then exit")
    args = ap.parse_args()

    if args.check:
        sys.exit(1 if any(check_install().values()) else 0)
    if not args.audio:
        ap.error("an audio file is required (or pass --check)")

    res = score_file_dict(args.audio, args.models,
                          aggregate=args.aggregate, device=args.device)
    width = max(len(k) for k in res)
    for k, v in res.items():
        bar = "#" * int(round(v * 30))
        print(f"{k:<{width}}  {v:.4f}  {bar:<30}  {'human' if v >= 0.5 else 'AI'}")
