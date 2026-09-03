r"""
cloners.py -- the generation half of the testbed, mirroring detectors.py.

Wraps the two voice-cloning stacks (wherever VOICE_CLONING_ROOT in .env points)
behind one pair of
functions:

    from cloners import enroll, generate

    enroll(r"D:\refs\alice.wav", "CosyVoice2", prompt_text="this is alice speaking")
    generate("Hello there, this is a test.", r"D:\refs\alice.wav")
    # -> WindowsPath('D:/Voicetune vs Voicefake/out/alice__CosyVoice2__0001.wav')

A note on "training"
--------------------
Neither model is trained by ``enroll``. Both are zero-shot: they extract a
speaker representation from the reference clip and condition on it at
synthesis time. No weights change, nothing is written to the checkpoints,
and enrolling a hundred voices costs a hundred short forward passes. The
function is called ``enroll`` rather than ``train`` for that reason -- what
it builds is a lookup entry, not a fine-tuned model.

What each stack actually does with the reference clip:

  CosyVoice2   tokenizes the reference speech, extracts a speaker embedding,
               and stores the bundle in the model's own ``spk2info`` table
               under an id. Synthesis conditions the LLM + flow on it.

  OpenVoiceV2  extracts a 'tone colour' embedding with the converter's
               reference encoder. Synthesis is two-stage: MeloTTS speaks the
               text in a stock base voice, then the converter reshapes that
               audio toward the enrolled embedding.

That asymmetry is why generate() needs to know which model a voice was
enrolled on -- the two produce completely different objects.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# torch and the repo code are imported lazily inside the functions that need
# them, so check_install() works before the environment is built.

# Where the cloning repos and weights live -- read from env_config, which
# loads .env next to this file. No hardcoded machine-specific path here.
from env_config import VOICE_CLONING_ROOT as ROOT

COSY_REPO = ROOT / "CosyVoice"
COSY_MODEL = ROOT / "models" / "CosyVoice2-0.5B"
OV_REPO = ROOT / "OpenVoice"
OV_CKPT = ROOT / "models" / "OpenVoiceV2"
MELO_REPO = ROOT / "MeloTTS"

MODEL_ORDER: tuple[str, ...] = ("CosyVoice2", "OpenVoiceV2")

# Temporary restriction: only CosyVoice2 is callable right now. OpenVoiceV2's
# code (_load_openvoice2, the OpenVoiceV2 branches in enroll()/generate())
# is untouched -- flip this back to MODEL_ORDER to re-enable it.
ENABLED_MODELS: tuple[str, ...] = ("CosyVoice2",)

# where generated audio lands, unless you pass out_path.
from env_config import GENERATORS_DIR as OUTPUT_DIR


class CloneError(RuntimeError):
    """Base class for this module's errors."""


class VoiceNotEnrolledError(CloneError):
    """generate() was given a reference clip that enroll() never saw."""


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

@dataclass
class Voice:
    """One reference clip, enrolled on one model."""
    audio_path: Path            # resolved, absolute -- the registry key
    model: str
    alias: str
    payload: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.model, str(self.audio_path).lower())


_VOICES: dict[tuple[str, str], Voice] = {}
_MODELS: dict[tuple[str, str], Any] = {}


def _resolve_audio(audio: str | Path) -> Path:
    return Path(audio).expanduser().resolve()


def list_voices(model: str | None = None) -> list[Voice]:
    """Every enrolled voice, optionally filtered to one model."""
    out = list(_VOICES.values())
    if model is not None:
        out = [v for v in out if v.model == _canon_model(model)]
    return sorted(out, key=lambda v: (v.model, v.alias))


def _canon_model(name: str) -> str:
    """Accept sloppy spellings: 'cosyvoice2', 'cosy', 'openvoice', 'ov2'."""
    if name in MODEL_ORDER:
        return name
    squashed = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    table = {
        "cosyvoice2": "CosyVoice2", "cosyvoice": "CosyVoice2",
        "cosy": "CosyVoice2", "cv2": "CosyVoice2",
        "openvoicev2": "OpenVoiceV2", "openvoice2": "OpenVoiceV2",
        "openvoice": "OpenVoiceV2", "ov2": "OpenVoiceV2", "ov": "OpenVoiceV2",
    }
    if squashed not in table:
        raise KeyError(f"unknown model {name!r}; expected one of {list(MODEL_ORDER)}")
    return table[squashed]


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

def _add_syspath(*paths: Path) -> None:
    for p in paths:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _pick_device(device: str | None) -> str:
    import torch
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cosyvoice2(device: str, **kw) -> Any:
    # CosyVoice needs both its own root and the vendored Matcha-TTS on the
    # path; this is what the repo's own README tells you to export.
    _add_syspath(COSY_REPO, COSY_REPO / "third_party" / "Matcha-TTS")
    from cosyvoice.cli.cosyvoice import CosyVoice2
    if not COSY_MODEL.is_dir():
        raise CloneError(f"CosyVoice2 weights not found at {COSY_MODEL}")
    # fp16 only makes sense on GPU
    fp16 = kw.pop("fp16", device.startswith("cuda"))
    return CosyVoice2(str(COSY_MODEL), load_jit=False, load_trt=False, fp16=fp16)


def _load_openvoice2(device: str, *, enable_watermark: bool = False, **kw) -> Any:
    """Load the tone-colour converter.

    enable_watermark defaults to False, unlike the upstream demo. Two
    reasons: it drops the ``wavmark`` dependency, and -- more importantly for
    this testbed -- an embedded watermark is a signal the anti-spoofing
    models could pick up on, which would flatter the detectors for the wrong
    reason. Set it True if you actually want watermarked output.
    """
    _add_syspath(OV_REPO)
    from openvoice.api import ToneColorConverter
    cfg = OV_CKPT / "converter" / "config.json"
    ckpt = OV_CKPT / "converter" / "checkpoint.pth"
    for p in (cfg, ckpt):
        if not p.is_file():
            raise CloneError(f"OpenVoiceV2 converter file missing: {p}")
    # NOTE: passing enable_watermark= to the constructor does not work in this
    # build. ToneColorConverter.__init__ forwards **kwargs to
    # OpenVoiceBaseClass.__init__, which only accepts (config_path, device),
    # so the kwarg raises TypeError before it is ever read. Construct plainly
    # and null the model instead -- add_watermark() returns audio untouched
    # when watermark_model is None.
    conv = ToneColorConverter(str(cfg), device=device)
    if not enable_watermark:
        conv.watermark_model = None
    conv.load_ckpt(str(ckpt))
    return conv


def load_model(model: str, *, device: str | None = None, **kw) -> Any:
    """Load (and cache) one cloning model. Loading dominates runtime."""
    name = _canon_model(model)
    if name not in ENABLED_MODELS:
        raise CloneError(
            f"{name} is temporarily disabled; only {list(ENABLED_MODELS)} is enabled "
            f"right now. Its code is untouched -- edit ENABLED_MODELS in cloners.py "
            f"to bring it back."
        )
    device = _pick_device(device)
    key = (name, device)
    if key not in _MODELS:
        loader = {"CosyVoice2": _load_cosyvoice2, "OpenVoiceV2": _load_openvoice2}[name]
        _MODELS[key] = loader(device, **kw)
    return _MODELS[key]


def loaded_models() -> list[str]:
    return sorted({name for name, _ in _MODELS})


def unload_all() -> None:
    _MODELS.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# 1. enrol a zero-shot reference clip
# --------------------------------------------------------------------------

def enroll(
    audio_path: str | Path,
    model: str,
    *,
    prompt_text: str | None = None,
    alias: str | None = None,
    device: str | None = None,
    use_vad: bool = False,
    overwrite: bool = True,
) -> Voice:
    """Register a reference clip with one cloning model.

    Parameters
    ----------
    audio_path  : the voice to clone. Loaded at 16 kHz mono.
    model       : "CosyVoice2" or "OpenVoiceV2" (loaded on demand if needed).
    prompt_text : CosyVoice2 only, and worth supplying -- it is the literal
                  transcript of ``audio_path``. CosyVoice2's zero-shot mode
                  conditions on (reference audio, reference text) together.
                  Without it this falls back to cross-lingual mode, which
                  needs no transcript but generally clones less faithfully.
                  Ignored by OpenVoiceV2, which never needs a transcript.
    alias       : friendly name; defaults to the file stem.
    use_vad     : OpenVoiceV2 only. False (default) embeds the clip directly.
                  True routes through the upstream se_extractor, which trims
                  silence with a Silero VAD for a cleaner embedding but drags
                  in faster-whisper, whisper-timestamped, pydub and ffmpeg.

    Returns
    -------
    Voice -- also retrievable later via list_voices().
    """
    import torch

    name = _canon_model(model)
    path = _resolve_audio(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"reference audio not found: {path}")

    alias = alias or path.stem
    device = _pick_device(device)
    net = load_model(name, device=device)

    if name == "CosyVoice2":
        _add_syspath(COSY_REPO, COSY_REPO / "third_party" / "Matcha-TTS")
        # Pass the PATH, not a preloaded tensor. This build's frontend calls
        # load_wav() itself, at three different rates -- 16 kHz for the speech
        # tokenizer and the campplus speaker embedding, 24 kHz for the flow's
        # mel features. Handing it a tensor raises inside soundfile, and
        # pre-loading at any single rate would starve one of those stages.
        prompt_wav = str(path)

        payload: dict[str, Any] = {"prompt_wav": prompt_wav, "prompt_text": prompt_text}
        if prompt_text:
            # store it inside the model's own speaker table, so synthesis can
            # skip re-encoding the reference every call
            spk_id = f"{alias}__{abs(hash(str(path))) % 10**8:08d}"
            net.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)
            payload["spk_id"] = spk_id
        else:
            payload["spk_id"] = ""

    elif name == "OpenVoiceV2":
        if use_vad:
            _add_syspath(OV_REPO)
            from openvoice import se_extractor
            target_se, _ = se_extractor.get_se(
                str(path), net, target_dir=str(OUTPUT_DIR / "_se"), vad=True
            )
        else:
            target_se = net.extract_se([str(path)])
        payload = {"target_se": target_se}

    else:                                            # pragma: no cover
        raise KeyError(name)

    voice = Voice(audio_path=path, model=name, alias=alias, payload=payload)
    if voice.key in _VOICES and not overwrite:
        raise CloneError(f"{path.name} is already enrolled on {name}; pass overwrite=True")
    _VOICES[voice.key] = voice
    return voice


# --------------------------------------------------------------------------
# 2. generate speech in an enrolled voice
# --------------------------------------------------------------------------

def _lookup(audio: str | Path, model: str | None) -> Voice:
    """Find the enrolled voice for this clip, or explain what went wrong."""
    path = _resolve_audio(audio)
    hits = [v for v in _VOICES.values() if str(v.audio_path).lower() == str(path).lower()]

    if model is not None:
        want = _canon_model(model)
        hits = [v for v in hits if v.model == want]
        if not hits:
            enrolled = list_voices(want)
            raise VoiceNotEnrolledError(
                f"{path.name!r} has not been enrolled on {want}.\n"
                f"  call enroll({str(path)!r}, {want!r}) first.\n"
                f"  currently enrolled on {want}: "
                f"{[v.alias for v in enrolled] or 'nothing'}"
            )
        return hits[0]

    if not hits:
        raise VoiceNotEnrolledError(
            f"{path.name!r} has not been enrolled on any model.\n"
            f"  call enroll({str(path)!r}, <model>) first.\n"
            f"  currently enrolled: "
            f"{[f'{v.alias} ({v.model})' for v in list_voices()] or 'nothing'}"
        )
    if len(hits) > 1:
        raise CloneError(
            f"{path.name!r} is enrolled on more than one model "
            f"({[v.model for v in hits]}); pass model= to choose."
        )
    return hits[0]


def _next_out_path(voice: Voice, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        p = out_dir / f"{voice.alias}__{voice.model}__{n:04d}.wav"
        if not p.exists():
            return p
        n += 1


def generate(
    text: str,
    audio: str | Path,
    *,
    model: str | None = None,
    out_path: str | Path | None = None,
    speed: float = 1.0,
    device: str | None = None,
    language: str = "EN",
    speaker: str | None = None,
    tau: float = 0.3,
) -> Path:
    """Speak ``text`` in the voice of a previously enrolled clip.

    Parameters
    ----------
    text     : what to say.
    audio    : the reference clip you passed to enroll(). Raises
               VoiceNotEnrolledError if that clip was never enrolled.
    model    : only needed to disambiguate when the same clip is enrolled on
               both models.
    language : OpenVoiceV2 only -- which MeloTTS base model speaks the text
               first. One of EN, ES, FR, ZH, JP, KR (and EN_NEWEST).
    speaker  : OpenVoiceV2 only -- which base speaker within that language,
               e.g. 'EN-Newest', 'EN-US'. Defaults to the first available.
    tau      : OpenVoiceV2 conversion strength.

    Returns
    -------
    Path to the written .wav.
    """
    import torch

    if not text or not text.strip():
        raise ValueError("text is empty")

    voice = _lookup(audio, model)
    device = _pick_device(device)
    net = load_model(voice.model, device=device)
    out = Path(out_path).expanduser().resolve() if out_path else _next_out_path(voice, OUTPUT_DIR)
    out.parent.mkdir(parents=True, exist_ok=True)

    if voice.model == "CosyVoice2":
        import soundfile as sf
        spk_id = voice.payload.get("spk_id") or ""
        prompt_text = voice.payload.get("prompt_text")
        prompt_wav = voice.payload["prompt_wav"]

        if prompt_text:
            chunks = net.inference_zero_shot(
                text, prompt_text, prompt_wav,
                zero_shot_spk_id=spk_id, stream=False, speed=speed,
            )
        else:
            # no transcript available -- cross-lingual mode conditions on the
            # reference audio alone
            chunks = net.inference_cross_lingual(
                text, prompt_wav, zero_shot_spk_id=spk_id, stream=False, speed=speed,
            )

        pieces = [c["tts_speech"] for c in chunks]
        if not pieces:
            raise CloneError("CosyVoice2 produced no audio")
        wav = torch.cat(pieces, dim=1).squeeze(0).cpu().numpy()
        sf.write(str(out), wav, net.sample_rate)

    elif voice.model == "OpenVoiceV2":
        _add_syspath(MELO_REPO)
        from melo.api import TTS

        # stage 1: speak the text in a stock base voice.
        # (built via explicit membership test, not setdefault -- setdefault
        # would construct a fresh TTS on every call just to discard it)
        melo_key = (f"melo:{language}", device)
        if melo_key not in _MODELS:
            _MODELS[melo_key] = TTS(language=language, device=device)
        tts = _MODELS[melo_key]
        # spk2id is MeloTTS's HParams, not a dict: it defines keys()/items()
        # but no __iter__, so iter() falls back to the legacy integer-index
        # protocol and calls getattr(self, 0) -> TypeError. Always go via keys().
        spk2id = tts.hps.data.spk2id
        keys = list(spk2id.keys())
        if not keys:
            raise CloneError(f"{language} base model exposes no speakers")
        if speaker is None:
            # prefer the newest English voice when present, else first listed
            key = next((k for k in keys if k.lower() == "en-newest"), keys[0])
        else:
            match = [k for k in keys if k.lower() == speaker.lower()]
            if not match:
                raise CloneError(
                    f"speaker {speaker!r} not in {language} base model; "
                    f"available: {keys}"
                )
            key = match[0]

        ses_name = key.lower().replace("_", "-") + ".pth"
        ses_file = OV_CKPT / "base_speakers" / "ses" / ses_name
        if not ses_file.is_file():
            raise CloneError(f"base speaker embedding missing: {ses_file}")
        source_se = torch.load(str(ses_file), map_location=device)

        tmp = out.with_name(out.stem + "__base.wav")
        tts.tts_to_file(text, spk2id[key], str(tmp), speed=speed)

        # stage 2: bend that audio toward the enrolled tone colour
        net.convert(
            audio_src_path=str(tmp),
            src_se=source_se,
            tgt_se=voice.payload["target_se"],
            output_path=str(out),
            tau=tau,
        )
        tmp.unlink(missing_ok=True)

    else:                                            # pragma: no cover
        raise KeyError(voice.model)

    return out


# --------------------------------------------------------------------------
# install check -- mirrors detectors.check_install()
# --------------------------------------------------------------------------

_REQUIRED: dict[str, tuple[str, ...]] = {
    "CosyVoice2": (
        r"CosyVoice\cosyvoice\cli\cosyvoice.py",
        r"CosyVoice\third_party\Matcha-TTS\matcha\__init__.py",
        r"models\CosyVoice2-0.5B\cosyvoice2.yaml",
        r"models\CosyVoice2-0.5B\llm.pt",
        r"models\CosyVoice2-0.5B\flow.pt",
        r"models\CosyVoice2-0.5B\hift.pt",
    ),
    "OpenVoiceV2": (
        r"OpenVoice\openvoice\api.py",
        r"MeloTTS\melo\api.py",
        r"models\OpenVoiceV2\converter\config.json",
        r"models\OpenVoiceV2\converter\checkpoint.pth",
        r"models\OpenVoiceV2\base_speakers\ses\en-newest.pth",
    ),
}


def check_install(verbose: bool = True) -> dict[str, list[str]]:
    """Report which cloning models ROOT can serve. Files only -- it does not
    check that the Python dependencies are importable."""
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
            print("\nedit VOICE_CLONING_ROOT in .env if the models live elsewhere.")
        print("\nnote: MeloTTS downloads its own base-speaker checkpoints from")
        print("HuggingFace on first use, so OpenVoiceV2 generation needs network")
        print("access the first time you call generate() for a given language.")
    return missing


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Clone a voice and speak text with it")
    ap.add_argument("--check", action="store_true", help="verify ROOT, then exit")
    ap.add_argument("--ref", help="reference audio to clone")
    ap.add_argument("--model", default="CosyVoice2", help=str(list(MODEL_ORDER)))
    ap.add_argument("--text", help="what to say")
    ap.add_argument("--prompt-text", default=None, help="transcript of --ref (CosyVoice2)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.check:
        sys.exit(1 if any(check_install().values()) else 0)
    if not (args.ref and args.text):
        ap.error("--ref and --text are required (or pass --check)")

    v = enroll(args.ref, args.model, prompt_text=args.prompt_text, device=args.device)
    print(f"enrolled {v.alias!r} on {v.model}")
    p = generate(args.text, args.ref, out_path=args.out, device=args.device)
    print(f"wrote {p}")
