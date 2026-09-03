r"""
env_config.py -- every filesystem path this project touches, in one place.

Nothing else in this repo hardcodes a machine-specific path. Copy
.env.example to .env in this same folder and fill in the paths for your
machine; every script imports this module and reads from there.

No external dependency: this repo's work is split across two conda envs
(voicefake, voiceclone) already pinned tightly enough that adding
python-dotenv to both felt like the wrong tradeoff for parsing four
KEY=VALUE lines, so the loader below is a few lines of stdlib.
"""

from __future__ import annotations

import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)   # a real env var always wins


_load_dotenv(THIS_DIR / ".env")


def _required(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env in {THIS_DIR} "
            f"and fill in {name} (see the comment above it in .env.example)."
        )
    return Path(value)


def _optional(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


# the four things that differ machine to machine
VOICE_DETECTION_ROOT = _required("VOICE_DETECTION_ROOT")   # anti-spoofing repos + weights
VOICE_CLONING_ROOT   = _required("VOICE_CLONING_ROOT")     # CosyVoice2 repo + weights
VOICE_DATA_ROOT       = _required("VOICE_DATA_ROOT")         # reference.wav / reference.txt per voice
RESULTS_ROOT           = _required("RESULTS_ROOT")             # generators/ + discriminator/ output tree

# derived, individually overridable if you want them somewhere else
GENERATORS_DIR    = _optional("GENERATORS_DIR",     RESULTS_ROOT / "generators")
DISCRIMINATOR_DIR = _optional("DISCRIMINATOR_DIR",  RESULTS_ROOT / "discriminator")
METRICS_DIR        = _optional("METRICS_DIR",        GENERATORS_DIR / "metrics")
CAMPPLUS_ONNX      = _optional(
    "CAMPPLUS_ONNX_PATH",
    VOICE_CLONING_ROOT / "models" / "CosyVoice2-0.5B" / "campplus.onnx",
)
