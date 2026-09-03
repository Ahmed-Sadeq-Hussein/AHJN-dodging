# AHJN-dodging

A codebase for trying to make code to make Cozyvoice 2 dodge detection of AI voice detection tools while maintaining MOS and WER scores.

Pipeline: clone a reference voice with CosyVoice2 → score the result with four anti-spoofing detectors → measure how much the clone actually sounds like its reference (similarity) and how intelligible it is (WER) → use those numbers to judge whether a change made the clone dodge detection without wrecking quality.

## Setup

### 1. Configure paths (`.env`)

Nothing in this repo hardcodes a path. Copy `.env.example` to `.env` in this folder and set these four:

| Variable | What goes there |
|---|---|
| `VOICE_DETECTION_ROOT` | Folder holding the anti-spoofing repos (`aasist`, `RawGAT-ST-antispoofing`, `asvspoof2021-baselines`) and their pretrained weights. |
| `VOICE_CLONING_ROOT` | Folder holding the CosyVoice2 repo and its `models\CosyVoice2-0.5B` weights (`llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`, …). |
| `VOICE_DATA_ROOT` | One subfolder per voice, each containing `reference.wav` and, ideally, `reference.txt` (its transcript — CosyVoice2 clones better with one, and `Tester.py` needs it to compute WER). |
| `RESULTS_ROOT` | Output root. `generators\` and `discriminator\` are created under it automatically. |

Four more variables (`GENERATORS_DIR`, `DISCRIMINATOR_DIR`, `METRICS_DIR`, `CAMPPLUS_ONNX_PATH`) are derived automatically from the four above — only set one if you want that particular output somewhere else. See the comments in `.env.example` for their derived defaults.

A real environment variable of the same name always overrides `.env`. Missing a required one fails loudly at import time rather than silently falling back to some machine's path.

### 2. Install dependencies

Work is split across **two conda environments** because the two halves pin incompatible torch/numpy versions. They never share a process — they hand off through `.wav` files and a JSON manifest on disk.

| Env | Runs | torch | numpy | Install |
|---|---|---|---|---|
| `voicefake` | `detectors.py`, `score_all.py` | 2.6.0+cu124 | 2.2.6 | `conda create -n voicefake -y python=3.10` then `pip install -r requirements.txt` |
| `voiceclone` | `cloners.py`, `generate_all.py`, `Tester.py` | 2.3.1+cu121 | 1.26.4 | `conda create -n voiceclone -y python=3.10` then `pip install -c constraints.txt -r requirements-voiceclone.txt` |

Both `requirements*.txt` files document *why* each pin exists inline (the two anti-spoofing baselines pin mutually-incompatible numpy/librosa/pyyaml versions; some of CosyVoice2's transitive deps will silently swap out the pinned CUDA torch for a CPU wheel unless installed against `constraints.txt`) — read the comments there before changing anything, they're the record of what actually broke during setup.

Two things pip can't install, needed once before things fully work:

```
# voiceclone env -- English g2p for MeloTTS/OpenVoiceV2
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
```

- **Whisper weights** (`Tester.py`'s WER) and **MeloTTS's base-speaker/BERT checkpoints** (OpenVoiceV2 generation) download from HuggingFace automatically the first time each is used — just needs network access once.

## Files

| File | Env | What it does |
|---|---|---|
| `env_config.py` | either | Loads `.env` and exposes every path (`VOICE_DETECTION_ROOT`, `VOICE_CLONING_ROOT`, `VOICE_DATA_ROOT`, `RESULTS_ROOT`, and the four derived dirs) as one shared import. No dependencies. |
| `detectors.py` | voicefake | Runs a `.wav` through the four anti-spoofing models (AASIST, AASIST-L, RawNet2, RawGAT-ST) and returns `P(human)` per model, 1.0 = human, 0.0 = AI. Handles model selection, windowing long clips, and caching loaded models. `--check` verifies the weights are where `.env` says they are. |
| `cloners.py` | voiceclone | Zero-shot voice cloning: `enroll(audio, model)` registers a reference clip, `generate(text, audio)` speaks new text in that voice. CosyVoice2 only right now — `ENABLED_MODELS` gates OpenVoiceV2 off (its code is intact, untouched, ready to flip back on). |
| `generate_all.py` | voiceclone | Phase 1: enrolls one reference voice on every enabled cloning model and writes each clone to `generators\<Model>\<voice>.wav`, plus a `manifest_<voice>.json` phase 2 reads. Merges into an existing manifest rather than overwriting it. |
| `score_all.py` | voicefake | Phase 2: reads a voice's manifest, scores the human reference *and* every generated clip with all four detectors, and writes a CSV, a JSON, a clip×detector heatmap, and a human-vs-AI grouped bar chart to `discriminator\`. |
| `Tester.py` | voiceclone | Quality metrics, independent of the detectors: a speaker-similarity heatmap (CAM++ cosine embedding — the same one CosyVoice2 itself conditions on) across an array of voices, and a word-error-rate ranking (Whisper transcript vs. target text) as both a table and a best-first bar chart. Writes to `generators\metrics\`. |
| `requirements.txt` | — | Pinned deps for the `voicefake` env, with inline notes on why each pin exists. |
| `requirements-voiceclone.txt` | — | Pinned deps for the `voiceclone` env, same treatment. |
| `constraints.txt` | — | Passed to pip alongside `requirements-voiceclone.txt` so a transitive dependency (`lightning`, `x-transformers`) can't silently replace the pinned CUDA torch with a CPU-only wheel. |
| `.env` / `.env.example` | — | Your machine's real paths (gitignored) / the template to copy it from. |

## Typical run

```
conda activate voiceclone
python generate_all.py --voice riven

conda activate voicefake
python score_all.py --voice riven

conda activate voiceclone
python Tester.py --voice riven
```
