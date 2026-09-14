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

## Shell setup (do this once)

`conda activate` only works in PowerShell after `conda init powershell` has been
run — without it the command **exits 0 and silently does nothing**, leaving you on
base Python and a later `ModuleNotFoundError: torch` that looks unrelated. Check
with `conda activate voiceclone; echo $env:CONDA_DEFAULT_ENV` — if that prints
empty, run:

```powershell
conda init powershell
```

then open a new terminal. (Reversible: `conda init --reverse powershell`.)

Also note PowerShell 5.1 has **no `&&` operator** — chain with `;` instead, or put
each command on its own line.

## Typical run

```powershell
conda activate voiceclone
python generate_all.py --voice riven

conda activate voicefake
python score_all.py --voice riven

conda activate voiceclone
python Tester.py --voice riven
```

## Best result: apply the distilled operator

This is the configuration that works. Two RL designs (below) produced nothing;
a small network trained by regression on PGD solutions, applied ten times, is
what moved the detectors. Full reasoning, numbers and caveats are in
[`Plan_X/RESULTS.md`](Plan_X/RESULTS.md).

```powershell
& "E:\envs\voiceclone\python.exe" apply_best.py --voice evelynn --text "I sentence you to death." --out out.wav --score
& "E:\envs\voiceclone\python.exe" apply_best.py --voice evelynn --text "I sentence you to death." --out base.wav --no-edit --score
```

On held-out voices with unseen sentences: mean detector score 0.34 -> 0.69
(t = 12.26), soft-min 0.11 -> 0.37, similarity and WER unchanged, PESQ -0.22
(at 4 passes PESQ is unchanged but the gain is half). Strongest on
5-15 s utterances; the gain disappears on ~70 s clips.

### Reproducing it

| step | command | time |
|---|---|---|
| 1. collect PGD targets | `python collect_pgd.py --max-hours 3.3` | ~3 h |
| 2. distil | `python distill.py --run-id final --dilations 1 --hidden 24 --weight-decay 1e-3 --lr 1e-4 --input-noise 0.15 --max-hours 0.5` | ~30 min |
| 3. evaluate | `python eval_final.py --distill-run distill_final --edit-iters 10` | ~45 min |
| 4. figures | `python plot_distill.py --run distill_final --control -0.155` | ~1 min |

### Plan_X files

| file | what it does |
|---|---|
| `apply_best.py` | Single entry point: speak text with the best configuration, optional A/B and scoring. |
| `diffscore.py` | Differentiable mel -> HiFT -> four detectors path; the gradient hint. Escapes `inference_mode` and cuDNN RNN restrictions. |
| `collect_pgd.py` | Builds the supervised dataset: PGD on real CosyVoice2 mels. Resumable, round-robin across voices. `--detectors` for leave-one-out, `--breathify` for expressive text. |
| `distill.py` | Trains `GradInformedActor` to predict PGD's edit. Reports recovered fraction of PGD's gain and a fixed-average-edit control. |
| `eval_final.py` | Baseline vs Design B vs distilled on held-out voices and unseen sentences, with `--edit-iters` / `--edit-scale`. |
| `quality.py` | SQUIM PESQ / STOI / MOS estimates -- the perceptual axis similarity and WER cannot see. |
| `textbank.py` | Train/eval split of sentences and voices, kept strictly apart. |
| `probe_pgd.py` | The original feasibility check: how far can white-box PGD move the detectors? |
| `probe_amplify.py` | Scans edit scale x number of passes on stored mels. Found that 10 passes recovers 97% of PGD. |
| `probe_teacher.py` | Sweeps PGD's own settings (`--hardest` selects clips the detectors actually catch). |
| `probe_longform.py` | Length curve (short / medium / ~70 s). |
| `probe_prosody.py`, `prosody.py` | Emphasis, breath, pause and instruct-mode variants of identical content. |
| `probe_reference.py` | Denoise / high-pass / normalise the reference before enrolment. |
| `demo_best.py`, `demo_listen.py` | Keep-the-wavs demos for listening. |
| `eval_ab_voices.py`, `eval_generalization.py`, `eval_policy.py` | Earlier paired evaluations of the RL policies. |
| `sac_agent.py`, `train_sac_mel.py` | Design B (SAC with gradient hint). Flat -- kept for the record. |
| `mel_policy.py`, `reward.py`, `train_ppo_mel.py` | Design A (PPO-Lagrangian). Degraded -- kept for the record. |
| `analyze_run.py`, `plot_plan_x.py`, `plot_ab_voices.py`, `plot_distill.py` | Trend analysis and every figure in `Plan_X/`. |

`cloners.generate` now also takes `instruct=` (style prompting via
`inference_instruct2`) and accepts CosyVoice2 prosody tags (`<strong>`,
`[breath]`, ...) in the text; `cloners.strip_markup` gives the spoken words for
WER.

## Design A: RL training

```powershell
conda activate voiceclone
cd "D:\AHJN-dodging"
python train_ppo_mel.py --voices riven swain samira jhin --iterations 182 --batch-size 8 --epochs 4 --lr 6e-4 --lambda-lr 0.2 --wer-max 0.15 --snapshot-every 10 --run-id budget6h
```

Interrupted? Re-run the identical command with `--resume auto` appended — nothing
is lost. Then:

```powershell
conda activate voicefake
python analyze_run.py --run designA_budget6h --wer-max 0.15 --plot
```

If `conda activate` is ever uncooperative, every command also works by calling the
env's interpreter directly, which needs no shell setup at all:

```powershell
& "E:\envs\voiceclone\python.exe" train_ppo_mel.py --voices riven ...
& "E:\envs\voicefake\python.exe" analyze_run.py --run designA_budget6h
```

See `MATH.md` for how the parameters above were derived.
