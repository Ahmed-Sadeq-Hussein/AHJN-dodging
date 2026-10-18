# action_string.md

Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved

Everything you need to run this project from a bare machine, in order, with
the command for each step. Placeholders look like `<THIS>` and you replace
them; everything else can be pasted as written.

Contents:

1. [What you need first](#1-what-you-need-first)
2. [Get the code](#2-get-the-code)
3. [Create the two environments](#3-create-the-two-environments)
4. [Download the models and data](#4-download-the-models-and-data)
5. [The .env file, explained](#5-the-env-file-explained)
6. [Verify the install](#6-verify-the-install)
7. [The four jobs](#7-the-four-jobs) - train, generate, compare, graph
8. [Known traps](#8-known-traps)

---

## 1. What you need first

| | |
|---|---|
| OS | Windows 10/11 (the project was built and measured there). Linux works, but the paths in the examples are Windows. |
| GPU | An NVIDIA card with CUDA. Training was done on one consumer GPU. CPU will run inference, very slowly, and is not realistic for training. |
| Disk | About 25 GB: roughly 7 GB per conda environment, 2 GB of model weights, and the rest for generated audio. |
| conda | Miniconda or Anaconda. |
| git | For the repositories in step 4. |

Two Python environments are required and they cannot be merged. The detector
repositories and CosyVoice2 pin incompatible versions of torch and numpy:

| | voiceclone | voicefake |
|---|---|---|
| torch | 2.3.1+cu121 | 2.6.0+cu124 |
| numpy | 1.26.4 | 2.2.6 |
| runs | CosyVoice2, training, the attack, all evaluation | the detectors in isolation |

Almost everything happens in `voiceclone`. See [section 8](#8-known-traps) for
why `voicefake` still exists.

---

## 2. Get the code

```bash
git clone <YOUR_REPO_URL> AHJN-dodging
cd AHJN-dodging
```

From here on, `<REPO>` means this folder.

---

## 3. Create the two environments

Order matters only in that `constraints.txt` must be passed when installing
`voiceclone`.

### voiceclone (the main one)

```bash
conda create -n voiceclone -y python=3.10
conda activate voiceclone
pip install -c constraints.txt -r requirements-voiceclone.txt
```

The `-c constraints.txt` is not optional. Several transitive dependencies
declare a bare `torch` requirement and will happily replace the pinned CUDA
build with a CPU-only wheel, which silently kills GPU support. Check
immediately afterwards:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `2.3.1+cu121 True`. If it says anything else, reinstall torch before
continuing.

One thing pip cannot do, needed once:

```bash
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
```

### voicefake (the detectors)

```bash
conda create -n voicefake -y python=3.10
conda activate voicefake
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

You want `2.6.0+cu124 True`.

Both requirements files explain inline why each pin exists. They are the
record of what broke during setup, so read the comments before changing a
version.

---

## 4. Download the models and data

Three roots, which you will point `.env` at in the next section. Put them
wherever you like; the names below are just the ones used here.

### 4a. The cloning root

```bash
mkdir "<CLONING_ROOT>"
cd "<CLONING_ROOT>"
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
```

`--recursive` matters: CosyVoice vendors Matcha-TTS as a submodule and the
flow decoder imports from it. If you already cloned without it:

```bash
cd CosyVoice && git submodule update --init --recursive
```

Then the weights, into `<CLONING_ROOT>\models\CosyVoice2-0.5B`:

```bash
# in the voiceclone env
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir=r'<CLONING_ROOT>\models\CosyVoice2-0.5B')"
```

The HuggingFace mirror is `FunAudioLLM/CosyVoice2-0.5B` if ModelScope is slow
from where you are. Both are listed in CosyVoice's own README.

Expected afterwards:

```
<CLONING_ROOT>\
    CosyVoice\                       the repo, with third_party\Matcha-TTS
    models\CosyVoice2-0.5B\
        llm.pt
        flow.pt
        hift.pt
        campplus.onnx                also used for speaker similarity
        cosyvoice2.yaml
        speech_tokenizer_v2.batch.onnx
```

### 4b. The detection root

```bash
mkdir "<DETECTION_ROOT>"
cd "<DETECTION_ROOT>"
git clone https://github.com/clovaai/aasist.git
git clone https://github.com/eurecom-asp/RawGAT-ST-antispoofing.git
git clone https://github.com/asvspoof-challenge/2021.git asvspoof2021-baselines
```

AASIST and RawGAT-ST ship their pretrained weights inside the repository, so
those two need nothing further. RawNet2 does not: the only pretrained model
is the deepfake-track one, and it is a separate download named in the
baseline's own README.

```bash
mkdir "<DETECTION_ROOT>\models\RawNet2-DF"
# download and unzip into that folder:
#   https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip
```

Expected afterwards (these exact paths are what `detectors.py` checks):

```
<DETECTION_ROOT>\
    aasist\config\AASIST.conf
    aasist\config\AASIST-L.conf
    aasist\models\AASIST.py
    aasist\models\weights\AASIST.pth
    aasist\models\weights\AASIST-L.pth
    RawGAT-ST-antispoofing\model.py
    RawGAT-ST-antispoofing\model_config_RawGAT_ST.yaml
    RawGAT-ST-antispoofing\Pre_trained_models\RawGAT_ST_mul\Best_epoch.pth
    asvspoof2021-baselines\LA\Baseline-RawNet2\model.py
    asvspoof2021-baselines\LA\Baseline-RawNet2\model_config_RawNet.yaml
    models\RawNet2-DF\pre_trained_DF_RawNet2.pth
```

Note that RawGAT-ST ships three variants. Only `RawGAT_ST_mul` is attacked;
`add` and `cat` are deliberately left untouched and used as never-seen models
in the transfer test.

### 4c. The voice data root

One folder per voice, each with a reference recording and its transcript:

```
<DATA_ROOT>\
    riven\
        reference.wav
        reference.txt
    jhin\
        reference.wav
        reference.txt
    ...
```

`reference.wav` should be clean speech of roughly 5 to 20 seconds, 16 kHz or
better. `reference.txt` is what is said in it: CosyVoice2 clones noticeably
better with the prompt transcript, and WER cannot be computed without it.
Fifteen voices were used in this project.

### 4d. The results root

Just an empty folder. Everything generated lands under it, and the
subdirectories are created automatically.

```bash
mkdir "<RESULTS_ROOT>"
```

---

## 5. The .env file, explained

Nothing in this repository hardcodes a path. All of them come from a single
`.env` file sitting next to the code, which is gitignored so your machine's
layout never reaches the repository.

```bash
cd <REPO>
copy .env.example .env        # Windows
cp .env.example .env          # bash
```

Then edit it. The format is plain `KEY=VALUE`, one per line, no quotes, no
spaces around the `=`. Backslashes are fine; do not escape them.

### The five you must set

```ini
VOICE_DETECTION_ROOT=D:\Voice detection
VOICE_CLONING_ROOT=D:\Voice cloning
VOICE_DATA_ROOT=D:\voice data\Voices
RESULTS_ROOT=D:\results and benchmarks
VOICEFAKE_PYTHON=E:\envs\voicefake\python.exe
```

| Variable | Points at | Used by |
|---|---|---|
| `VOICE_DETECTION_ROOT` | the folder from step 4b | `detectors.py` to find the four models |
| `VOICE_CLONING_ROOT` | the folder from step 4a | `cloners.py` to find the CosyVoice repo and weights |
| `VOICE_DATA_ROOT` | the folder from step 4c | everything that loads a reference voice |
| `RESULTS_ROOT` | the folder from step 4d | training runs, generated audio, metrics |
| `VOICEFAKE_PYTHON` | the `voicefake` interpreter | `reward.py`, to reach the detectors while training |

To find the last one:

```bash
conda activate voicefake
python -c "import sys; print(sys.executable)"
```

### The five you can usually leave alone

These are derived from the ones above and only need setting if you want that
particular output somewhere else.

```ini
# GENERATORS_DIR=<RESULTS_ROOT>\generators
# DISCRIMINATOR_DIR=<RESULTS_ROOT>\discriminator
# METRICS_DIR=<GENERATORS_DIR>\metrics
# CAMPPLUS_ONNX_PATH=<VOICE_CLONING_ROOT>\models\CosyVoice2-0.5B\campplus.onnx
# GRAPHS_DIR=<REPO>\graphs
```

### How it behaves

- A real environment variable of the same name **overrides** `.env`. Useful
  for a one-off run against a different results folder.
- A **missing required** variable fails loudly at import time rather than
  falling back to some other machine's path.
- `GRAPHS_DIR` defaults into the repository on purpose: the figures are small,
  reviewable and worth keeping under version control next to the write-up.

---

## 6. Verify the install

```bash
conda activate voiceclone
cd <REPO>
python ahjn.py check
```

This prints the interpreter and CUDA state, every configured path with
whether it exists, whether all four detector checkpoints are findable, and
whether the listening-test corpus is present. Fix anything it flags before
running anything slow.

A quick end-to-end test once it is clean:

```bash
python ahjn.py detect "<REPO>\MOS TEST RL\baseline\Ego_L1.wav"
```

Four numbers between 0 and 1, where 1 means the detector thinks it is human.

---

## 7. The four jobs

Each has a script in `scripts/`. They are thin: the real work lives in the
modules they call, so the scripts cannot drift from the code that produced
the results. All of them run in the `voiceclone` environment.

### 7.1 Training

Two designs, one script each.

**Design A, PPO with Lagrangian constraints.** One utterance is one episode:
observe the mel, apply a single bounded edit, receive one terminal reward.
Roughly 4 GPU hours to 400 iterations.

```bash
python scripts/train_design_a.py --dry-run                  # show settings, train nothing
python scripts/train_design_a.py                            # published configuration
python scripts/train_design_a.py --iters 400 --resume       # continue the existing run
python scripts/train_design_a.py --run <NEW_RUN_ID> --lr <LR> --voices <V1> <V2>
```

**Design B, SAC with a detector gradient hint.** Same one-step formulation,
but off-policy, with twin Q critics, an automatically tuned temperature, and
the detector gradient fed to the actor as extra input channels. Roughly 9 GPU
hours to 400 iterations.

```bash
python scripts/train_design_b.py --dry-run
python scripts/train_design_b.py --iters 400 --resume
python scripts/train_design_b.py --run <NEW_RUN_ID> --no-grad-hint   # the ablation
```

Useful flags on both:

| Flag | Does |
|---|---|
| `--dry-run` | print the settings and the target directory, train nothing |
| `--resume` | continue the newest checkpoint, carrying optimiser state, the Lagrange multipliers, and for Design B the replay buffer |
| `--run <ID>` | a new run id starts a separate directory; reusing one continues it |
| `--iters <N>` | total iterations to reach, not additional ones |
| `--voices <...>` | which voices to train on |

**Start a new run id when you change a hyperparameter.** Reusing a run id
without `--resume` starts at iteration 0 and overwrites the checkpoints that
are there. Both scripts warn you when that is about to happen.

Watch progress at any time:

```bash
python ahjn.py status
```

Output lands in `<RESULTS_ROOT>\rl_runs\design<A|B>_<RUN_ID>\`:
`train_log.jsonl` (one row per iteration), `latest.pt`, `iter_NNNN.pt`
snapshots, and `config.json`.

### 7.2 Generating audio from a trained model

```bash
# one clip from one policy
python scripts/generate_audio.py --design b --voice <VOICE> --text "<WHAT TO SAY>" --out <OUT.wav>

# the plain clone, no policy
python scripts/generate_audio.py --voice <VOICE> --text "<WHAT TO SAY>"

# all three arms on one seed, and measure them
python scripts/generate_audio.py --all-arms --voice <VOICE> --text "<WHAT TO SAY>" --score
```

`--all-arms` is the one to use for listening. It writes the same line
unedited, through Design A and through Design B, **under one seed**. Both
policies are deterministic at inference and consume no RNG, so all three
replay the same speech tokens and the same vocoder noise, and the mel edit is
the only audible difference. Identical file lengths are the check that it
held. Two clips rendered under different seeds are different takes, and
comparing them tells you nothing about the edit.

`--score` adds the four detectors, the soft-min, speaker similarity, WER and
SQUIM PESQ/STOI for every clip produced.

Other flags: `--seed <N>`, `--checkpoint iter_0200.pt`, `--run <RUN_ID>` to
point at a run other than the published one, `--out-dir <DIR>`.

### 7.3 Comparing models

```bash
python scripts/compare_models.py                                  # quick: 3 voices, 1 line
python scripts/compare_models.py --voices all --texts 2           # the full 30-item comparison
python scripts/compare_models.py --arms baseline b --voices <V1> <V2>
python scripts/compare_models.py --from-json out/comparison/comparison.json
```

Renders every arm of every item under one shared seed, scores all of it, and
reports a paired comparison: mean soft-min, similarity, WER and PESQ per arm,
then each edited arm against the unedited one with a paired t-test, a
Wilcoxon signed rank test, and a count of how many items moved the right way.

Paired is the point. Per-clip variance across voices is far larger than any
effect being looked for, with baseline soft-min spanning roughly 0.24 to 0.96
in this project, so an unpaired comparison would drown the signal.

Sentences come from the held-out `EVAL` split that no policy trained on.

`--from-json` re-reports a finished run instantly, which is what you want
while writing up: rendering is the slow part and the numbers do not change.

### 7.4 Printing the graphs

```bash
python scripts/make_graphs.py --list        # what each one draws
python scripts/make_graphs.py               # all of them
python scripts/make_graphs.py --only mos architecture
python scripts/make_graphs.py --open        # and open the folder afterwards
```

Everything is drawn from stored results and never from a model, so this is
fast and repeatable. Figures land in `GRAPHS_DIR` (by default `<REPO>\graphs`)
and `graphs/INDEX.md` says what each one shows.

A script that fails is reported and does not stop the others, because the
usual cause is a results file that has not been produced on this machine
rather than a broken figure.

### Reproducing the published numbers

```bash
python ahjn.py verify
```

`verify_claims.py` re-derives 41 headline numbers from stored JSON and prints
each beside its published value. `verify_wer.py` checks the jiwer-based word
error rate against the hand-written implementation it replaced, over 20098
cases.

---

## 8. Known traps

**Windows Smart App Control blocks DLLs.** It silently refused to load
`tiktoken`, `llvmlite` and `pyarrow` during this project's setup. The symptom
is an import error that mentions a DLL and makes no sense. The pins in
`requirements-voiceclone.txt` already work around the known ones
(`llvmlite==0.42.0`, `pyarrow==14.0.2`); if you hit a new one, downgrading
that package is usually the fix. Do not disable the security setting.

**The CPU torch wheel.** Covered in step 3, and worth repeating because it is
silent: install `voiceclone` with `-c constraints.txt` or a transitive
dependency will swap your CUDA torch for a CPU build and training will be
unusably slow with no error.

**`VOICEFAKE_PYTHON` is only needed for training.** Scoring and generating
work without it. `reward.py` is the one caller, and it shells out so that a
whole rollout is scored in a single subprocess: loading the four detectors
costs about 13 seconds, which would otherwise be paid every episode.

**The detectors run in either environment.** They import and give the same
answer under `voiceclone`, agreeing with `voicefake` to about 1e-7, which is
how the stored scores were produced. The separate environment exists for the
batching above and because the detector repositories pin versions CosyVoice2
cannot use, not because the models refuse to load.

**Rendering is the slow part of every comparison.** Loading CosyVoice2 takes
on the order of a minute, and each clip is a few seconds after that. Prefer
`--from-json` and `ahjn.py figures` when you are only changing how results are
presented.

**Anything else using the GPU will dominate.** CosyVoice2 normally
synthesises faster than real time. Measured on this machine with a game
running in the background, one 3.4 second clip took seven minutes, a real
time factor of 124. Check before starting a run:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

If utilisation is already high and most of the memory is gone, close whatever
owns it first. The training estimates above assume the card is yours.

**No ffmpeg needed.** Whisper would normally shell out to it; `Tester.py`
decodes with librosa and hands Whisper an array instead.
