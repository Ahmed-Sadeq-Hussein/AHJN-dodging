# AHJN-dodging

Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved

Can a CosyVoice2 voice clone be made to read as **human** to four anti-spoofing
detectors, without losing the speaker, the words, or the audio quality?

Framed as a reinforcement-learning problem: an agent observes the mel
spectrogram inside the synthesiser, applies one bounded edit to it, and is
rewarded for how human the four detectors judge the result, under constraints
on speaker similarity and word error rate.

**The headline result is negative, and it is the finding.** Two RL designs
were built, trained to 400 iterations each, and measured end to end. Both are
significantly *worse* than not editing at all, on the detectors and on human
listening alike. A white-box attack on the same intervention point, which gets
the gradient instead of a scalar, works well. The gap between those two
outcomes is what this project is about.

| arm | soft-min detector | MOS (human) | PESQ |
|---|---|---|---|
| unedited clone | **0.185** | **3.97** | **3.47** |
| Design A, PPO-Lagrangian | 0.085 | 2.53 | 2.82 |
| Design B, SAC + gradient hint | 0.135 | 3.20 | 3.29 |

*30 paired items, 15 voices in two registers. 1 = human on the detectors.*

---

## Where to start reading

| If you want | Read |
|---|---|
| How the pieces fit together | `Plan_X/ARCHITECTURE.md` and `graphs/21_architecture.svg` |
| The RL designs, results and what went wrong | `rl/ANALYSIS.md` |
| The formal problem: states, actions, reward, bandit framing, power analysis | `MATH.md` |
| Every experiment in order, including the failures | `Plan_X/RESULTS.md` |
| Papers, models and libraries used, with links | `Plan_X/SOURCES.md` |
| What each figure shows | `graphs/INDEX.md` |

---

## Setup

### 1. Configure paths (`.env`)

Nothing hardcodes a path. Copy `.env.example` to `.env` and set:

| Variable | What goes there |
|---|---|
| `VOICE_DETECTION_ROOT` | The anti-spoofing repos (`aasist`, `RawGAT-ST-antispoofing`, `asvspoof2021-baselines`) and their pretrained weights. |
| `VOICE_CLONING_ROOT` | The CosyVoice2 repo and its `models\CosyVoice2-0.5B` weights (`llm.pt`, `flow.pt`, `hift.pt`, `campplus.onnx`). |
| `VOICE_DATA_ROOT` | One subfolder per voice, each with `reference.wav` and `reference.txt`. 15 voices were used here. |
| `RESULTS_ROOT` | Output root. `generators\`, `discriminator\` and `rl_runs\` are created under it. |
| `VOICEFAKE_PYTHON` | Path to the `voicefake` interpreter. `reward.py` needs it to reach the detectors during training. |

`GENERATORS_DIR`, `DISCRIMINATOR_DIR`, `METRICS_DIR`, `CAMPPLUS_ONNX_PATH` and
`GRAPHS_DIR` are derived automatically; set one only to move that output. A
real environment variable always overrides `.env`, and a missing required one
fails at import rather than silently using someone else's path.

### 2. Install the two environments

The two halves pin incompatible torch/numpy versions:

| Env | Runs | torch | numpy | Install |
|---|---|---|---|---|
| `voiceclone` | CosyVoice2, the trainers, the attack, every evaluation | 2.3.1+cu121 | 1.26.4 | `conda create -n voiceclone -y python=3.10` then `pip install -c constraints.txt -r requirements-voiceclone.txt` |
| `voicefake` | the detectors in isolation | 2.6.0+cu124 | 2.2.6 | `conda create -n voicefake -y python=3.10` then `pip install -r requirements.txt` |

**Most work happens in `voiceclone`.** The detectors import and run correctly
there too, which is how the stored scores were produced; the two environments
agree to about 1e-7. The separate `voicefake` env exists because `reward.py`
batches a whole RL rollout through one subprocess call, and because the
detector repos pin versions CosyVoice2 cannot use.

Both `requirements*.txt` files explain inline *why* each pin exists. They are
the record of what actually broke during setup, including the Windows Smart
App Control traps; read them before changing a version.

One thing pip cannot do, needed once:

```
python -c "import nltk; [nltk.download(p) for p in ['averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab']]"
```

Whisper's `base` weights download from HuggingFace on first use.

### 3. Check the install

```
python ahjn.py check
```

Reports the interpreter, CUDA, every configured path, whether all four
detector checkpoints are findable, and whether the listening-test corpus is
present. Run this before anything slow.

---

## Using it

`ahjn.py` is the front door. Everything below is a verb on it.

```
python ahjn.py check                      # environment and model files
python ahjn.py detect clip.wav            # the four detectors, 1 = human
python ahjn.py score  clip.wav --reference ref.wav --text "what was said"
python ahjn.py status                     # checkpoints and training trend
python ahjn.py speak --design b --voice riven --text "hello" --out b.wav
python ahjn.py speak           --voice riven --text "hello" --out plain.wav
python ahjn.py figures                    # redraw every figure
python ahjn.py verify                     # re-derive the published numbers
```

`score` is the one call that returns everything for a clip: the four
detectors, their soft-min, speaker similarity, WER, and SQUIM PESQ / STOI /
SI-SDR. It never raises; whatever it could not compute is absent and listed
under `problems`.

### Training

```
python ahjn.py train b --iters 400 --resume      # continue Design B
python ahjn.py train a --iters 400 --resume      # continue Design A
```

`--resume` picks up the newest checkpoint in the run directory and carries the
optimiser state, the Lagrange multipliers and (for Design B) the replay
buffer. Without `--iters` each design uses the defaults in `rl/design_a.py`
and `rl/design_b.py`. The underlying scripts, `train_ppo_mel.py` and
`train_sac_mel.py`, can still be called directly for anything the wrapper does
not expose; `--help` on either lists every knob.

### The listening test

```
python ahjn.py rate build        # render the corpus: 15 voices x 2 lines x 3 arms
python ahjn.py rate page         # build the rating page
python ahjn.py rate serve        # serve it and open a browser
python ahjn.py rate import ~/Downloads/mos_ratings.json
python ahjn.py rate results --split
```

Use `rate serve` rather than opening the HTML from disk. A `file://` page has
no storage and no download, so ratings go nowhere.

All three arms of an item share one seed. Both policies are deterministic and
consume no RNG, so the arms replay the same speech tokens and the same vocoder
noise, and the mel edit is the only audible difference. Identical file lengths
are the check that it held.

---

## Reproducing the results

```
python ahjn.py verify
```

`verify_claims.py` re-derives 41 headline numbers from the stored JSON and
prints each beside its published value (currently 41/41). `verify_wer.py`
checks the jiwer-based WER against the hand-written Levenshtein pass it
replaced, over 20098 cases.

```
python ahjn.py figures
```

Runs every `plot_*.py` against the stored results, writing to `graphs/`:
21 numbered figures across 27 files, since figure 20 also ships its panels
separately. Nothing is regenerated from a model, so this is fast and
deterministic.

Full retraining is not fast. Measured from the training logs themselves,
Design A took 6.2 GPU hours to 400 iterations and Design B took 12.0, split
across two run directories (`designB_B3h` holds iterations 0 to 80 and
`designB_B9h` the rest). The checkpoints and logs under
`RESULTS_ROOT\rl_runs\` are what the analysis reads.

---

## The code

| File | Env | What it does |
|---|---|---|
| `ahjn.py` | voiceclone | The command line. Every verb above. |
| `env_config.py` | either | Loads `.env` and exposes every path as one import. |
| `cloners.py` | voiceclone | Zero-shot cloning: `enroll`, `generate`. CosyVoice2 only. |
| `detectors.py` | either | The four anti-spoofing models; `P(human)` per model. `--check` verifies the weights. |
| `diffscore.py` | voiceclone | The differentiable path: reaches past `inference_mode`, turns cuDNN off for RawNet2's GRU, and backpropagates detector scores to the mel. |
| `mel_policy.py` | voiceclone | `MelResidualPolicy`, `MelValueNet`, and `mel_edit_hook`, which patches `hift.inference` so the mel can be edited mid-synthesis. |
| `reward.py` | voiceclone | Soft-min over the detectors, two Lagrange multipliers, a hard floor. Reaches the detectors through `VOICEFAKE_PYTHON`. |
| `train_ppo_mel.py` | voiceclone | Design A: PPO with Lagrangian constraints. |
| `train_sac_mel.py` | voiceclone | Design B: SAC, twin critics, detector gradient as extra input channels. |
| `sac_agent.py` | voiceclone | The SAC actor, critics and temperature. |
| `collect_pgd.py` | voiceclone | The white-box teacher: PGD on the mel, per clip. |
| `distill.py` | voiceclone | Trains a one-pass student to imitate the teacher's edits. |
| `rl/` | voiceclone | A stable interface over both designs: `train`, `summarise`, `latest_checkpoint`, `generate`, plus `mos_corpus.py` and `ANALYSIS.md`. |
| `scorecard.py` | voiceclone | Every metric for a clip in one call. |
| `Tester.py` | voiceclone | CAM++ speaker embedding, Whisper transcript, WER through jiwer. |
| `quality.py` | voiceclone | TorchAudio-SQUIM: PESQ, STOI, SI-SDR, estimated MOS. No reference needed. |
| `textbank.py` | either | The train / eval sentence split. Eval sentences are never trained on. |
| `eval_*.py`, `probe_*.py` | voiceclone | One script per experiment, in the order `Plan_X/RESULTS.md` tells them. |
| `plot_*.py` | either | One script per figure, all writing to `graphs/`. |
| `verify_claims.py`, `verify_wer.py` | voiceclone | Re-derive the published numbers; check the WER change. |
| `generate_all.py`, `score_all.py` | both | The original two-phase survey that preceded the RL work. |
| `requirements*.txt`, `constraints.txt` | - | Pinned deps, with the reason for each pin inline. |
| `.env` / `.env.example` | - | Your paths (gitignored) / the template. |

---

## A note on the two environments

`detectors.py` runs under either interpreter and gives the same answer. The
split matters in exactly one place: `reward.py` shells out to
`VOICEFAKE_PYTHON` so a whole rollout is scored in one subprocess, because
loading the four models costs about 13 seconds and would otherwise dominate
every episode. The white-box path does the opposite and keeps everything in
one process, because it needs CosyVoice2 and the detectors in the same
autograd graph.
