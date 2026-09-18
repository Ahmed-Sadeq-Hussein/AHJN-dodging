# Plan_X — source material

Every external paper, model, library and standard this project relied on,
grouped by the process it supports, with the repo file where it is used.

**Verification.** All 39 arXiv IDs were checked against the arXiv API, all
GitHub and web links returned HTTP 200, and all four DOIs were confirmed
through Crossref (2026-09-14); the three added for sections 7m and
7q were checked the same way (2026-09-18).

**How to read "role".**
- **used** — we run this model/code or implement this method directly.
- **basis** — our method is a direct adaptation of this.
- **related** — the closest prior work to something we built ourselves. We did
  not take the method from it, but it is the right place to read further.

---

## 1. Voice cloning — the thing being made to evade

| source | role | where in repo |
|---|---|---|
| Du et al. (2024). *CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.* [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | **used** — the cloning model. Also the source of the prosody tags (`<strong>`, `[breath]`) and instruction mode tested in 7f | `cloners.py`, `prosody.py` |
| Du et al. (2024). *CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens.* [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | related — predecessor; describes the speech-token → flow → vocoder pipeline CosyVoice 2 keeps | — |
| Code: [github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | **used** | `cloners.py` |
| Lipman et al. (2022). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | related — the flow decoder that produces the mel we edit | — |
| Mehta et al. (2023). *Matcha-TTS: A fast TTS architecture with conditional flow matching.* [arXiv:2309.03199](https://arxiv.org/abs/2309.03199) · [code](https://github.com/shivammehta25/Matcha-TTS) | **used** indirectly — CosyVoice 2's flow decoder imports from it (why `lightning` is a dependency) | `requirements-voiceclone.txt` |
| Li et al. (2023). *HiFTNet: A Fast High-Quality Neural Vocoder with Harmonic-plus-Noise Filter and Inverse Short Time Fourier Transform.* [arXiv:2309.09493](https://arxiv.org/abs/2309.09493) | related — design family of CosyVoice's HiFT vocoder, the network we backpropagate through | `diffscore.py` |
| Kong et al. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* [arXiv:2010.05646](https://arxiv.org/abs/2010.05646) | related — the GAN vocoder lineage HiFT builds on | — |

## 2. Detection — the anti-spoofing models being evaded

| source | role | where in repo |
|---|---|---|
| Jung et al. (2021). *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.* [arXiv:2110.01200](https://arxiv.org/abs/2110.01200) · [code + weights](https://github.com/clovaai/aasist) | **used** — AASIST and AASIST-L | `detectors.py` |
| Tak et al. (2020). *End-to-end anti-spoofing with RawNet2.* [arXiv:2011.01108](https://arxiv.org/abs/2011.01108) | **used** — RawNet2 (ASVspoof 2021 DF baseline weights) | `detectors.py` |
| Tak et al. (2021). *End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection.* [arXiv:2107.12710](https://arxiv.org/abs/2107.12710) · [code + weights](https://github.com/eurecom-asp/RawGAT-ST-antispoofing) | **used** — RawGAT-ST `mul` (attacked); `add` and `cat` (never-seen, 7k) | `detectors.py`, `probe_transfer.py` |
| Yamagishi et al. (2021). *ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection.* [arXiv:2109.00537](https://arxiv.org/abs/2109.00537) · [baselines](https://github.com/asvspoof-challenge/2021) · [asvspoof.org](https://www.asvspoof.org/) | **used** — challenge the detectors come from; source of the RawNet2 baseline code | `detectors.py` |
| Wang & Yamagishi (2021). *A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection.* [arXiv:2103.11326](https://arxiv.org/abs/2103.11326) | related — the LFCC-LCNN architecture proposed as the far-transfer test in 7k (not yet run) | — |

## 3. Quality metrics — what the edit costs

| source | role | where in repo |
|---|---|---|
| Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) · [code](https://github.com/openai/whisper) | **used** — transcripts for WER | `Tester.py` |
| Word error rate | used — Levenshtein distance written directly in `Tester.wer` rather than via a library ([jiwer](https://github.com/jitsi/jiwer) is the usual equivalent) | `Tester.py` |
| Wang et al. (2023). *CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking.* [arXiv:2303.00332](https://arxiv.org/abs/2303.00332) · [3D-Speaker](https://github.com/modelscope/3D-Speaker) | **used** — speaker embeddings for similarity (`campplus.onnx`, the model shipped with CosyVoice 2) | `Tester.py` |
| Kumar et al. (2023). *TorchAudio-Squim: Reference-less Speech Quality and Intelligibility measures in TorchAudio.* [arXiv:2304.01448](https://arxiv.org/abs/2304.01448) · [tutorial](https://pytorch.org/audio/stable/tutorials/squim_tutorial.html) | **used** — PESQ / STOI / SI-SDR / MOS estimates without a clean reference | `quality.py` |
| ITU-T Recommendation P.862, *PESQ.* [itu.int/rec/T-REC-P.862](https://www.itu.int/rec/T-REC-P.862) | basis — the metric SQUIM estimates | `quality.py` |
| Taal et al. (2011). *An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech* (STOI). IEEE TASLP. [doi:10.1109/TASL.2011.2114881](https://doi.org/10.1109/TASL.2011.2114881) · [pystoi](https://github.com/mpariente/pystoi) | basis — STOI | `quality.py` |
| Le Roux et al. (2018). *SDR – half-baked or well done?* [arXiv:1811.02508](https://arxiv.org/abs/1811.02508) | basis — SI-SDR | `quality.py` |

## 4. Design A — constrained PPO (sections 3, 5)

| source | role | where in repo |
|---|---|---|
| Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | **basis** — clipped surrogate objective | `train_ppo_mel.py` |
| Achiam et al. (2017). *Constrained Policy Optimization.* [arXiv:1705.10528](https://arxiv.org/abs/1705.10528) | related — the constrained-MDP framing for "evade, but keep similarity and WER" | `reward.py` |
| Ray, Achiam & Amodei (2019). *Benchmarking Safe Exploration in Deep Reinforcement Learning* (PPO-Lagrangian). [PDF](https://cdn.openai.com/safexp-short.pdf) | related — the method Design A was named after. `MATH.md` notes the implementation is actually a quadratic penalty with a dual-adapted weight, not textbook PPO-Lagrangian | `reward.py`, `MATH.md` |
| Boyd & Vandenberghe (2004). *Convex Optimization* (log-sum-exp function, ch. 3). [free PDF](https://web.stanford.edu/~boyd/cvxbook/) | basis — soft-min as a scaled negative log-sum-exp, the detector aggregator | `reward.py`, `diffscore.py` |

## 5. Design B — SAC with a gradient hint (section 4)

| source | role | where in repo |
|---|---|---|
| Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) | **basis** — SAC, including the tanh-squashed Gaussian and its log-probability correction | `sac_agent.py`, `train_sac_mel.py` |
| Haarnoja et al. (2018). *Soft Actor-Critic Algorithms and Applications.* [arXiv:1812.05905](https://arxiv.org/abs/1812.05905) | **basis** — automatic temperature (α) tuning, whose steady rise diagnosed Design B's failure | `sac_agent.py` |
| Fujimoto et al. (2018). *Addressing Function Approximation Error in Actor-Critic Methods* (TD3). [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) | **basis** — twin Q critics with a min, against overestimation | `sac_agent.py` |
| Wu & He (2018). *Group Normalization.* [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) | **used** — normalisation in the actor (why `hidden` must divide by 8) | `sac_agent.py` |
| Kingma & Ba (2014). *Adam.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) · Loshchilov & Hutter (2017). *Decoupled Weight Decay Regularization.* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | **used** — optimiser and weight decay | all trainers |

## 6. The white-box attack — PGD (sections 2, 7)

| source | role | where in repo |
|---|---|---|
| Madry et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD). [arXiv:1706.06083](https://arxiv.org/abs/1706.06083) | **basis** — iterative gradient ascent under a bound. Ours bounds with `max_delta · tanh(raw)` and Adam instead of projection and sign steps | `probe_pgd.py`, `collect_pgd.py` |
| Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* [arXiv:1412.6572](https://arxiv.org/abs/1412.6572) | related — origin of gradient-based adversarial perturbations | — |
| Carlini & Wagner (2018). *Audio Adversarial Examples: Targeted Attacks on Speech-to-Text.* [arXiv:1801.01944](https://arxiv.org/abs/1801.01944) | related — optimisation attacks on audio models | — |
| Liu et al. (2019). *Adversarial Attacks on Spoofing Countermeasures of automatic speaker verification.* [arXiv:1910.08716](https://arxiv.org/abs/1910.08716) | related — **the closest prior work to this project**: adversarial attacks specifically against anti-spoofing detectors | — |
| Carlini & Wagner (2016; IEEE S&P 2017). *Towards Evaluating the Robustness of Neural Networks.* [arXiv:1608.04644](https://arxiv.org/abs/1608.04644) | **basis** — optimise the pre-softmax logit margin rather than a probability. Section 7m: the two detectors deciding soft-min were saturated at P ~ 1e-6 with gradients ~5,000x smaller than the rest, so the probability objective had been pushing against a wall. Switching space took the teacher 0.574 -> 0.9949 | `diffscore.detector_margins`, `collect_pgd.py --objective margin` |

## 7. Distillation — the method that worked (section 7)

| source | role | where in repo |
|---|---|---|
| Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) | related — the general idea of training a cheap student on an expensive teacher's outputs. Ours regresses onto PGD edits, not soft labels | `distill.py` |
| Baluja & Fischer (2017). *Adversarial Transformation Networks: Learning to Generate Adversarial Examples.* [arXiv:1703.09387](https://arxiv.org/abs/1703.09387) | related — a network that *generates* adversarial perturbations in one pass instead of optimising each input | `distill.py` |
| Poursaeed et al. (2017). *Generative Adversarial Perturbations.* [arXiv:1712.02328](https://arxiv.org/abs/1712.02328) | related — same idea, and the image-attack counterpart of our fixed-average-edit control (universal vs input-conditioned perturbations) | `distill.py` |
| Andrychowicz et al. (2016). *Learning to learn by gradient descent by gradient descent.* [arXiv:1606.04474](https://arxiv.org/abs/1606.04474) | related — learned update rules applied repeatedly; the nearest analogue to finding that 10 re-hinted passes beat 1 (7b, 7h) | `probe_amplify.py`, `eval_final.py` |
| Yu & Koltun (2015). *Multi-Scale Context Aggregation by Dilated Convolutions.* [arXiv:1511.07122](https://arxiv.org/abs/1511.07122) · van den Oord et al. (2016). *WaveNet.* [arXiv:1609.03499](https://arxiv.org/abs/1609.03499) | used — dilated convolutions for the wide-receptive-field variants in the capacity sweep (which lost to the 5-frame model) | `sac_agent.py`, `distill.py` |

## 7b. Perceptual cost in the loss (section 7q)

The direct-training work needed a penalty that tracks *audible* damage;
`mean(delta^2)` does not, and raising its weight made PESQ worse while the
optimiser simply found loud edits that scored well.

| source | role | where in repo |
|---|---|---|
| Qin et al. (2019). *Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition.* [arXiv:1903.10346](https://arxiv.org/abs/1903.10346) | **basis** — psychoacoustic frequency masking, so an edit is cheap under loud speech and expensive in near-silence. Ours is a simplified per-band approximation of their masking threshold, not the full model | `perceptual.masking_weights`, `perceptual.masked_l2` |
| Yamamoto et al. (2019). *Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram.* [arXiv:1910.11480](https://arxiv.org/abs/1910.11480) | **basis** — multi-resolution STFT distance, computed on the waveform actually heard rather than on the mel. Raised quality efficiency from 0.18 to 2.30 soft-min per PESQ point | `perceptual.mrstft` |

## 8. Transfer to unseen detectors (sections 7d, 7k)

| source | role | where in repo |
|---|---|---|
| Papernot et al. (2016). *Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples.* [arXiv:1605.07277](https://arxiv.org/abs/1605.07277) | related — why an edit optimised against some models may fool others | `probe_transfer.py` |
| Liu et al. (2016). *Delving into Transferable Adversarial Examples and Black-box Attacks.* [arXiv:1611.02770](https://arxiv.org/abs/1611.02770) | related — ensemble attacks transfer better; relevant to attacking four detectors at once | `probe_transfer.py` |
| Tramèr et al. (2017). *Ensemble Adversarial Training: Attacks and Defenses.* [arXiv:1705.07204](https://arxiv.org/abs/1705.07204) | related — ensemble perspective on attack transfer | — |

## 9. Reference cleaning (section 7i)

| source | role | where in repo |
|---|---|---|
| Sainburg et al. (2020). *Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires.* PLOS Comp. Biol. [doi:10.1371/journal.pcbi.1008228](https://doi.org/10.1371/journal.pcbi.1008228) · [noisereduce](https://github.com/timsainb/noisereduce) | **used** — spectral-gate denoising | `probe_reference.py` |

## 10. Statistics — why several effects vanished (sections 7f, 7g)

| source | role | where in repo |
|---|---|---|
| Button et al. (2013). *Power failure: why small sample size undermines the reliability of neuroscience.* Nature Reviews Neuroscience. [doi:10.1038/nrn3475](https://doi.org/10.1038/nrn3475) | related — small samples produce inflated, unreplicable effects, which is exactly what happened to the breath-token result | `RESULTS.md` 7f |
| Ioannidis (2008). *Why Most Discovered True Associations Are Inflated.* Epidemiology. [doi:10.1097/EDE.0b013e31818131e7](https://doi.org/10.1097/EDE.0b013e31818131e7) | related — the "winner's curse": the first significant estimate overstates the true effect, matching the halving seen each time n doubled | `RESULTS.md` 7f, 7g |

## 11. Environment

| source | role |
|---|---|
| Microsoft. *What is Smart App Control?* [support.microsoft.com](https://support.microsoft.com/en-us/topic/what-is-smart-app-control-285ea03d-fa88-4d56-882e-6698afdb7003) · [App Control for Business](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/appcontrol) | the Windows policy that blocked `tiktoken` and `llvmlite` mid-session (`requirements-voiceclone.txt`, trap 5) |

---

## Built here, with no direct external source

These are this project's own design decisions. The "related" rows above are
the nearest literature, not their origin.

| process | file | RESULTS.md |
|---|---|---|
| Hooking the mel between CosyVoice 2's flow decoder and HiFT; escaping `inference_mode` via `__wrapped__` | `mel_policy.py`, `diffscore.py` | §1 |
| Blending soft-min with the mean (0.5 / 0.5) so already-passing detectors keep gradient weight | `diffscore.py` | §3, §4 |
| Gradient hint as extra actor input channels | `sac_agent.py` | §4 |
| Fixed-average-edit control to test for a universal perturbation | `distill.py` | §7 |
| Applying the student repeatedly with a re-computed hint each pass | `eval_final.py` (`AmplifiedAdapter`) | §7b, §7h |
| Paired same-seed evaluation, so the edit is the only difference between arms | `eval_*.py`, `probe_*.py` | §5 onward |
| Leave-one-detector-out collection with the held-out detector excluded from both target and hint | `collect_pgd.py`, `eval_final.py` | §7d |
| Duration and emphasis controls for the prosody test | `prosody.py` | §7f |
