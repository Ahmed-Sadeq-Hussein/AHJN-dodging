# Plan_X — results and reasoning

What was run, what came out, and why each conclusion follows. Every figure
lives in [`../graphs/`](../graphs/INDEX.md) with an index saying what each one
shows; every number below is reproducible from the scripts named against it.

Status: complete. Sections 0-6 cover the two RL designs and the audits that
explain why they failed; section 7 covers the distillation run that worked and
the eighteen follow-ups that refined, corrected and bounded it. Sections 7m and
7r are the two places where a conclusion recorded as a ceiling turned out to be
a measurement artefact -- read those first if you read nothing else.

A readable narrative version of this document, with the figures inline, is the
final report; this file remains the complete record.

---

## 0. The premise, and how it changed

The project goal is to make CosyVoice2 clones read as *human* to four
anti-spoofing detectors (AASIST, AASIST-L, RawNet2, RawGAT-ST; 1 = human,
0 = AI) without losing speaker similarity or intelligibility.

The first measurement undercut the premise. A clone of `ahmed`, from a clean
recording, scores **0.997 / 0.984 / 1.000 / 1.000**. CosyVoice2 already
defeats all four detectors outright when the reference is clean; there is
nothing to evade.

The voices with headroom — riven 0.089, swain 0.087, samira 0.170 — are game
voice-over. Their *reference files* fail RawNet2 and RawGAT-ST before any
cloning happens. So the detectable artefacts are inherited from studio
processing in the reference, not produced by the synthesiser. That reframes
the problem, and it is worth stating plainly because it means a large part of
the measured "detection" is a property of the source audio.

This spread is the single largest effect in the whole project: **0.24
(evelynn) to 0.96 (ahmed) at baseline**, larger than any intervention
described below.

---

## 1. The intervention point

CosyVoice2 runs text → speech tokens → flow decoder → **mel** → HiFT vocoder →
waveform. Everything from the mel onward is differentiable and locally held,
so the mel is the natural place to intervene: late enough to have direct
acoustic effect, early enough that the vocoder smooths discontinuities.

`mel_policy.mel_edit_hook` intercepts the mel and adds a learned residual Δ
before the vocoder. Two implementation facts cost real time and are recorded
so they are not rediscovered:

- **`torch.inference_mode()` poisons autograd.** CosyVoice2 generates under
  it, and tensors created there can never enter a graph. The first PGD probe
  silently optimised nothing for 60 steps while reporting `|delta| = 0.0000`,
  and nearly produced the conclusion "mel editing cannot work". The escape is
  `torch.inference_mode(False)` for policy tensors and the decorator's
  `__wrapped__` attribute for `hift.inference`.
- **cuDNN refuses RNN backward in `eval()`**, and RawNet2 contains a GRU.
  Switching to `train()` would flip its BatchNorm to batch statistics and
  change the very scores being measured, so the fix is to disable cuDNN
  (`diffscore.differentiable()`) and accept native kernels.

---

## 2. Feasibility: what is reachable at all (figure 6)

`probe_pgd.py` answers the question RL is an expensive way to ask: with full
white-box access, how far can a bounded mel edit move the objective?

| | baseline | PGD, 60 steps |
|---|---|---|
| soft-min | 0.085 | **0.666** |
| similarity | 0.836 | 0.860 (*improved*) |
| WER | 11.8% | 11.8% (unchanged) |

Evasion is reachable, cheaply, without damaging the voice. This is an upper
bound on what any policy could learn, and it establishes that later failures
are failures of *optimisation*, not of the intervention point.

The trajectory matters as much as the endpoint. PGD needs **|Δ| ≈ 0.16** to
break through — the jump happens between step 20 and step 30, as |Δ| goes
0.135 → 0.162. At |Δ| = 0.0065 (step 1) the objective has not moved at all.

---

## 3. Design A — PPO-Lagrangian (figures 1, 2, 3)

Constrained policy gradient: maximise the detector soft-min subject to
similarity ≥ 0.70 and WER ≤ 0.15, with dual-adapted multipliers. 130
iterations, 1,040 episodes.

**Result: significant degradation.** Slope −0.000727, **t = −7.26**. The
policy made detection *easier* than doing nothing.

**Cause, from figure 2:** pure soft-min at k = 8 hands 50.2% / 49.8% of its
gradient to RawNet2 / RawGAT-ST and **0.06% / 0.03% to AASIST / AASIST-L**.
The two detectors that were already passing were undefended, and the policy
degraded them (AASIST t = −3.54, AASIST-L t = −2.47) while never improving
the two it was optimising. The aggregator, not the algorithm, chose that
outcome.

---

## 4. Design B — SAC with a gradient hint (figures 1b, 4, 5)

Three changes: a blended objective (0.5·soft-min + 0.5·mean) so no detector
carries near-zero weight; a **gradient hint** — d(objective)/d(mel), the
signal Design A discarded — as extra actor input channels; and twin Q critics
conditioned on the action, since Design A's V(mel) could say an utterance was
hard but nothing about whether an edit was good.

396 iterations across two sessions, 2,520 episodes in the 9-hour run.

**Result: flat. Slope +0.000033, t = +0.46**, against a resolvable threshold
of ±0.045. This is a *well-powered* null, not an underpowered one.

What Design B did fix:

- degradation eliminated: t = −7.26 → **+0.46**
- the previously-undefended detectors now improve: AASIST **+0.034**,
  AASIST-L **+0.016**

What it did not fix: anything about the actual objective.

**Figure 4 rules out action scale.** The `max_delta` cap of 0.05 held |Δ| at
0.0072 mean, with 0.0% of iterations above 0.02 — it never approached its own
ceiling, and in fact *shrank* (t = −4.24).

**Figure 5 says what went wrong.** `log_std` climbed −4.30 → −3.43
(t = +1440) and the SAC temperature α rose 0.060 → 0.115 (t = +316) — the
entropy controller *increasing* exploration for nine hours straight. Q-loss
was low (~0.03), so the critic predicts reward accurately but cannot **rank**
two candidate edits. When ranking fails, entropy is the only term left that
the objective can improve, and it inflates.

**Constraints never bound**: similarity 0.827, WER 5.2%, λ pinned at its
0.250 floor throughout. The penalties the brief asked for worked; they were
never the limiting factor.

---

## 5. A vs B on unseen voices (figures 7, 8)

`eval_ab_voices.py`. Five voices drawn at random
(`random.Random(20260906).sample`) — Tywin_2, ahmed, evelynn, mordekaiser,
thresh, all held-out — plus the four training voices as a reference group.
Four utterances each, three arms per utterance from one shared seed, so the
mel edit is the only difference between arms.

| | mean detector | | | soft-min | | |
|---|---|---|---|---|---|---|
| | base | A | B | base | A | B |
| 5 random voices | 0.437 | 0.412 (−0.025) | 0.420 (−0.017) | 0.271 | 0.261 | 0.276 |
| 4 trained voices | 0.366 | 0.304 (**−0.062**) | 0.366 (−0.001) | 0.199 | 0.167 | 0.199 |

Pooled over 36 clips: **Design A −0.0417 mean detector, t = −3.78**
(significant, worse than doing nothing). **Design B −0.0095, t = −0.80** —
indistinguishable from doing nothing, and neutral on its own training voices
to three decimal places.

**A correction this evaluation forced.** Earlier reporting claimed Design B
"sustained soft-min at 2.4× baseline". That compared B's four-voice average
(0.21) against *riven's* baseline alone (0.087). The correct paired control
across the same four voices is **0.199**, and B scores **0.199**. There was
no gain; there was a mismatched denominator.

**Why the average is zero:** B moves riven **+0.075** but swain −0.027,
samira −0.023, jhin −0.028. Not a learned function — one voice benefiting
while the rest pay for it.

Neither design costs quality (figure 8): similarity 0.78–0.83 against a 0.70
floor, WER 5.5% → 6.0% / 6.2% against a 15% ceiling.

---

## 6. Two findings that came from listening

### 6a. The magnitude was never the barrier

The obvious reading of section 4 is that capping |Δ| at 0.05 made PGD's
0.16-magnitude solution unreachable. It did — but that is not why Design B
failed. Run **B3h**, before the cap, sat at **|Δ| 0.12–0.20 for its entire
length**, squarely in PGD's solution range, and reached soft-min 0.186. Over
that run, **corr(|Δ|, soft-min) = −0.554**: at the right magnitude, larger
edits were *worse*.

So the barrier is **direction**, not scale. PGD's solution is one specific
vector in 80×T dimensions; moving 0.16 in any other direction is loud noise.
Gradient descent finds it in 60 steps and sampling does not. This is the
central negative result of the project, and it is what motivates section 7.

### 6b. Detection difficulty scales with clip length

`demo_listen.py` on evelynn, same voice, three script lengths:

| script | length | baseline soft-min | RawNet2 | RawGAT-ST |
|---|---|---|---|---|
| short | 4.6 s | 0.356 | 0.500 | 0.499 |
| medium | 14.5 s | 0.154 | 0.251 | 0.117 |
| long | 50 s | **0.050** | **0.0007** | **0.0001** |

A **7× collapse** in baseline soft-min from 4.6 s to 50 s. More audio means
more evidence and the detectors converge hard. Every training run in this
project used 5–15 s utterances — the easiest regime — so all results above
describe short-form only. This was not measured until late and should be
treated as a scope limit on everything else.

### 6c. Perceptual quality was never measured until now

The brief says "maintain MOS and WER". Only WER and speaker similarity were
tracked, and both are blind to buzz, roughness and phase smear — speaker
embeddings are *trained* to be invariant to channel distortion. `quality.py`
now adds torchaudio SQUIM. Applied to the PGD attack:

| | baseline | PGD | Δ |
|---|---|---|---|
| STOI | 0.997 | 0.985 | −0.012 |
| PESQ | 4.030 | 3.358 | **−0.672** |
| SI-SDR | 26.19 dB | 21.54 dB | −4.65 dB |

PGD costs real but moderate quality: intelligibility intact, PESQ from
excellent to good. Note SQUIM's MOS estimate saturated at 4.995 for both arms
and was uninformative here.

---

## 7. Distillation from PGD (figures 9, 12)

### Rationale

Sections 3–6 converge on one diagnosis. The intervention point works
(section 2), the action magnitude is reachable (6a), the constraints never
bind (4), and the aggregator bug is fixed (4) — yet RL produces nothing,
because finding the right *direction* in 80×T dimensions by sampling is
hopeless while gradient descent does it in 60 steps.

So supply the target instead of discovering it. `collect_pgd.py` runs PGD on
242 (voice, sentence) pairs and stores (mel, gradient hint, Δ*);
`distill.py` trains the actor to predict Δ* by regression. The network never
explores — it only interpolates between solved instances.

### The dataset

242 examples, 11 voices × 22 sentences, ~44 s each (≈3 h).

| | value |
|---|---|
| soft-min | 0.284 → **0.574** (gain +0.290) |
| objective | 0.358 → 0.683 |
| similarity cost | **+0.0115** mean (negative — i.e. *improved* — on 6 of 11 voices) |
| \|Δ\| | 0.161 |

PGD's success is strongly voice-dependent: swain +0.436, jhin +0.424,
tahmkench +0.424, but ahmed +0.042 (already at 0.951, nothing to gain) and
david reaches only 0.159 from a baseline of 0.065. The 0.666 headline from
section 2 was one favourable voice, not a universal ceiling.

**Target filtering.** PGD is unconstrained — it maximises the detector
objective and nothing else. On `ahmed` it spends |Δ| = 0.17 to gain nothing
while dropping similarity 0.824 → 0.664. Regressing onto that would teach the
network to make a large damaging edit on exactly the inputs needing none.
Targets with gain < 0.02 or similarity cost > 0.10 are dropped: **221 of 242
kept** (12 no-gain, 9 similarity-cost).

### The control that settles the open question

Before reading any learning curve: apply **one fixed per-channel average
edit** — the mean of all PGD solutions — blind to every utterance.

**Result: −5.4% of PGD's gain** (−15.5% on the final run's larger sample).
Worse than no edit at all.

So no universal perturbation exists. PGD's solution is genuinely
utterance-specific. This is the result that makes the RL failure
*intelligible* rather than merely disappointing: there was never a single
constant edit to converge on, and the useful function is conditional on the
input.

### Capacity is the enemy (figure 12)

Every sweep run trains on the same 221 targets. Judged by the fraction of
PGD's objective gain recovered on held-out sentences, pushed through HiFT and
the real detectors:

| config | params | receptive field | recovered |
|---|---|---|---|
| s_wide | 270k | 1.33 s | **−0.4%** |
| s_narrow | 98k | 0.14 s | +10.0% |
| s_wd (wd 1e-2) | 270k | 1.33 s | +13.5% |
| s_lowlr_wd | 98k | 0.31 s | +21.1% |
| s_noise | 270k | 1.33 s | +23.8% |
| s_tiny | 33k | 0.10 s | +24.6% |
| s_lowlr | 98k | 0.31 s | +25.9% |
| **r_micro** | **21k** | **0.05 s** | **+33.6%** |
| final (r_micro, longer) | 21k | 0.05 s | +27.0%, **best +41.8%** |

Monotone in capacity: with 221 examples in an 80×T output space, every
reduction helped. The initial instinct to widen the receptive field to 1.33 s
was exactly backwards.

**The transferable signal is nearly pointwise.** The best model sees five mel
frames — 0.05 s. Whatever structure in PGD's attack generalises across
utterances is a local per-frame spectral reshaping, not a temporal pattern.

Two other diagnostics from figure 9. Direction agreement generalises weakly
but genuinely: train cosine +0.163 against val +0.139 — a small gap, unlike
the wide net's +0.449 / +0.047, which was almost pure memorisation. And the
magnitude ratio settles at **0.08**: the distilled edit is one-twelfth of
PGD's size, yet recovers a third of its gain. It is not copying PGD, it is
extracting the cheap, shared part of the attack.

*(One sweep config, `r_nano` at hidden = 12, failed to build — GroupNorm
requires channels divisible by its 8 groups. A configuration error in the
sweep, not a code defect; hidden = 24 was already the winner.)*

### Held-out voices and held-out sentences (figures 10, 11)

`eval_final.py`. Four voices in no training set (Tywin_2, evelynn,
mordekaiser, thresh) and six sentences in no collection run, plus four
collection voices as a reference group. Three arms from one shared seed per
utterance, so the mel edit is the only difference.

**HELD-OUT voices, unseen sentences** (24 clips), distilled vs baseline:

| metric | baseline | Design B | distilled | Δ | t |
|---|---|---|---|---|---|
| mean detector | 0.3400 | 0.3345 | **0.4106** | **+0.0707** | **3.97** |
| soft-min | 0.1132 | 0.1161 | **0.1496** | **+0.0365** | **3.03** |
| similarity | 0.7888 | 0.7897 | 0.7873 | −0.0015 | −0.65 |
| WER | 6.74% | 7.31% | 7.56% | +0.82pp | 1.01 |
| PESQ | 2.832 | 2.878 | 2.866 | +0.034 | 1.04 |
| STOI | 0.9778 | 0.9791 | 0.9766 | −0.0011 | −0.72 |

**REFERENCE voices, unseen sentences** (24 clips): mean detector +0.1000
(t = 6.44), soft-min +0.0649 (t = 3.34).

**Every one of the eight voices improved**, four of them never seen at any
stage of training:

| voice | set | baseline | Design B | distilled |
|---|---|---|---|---|
| Tywin_2 | held-out | 0.1346 | 0.1398 | 0.1776 |
| evelynn | held-out | 0.1131 | 0.1135 | 0.1815 |
| mordekaiser | held-out | 0.0987 | 0.0887 | 0.1010 |
| thresh | held-out | 0.1062 | 0.1223 | 0.1385 |
| Ego | reference | 0.3060 | 0.3290 | 0.4582 |
| Joffrey | reference | 0.5155 | 0.5263 | 0.5776 |
| Silco | reference | 0.1094 | 0.1090 | 0.1250 |
| Tywin | reference | 0.5535 | 0.5072 | 0.5833 |

**The cost is nil on every axis measured** (figure 11). Similarity, WER,
PESQ and STOI are all statistically indistinguishable from baseline; PESQ is
nominally *up*. This is the first intervention in the project that buys
evasion without paying for it, and the only reason we can assert the
perceptual half of that claim is that section 6c added the metric.

### What this establishes

Same objective, same intervention point, same four detectors, all evaluated
on held-out voices with unseen sentences:

| approach | cost | soft-min | vs baseline |
|---|---|---|---|
| baseline | — | 0.113 | — |
| Design A (PPO, 1,040 episodes) | 4 GPU-h | — | significantly **worse**, t = −3.78 |
| Design B (SAC, 2,520 episodes) | 12 GPU-h | 0.116 | +0.003, null |
| distilled x1 (21k params) | 1 forward pass | 0.150 | +0.037, t = 3.03 |
| **distilled x4** | **4 forward passes** | **0.239** | **+0.126, t = 4.17** |
| PGD (per-utterance oracle) | 60 gradient steps | 0.574 | +0.290 |

The RL failure was never about the intervention point, the action bound, the
constraints, or the detector aggregator. It was that the useful edit function
is conditional on the input, has no constant component to converge on
(control: −5.4%), and lives in a direction that sampling cannot find. Given
the answer as a supervised target, a **21,352-parameter network with a 0.05 s
receptive field** learns roughly a third of it and transfers to unseen voices
and unseen sentences at no measurable quality cost.

### 7b. Iterating the distilled edit

The distilled network predicts edits at |Δ| ≈ 0.08× PGD's magnitude while
recovering ~35% of its gain. Two readings: it found a cheaper direction that
would overshoot if scaled, or regression-to-the-mean shrank a correct
direction. `probe_amplify.py` separates them on stored held-out-sentence
mels, where nothing needs regenerating.

**Scaling the edit fails. Iterating it compounds.**

| scale | iters=1 | iters=2 | iters=3 | iters=4 | iters=5 |
|---|---|---|---|---|---|
| 0.5 | +33% | +44% | +53% | +56% | +65% |
| **1.0** | +38% | +51% | +55% | +68% | **+76%** |
| 1.5 | +34% | +52% | +65% | **+76%** | +71% |
| 2.0 | +34% | −7% | +57% | | |
| 4.0 | −43% | +4% | +4% | | |
| 8.0 | −123% | −95% | −100% | | |

At scale 1.0 the progression is monotone (38 → 51 → 55 → 68 → 76%), and
absolute soft-min reaches **0.582 against PGD's 0.641** — 91% of the oracle's
level — at |Δ| = 0.047, still 3.5× smaller than PGD's 0.161. Multiplying the
edit by 4 or more is catastrophic (−43%, −123%).

That asymmetry is the mechanism. A memorised displacement would improve under
scaling and gain nothing from re-application. A learned *descent direction*
behaves exactly the opposite way: one step of the right size helps, four
steps help more, and one step of four times the size overshoots into
nonsense. Each iteration re-computes the gradient hint on the already-edited
mel, so the network is being asked a fresh question each pass rather than
repeating an answer.

This also explains the shrunken magnitude benignly. The network is not a
weaker version of PGD; it is a *step operator* that PGD-style iteration can
be built on, at four forward passes instead of sixty gradient steps.

### 7c. Confirmed on held-out voices with 4 iterations (figures 10, 11, 13)

Same protocol as above -- unseen voices, unseen sentences, shared seed -- with
the distilled edit applied four times at scale 1.0.

**HELD-OUT voices, unseen sentences** (24 clips):

| metric | baseline | Design B | distilled x1 | **distilled x4** | delta | t |
|---|---|---|---|---|---|---|
| mean detector | 0.3400 | 0.3345 | 0.4106 | **0.5573** | **+0.2173** | **9.66** |
| soft-min | 0.1132 | 0.1161 | 0.1496 | **0.2388** | **+0.1257** | **4.17** |
| similarity | 0.7888 | 0.7897 | 0.7873 | **0.7952** | **+0.0064** | **2.05** |
| WER | 6.74% | 7.31% | 7.56% | 6.78% | +0.03pp | 0.04 |
| PESQ | 2.832 | 2.878 | 2.866 | 2.874 | +0.043 | 1.15 |
| STOI | 0.9778 | 0.9791 | 0.9766 | 0.9764 | -0.0013 | -0.95 |

**REFERENCE voices** (12 clips): mean detector +0.1919 (t = 4.20), soft-min
+0.1578 (t = 3.76).

| voice | set | baseline | Design B | distilled x4 |
|---|---|---|---|---|
| Tywin_2 | held-out | 0.1346 | 0.1398 | **0.3406** |
| evelynn | held-out | 0.1131 | 0.1135 | **0.2399** |
| mordekaiser | held-out | 0.0987 | 0.0887 | **0.1486** |
| thresh | held-out | 0.1062 | 0.1223 | **0.2262** |
| Ego | reference | 0.3060 | 0.3290 | **0.4843** |
| Joffrey | reference | 0.5155 | 0.5263 | **0.6528** |

Iterating roughly **triples** the single-pass gain: soft-min +0.037 -> +0.126,
mean detector +0.071 -> +0.217. Held-out soft-min reaches **2.1x baseline**.

And the cost is still nothing. WER is unchanged (+0.03 percentage points),
PESQ and STOI are flat, and **similarity significantly improves** (+0.0064,
t = 2.05) -- the same effect PGD showed in section 2, where the attack raised
similarity from 0.836 to 0.860. Whatever the edit removes appears to be
artefact rather than speaker identity.

### 7d. Leave-one-detector-out: does it transfer to a detector never optimised?

The strongest claim so far is bounded by four specific checkpoints. To test
whether the operator learned something about synthetic-speech artefacts in
general, PGD targets were re-collected optimising **only AASIST, AASIST-L and
RawNet2** (132 examples, 11 voices x 12 sentences). RawGAT-ST is scored
throughout but influences neither the PGD target, nor the gradient hint at
collection, nor the hint at inference -- `AmplifiedAdapter` takes the
detector subset from the checkpoint, because computing the hint from all four
would have quietly re-introduced the held-out detector and made the
experiment meaningless.

**PGD itself transfers**, measured on the targets:

| detector | before | after | change | role |
|---|---|---|---|---|
| AASIST | 0.6476 | 0.9997 | +0.3521 | optimised |
| AASIST-L | 0.4779 | 0.9994 | +0.5214 | optimised |
| RawNet2 | 0.3228 | 0.5337 | +0.2109 | optimised |
| **RawGAT-ST** | 0.2890 | 0.3604 | **+0.0714** | **HELD OUT** |

t = +5.95, improved on 79% of examples.

**So does the distilled operator**, on held-out voices with unseen sentences
(24 clips, distilled x4):

| detector | baseline | Design B | distilled x4 | change | t | role |
|---|---|---|---|---|---|---|
| AASIST | 0.3988 | 0.3506 | 0.5945 | +0.1957 | 2.66 | optimised |
| AASIST-L | 0.4962 | 0.5227 | 0.7845 | +0.2883 | 4.28 | optimised |
| RawNet2 | 0.3545 | 0.3357 | 0.5071 | +0.1526 | 2.74 | optimised |
| **RawGAT-ST** | 0.1103 | 0.1290 | **0.1819** | **+0.0716** | **2.44** | **HELD OUT** |

Aggregate on held-out voices: mean detector +0.1770 (t = 5.54), soft-min
+0.1055 (t = 2.78), similarity -0.0034 (not significant), WER +1.4pp (not
significant).

The held-out gain (+0.0716) is almost identical to what PGD transferred at
target level (+0.0714). The operator inherits the attack's cross-detector
generality rather than memorising three detectors' idiosyncrasies -- so the
edit is acting on something the detectors share, not on per-model quirks.

Two caveats. The transfer is real but **much smaller** than the optimised
detectors' gains (+0.07 against +0.15 to +0.29), so most of the effect is
still detector-specific. And this is one fold: RawGAT-ST was chosen because
it is one of the two detectors that actually catch these voices, but a single
holdout cannot say whether every detector transfers equally.

### 7e. Long form: the gain does not survive length

> **Superseded in part by 7r.** The measurement below is correct for the
> learned operator applied per chunk, and that result stands. The *conclusion*
> that long audio is a ceiling does not: the suspected aggregation mismatch
> named at the end of this section turned out to be the whole story. Optimising
> each chunk against the WHOLE FILE's score reaches +0.427 soft-min at 78 s.
> Read 7r before taking this section as a scope limit.

Section 6b showed baseline soft-min collapsing ~7x from 4.6 s to 50 s but
never tested the working operator there. `probe_longform.py` runs three
lengths built from unseen `EVAL_TEXTS`: 15 words (~4.6 s), 44 words
(~13.6 s), 182 words (~72 s, which CosyVoice2 splits into 4 chunks).

| voice | length | audio | baseline | distilled | gain |
|---|---|---|---|---|---|
| evelynn | short | 4.6 s | 0.0587 | 0.0868 (x4) | +0.028 |
| evelynn | medium | 13.6 s | 0.0690 | **0.3277** (x4) | **+0.259** |
| evelynn | long | ~72 s | 0.0717 | 0.0944 (x4) | +0.023 |
| thresh | short | 4.6 s | 0.0684 | 0.0880 (x4) | +0.020 |
| thresh | medium | 13.6 s | 0.0861 | 0.1551 (x4) | +0.069 |
| thresh | long | ~72 s | 0.1876 | 0.1725 (x2) | **-0.015** |

**Medium length is where the operator is strongest** -- evelynn's +0.259 is
larger than anything in the main evaluation, which makes sense: 13.6 s is
closest to the 5-15 s utterances every PGD target was collected on. At 72 s
the gain is +0.023 and -0.015, i.e. gone.

The likely mechanism is an aggregation mismatch rather than a capability
failure. CosyVoice2 chunks long text and `mel_edit_hook` fires once per
chunk, so the operator edits ~20 s mels independently. The detectors window
the *concatenated* file into 64600-sample (~4 s) slices and average -- ~19
windows for a 72 s clip against 1-2 for a training utterance. An edit tuned
to raise a local objective is being diluted across an average it never saw.

**An engineering constraint fell out of this too.** Four iterations on a 72 s
clip exhaust a 10 GB card: each hint backpropagates through HiFT over a ~20 s
mel plus all four detectors, and the peak is reached partway through the
third voice. Two iterations complete. Long-form work therefore needs either
gradient checkpointing, chunked hint computation, or a smaller detector
batch -- `diffscore.detector_scores` now takes `max_windows` (default 8 in
the hint) as a partial guard, and `probe_longform.py` frees cached blocks
between clips, but neither is sufficient at four iterations.

### 7f. Does dramatising the delivery help? Mostly no

CosyVoice2's tokenizer accepts fine-grained prosody tags (`<strong>`,
`[breath]`, `[quick_breath]`, `[laughter]`) and the build exposes
`inference_instruct2` for natural-language style prompting. Neither was wired
into `cloners.py`; both now are. Eight renditions of identical content were
compared per voice and line (`prosody.py`, `probe_prosody.py`).

**Two bugs had to be fixed before any number here meant anything.**

The markup route needs `text_frontend=False` -- the English text normaliser
rewrites punctuation and would strip the tags before the tokenizer sees them.
And `inference_instruct2` silently ignored the instruction: `frontend_zero_shot`
loads a cached speaker and discards `prompt_text` entirely whenever
`zero_shot_spk_id` is non-empty (frontend.py:186), so two opposite
instructions produced **byte-identical audio** (verified by md5). Passing an
empty speaker id fixes it.

**The first measurement was a spectacular false positive.** On one clip,
`[breath]` tags took soft-min from 0.086 to 0.804 -- larger than every result
in sections 3-7 combined. Controls were added to separate the mechanisms, and
they held up at n=1: `neutral_long` (matched duration, no tags) gave +0.042,
`emphasis` alone +0.002, `breath_only` +0.694. It looked like breath sounds
specifically.

Then it was run properly.

| n | style | d soft-min | t | win rate |
|---|---|---|---|---|
| 36 (6 voices) | breathy | +0.0789 | 1.57 | 56% |
| **72 (12 voices)** | **breathy** | **+0.0133** | **0.39** | **51%** |
| 72 | breath_only | +0.0404 | 1.20 | 50% |
| 72 | emphasis | +0.008 | 0.18 | -- |
| 72 | pauses | -0.030 | -0.64 | -- |
| 72 | instruct_dramatic | -0.039 | -0.84 | -- |

**A 51% win rate is a coin flip.** The effect halved from n=36 to n=72 and the
single clip that started this sat near the top of a range spanning -0.83 to
+0.90. Per-pair variance (sd 0.30) is roughly four times the effect.

**But a real mechanism is hiding inside the null.** Per detector, breathy vs
neutral over 72 pairs:

| detector | change | t | front end |
|---|---|---|---|
| AASIST | +0.0667 | +1.59 | spectro-temporal graph attention |
| AASIST-L | +0.0790 | +1.79 | spectro-temporal graph attention |
| RawNet2 | **-0.0756** | **-1.55** | raw waveform (SincNet) |
| RawGAT-ST | +0.0290 | +0.61 | raw waveform + graph attention |

Breath tokens help the spectral-attention family and **hurt** the raw-waveform
one. Inserted non-speech reads as human to a model looking at spectral
structure and as artefact to a model looking at the waveform directly. The two
cancel, which is exactly why soft-min -- a worst-case aggregator -- shows
nothing. This is the same failure mode as Design A's aggregator problem
(section 3), arriving from a completely different direction.

**Instruct mode is currently unusable** regardless: WER runs 82-160%, i.e. the
model speaks material that is not in the target text. Worth revisiting only if
the instruct pathway is needed for its own sake.

### 7g. Training on dramatised speech, and whether prosody multiplies the edit

Two further questions: does PGD behave differently on expressive speech, and
does an operator *trained* on it do better?

**PGD is completely indifferent to delivery.** 82 targets collected on
breathified text (`collect_pgd.py --breathify`), paired against the same
(voice, sentence) pairs in the neutral corpus, both optimising all four
detectors:

| | neutral | breathy | diff | t |
|---|---|---|---|---|
| soft-min before | 0.2982 | 0.3401 | +0.042 | +1.98 |
| soft-min after PGD | 0.5990 | 0.6516 | +0.053 | +1.63 |
| **PGD objective gain** | +0.3310 | +0.3292 | **-0.0017** | **-0.08** |
| \|delta\| | 0.1607 | 0.1607 | -0.0000 | -0.01 |
| similarity cost | +0.0094 | +0.0089 | -0.0005 | -0.13 |

Identical gain, identical edit magnitude, identical cost. Expressive prosody
neither helps nor hinders the white-box attack.

**Training on dramatised speech does not help either.** A distilled operator
on the breathy corpus reached +36.6% best recovery from 51 training examples,
against +41.8% from 221 neutral ones -- equivalent per example. Cross-evaluated
on 4 held-out voices, both operators on both styles:

| operator | test style | plain | edited | gain | t |
|---|---|---|---|---|---|
| neutral-trained | neutral | 0.1061 | 0.1652 | +0.0592 | 2.94 |
| neutral-trained | breathy | 0.2297 | 0.3947 | +0.1650 | 4.31 |
| breath-trained | neutral | 0.1061 | 0.1604 | +0.0543 | 2.93 |
| breath-trained | breathy | 0.2297 | 0.3647 | +0.1350 | 4.03 |

The neutral-trained operator is at least as good in every cell. Corpus style
is not a useful training variable.

**Does prosody multiply the edit?** On the first 4 voices the interaction
looked strong (+0.1058, t = +2.38, larger on 71% of pairs). Given that this
session had already produced two effects that evaporated under power, it was
re-run on 6 more voices. Pooled over 10 voices, 60 pairs:

| effect | value | t |
|---|---|---|
| edit gain on neutral speech | +0.1252 | **+5.14** |
| edit gain on breathy speech | +0.1786 | **+7.56** |
| interaction (breathy - neutral gain) | +0.0534 | +1.68 |
| prosody alone | +0.0603 | +1.73 |

The interaction halved and fell below significance, exactly as the
prosody-alone effect did when n went 36 -> 72. **Three separate prosody
effects in this session shrank by about half each time the sample doubled**,
which is the signature of noise, not of a small real effect being resolved.

What *is* solid is the operator itself: +0.125 on neutral speech (t = 5.14)
and +0.179 on breathy (t = 7.56), the strongest and best-powered result in the
project. Absolute levels, pooled over 10 voices:

| condition | soft-min |
|---|---|
| neutral, no edit | 0.2223 |
| neutral + edit | 0.3475 |
| breathy, no edit | 0.2826 |
| **breathy + edit** | **0.4611** |

The best condition is 2.07x the worst. Whether the breathy half of that is
real remains unproven; the edit half is not in doubt.

### 7h. Ten passes, not four -- the operator catches its teacher

Section 7b stopped the iteration scan at 5 passes with the curve still
rising, and every evaluation since used 4. Extending the scan on stored
held-out-sentence mels:

| passes | 1 | 2 | 4 | 6 | 8 | **10** | 12 |
|---|---|---|---|---|---|---|---|
| soft-min | 0.446 | 0.477 | 0.525 | 0.574 | 0.598 | **0.642** | 0.608 |
| % of PGD's gain recovered | +36 | +54 | +73 | +82 | +89 | **+97** | +95 |

PGD reaches 0.6325 on the same mels in 60 gradient steps. **Ten forward
passes reach 0.6416 -- 97.1% of the gain, at |delta| = 0.068, an edit 2.4x
smaller than PGD's 0.161.** Twelve passes is slightly worse, so the optimum
is near ten.

> **The 97.1% is measured on TRAINING voices only.** Section 7l re-measured it
> properly across all 15 voices: **79% on trained voices, 45% on held-out
> ones.** The conclusion drawn from this number at the time -- that the student
> had caught its teacher, so more data could not help -- was wrong, and adding
> voices is now the largest measured lever remaining. The figures in this
> section are correct for the mels they were computed on; the generalisation
> claim was not.

Confirmed end-to-end on held-out voices with unseen sentences (24 clips):

| metric | baseline | Design B | distilled x4 | **distilled x10** | delta | t |
|---|---|---|---|---|---|---|
| mean detector | 0.3400 | 0.3345 | 0.5573 | **0.6934** | **+0.3535** | **12.26** |
| soft-min | 0.1132 | 0.1161 | 0.2388 | **0.3727** | **+0.2595** | **4.52** |
| similarity | 0.7888 | 0.7897 | 0.7952 | 0.7907 | +0.0019 | 0.27 |
| WER | 6.74% | 7.31% | 6.78% | **6.24%** | **-0.50pp** | -0.45 |

Mean detector score more than doubles, and WER *improves*. This doubles the
previous best (+0.126 at four passes) and it is the strongest result in the
project by a wide margin (t = 12.26).

**Correction, added after 7k measured it:** ten passes is *not* free on
perceptual quality. PESQ falls 3.258 -> 3.037 (**-0.22, t = -4.60**) and STOI
-0.004, where four passes had cost nothing measurable. An earlier version of
this section said the gain came "for no cost beyond six more forward passes";
that was wrong -- similarity and WER were measured at ten passes, PESQ was
not. The cost is about a third of PGD's own -0.64, so the pass count is a real
evasion-versus-quality dial rather than a free setting.

It is also the cleanest evidence yet for the step-operator reading in 7b: a
memorised displacement could not keep improving through ten successive
re-hinted applications.

### 7i. Cleaning the reference: partially confirmed, with a real trade-off

Section 0 identified recording provenance as the largest effect anywhere in
this project (0.24 to 0.96 across voices) and it had never been tested.
`probe_reference.py` treats reference.wav before enrolment and measures both
the reference's own detector scores and the resulting clone's.

| treatment | reference soft-min | clone soft-min | d vs raw | t | d sim |
|---|---|---|---|---|---|
| raw | 0.0577 | 0.1447 | -- | -- | -- |
| denoise (spectral gate) | **0.1283** | 0.2382 | **+0.0935** | 1.58 | **-0.1026** |
| highpass 80 Hz | 0.0585 | 0.0831 | -0.0616 | **-2.36** | -0.0073 |
| normalise to -3 dBFS | 0.0661 | 0.1902 | +0.0455 | 1.67 | -0.0037 |
| combined | 0.1069 | 0.1579 | +0.0132 | 0.32 | -0.1308 |

Denoising **doubles the reference file's own soft-min** (0.058 -> 0.128),
which directly supports the section 0 hypothesis: part of what the detectors
react to is present in the reference before synthesis. It lifts the clone by
+0.094 -- but at 0.10 similarity, taking the clone from 0.81 to 0.71, against
a 0.70 constraint floor. That is not free.

Normalisation is the interesting one: +0.046 soft-min for -0.004 similarity
and 2.4 points *better* WER. Essentially free, if small.

High-pass filtering makes things significantly **worse** (t = -2.36),
which is informative in itself -- the low-frequency content the filter
removes is apparently doing some of the work of looking human.

Neither positive result is significant at n = 24, and per-voice the variance
is severe: evelynn's clone goes 0.129 -> 0.420 under denoising while jhin's
goes 0.254 -> 0.107. Given that three prosody effects in this session halved
under power, these should be treated as suggestive only.

### 7j. Can the teacher go higher? Yes -- but not usefully

Since the operator recovers 97% of PGD's gain, further progress has to come
from a better teacher. PGD's settings had never been tuned.

> **That premise is false, and it makes this section's conclusion narrower than
> it reads.** The 97% was measured on training voices only; 7l puts held-out
> recovery at 45%. So "further progress has to come from a better teacher" was
> wrong -- on unseen voices most of the available gain was still unclaimed by
> the student, and more voices in the dataset was the cheaper lever. What
> survives below is the *pricing* of a stronger teacher, which is sound and
> still worth reading: a stronger teacher costs more quality than it buys.

**A first sweep was invalid and is recorded as a warning.** Selecting examples
alphabetically picked ten whose baseline soft-min was already 0.92; PGD
saturates those at 1.0 under any setting, so the grid came out flat and
appeared to show the teacher was maxed out.

On the eight hardest clips (baseline 0.013), 150 steps at lr 0.12 reached
soft-min 0.2532 against the current teacher's 0.1698 -- 1.53x the gain, with
an edit twice as large. `probe_teacher_quality.py` then measured what that
costs on 6 hardest + 6 median clips:

| config | \|delta\| | soft-min | d sim | t | d PESQ | t | d STOI |
|---|---|---|---|---|---|---|---|
| current 60 / 0.05 | 0.155 | 0.3737 | +0.001 | 0.17 | -0.64 | -5.61 | -0.008 |
| 150 / 0.05 | 0.164 | 0.4049 | -0.002 | -0.27 | -0.77 | -6.33 | -0.008 |
| strong 150 / 0.12 | 0.320 | 0.4119 | **-0.080** | **-6.19** | **-1.87** | **-12.53** | -0.040 |

**The strong teacher is rejected.** It buys +0.038 soft-min for 0.08
similarity and almost two PESQ points -- the edit becomes audible damage.

150 steps at lr 0.05 is nearly free (+0.031 soft-min, no similarity cost,
-0.13 extra PESQ), but re-collecting the dataset at 2.5x the steps is roughly
7.5 hours for a gain the student would only partly inherit. **Conclusion:
further training on this formulation is not worth the compute.**

This also exposes a quality cost that had been under-reported: the current
PGD teacher itself costs **-0.64 PESQ (t = -5.61)**, consistent with the
single-clip audit in section 6c. The distilled student was measured as free at
4 passes (PESQ +0.04) but its PESQ at 10 passes has not yet been measured --
see 7k.

### 7k. Transfer to detectors that influenced nothing, and the real quality cost

`probe_transfer.py` scores the ten-pass operator against detectors that played
no part in the PGD targets, the gradient hint, or checkpoint selection. Six
voices (four held out) x six unseen sentences, paired on a shared seed.

The only unused detectors available locally are the two other fusion variants
of RawGAT-ST (the attacked one is `mul`). They share the attacked model's
architecture and training data but have separately trained weights, so this is
a **near** transfer test -- success here is expected and does not by itself
show the effect generalises beyond this model family.

| detector | role | baseline | edited | gain | t | win rate |
|---|---|---|---|---|---|---|
| AASIST | attacked | 0.4203 | 0.8685 | +0.4481 | 7.58 | 97% |
| AASIST-L | attacked | 0.4144 | 0.9194 | +0.5051 | 8.95 | 100% |
| RawNet2 | attacked | 0.2829 | 0.5130 | +0.2301 | 4.44 | 94% |
| RawGAT-ST (mul) | attacked | 0.1292 | 0.3915 | +0.2623 | 6.19 | 92% |
| **RawGAT-ST-add** | **never seen** | 0.1023 | 0.1943 | **+0.0920** | **3.20** | 89% |
| **RawGAT-ST-cat** | **never seen** | 0.2473 | 0.5518 | **+0.3046** | **5.96** | 83% |

Both unseen variants move significantly. But the spread is the finding: the
`cat` sibling picks up 84% of the attacked detectors' mean gain and the `add`
sibling only 25%, despite identical architecture and training data. **Even
separately trained copies of one model differ by more than 3x in how much of
the edit transfers**, which is a warning against assuming any unseen detector
behaves like the attacked ones. Consistent with leave-one-detector-out (7d),
where a held-out detector received about a third of the optimised gain.

Similarity *rose* on these clips (+0.0087, t = 3.08). Perceptual quality did
not come free:

| | baseline | edit x10 | change | t | worse on |
|---|---|---|---|---|---|
| PESQ | 3.258 | 3.037 | **-0.220** | -4.60 | 72% of clips |
| STOI | 0.981 | 0.977 | -0.004 | -2.80 | 72% of clips |

This corrects section 7h (see the note there). Four passes measured PESQ
+0.04; ten passes cost 0.22 -- about a third of PGD's -0.64. The pass count is
therefore a trade-off, not a free setting.

**The far-transfer test has not been run.** No detector of a genuinely
different design has weights on this machine. The official ASVspoof 2021
LFCC-LCNN baseline (hand-crafted LFCC features plus a light CNN, 1.0 MB from
asvspoof.org) would be the right one. `probe_transfer.py` already picks it up
through a `lcnn_detector` adapter if present and `--rescore` re-scores the
kept clips without regenerating. Until that runs, every result in this
document is established only for the RawGAT-ST / AASIST / RawNet2 families.

### 7l. The student has NOT caught the teacher on unseen voices (figure 14)

`probe_gap.py` scores the same mel three ways -- unedited, student x10,
teacher (60-step PGD) -- for all 15 voices on 3 unseen sentences each.

**This corrects the 97% figure in 7h.** That was measured on 14 stored mels
from *training* voices only. Measured properly:

| | soft-min recovery | objective recovery |
|---|---|---|
| trained voices (n = 30 clips) | **79%** | 82% |
| held-out voices (n = 12 clips) | **45%** | 61% |

So on voices the student never trained on it captures **less than half** of
what the teacher achieves. The earlier conclusion that "the student has caught
its teacher, so more data cannot help" was wrong: for unseen speakers, more
than half the available gain is still unclaimed, and **adding voices to the
dataset is the strongest remaining lever.**

Recovery is highly voice-specific, and not simply trained-vs-held-out --
`evelynn` is held out and recovers 96%, while `riven` is a training voice and
recovers 37%:

| voice | set | base | student | teacher | gain | recovery |
|---|---|---|---|---|---|---|
| swain | train | 0.136 | 0.635 | 0.734 | +0.500 | 84% |
| Ego | train | 0.192 | 0.642 | 0.638 | +0.450 | 101% |
| mordekaiser | HELD-OUT | 0.105 | 0.501 | 0.722 | +0.396 | 64% |
| tahmkench | train | 0.073 | 0.411 | 0.446 | +0.338 | 91% |
| Joffrey | train | 0.301 | 0.638 | 0.777 | +0.337 | 71% |
| jhin | train | 0.184 | 0.520 | 0.776 | +0.336 | 57% |
| Silco | train | 0.259 | 0.573 | 0.585 | +0.313 | 96% |
| samira | train | 0.266 | 0.391 | 0.392 | +0.125 | 99% |
| evelynn | HELD-OUT | 0.330 | 0.445 | 0.449 | +0.115 | 96% |
| Tywin | train | 0.456 | 0.559 | 0.585 | +0.103 | 80% |
| thresh | HELD-OUT | 0.116 | 0.203 | 0.612 | +0.087 | 18% |
| riven | train | 0.094 | 0.174 | 0.308 | +0.080 | 37% |
| david | train | 0.052 | 0.112 | 0.115 | +0.060 | 96% |
| Tywin_2 | HELD-OUT | 0.091 | 0.115 | 0.249 | +0.024 | 15% |
| ahmed | train | 0.984 | 1.000 | 1.000 | +0.016 | no headroom |

### Why do some voices work? Mostly not for the reasons section 0 suggested

`analyze_voices.py` measures acoustic properties of each reference.wav and
correlates them with the gain (Spearman, n = 15).

**A bug found and fixed here.** A hand-rolled rank function gave distinct
ranks to tied values, and `clip_fraction` -- identically zero for all 15
voices -- came out at a perfect rho = -1.00. Constants are now rejected and
ties get average ranks via scipy.

| predictor | soft-min gain | mean-detector gain | recovery vs teacher |
|---|---|---|---|
| **teacher's own gain** | **+0.74 (p = 0.003)** | +0.51 (p = 0.05) | -0.37 |
| f0 spread (IQR) | +0.17 | +0.20 | **+0.76 (p = 0.001)** |
| reference duration | +0.31 | **+0.69 (p = 0.004)** | +0.28 |
| reference loudness (RMS) | +0.51 (p = 0.05) | +0.41 | -0.29 |
| crest factor | -0.51 (p = 0.05) | -0.39 | +0.29 |
| baseline soft-min | -0.02 | -0.53 (p = 0.04) | +0.38 |
| **energy above 8 kHz** | +0.31 | +0.21 | +0.27 |
| reference's own detector score | -0.28 | -0.34 | -0.08 |
| estimated SNR | +0.08 | -0.12 | +0.28 |

**Multiple comparisons matter here**: 16 predictors x 3 targets = 48 tests, so
a Bonferroni threshold is p < 0.001 and roughly 2 false positives at p < 0.05
are expected. Only f0 spread clears the strict bar; the teacher's gain and
duration are close.

Three readings, in descending confidence:

1. **The student succeeds where the attack itself succeeds** (rho = +0.74).
   Voice-level difficulty is mostly a property of how attackable that voice's
   audio is, not of the student.
2. **Reference "quality" does not explain the gain.** Energy above 8 kHz --
   the marker of band-limited, processed game voice-over that section 0 blamed
   for detectability -- has rho = +0.31 (p = 0.26). Estimated SNR, spectral
   brightness and the reference's own detector score are all flat too.
   Section 0's observation stands (references *are* part of why baselines
   differ), but it does **not** predict who benefits from the edit.
3. **Pitch variability may predict how well the student tracks its teacher**
   (f0 IQR, rho = +0.76). The four worst-recovering voices (Tywin_2, thresh,
   riven, jhin) have f0 IQR 42-56 Hz; the best (samira, tahmkench, evelynn,
   Ego) have 88-101 Hz. A plausible mechanism is that a 5-frame, 0.05 s
   receptive field generalises better on speech with more local pitch motion,
   but with n = 15 and 48 tests this is a hypothesis to test, not a finding.

### Which detector actually blocks each voice

Taking the lowest detector score after the edit:

| binding detector | voices |
|---|---|
| RawGAT-ST | 8 |
| RawNet2 | 5 |
| AASIST | 2 |

AASIST and AASIST-L are at or near 1.0 for almost every voice after the edit;
**13 of 15 voices are held back by one of the two raw-waveform detectors.**
Several are extreme -- `Tywin_2` and `david` sit at RawNet2 = 0.00 while their
AASIST scores are 0.99. Effort spent on the two graph-attention detectors is
now wasted, which makes re-weighting the objective toward RawNet2 and
RawGAT-ST the clearest cheap experiment available.

### 7m. The objective was measured in the wrong space all along (figures 15, 16)

Section 7l left one question: why does soft-min stall on some voices even for
the teacher? `probe_objective.py` swept six probability-space objectives --
sharper soft-min, capped, hinge, reweighted -- and every one returned an
**identical** RawNet2 score of 0.292. That is not a weighting problem.

Measuring the gradient directly on `david_t00` explains it:

| detector | P(human) | logit margin | gradient w.r.t. mel |
|---|---|---|---|
| AASIST | 0.001 | -6.7 | 3.8e-04 |
| AASIST-L | 0.003 | -5.8 | 1.1e-03 |
| **RawNet2** | **9e-06** | -11.6 | **6.4e-08** |
| **RawGAT-ST** | **1.6e-06** | -13.4 | **1.7e-07** |

The two detectors that block 13 of 15 voices are saturated, and their
gradient is ~5,000x smaller than AASIST's. **PGD was pushing against a wall on
exactly the detectors that decide soft-min.** Every result in sections 2-7l --
the teacher's "ceiling", the 45%/79% recovery, the conclusion that further
training was not worth the compute -- was measured against a teacher crippled
this way.

The fix is standard for saturated classifiers: optimise the pre-softmax logit
margin (Carlini & Wagner 2017). At P = 1e-6 the margin is -11.6 and still
moves linearly.

**Teacher, before and after** (`--objective margin --margin-mode softmin`):

| clips | probability objective | margin objective |
|---|---|---|
| `david` (hardest) | 0.115 | **0.989 - 0.996** |
| median | 0.676 | **0.9998** (all four detectors 1.000) |
| full 268-target dataset | 0.574 | **0.9949**, >= 0.9 on 99% of targets |

RawNet2 goes from 0.0000 to 0.995 on the voice where it had never moved.

### 7n. The student cannot yet spend that gain cheaply

268 targets were re-collected on **all 15 voices** with the margin objective,
storing similarity *and* WER per target so damaging edits can be filtered
(43 dropped for similarity, 23 for WER, 202 kept).

**Voice conditioning did nothing.** A FiLM speaker embedding (CAM++, 192-d,
zero-initialised so it starts identical to the unconditioned net) recovered
25.3% against 25.2% unconditioned; a wider net gave 25.3%. Hypothesis
rejected -- one shared model is sufficient, per-voice adaptation is not the
missing piece.

**Confirmed end to end (2026-09-18).** The above compares *recovered fraction*
on held-out sentences, which is one step removed from the metric that matters.
The two checkpoints were re-run head to head through the full pipeline, all 15
voices x 4 unseen sentences at 4 passes, three arms from one shared seed
(`eval_all15.py --new-run distill_margin_base --old-run distill_margin_spk`):

| | soft-min | d PESQ | d similarity | WER | voices >= 0.5 |
|---|---|---|---|---|---|
| unedited | 0.2764 | -- | -- | 3.6% | 1/15 |
| **base, 21,352 params** | **0.4540** | -0.172 | +0.0022 | 5.3% | 6/15 |
| spk-conditioned, 30,616 params | 0.4632 | -0.178 | +0.0052 | 3.9% | 6/15 |

Paired over 60 clips, base minus spk:

| metric | difference | t | p |
|---|---|---|---|
| soft-min | -0.0092 | -0.98 | 0.331 |
| WER | +1.33 pp | +1.72 | 0.090 |
| PESQ | +0.006 | +0.17 | 0.867 |
| similarity | -0.0030 | **-3.97** | **<0.001** |

**The two are indistinguishable on the objective** (p = 0.33), and the WER gap
that looks large in the means does not survive a paired test either (p = 0.09).
The one significant difference is similarity, at 0.003 in the conditioned
model's favour -- real but far below anything that matters, and both arms
*improve* on the unedited clone.

So the conclusion stands on the metric that counts, not just on recovered
fraction: **use the 21k unconditioned model.** It matches a model with 43% more
parameters. Both reach +0.18 soft-min over unedited at t = 7.5, p < 1e-9.

One correction this forced: the 0.463 reported elsewhere in this section came
from the **spk** checkpoint. The recommended 21k model's own figure is
**0.454**, and the two differ by less than noise.

**A warning sign:** the fixed-average-edit control now recovers **+34.5%**,
*more than the trained network's 25%*. On the probability dataset that control
was -5.4% (section 7). Margin-space solutions evidently share a strong
universal component that the network is not capturing.

Applied repeatedly, the new model does climb -- but each pass costs audio
quality. On all 15 voices with unseen sentences:

| model | soft-min | similarity | WER | PESQ change | voices >= 0.5 |
|---|---|---|---|---|---|
| unedited | 0.276 | 0.819 | 3.6% | -- | 1/15 |
| previous model x10 | 0.489 | 0.823 | 4.8% | -0.28 | 6/15 |
| **margin model x4** | 0.463 | 0.824 | 3.9% | **-0.18** | 6/15 |
| **margin model x14** | **0.800** | 0.819 | 4.8% | **-0.85** | **14/15** |

At 14 passes soft-min nearly triples (+0.523, t = +14.84) with **similarity
unchanged to four decimal places** (t = -0.00) and WER inside tolerance -- but
PESQ falls 0.85, far outside a 0.30 budget. Per detector at that setting:
AASIST 0.85, AASIST-L 0.91, RawNet2 0.89, RawGAT-ST 0.79.

**Within the quality budget the new model does not beat the old one**
(0.463 vs 0.489). What it buys is efficiency and headroom:

| | soft-min gained per PESQ point |
|---|---|
| previous model x10 | 0.76 |
| **margin model x4** | **1.04** |
| margin model x14 | 0.62 |

So the honest summary of this work: **the teacher's ceiling was raised from
0.574 to 0.995, and the student can reach 0.800 / 14-of-15 voices, but only by
spending audio quality the brief does not allow.** Closing that gap is a
distillation problem, not an attack problem -- and the constant-edit control
beating the network says where to start: initialise at the average edit and
learn the residual, which needs no new collection.

### 7o. Two attempts to close the student-teacher gap, both instructive

Section 7n left the teacher at 0.9949 and the student at 0.463 within the
quality budget, with one clue: the fixed average edit recovered **34.5%** while
the trained network recovered **25%**.

**Attempt 1 -- start at the average edit and learn the residual.** A per-channel
pre-tanh bias initialised so the network emits exactly the mean PGD edit
(verified to 1.9e-09), leaving it only the correction to learn.

| | recovered |
|---|---|
| fixed average edit alone | +34.5% |
| network, no profile init | +25.2% |
| **network, profile init** | **+24.6%** |
| network, profile init, lr 3e-5 | +24.9% |

**No effect.** Training pulls the network back to the same solution regardless
of where it starts. Combined with the control beating the network, that points
at the training signal rather than the initialisation: MSE against noisy
per-utterance targets shrinks predictions toward their mean, and the shrunken
edit has lower MSE but a worse objective. The network was doing exactly what
it was asked to do.

**Attempt 2 -- stop imitating; optimise the objective directly.** The path
mel -> HiFT -> detectors is differentiable end to end, which is what makes PGD
possible in the first place. So the network can be trained on

    loss = -margin_objective(detectors(hift(mel + delta))) + beta * mean(delta^2)

with no targets at all (`train_direct.py`). Design A sampled this objective
with PPO, Design B with SAC, and distillation approximated it through MSE --
**nobody had simply backpropagated it into the weights.**

It does beat distillation on the objective, confirming the diagnosis:

| | val soft-min, single pass | \|delta\| |
|---|---|---|
| unedited | 0.257 | -- |
| distilled (the starting point) | 0.332 | 0.019 |
| direct, beta = 2.0 | **0.383** | 0.063 |
| direct, beta = 0.5 | **0.412** | 0.132 |

**But it is far less quality-efficient**, measured on 6 voices:

| config | soft-min | d PESQ |
|---|---|---|
| margin-distilled x4 | 0.465 | **-0.25** |
| direct (beta 2.0) x1 | 0.268 | -0.78 |
| direct (beta 2.0) x4 | 0.567 | -1.89 |
| direct (beta 20) x1 | 0.255 | -0.89 |

Raising beta from 2 to 20 made PESQ *worse* (-0.78 -> -0.89) while |delta|
still grew to 0.08. **`mean(delta^2)` is not a proxy for audible damage**: the
optimiser simply finds loud edits that score well, and penalising their mean
square does not stop it. A perceptual term -- frequency-weighted, or a
differentiable quality estimate in the loss -- is what this needs.

**So the best configuration within the quality budget is unchanged**: the
distilled model, applied 4-10 times. Direct training is the methodologically
correct route and it is now implemented and working, but it cannot yet be
spent without damaging the audio.

### 7p. Where this leaves the project

| approach | soft-min (all 15 voices) | d PESQ | inside budget? |
|---|---|---|---|
| unedited | 0.276 | -- | -- |
| Design A (PPO, 4 GPU-h) | worse than unedited | -- | no |
| Design B (SAC, 12 GPU-h) | no gain | -- | no |
| distilled, probability objective, x10 | 0.489 | -0.28 | **yes** |
| distilled, margin objective, x4 | 0.463 | **-0.18** | **yes** |
| distilled, margin objective, x14 | **0.800** (14/15 voices >= 0.5) | -0.85 | no |
| direct objective training | 0.268-0.567 | -0.78 to -1.89 | no |
| PGD teacher (per utterance) | **0.995** | -- | -- |

Three bottlenecks have been found and two are fixed. The objective was being
measured in a saturated space (fixed, 7m). The training signal was a proxy for
the wrong thing (diagnosed and bypassed, 7o). What remains is that **every
route to higher soft-min currently spends audio quality faster than the budget
allows**, and the loss has no term that knows what "quality" means.

The next step is therefore not more compute on any existing method. It is a
perceptual term in the direct-training loss, at which point `train_direct.py`
can be run with a larger network for hours rather than thirty minutes.

### 7q. Making the loss perceptual: three fixes, frontier unmoved (figure 17)

Section 7o ended with direct training working but spending quality far too
fast, and one diagnosis: `mean(delta^2)` is not a measure of audible damage.
Three things were fixed here. Each is real; together they did not beat
distillation inside the quality budget.

**Fix 1 -- the penalty has to be perceptual.** Two replacements
(`perceptual.py`): a **masking** weight that makes edits cheap under loud
speech and expensive in near-silence (the approximation of Qin et al. 2019,
arXiv:1903.10346), and **multi-resolution STFT distance** between the edited
and unedited waveform, the standard vocoder reconstruction loss (Parallel
WaveGAN, arXiv:1910.11480) -- computed on the signal that is actually heard.

A trap worth recording: HiFT injects noise, so an unedited/edited pair drawn
from two separate calls differs by 0.025 peak and **0.059 of STFT distance on
its own** -- larger than the penalty being measured. Reseeding before each
vocoder call makes the two passes bit-identical (max diff exactly 0), leaving
the edit as the only difference.

The effect on quality efficiency is large:

| loss | passes | soft-min | d PESQ | soft-min per PESQ point |
|---|---|---|---|---|
| mean(delta^2) | 1 | 0.268 | -0.78 | 0.18 |
| mean(delta^2) | 4 | 0.567 | -1.89 | 0.23 |
| **masking** | 1 | 0.278 | **-0.07** | **2.02** |
| **STFT + masking** | 2 | 0.342 | -0.14 | 1.50 |

**Fix 2 -- capacity helps, reversing the distillation finding.** Section 7
found smaller networks strictly better (270k -> -0.4%, 21k -> +33.6%). That
was a statement about overfitting 221 fixed targets, and it does not survive
the change of objective. Trained directly, a 98k network with a 0.31 s
receptive field beat the 21k / 0.05 s one: **val soft-min 0.637 vs 0.534**.

**Fix 3 -- quality has to be a constraint, not a price.** A weighted penalty
is always outbid by the objective given enough steps. At beta = 2 the edit
magnitude held for thirty minutes and then drifted: |delta| 0.023 -> 0.067 ->
0.15 -> 0.19 over three hours, and efficiency collapsed to 0.23. Replacing the
price with a hinge -- nothing until the penalty passes a target, then steeply
-- held |delta| at 0.028 for a whole run. This is the same lesson Design A's
Lagrangian already taught (section 4: the similarity and WER constraints never
once bound); I used a plain penalty here and had to relearn it.

**And yet the frontier did not move.** Every configuration measured on the
same six harder voices (unedited soft-min 0.131, PESQ 3.78):

| config | soft-min | d PESQ | efficiency |
|---|---|---|---|
| **distilled from PGD x4** | **0.465** | -0.251 | 1.33 |
| direct, constrained x8 | 0.385 | -0.259 | 0.98 |
| direct, STFT+mask x2 | 0.342 | -0.141 | 1.50 |
| direct, constrained x4 | 0.316 | -0.081 | 2.30 |
| distilled x2 | 0.302 | -0.040 | 4.25 |
| direct, masking x1 | 0.278 | -0.073 | 2.02 |

Confirmed on all 15 voices with unseen sentences: distilled x4 reaches
**0.463** soft-min, constrained-direct x8 reaches **0.412** and costs more
quality (-0.386). Distillation wins.

**What this means.** Direct training is now methodologically correct --
the right objective, a perceptual cost, and a constraint that holds -- and it
is far more quality-efficient than it was (0.18 -> 2.30 at the same budget).
It is still second. The most likely reason is training time rather than
method: the constrained run had fifty minutes against distillation's full
collect-and-train pipeline, and its curve was still rising when the clock
stopped. That is a cheap, specific thing to test, not another hypothesis.

**Best configuration remains unchanged**: the distilled model, 4 passes, for
soft-min 0.463 at PESQ -0.18 on all 15 voices with similarity and WER intact.

### 7r. Direct training is genuinely second -- and long form was never a ceiling

Two questions closed, in opposite directions.

**Was direct training merely under-trained? No.** Section 7q ended by
guessing that the constrained run had lost only because it got fifty minutes
against distillation's full pipeline. Given 7x the compute:

| compute | best val soft-min |
|---|---|
| 50 min | 0.382 |
| **6 hours** | **0.404** |

Seven times the budget bought **+0.02**, on a visibly saturating curve
(0.306 at 27 min, 0.330 at 67, 0.360 at 148, 0.363 at 228, 0.376 at 349). The
constraint held throughout -- |delta| 0.017 -> 0.044, against the drift to
0.28 that an unconstrained penalty allowed -- so this is convergence, not
instability. On the real metric it still trades badly: soft-min 0.602 costs
-1.20 PESQ, 0.680 costs -1.93.

**Direct training converges around 0.40 and stops.** It is genuinely below
distillation (0.465 in budget), not behind it for want of compute. My
prediction in 7q was wrong, and this is what settles it.

Why imitation beats direct optimisation here is worth stating, because the
intuition runs the other way. PGD gets sixty gradient steps *per clip* and
finds a solution no single forward pass can express; distillation inherits the
shape of those solutions. Direct training has to discover a one-shot function
unaided and settles into a weaker local optimum. The teacher is not just a
label source -- it is search the student cannot perform.

**Long form: the aggregation hypothesis was right** (figure 18). Section 7e
measured the gain vanishing at ~70 s and recorded it as a scope limit. The
suspected cause was that CosyVoice2 chunks long text, so the edit optimises a
~20 s local score while the detectors average ~19 windows across the whole
file. `longform.py` tests it: soft-min's derivative w.r.t. each detector is a
weight obtainable from a no-grad pass over the concatenated audio, and the
window mean decomposes, so each chunk can be optimised **against the file's
score** with only one chunk in the graph at a time. Memory is bounded by chunk
length, which is what OOM'd before.

| length | audio | chunks | n | soft-min | gain | d PESQ | d similarity |
|---|---|---|---|---|---|---|---|
| medium | 13 s | 1 | 4 | 0.174 -> **0.946** | **+0.771** | **-0.10** | +0.011 |
| long | 50 s | 3 | 4 | 0.263 -> **0.668** | +0.405 | -0.33 | +0.010 |
| xlong | 78 s | 5 | 3 | 0.118 -> **0.545** | +0.427 | **-0.28** | -0.011 |

**The 72 s case is the one that matters**, because that is the exact length
section 7e tested, and evelynn was in both runs -- same voice, same length,
only the optimised quantity changed:

| objective | evelynn at ~72 s | gain |
|---|---|---|
| per-chunk learned operator (7e) | 0.0717 -> 0.0944 | +0.023 |
| **global file objective (7r)** | 0.097 -> **0.461** | **+0.364** |

Across the three voices at ~78 s: **+0.427 mean gain, inside the -0.30 PESQ
budget, with WER unmoved (-0.007) and similarity -0.011.** Two of three clear
soft-min 0.5. `skip_frac` is 0.000 on all three, so these are whole-file
optimisations, not partial ones -- worth stating because an earlier attempt at
six voices ran the card to 9988/10240 MiB, where silent OOM skips would have
produced numbers that looked identical to clean ones.

And the decay is not monotonic: 78 s (+0.427) slightly beats 50 s (+0.405).
The cost of length lands between 13 s and 50 s and then flattens, rather than
compounding with every additional chunk. The 50 s mean is also dragged by one
voice; those means hide a wide spread at 50 s -- thresh 0.233 -> 0.779, Tywin_2
0.541 -> 0.831, swain 0.169 -> 0.707, but riven only 0.109 -> 0.354. Three of
four clear the 0.5 bar; one does not. At 13 s all four land between 0.878 and
1.000, so the difficulty is length, not voice.

At 13 s that is near-total evasion for almost no quality cost, and similarity
*improves* on both. Against section 7e's per-chunk gains (+0.164 medium,
+0.004 long) the difference is entirely in what was optimised, not in the
audio.

**Two of this project's "this does not work" conclusions turned out to be
measurement problems**: the objective measured in a saturated probability
space (7m), and the objective measured per chunk while scoring per file (here).
Both looked like ceilings and both moved once the thing optimised matched the
thing scored. That is the most transferable lesson in this document.

**Caveat:** the long-form numbers are the TEACHER's reach (per-utterance
optimisation), not a learned operator's. They establish that long audio is
attackable and that a distillation corpus could be built there; they do not
show a network doing it in a forward pass. No long-form targets have been
collected, and the learned operator's long-form result in 7e still stands.

### 7s. Constraining the TEACHER moves the frontier

Every section from 7m to 7r ended the same way: more evasion was available,
and it cost more audio quality than the -0.30 PESQ budget allowed. The
frontier had not moved since 7q. This is the section where it moves, and the
change is one line of the pipeline nobody had touched.

**The teacher was never quality-constrained.** PGD maximises evasion and
nothing else. Targets were filtered AFTERWARDS for similarity and WER -- but
never for audio quality, and 7j had already measured what the teacher's own
solutions cost: **-0.64 PESQ**, itself outside the budget its students are
held to. So the corpus taught the student to make expensive edits, and no
filter in the pipeline was looking for that.

`collect_pgd.py --penalty mrstft+mask` puts the perceptual cost inside the PGD
loop as a hinge (7q's lesson: a price gets outbid, a constraint does not).
180 targets, 15 voices x 12 sentences.

**At the teacher level the constraint is nearly free** (`compare_corpora.py`,
paired on all 180 shared pairs):

| | unconstrained | constrained | difference | t |
|---|---|---|---|---|
| soft-min reached | 0.995 | 0.969 | -0.026 | -4.54 |
| \|delta\| | 0.196 | **0.108** | **-45%** | -75.63 |
| similarity cost | +0.0396 | **-0.0054** | -0.045 | -11.95 |
| WER cost | +0.23 pp | -0.36 pp | -0.60 | -1.55 |
| soft-min per unit of edit | 3.85 | **6.72** | +75% | |

2.6% less evasion for an edit less than half the size, and a similarity cost
that turns into a similarity gain. The effect was stable as the sample grew
(-0.047 / -0.053 / -0.034 / -0.026 at n = 10 / 45 / 95 / 180), unlike the
three prosody effects that halved each time n doubled.

**It also produces a corpus that survives filtering.** On the same 180 pairs,
same thresholds: 169/180 kept against **134/180** for the unconstrained
teacher, and 1 target dropped for similarity damage against **29**.

**The student inherits the cheapness, not the evasion** (`eval_all15.py`, all
15 voices x 4 unseen sentences, both students trained on the same 180 pairs):

| | soft-min | d PESQ | similarity | WER | >= 0.5 |
|---|---|---|---|---|---|
| unedited | 0.2764 | -- | 0.819 | 3.6% | 1/15 |
| control, unconstrained | 0.4580 | -0.125 | 0.823 | 3.9% | 6/15 |
| **quality-constrained** | 0.4049 | **+0.004** | 0.819 | 4.3% | 5/15 |

Paired: soft-min -0.053 (p = 0.002), PESQ **+0.129 (p = 0.0002)**. 71% of the
control's evasion for a quality cost indistinguishable from zero -- PESQ is
fractionally ABOVE unedited.

*(An efficiency ratio is not reported for that row. Soft-min gained per PESQ
point divides by +0.004 there and prints 36.18, which is an artefact of a
near-zero denominator, not a measurement.)*

**And the unspent budget is what pays.** The pass count has always been the
evasion-versus-quality dial (7h, 7k, 7n). A student whose every pass is
perceptually free can afford three times as many:

| | soft-min | d PESQ | similarity | WER | >= 0.5 |
|---|---|---|---|---|---|
| unedited | 0.2764 | -- | 0.819 | 3.6% | 1/15 |
| margin x4 (7n's best in budget) | 0.4540 | -0.172 | 0.821 | 5.3% | 6/15 |
| **quality-constrained x12** | **0.5881** | -0.209 | 0.817 | **4.3%** | **11/15** |

**+0.134 soft-min, t = 7.24, p < 1e-4**, at a PESQ cost statistically
indistinguishable from the old best (p = 0.31) and still inside the budget.
WER improves. Voices clearing 0.5 go 6 -> 11. Against unedited: +0.312,
t = 11.28, p = 2e-16.

It beats the other in-budget configuration on BOTH axes at once -- the
probability model at 10 passes reached 0.489 at -0.28. Efficiency across the
three: 0.76, 1.03, **1.49**.

**Where the budget runs out.** Sweeping the pass count on the same model:

| passes | soft-min | d PESQ | efficiency | >= 0.5 | inside budget |
|---|---|---|---|---|---|
| **12** | 0.5881 | **-0.209** | **1.49** | 11/15 | **yes** |
| 16 | 0.6894 | -0.355 | 1.16 | 13/15 | no |
| 20 | 0.7505 | -0.539 | 0.88 | 14/15 | no |

12 is both the best in-budget point and the peak of efficiency, which falls
away on either side of it. 16 overshoots the budget slightly for 13 of 15
voices; 20 reaches 14 of 15 at -0.54, which is the same trade 7n's x14 made
and the same answer applies -- available, but not inside the brief.

Two voices worth naming. `david` clears 0.5 for the first time in the project
(0.048 -> 0.506 at x20); it has RawNet2 pinned at 0.00 and every earlier
configuration either failed on it or made it slightly worse. `tahmkench` goes
0.083 -> 0.551.

**More sentences do not help.** The corpus was extended from 12 to all 22
sentences per voice -- 180 -> 330 targets, 312 surviving the filter (94.5%,
holding the constrained teacher's rate at the larger size) -- and retrained
identically:

| corpus | soft-min | d PESQ | similarity | WER | >= 0.5 |
|---|---|---|---|---|---|
| 180 targets x12 | 0.5881 | -0.209 | 0.817 | 4.3% | 11/15 |
| 330 targets x12 | 0.5960 | -0.179 | 0.814 | 4.8% | 11/15 |

Paired: **+0.008 soft-min, t = 0.79, p = 0.43**. Nothing moves on any axis.
83% more data buys nothing measurable.

This closes the data lever. 7l named "adding voices to the dataset" as the
strongest remaining gap, on the strength of 79% recovery on trained voices
against 45% on held-out ones -- but VOICE_DATA_ROOT holds exactly 15 voices
and all 15 are already in every corpus, so that lever was never actually
available. The only axis that could be extended was sentences, and extending
it to exhaustion does nothing. Whatever limits held-out generalisation here,
it is not the number of training examples.

**What this says.** The binding constraint was never the student's capacity or
the amount of data. It was that the targets themselves were too expensive, and
every downstream fix -- capacity sweeps, conditioning, profile init, direct
training, more compute -- was working around a corpus that had been built
without the constraint the results were graded against. Constrain what you
imitate, and the budget spends itself on evasion instead.

### 7t. Adding the objective to the distillation loss: a clear failure

7n and 7o diagnosed the training signal. MSE against noisy per-utterance
targets shrinks predictions toward their mean, a smaller edit scores better on
MSE and worse on the objective, and the evidence was the magnitude ratio
settling at 0.08 while a fixed average edit (34.5%) beat the trained network
(25%). The obvious fix was never tried: add the real margin objective to the
loss, so shrinkage costs something.

    loss = mse - obj_weight * margin_objective(detectors(hift(mel + pred)))

`distill.py --obj-weight 0.01 --obj-clips 1`, on the quality-constrained
corpus, against the identical corpus trained with pure MSE.

**It does exactly what it was designed to do, and that turns out to be the
problem.** The magnitude ratio climbs 0.07 -> 0.46 across training instead of
collapsing to 0.06, so the shrinkage is genuinely reversed. Single-pass
validation soft-min reaches 0.516 against 0.484, and best recovery 18.7%
against 17.9%. Every training-time indicator says it worked.

**On the real metric it is worse on every axis** (all 15 voices, unseen
sentences, both models from the same corpus at 12 passes):

| | soft-min | d PESQ | similarity | WER | >= 0.5 |
|---|---|---|---|---|---|
| unedited | 0.2764 | -- | 0.819 | 3.6% | 1/15 |
| **pure MSE x12** | **0.5881** | **-0.209** | **0.817** | **4.3%** | **11/15** |
| hybrid loss x12 | 0.4126 | **-2.147** | 0.756 | 8.4% | 6/15 |

Paired: soft-min -0.176 (t = -6.29), PESQ **-1.94 (t = -26.26)**, similarity
-0.060 (t = -8.60), WER +4.2 points (t = 2.97). All at p < 0.005.

**Why, and it is the same lesson for the third time.** The objective was added
as a PRICE -- a weighted term in a sum -- and 7q already established that a
price gets outbid: "a weighted penalty is always outbid by the objective given
enough steps". Here the roles are swapped, with MSE as the price and the
objective as the reward, and the outcome is identical. The magnitude did not
converge; it climbed all the way to 0.46, and a per-pass edit that large
compounds over 12 passes into -2.1 PESQ.

The warning was visible during training and misread. At epoch 120 the ratio
was 0.34 and still rising; it levelled around 0.31-0.35 for forty epochs,
which looked like convergence, and then resumed climbing to 0.46. Training
indicators alone could not distinguish "shrinkage corrected" from "price being
outbid" -- only the end-to-end evaluation separated them.

**What would be worth trying instead**, if this is picked up again: make the
objective term a constraint rather than a price, the same way 7q fixed the
perceptual penalty. Hold |delta| or the perceptual cost at a hinge and let the
objective push only up to that line. Nothing in this section argues the
diagnosis in 7n was wrong -- the shrinkage is real and the term does reverse
it. It argues that reversing it without a bound is worse than living with it.

**The recommended configuration is unchanged by this**: the
quality-constrained corpus, pure MSE, 12 passes (7s).

### 7u. DAgger: collecting where the operator actually goes

Every target in every corpus up to here was solved on an **unedited** mel.
The operator is applied 12 times at inference, so passes 2 onward run on mels
no target ever covered -- textbook covariate shift, and it sits exactly where
the gains come from, since x12 beats x4 by a wide margin.

`collect_pgd.py --warm-start` applies the trained student first and lets PGD
solve the correction from there, so the corpus contains the states the
student will really meet. The pass depth is cycled over {2, 4, 8} across
examples rather than fixed, so the trajectory is covered rather than one
point on it. 180 warm-started targets were **aggregated** with the existing
330 (DAgger aggregates; it does not replace), giving 510, of which 476
survive filtering.

| | soft-min | d PESQ | similarity | WER | efficiency | >= 0.5 |
|---|---|---|---|---|---|---|
| unedited | 0.2764 | -- | 0.819 | 3.6% | -- | 1/15 |
| qc330 x12 | **0.5960** | -0.179 | 0.814 | 4.8% | 1.79 | **11/15** |
| DAgger x12 | 0.5292 | **-0.060** | **0.820** | 4.1% | **4.25** | 7/15 |

Paired: soft-min **-0.067 (t = -5.45)**, PESQ **+0.119 (t = 4.23)**,
similarity **+0.005 (t = 3.18)**. Against unedited, DAgger x12 is +0.253 at
t = 9.62.

**It did not do what it was meant to do, and did something else instead.** The
intent was better behaviour on the later passes -- higher soft-min at the same
pass count. Soft-min went *down*. What improved was cost: at 12 passes the
edit runs **-0.060 PESQ against -0.179**, roughly a third, with similarity
rising above the unedited clone. Quality efficiency goes 1.79 -> **4.25**, the
highest in the project by a factor of two.

The honest reading is that training on already-edited states teaches a
*gentler* operator rather than a better-aimed one. That makes sense in
hindsight: a warm-started mel is already most of the way there, so the
correction PGD solves from it is small, and a corpus of small corrections
teaches small steps.

**Which leaves the budget unspent**, and the same move as 7s applies -- the
pass count is the dial, and this operator is cheap enough to turn it much
further:

| passes | soft-min | d PESQ | similarity | efficiency | >= 0.5 | in budget |
|---|---|---|---|---|---|---|
| 12 | 0.5292 | -0.060 | 0.820 | 4.25 | 7/15 | yes |
| **18** | **0.6271** | **-0.170** | 0.816 | 2.07 | 10/15 | **yes** |
| 24 | 0.7262 | -0.302 | 0.813 | 1.49 | 13/15 | no, by 0.002 |

**18 passes is the best configuration the project has produced.** Paired
against the previous best (qc330 at 12 passes) on the same 60 clips:

| | difference | t | p |
|---|---|---|---|
| soft-min | **+0.0311** | 2.57 | **0.013** |
| PESQ | +0.0088 | 0.28 | 0.78 |
| similarity | +0.0021 | 1.63 | 0.11 |
| WER | -0.44 pp | -0.68 | 0.50 |

Significantly more evasion at a quality cost that is statistically identical.
Modest -- +0.031 soft-min -- but real, and free.

**One honest wrinkle.** DAgger x18 has the higher *mean* soft-min yet clears
0.5 on **10 of 15** voices against qc330's 11. The gain is distributional:
it lifts voices that were already moving rather than converting a stubborn
one. If the threshold count is what matters to you rather than the mean, the
two are a tie and qc330 x12 is the simpler pipeline.

24 passes misses the budget by 0.002 PESQ, which is inside anyone's noise --
treat it as the boundary rather than as a failure, and note it reaches
13 of 15 voices.

**So the final recommendation is the DAgger operator at 18 passes**, with
qc330 at 12 as the simpler alternative that is not meaningfully worse.

### Honest limits

Current as of 7r. Earlier revisions of this list were scoped to section 7c and
understated both the reach and the costs; the numbers below are the final ones.

- **This is not evasion.** The best in-budget configuration reaches soft-min
  0.463 across all 15 voices. Most of those clips would still be caught by a
  detector making a hard call at 0.5. It is a large *relative* improvement.
  Clearing 0.5 on 14 of 15 voices is reachable (0.800) but costs -0.85 PESQ,
  well outside the quality budget.
- **The student is far from the teacher on unseen voices.** Recovery is 79%
  on trained voices but **45% on held-out ones** (7l). The "97%" figure that
  appeared in 7h was measured on training voices only. More than half the
  available gain on new speakers is unclaimed, which makes adding voices to
  the dataset the largest measured lever remaining.
- **Pass count is a quality dial, not a free setting.** Four passes cost
  nothing measurable; ten cost -0.22 PESQ (t = -4.60); fourteen cost -0.85.
- **Far transfer is untested.** Every result here is established for the
  AASIST / AASIST-L / RawNet2 / RawGAT-ST families only. Two separately
  trained variants of a *single* architecture differed by more than 3x in how
  much of the edit transferred (7k), which is a direct warning against
  assuming any unseen detector behaves like these. The LFCC-LCNN far-transfer
  test remains the right next check and has not been run.
- **Long-form results are the teacher's, not a learned operator's.** 7r shows
  +0.427 soft-min at 78 s under the global-file objective, but no distillation
  corpus has been collected at that length and no network reaches it in a
  forward pass. 7e's learned-operator result at length still stands.
- **202-268 training targets is small.** The capacity sweep is monotone all
  the way down, which usually means more data would help more than more model.
- **Statistical caution earned the hard way.** Three separate prosody effects
  in 7f-7g halved each time the sample doubled. Anything here at n < 30 should
  be read as suggestive; the load-bearing results are at n >= 60.

---

## Reproducing

```bash
python probe_pgd.py --voice riven --steps 60          # section 2
python analyze_run.py --run designA_budget4h          # section 3
python analyze_run.py --run designB_B9h               # section 4
python eval_ab_voices.py --voices Tywin_2 ahmed evelynn mordekaiser thresh riven swain samira jhin --pairs 4
python plot_ab_voices.py                              # figures 7, 8
python demo_listen.py --voice evelynn --run designB_B9h   # section 6b
python quality.py --pairs base.wav edited.wav --reference reference.wav

python collect_pgd.py --max-hours 3.3                 # section 7 dataset
python distill.py --run-id final --dilations 1 --hidden 24     --weight-decay 1e-3 --lr 1e-4 --input-noise 0.15 --max-hours 0.5
python eval_final.py --distill-run distill_final      # held-out evaluation
python plot_distill.py --run distill_final --control -0.155   # figures 9-12

python probe_amplify.py                               # 7b, figure 13
python probe_transfer.py                              # 7k
python probe_gap.py                                   # 7l, the 45%/79% split
python analyze_voices.py                              # 7l, figure 14
python probe_objective.py                             # 7m, the saturation finding
```

The margin objective (7m onward) is the current best configuration:

```bash
python collect_pgd.py --objective margin --margin-mode softmin --voices all     --measure-wer                                     # 268 targets, all 15 voices
python distill.py --run-id margin_final --objective margin --hidden 24
python eval_all15.py --new-run margin_final --new-iters 4   # headline number
python plot_margin.py                                 # figures 15, 16
```

Perceptual training and long form (7q, 7r):

```bash
python train_direct.py --penalty mrstft+mask --pen-target 0.35 --beta 50
python plot_frontier.py                               # figure 17
python longform.py --lengths xlong --steps 40 --tag longform_xlong
python plot_longform.py                               # figure 18
python collect_graphs.py                              # refresh graphs/INDEX.md
```
