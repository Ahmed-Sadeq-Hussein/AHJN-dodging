# The reinforcement-learning approaches: what they did, and why

Two designs were trained. Neither improved the objective. This document is the
account of what each one was, what it measurably achieved, and what the
instrumentation says went wrong, because the diagnosis is the result.

Everything here is reproducible from the recorded runs:

```python
import rl
rl.summarise("designA_budget4h")   # slope -0.000727, t = -7.26, n = 130
rl.summarise("designB_B9h")        # slope +0.000033, t = +0.46, n = 315
```

---

## The problem both designs were given

CosyVoice2 runs text -> speech tokens -> flow decoder -> **mel** -> HiFT
vocoder -> waveform. A hook intercepts the mel and adds a residual. The policy
chooses that residual.

The reward is the smooth minimum over four anti-spoofing detectors, minus
hinged quadratic penalties on speaker similarity and word error rate, with
multipliers adapted during training.

Two structural facts matter for reading what follows:

- **The action space is continuous and enormous**, roughly 80 x T, about
  40,000 dimensions for a ten-second clip.
- **Episodes are single-step.** One mel in, one edit out, one reward, no state
  transition. Formally these are contextual bandits, not Markov decision
  processes. Nothing in either design does temporal credit assignment.

---

## Design A: PPO with Lagrangian constraints

A stochastic policy proposes the residual; the clip is rendered and scored;
clipped policy gradient updates the policy. Similarity and WER enter as
dual-adapted multipliers rather than fixed penalties. 130 iterations, 1,040
episodes, 4 GPU-hours.

### Result: significantly worse than doing nothing

| | value |
|---|---|
| reward slope | **-0.000727** |
| t | **-7.26** |
| reward, first -> last | 0.188 -> 0.065 |

Not noise, and not merely a failure to learn: the policy actively made
detection easier.

### Cause: the aggregator, not the algorithm

Soft-min at k = 8 is a worst-case objective, which is what the brief asks for.
On an ensemble where two detectors are already failing and two are already
passing, its gradient allocation is pathological. Measured on the real scores:

| detector | share of the objective's gradient |
|---|---|
| RawNet2 | 50.2% |
| RawGAT-ST | 49.8% |
| AASIST | **0.06%** |
| AASIST-L | **0.03%** |

The two passing detectors were effectively unconstrained, so the policy was
free to degrade them at no cost, and it did: AASIST t = -3.54, AASIST-L
t = -2.47. It never improved the two it was actually optimising.

**This is an objective-design failure, not a PPO failure.** Any learner
maximising that scalar would have found the same thing.

---

## Design B: SAC with a gradient hint

Three changes, each targeting one diagnosed problem:

1. **Blended objective**, `0.5 * soft-min + 0.5 * mean`, so no detector can
   carry a near-zero weight.
2. **Gradient hint**: d(objective)/d(mel) fed to the actor as extra input
   channels. Design A discarded this signal entirely; the environment is
   differentiable and the policy was not being told which way helped.
3. **Twin Q critics conditioned on the action.** Design A's state-value critic
   could say an utterance was hard but nothing about whether a given edit was
   good.

396 iterations across two sessions, 2,520 episodes in the main nine-hour run,
12 GPU-hours.

### Result: flat, and well-powered

| | value |
|---|---|
| reward slope | **+0.000033** |
| t | **+0.46** |
| resolvable threshold | +/- 0.045 |

This is a **well-powered null**, not an inconclusive run. The experiment could
have detected an effect well below the size being sought, and there was none.

Confirmed out of sample. On five randomly drawn unseen voices, Design B scored
**0.199 soft-min against a baseline of 0.199** -- indistinguishable from doing
nothing to three decimal places. Pooled over 36 clips, the mean detector
change was -0.0095 at t = -0.80.

### What Design B did fix

The three changes worked, in the sense that each eliminated the failure it
targeted:

| | Design A | Design B |
|---|---|---|
| overall trend | t = -7.26 | t = +0.46 |
| AASIST | -0.034 (t = -3.54) | **+0.034** |
| AASIST-L | -0.016 (t = -2.47) | **+0.016** |

The degradation is gone and the previously undefended detectors now improve.
The objective still did not move.

### Why it did not move: three diagnostics

**1. The critic could predict but not rank.** Q loss stayed low, around 0.03,
so reward estimation was accurate. Meanwhile the entropy coefficient climbed
for nine hours straight:

| | start | end | t |
|---|---|---|---|
| `log_std` | -4.30 | -3.43 | +1440 |
| SAC temperature | 0.060 | 0.115 | +316 |

A temperature controller that increases exploration monotonically for the
whole run is the signature of an actor whose gradient carries no useful
ranking information. When the critic cannot distinguish two candidate edits,
entropy is the only term left that the objective can improve, so it inflates.

**2. The constraints were never the limit.** Similarity held at 0.827, WER at
5.2%, and both Lagrange multipliers sat on their 0.250 floor for the entire
run. A multiplier that never leaves its floor is evidence: that constraint is
not what is binding. This rules out "the constraints were too tight" as an
explanation.

**3. Action magnitude was never the limit either.** The `max_delta` cap of
0.05 held the edit at 0.0072 mean, with 0.0% of iterations above 0.02 -- it
never approached its own ceiling, and in fact *shrank* over training
(t = -4.24). An earlier run without the cap sat at an edit magnitude of
0.12-0.20 for its whole length, squarely inside the range the white-box attack
uses, and still reached only 0.186. Over that run:

> **corr(|delta|, soft-min) = -0.554**

At the right magnitude, larger edits were **worse**.

---

## The conclusion both designs point to

The barrier is **direction, not scale, and not constraint tightness**.

The useful edit is one specific vector in ~40,000 dimensions. It is
**conditional on the input**, not constant: applying a single fixed
per-channel average edit, blind to the utterance, recovers **-5.4%** of the
available gain, which is worse than doing nothing. So there is no constant
action for a policy to converge on.

Finding that vector by sampling is hopeless at this dimensionality, and random
exploration is not merely inefficient here but actively counterproductive, per
the negative correlation above. This is why increasing exploration (which the
entropy controller did automatically, for nine hours) bought nothing.

### The structural point

The environment is **differentiable end to end**: mel -> vocoder -> detectors
is one continuous chain. Reinforcement learning exists for rewards you can
only sample. Applying it here discards the most valuable signal available, and
the comparison is stark: white-box gradient ascent solves the same objective
in **60 gradient steps per clip**, where 2,520 sampled episodes moved nothing.

**When the reward is differentiable with respect to the action, RL is the
wrong tool** -- and the failure presents as an algorithm problem long before
it presents as a formulation problem. Both designs were implemented
correctly. Neither could have worked.

### What would make this an RL problem again

Two changes would restore RL as the right choice rather than a worse one:

- **A black-box detector.** If the detector could only be queried, with no
  gradients -- a remote API, or a hard accept/reject -- then gradient ascent
  is unavailable and sampling is all there is.
- **A genuinely sequential formulation.** The single-step framing above is a
  bandit. Treating the already-edited mel as a state and the next increment as
  an action is a real MDP, and nothing here has trained that. Even then the
  environment stays differentiable, so analytic gradients through k steps
  would likely beat sampling again.

---

## Extending both runs to 400 iterations

Both designs were resumed from their last checkpoints and trained to 400
iterations, to test whether either result was an artefact of stopping early.
Design A gained 270 iterations (129 -> 400); Design B only 5, because its
counter was already at 395 across two sessions, so its figures below are
effectively unchanged and are reported for completeness rather than as new
evidence.

### Design A: the degradation is not an early-training transient

| | at 130 iterations | at 400 iterations |
|---|---|---|
| reward slope | -0.000727 | -0.000203 |
| t | -7.26 | **-8.18** |
| reward, first -> last | 0.188 -> 0.065 | 0.188 -> 0.104 |

The slope **magnitude** falls because the steep early decline is now averaged
over a longer, flatter tail. That is not a recovery. Two readings make this
unambiguous.

Fitting the trend to the NEW 270 iterations alone, ignoring everything that
came before:

> slope = **-0.000168**, **t = -3.49**

Still significantly downward on data the earlier conclusion never saw. And
the reward keeps falling in thirds:

| iterations | mean reward |
|---|---|
| 1 - 130 | 0.118 |
| 131 - 260 | 0.092 |
| 261 - 400 | **0.068** |

Tripling the training budget made Design A monotonically worse. This matters
for how strongly the section 3 conclusion can be stated: the failure is not
"the run stopped before it turned around", it is a policy descending a
gradient that points the wrong way, and it descends for as long as you let it.
The cause is the aggregator, and no amount of training changes which detectors
the objective pays attention to.

### Design B: unchanged

| | at 315 rows | at 400 iterations |
|---|---|---|
| reward slope | +0.000033 | +0.000023 |
| t | +0.46 | +0.32 |
| soft-min slope | -- | +0.000013 (t = +0.34) |

Five additional iterations cannot move a 400-point trend and did not. The null
stands exactly where it did.

---

## Replication on an unseen sentence

The corpus built by `python -m rl.mos_corpus` is an independent check: a line
neither policy was trained on, all 15 voices, three arms from one shared seed.

| arm | soft-min | vs baseline | t | p | voices improved |
|---|---|---|---|---|---|
| baseline | 0.307 | -- | -- | -- | -- |
| **Design A** | 0.210 | **-0.097** | **-3.77** | **0.002** | **0 / 15** |
| Design B | 0.314 | +0.007 | 0.62 | 0.54 | 9 / 15 |

Both results reproduce. Design A degrades significantly and loses on **every
single voice**, which is a stronger statement than the training trend made.
Design B is a coin flip: nine voices up, six down, mean movement well inside
noise.

Speaker similarity (0.829-0.831) and WER (8.1%) are identical across all three
arms to the precision reported, which is the shared seed doing its job -- the
arms differ only by the Mel edit.

Audio quality separates them slightly: PESQ 3.40 baseline, **3.16** for
Design A, 3.44 for Design B. Design A pays real perceptual cost for a result
that is worse than doing nothing.

---

## A sharper test: two lines, 30 paired items

The replication above used one sentence and 15 voices. Repeating it with two
sentences in different registers, and the 400-iteration checkpoints, doubles
the paired sample to 30 and changes one conclusion.

Line 1 is held out from every training set (`textbank.EVAL_TEXTS`):
*"Whatever happens tomorrow, remember that the tide comes in twice and asks
nobody first."* Line 2 is declamatory: *"Beyond heaven and earth, He alone was
the honored one. The one to stand above all the other men."* Both lines are
harder than the earlier sentence, with an unedited soft-min of 0.185 against
0.307.

| arm | soft-min | vs baseline | t | p | improved | d PESQ |
|---|---|---|---|---|---|---|
| baseline | 0.185 | -- | -- | -- | -- | -- |
| Design A | 0.085 | -0.100 | -4.34 | 0.0002 | 1 / 30 | -0.649 |
| **Design B** | 0.135 | **-0.050** | **-4.89** | **<0.0001** | **3 / 30** | -0.178 |

### Design B is mildly harmful, not neutral

This revises the earlier characterisation. Design B's *training* trend is a
well-powered null and that stands: slope +0.000023 at t = +0.32 over 400
iterations. But measured end to end on held-out sentences, with enough paired
samples to resolve it, the deployed policy is **significantly worse than doing
nothing**: -0.050 soft-min at t = -4.89, improving only 3 of 30 items.

This is a sharpening rather than a reversal. The sign was always negative and
the project simply lacked the power to say so:

| evaluation | n | effect | t | significant |
|---|---|---|---|---|
| held-out voices (section 5) | 36 clips | -0.0095 mean detector | -0.80 | no |
| one sentence, 15 voices | 15 | +0.007 soft-min | +0.62 | no |
| **two lines, 15 voices** | **30** | **-0.050 soft-min** | **-4.89** | **yes** |

The single-sentence test pointed the other way by +0.007, which is a reminder
that a null at n = 15 is weak evidence of absence. The honest statement is now
that **neither RL design is neutral: Design A is badly harmful and Design B is
mildly harmful**, and the gap between them is the degree, not the direction.

Design A also costs real perceptual quality for this, -0.649 PESQ, against
Design B's -0.178.

By line, the pattern holds in both registers:

| | baseline | Design A | Design B |
|---|---|---|---|
| line 1, conversational | 0.139 | 0.059 | 0.083 |
| line 2, declamatory | 0.231 | 0.110 | 0.187 |

---

## Reproducing

```python
from rl import design_a, design_b

design_a.summarise("designA_budget4h")          # trend and t statistic
design_a.history("designA_budget4h")            # every logged iteration
design_b.summarise("designB_B9h", "mean_softmin")

design_b.train(run_id="b2", iterations=400)     # a fresh run
design_b.train(run_id="b2", resume="auto")      # continue it, buffer intact

design_b.generate("designB_B9h", "riven", "some text", "out.wav")
```

The listening-test corpus for these arms is built by
`python -m rl.mos_corpus`, which renders every voice under baseline, Design A
and Design B from one shared seed, scores all of it, and writes the folder the
rating page reads.
