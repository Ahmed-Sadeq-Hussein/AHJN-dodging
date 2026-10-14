# Design A  -  the math

Every formula used by `mel_policy.py`, `reward.py` and `train_ppo_mel.py`, stated in
general form first and then applied to this repo's measured numbers.

> **Scope.** This documents Design A, which **failed**  -  it made detection
> significantly easier than doing nothing (slope -0.000727, t = -7.26). It is
> kept because the derivations are correct and the failure is instructive: the
> soft-min aggregator handed 99.9% of its gradient to two of the four detectors
> and left the other two undefended, so the policy degraded them. The method
> that worked is distillation from a PGD teacher  -  see
> [`Plan_X/RESULTS.md`](Plan_X/RESULTS.md) section 7  -  and it optimises the
> logit margin of section 7m, not the probability-space reward derived here.

> **Viewing this file:** VS Code's built-in Markdown preview renders the math
> (KaTeX). `Ctrl+Shift+V`, or `Ctrl+K V` for a side-by-side preview. No extension
> needed.

---

## 0. Notation

| Symbol | Meaning | Where it lives |
|---|---|---|
| $M$ | mel spectrogram from CosyVoice2's flow decoder, $\mathbb{R}^{80\times T}$ | the state $s$ |
| $T$ | frames in an utterance | $T = 50\cdot(\text{seconds})$, measured |
| $N$ | action dimensionality, $N = 80T$ | ~22,000 for a 5-6 s clip |
| $\Delta$ | the additive edit applied to $M$ | the action $a$ |
| $u$ | pre-squash Gaussian sample | internal to the policy |
| $\mu_\theta(M)$ | policy mean, conv net output | `MelResidualPolicy.forward` |
| $\sigma$ | policy std, $\sigma = e^{\psi}$, $\psi$ learned per channel | `log_std` |
| $\varepsilon_{\max}$ | edit bound | `--max-delta`, default $0.6$ |
| $\varepsilon_{\text{clip}}$ | PPO trust-region width | `--clip-eps`, default $0.2$ |
| $V_\omega$ | critic, the advantage baseline | `MelValueNet` |
| $d_j$ | detector $j$'s $P(\text{human})$, $j=1..4$ | `detectors.py` |
| $\lambda_s,\lambda_w$ | dual variables on similarity / WER | `LagrangianMultipliers` |
| $\eta_\lambda$ | dual step size | `--lambda-lr`, default $0.05$ |
| $B$ | episodes per iteration | `--batch-size` |
| $n$ | number of iterations | `--iterations` |

**Measured constants** (832 mel frames across 3 generated clips, and this repo's own
detector/similarity/WER runs):

| Quantity | Value |
|---|---|
| mel mean | $-4.3731$ |
| mel std, $s_M$ | $2.2879$ |
| mel IQR | $2.6919$ |
| mel range | $[-11.625,\ +2.209]$ |
| mean $\lvert$frame-to-frame $\Delta\rvert$, $\bar\delta_t$ | $0.6367$ |
| frame rate | $50.0$ frames/sec (exact) |
| unedited clone softmin reward | $0.0871$ |
| clone similarity to own reference | $0.836$ |
| cross-speaker similarity range | $0.16 - 0.56$ |
| clone WER | $5.3\%$ |
| riven **reference** WER | $13.3\%$ |

---

## 1. The problem is a contextual bandit, not an MDP

One episode = one utterance: observe $M$, apply one edit, receive one terminal reward.
There is no next state, so the return is the reward itself:

$$G = r$$

**Consequence.** No discount factor $\gamma$, no bootstrapping, no GAE. The usual

$$\hat A_t^{\text{GAE}} = \sum_{l=0}^{\infty}(\gamma\eta)^l\big(r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})\big)$$

collapses, with a single step and no successor, to

$$\hat A = r - V_\omega(M)$$

That is the only advantage this design needs. (The *iterative-refinement* variant of
Design A  -  edit, re-score, edit again  -  would need the full form.)

---

## 2. Policy: bounded, stochastic mel residual

### 2.1 Sampling

**General.** A diagonal Gaussian squashed through $\tanh$ and scaled:

$$u \sim \mathcal{N}\!\big(\mu_\theta(M),\ \operatorname{diag}(\sigma^2)\big), \qquad
\Delta = \varepsilon_{\max}\tanh(u)$$

The squash makes $\lvert\Delta\rvert < \varepsilon_{\max}$ **by construction**, not by a
penalty the policy has to learn to respect.

### 2.2 Log-density under the squash

**General.** For an invertible map $\Delta = g(u)$, the change-of-variables formula is

$$\log \pi(\Delta) = \log \mathcal{N}(u;\mu,\sigma) - \log\left|\frac{dg}{du}\right|$$

Here $\dfrac{d\Delta}{du} = \varepsilon_{\max}\big(1-\tanh^2 u\big)$, so per element:

$$\log\pi(\Delta_i) = \underbrace{-\tfrac{1}{2}\!\left[\frac{(u_i-\mu_i)^2}{\sigma_i^2} + 2\log\sigma_i + \log 2\pi\right]}_{\text{Gaussian}} \;-\; \underbrace{\log\!\big(1-\tanh^2 u_i\big)}_{\text{squash correction}} \;-\; \underbrace{\log \varepsilon_{\max}}_{\text{constant}}$$

**As implemented.** `mel_policy.py` computes the first two terms and **drops
$\log\varepsilon_{\max}$**. Since $\varepsilon_{\max}$ is fixed for a run, that term is a
constant: it cancels exactly in the PPO ratio (section 3.1) and has zero gradient. Harmless for
optimisation  -  but it means `log_prob` is *not* a calibrated density, so don't read it as one.

### 2.3 The $1/N$ normalisation, and why it is load-bearing

**General.** The joint log-density of $N$ independent dimensions is the sum
$\sum_{i=1}^{N}\log\pi_i$. This code instead reports the **mean**:

$$\log\bar\pi(\Delta\mid M) = \frac{1}{N}\sum_{i=1}^{N}\log\pi(\Delta_i), \qquad N = 80T$$

**Why.** The PPO ratio becomes

$$\rho = \exp\!\big(\log\bar\pi_{\text{new}} - \log\bar\pi_{\text{old}}\big) = \left(\frac{\pi_{\text{new}}}{\pi_{\text{old}}}\right)^{1/N}$$

the **geometric mean** per-element likelihood ratio. Applied: $N\approx 22{,}000$, so an
unnormalised joint ratio would be $e^{\pm\text{(thousands)}}$ and clip on every single
step. The normalisation is what makes PPO tractable on this action space  -  at the cost
that $\varepsilon_{\text{clip}}$ no longer carries its textbook meaning (section 4.3).

### 2.4 Expected edit magnitude  -  a closed form worth having

**General.** For $X\sim\mathcal{N}(0,\sigma^2)$, the mean absolute value is

$$\mathbb{E}\lvert X\rvert = \sigma\sqrt{\tfrac{2}{\pi}}$$

For small $\sigma$, $\tanh$ is near-linear ($\tanh x \approx x$), so

$$\mathbb{E}\lvert\Delta\rvert \;\approx\; \varepsilon_{\max}\,\sigma\sqrt{\tfrac{2}{\pi}}$$

**Applied** at $\sigma = e^{-2} = 0.1353$, $\varepsilon_{\max}=0.6$:

$$\mathbb{E}\lvert\Delta\rvert \approx 0.6 \times 0.1353 \times 0.7979 = \mathbf{0.0648}$$

Measured in the running system: $0.0638$. Agreement to ~1.5%  -  this is the check that
confirms the parameterisation behaves as derived.

---

## 3. PPO

### 3.1 Probability ratio

$$\rho(\theta) = \frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}$$

evaluated here in the normalised form of section 2.3. **Sanity property:** before any gradient
step, $\theta = \theta_{\text{old}}$, so $\rho \equiv 1$ exactly. This is the single most
useful correctness test of the whole implementation  -  it fails loudly if `sample()` and
`log_prob_of()` disagree.

### 3.2 Clipped surrogate

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\min\big(\rho\hat A,\; \operatorname{clip}(\rho,\,1-\varepsilon_{\text{clip}},\,1+\varepsilon_{\text{clip}})\,\hat A\big)\Big]$$

### 3.3 Advantage and its normalisation

$$\hat A = r - V_\omega(M), \qquad
\hat A_{\text{norm}} = \frac{\hat A - \operatorname{mean}(\hat A)}{\operatorname{std}(\hat A) + 10^{-8}}$$

**Computed once per rollout, before the epoch loop.** Recomputing it inside the loop  - 
with the same critic that same update is training on the same batch  -  makes $V\to r$, so
$\hat A\to 0$ and each successive epoch contributes less gradient than the last. Measured
decay when done wrongly: $0.82 \to 0.11$ over six epochs.

### 3.4 Value loss

$$L^{V}(\omega) = \mathbb{E}\big[(V_\omega(M) - r)^2\big]$$

The target is $r$, not a bootstrapped return, for the reason in section 1.

### 3.5 Entropy (a proxy)

**General.** Differential entropy of a diagonal Gaussian, per dimension:

$$H = \log\sigma + \tfrac{1}{2}\log(2\pi e) = \log\sigma + 1.4189$$

The $\tanh$-squashed distribution has no closed-form entropy; this pre-squash quantity is
the standard proxy, and it only shapes exploration pressure  -  it is not part of the
objective's correctness.

**Applied** at $\sigma = e^{-2}$: $H = -2 + 1.4189 = \mathbf{-0.581}$ per dimension.

> **Scale note.** Summing over 80 channels gives $-46.5$, four orders of magnitude above a
> typical policy loss of $\sim 0.003$  -  no sane $c_H$ could balance them. The code uses the
> per-dimension mean so that $c_H$ is a usable knob; sensible range $0.0005$-$0.005$.

### 3.6 Total loss

$$L = -L^{\text{CLIP}} + c_v\,L^{V} - c_H\,H$$

with $c_v = 0.5$, $c_H = 0$ by default.

---

## 4. The constrained problem

### 4.1 Primal

$$\max_\theta\ \ \mathbb{E}\big[\operatorname{softmin}_k(d_1,\dots,d_4)\big]
\quad\text{s.t.}\quad
\mathbb{E}[\text{sim}] \ge s_{\min},\qquad \mathbb{E}[\text{wer}] \le w_{\max}$$

### 4.2 Penalised objective (what is actually optimised)

**General.** A classical Lagrangian uses the constraint linearly, $\mathcal{L} = J - \lambda c$.
This code uses a **squared hinge**:

$$J = \mathbb{E}\big[\operatorname{softmin}_k(d)\big]
\;-\;\lambda_s\,\mathbb{E}\big[\max(0,\ s_{\min}-\text{sim})^2\big]
\;-\;\lambda_w\,\mathbb{E}\big[\max(0,\ \text{wer}-w_{\max})^2\big]$$

which makes this a **quadratic-penalty method with a dual-adapted weight** (augmented-Lagrangian
flavour), *not* textbook PPO-Lagrangian. Same spirit, different math  -  worth naming precisely.

The hinge $\max(0,\cdot)$ means a satisfied constraint costs **exactly zero**, whatever
$\lambda$ currently is. The square means the cost grows super-linearly with the violation,
so identity/intelligibility cannot be traded away at a constant exchange rate.

### 4.3 Dual ascent on the *signed* slack

**General.** Projected dual ascent on an inequality constraint:

$$\lambda \leftarrow \operatorname{clip}\big(\lambda + \eta_\lambda\,c,\ 0,\ \lambda_{\max}\big)$$

Using the **signed** slack $c$ (negative when comfortably satisfied) rather than the
hinge-clamped violation:

$$c_s = s_{\min} - \text{sim}, \qquad c_w = \text{wer} - w_{\max}$$

With the hinge-clamped value ($c\ge 0$ always), $\lambda$ is monotonically non-decreasing  - 
it can tighten but never ease, even long after the policy has moved well clear of the
threshold. The signed form lets it relax. The reward's penalty term still uses the clamped
version, so a satisfied episode is never charged.

### 4.4 Fixed point  -  and an asymmetry to watch

Setting the update to zero: $\lambda$ is stationary when $\mathbb{E}[c] = 0$, i.e. when
similarity sits **exactly at** $s_{\min}$  -  not comfortably above it. So whenever the
constraint is comfortably satisfied, $\lambda$ decays toward $0$ and the penalty
switches off entirely until a violation reappears.

**Applied.** Measured $\text{sim}\approx 0.855$, $s_{\min}=0.70$, $\eta_\lambda = 0.05$:

$$\Delta\lambda = \eta_\lambda\,c_s = 0.05\times(0.70-0.855) = -0.00775 \ \text{per iteration}$$

$$\text{iterations for } \lambda:1.0\to 0 \;=\; \frac{1.0}{0.00775} \approx \mathbf{129}$$

If similarity then degrades to $0.65$ ($c_s = +0.05$), regrowth is
$0.05\times0.05 = 0.0025$/iter:

$$\text{iterations for } \lambda:0\to 1.0 \;=\; \frac{1.0}{0.0025} = \mathbf{400}$$

**The dual disarms ~3x faster than it re-engages.** Mitigations: raise $\eta_\lambda$,
floor $\lambda$ above zero, or make the up-rate several times the down-rate.

> **Measured, and a caveat on raising $\eta_\lambda$.** In the `designA_budget4h` run
> at $\eta_\lambda = 0.2$ with similarity holding near $0.84$ (slack $\approx-0.13$):
>
> $$\Delta\lambda = 0.2\times(-0.13) = -0.026\ \text{per iteration}$$
>
> and $\lambda_s$ fell $0.973 \to 0.337$ over 23 iterations  -  a drop of $0.636$
> against $23\times0.026 = 0.598$ predicted. The formula holds.
>
> But note what raising $\eta_\lambda$ actually bought: it speeds the dual in **both**
> directions, and while the constraints are comfortably satisfied that means it
> **disarms four times faster**. $\lambda_s$ reaches zero around iteration ~36 of 130,
> leaving the similarity penalty switched off for the remaining ~70% of the run, with
> only the section 4.5 hard floor still guarding. Faster re-engagement only helps *after* a
> violation appears. If the goal is standing protection rather than reactive
> protection, a floor $\lambda_{\min}\approx0.5$ is the better lever  -  or an
> asymmetric rate (fast up, slow down), which raising $\eta_\lambda$ alone does not
> give you.

### 4.5 Hard floor

$$r \leftarrow \min(r,\ r_{\text{floor}}) \quad\text{if}\quad
\max(0, s_{\min}-\text{sim}) > m_s \ \ \text{or}\ \ \max(0,\text{wer}-w_{\max}) > m_w$$

**Why absolute margins $m_s, m_w$ and not a multiple of the threshold.** The earlier form
tested $\text{violation} > 2 s_{\min} = 1.40$. But
$\max(0,\,0.70-\text{sim}) \le 0.70$ for any non-negative cosine, so **that test could
never fire**  -  dead code. WER is unbounded above, so its version did work. Current values:
$m_s = 0.25$ (fires below $\text{sim}=0.45$, under every cross-speaker pair measured here)
and $m_w = 0.20$ (fires above $30\%$ WER).

---

## 5. softmin over the detector ensemble

### 5.1 Definition

$$\operatorname{softmin}_k(x_1,\dots,x_n) = -\frac{1}{k}\log\!\left(\frac{1}{n}\sum_{i=1}^{n} e^{-k x_i}\right)$$

### 5.2 Bounds

$$\min_i x_i \;\le\; \operatorname{softmin}_k(x) \;\le\; \frac{1}{n}\sum_i x_i$$

Lower bound by replacing every $x_i$ with $\min x$; upper bound by Jensen. Limits:
$k\to\infty$ recovers $\min$, and $k\to 0$ recovers the mean.

### 5.3 Gradient weights  -  where the learning signal actually goes

$$\frac{\partial}{\partial x_i}\operatorname{softmin}_k(x) = \frac{e^{-k x_i}}{\sum_j e^{-k x_j}} = \operatorname{softmax}(-k x)_i$$

A useful summary is the **participation ratio**, the effective number of detectors
receiving gradient:

$$n_{\text{eff}} = \frac{1}{\sum_i w_i^2}$$

**Applied** to the measured clone scores $x = (0.845,\ 0.943,\ 0.0000916,\ 0.00108)$:

| $k$ | AASIST | AASIST-L | RawNet2 | RawGAT-ST | $n_{\text{eff}}$ |
|---|---|---|---|---|---|
| 4 | 1.66% | 1.12% | 48.71% | 48.51% | 2.11 |
| 8 | 0.06% | 0.03% | 50.16% | 49.76% | 2.00 |
| 20 | 0.00% | 0.00% | 50.49% | 49.51% | 2.00 |

**Reading.** At $k=8$ the reward is driven almost entirely by the two detectors that
currently *catch* the clone. That is the intended behaviour  -  but AASIST contributes no
gradient at all until the other two rise. Use $k=4$ to keep it marginally in play.

Value check at $k=8$: $\operatorname{softmin} = 0.08712$ against a logged reward of
$0.08712$. yes

---

## 6. Parameter derivations

Each below: the general relation, then this repo's numbers.

### 6.1 Edit bound $\varepsilon_{\max}$

Three ways to express the same bound, all against measured mel statistics:

$$\frac{\varepsilon_{\max}}{s_M},\qquad
\frac{\varepsilon_{\max}}{\bar\delta_t},\qquad
e^{\varepsilon_{\max}}\ \text{(energy ratio, since the mel is logarithmic)}$$

| $\varepsilon_{\max}$ | % of mel std | % of a natural frame step | energy $\times$ at full deflection |
|---|---|---|---|
| 0.2 | 8.7% | 31.4% | 1.22 |
| **0.6** | **26.2%** | **94.2%** | **1.82** |
| 1.0 | 43.7% | 157.1% | 2.72 |

The middle column is the honest anchor: at $\varepsilon_{\max}=0.6$ a full-deflection edit
moves a frame by about **one natural frame-to-frame transition**. Large enough to matter,
still inside the signal's own variation.

### 6.2 Exploration scale $\sigma$

From section 2.4, $\mathbb{E}\lvert\Delta\rvert = \varepsilon_{\max}\sigma\sqrt{2/\pi}$, expressed
as a fraction of $\bar\delta_t = 0.6367$:

| $\psi = \log\sigma$ | $\sigma$ | $\mathbb{E}\lvert\Delta\rvert$ | % of a frame step |
|---|---|---|---|
| $-2$ (init) | 0.1353 | 0.0648 | 10.2% |
| $-3$ (floor) | 0.0498 | 0.0238 | 3.7% |

Both verified against the running policy.

### 6.3 Why $\varepsilon_{\text{clip}} = 0.2$ never binds

$$\rho = e^{\log\bar\pi_{\text{new}} - \log\bar\pi_{\text{old}}} \approx e^{\pm\text{KL}_{\text{per-element}}}$$

**Applied** with the measured per-element $\text{KL} = 0.0032$:

$$\rho \approx e^{0.0032} = 1.0032 \quad\text{vs. the clip band } [0.8,\ 1.2]$$

Clipping is an inert safety net at this learning rate; the real trust region is set by
$lr \times \text{epochs}$. Measured `clip_frac` $= 0.00$, confirming it.

### 6.4 Learning rate

**General.** For small steps, $\text{KL}\approx\tfrac12\Delta\theta^\top F\,\Delta\theta$
and $\Delta\theta \propto lr$, so

$$\text{KL} \propto lr^2 \qquad\Longrightarrow\qquad lr_{\text{new}} = lr_{\text{old}}\sqrt{\frac{\text{KL}_{\text{target}}}{\text{KL}_{\text{measured}}}}$$

**Applied**, targeting a per-element KL of $0.015$ from a measured $0.0032$ at $3\times10^{-4}$:

$$lr_{\text{new}} = 3\times10^{-4}\sqrt{\frac{0.015}{0.0032}} = 3\times10^{-4}\times 2.165 \approx \mathbf{6.5\times10^{-4}}$$

> **Caveat on KL targets.** Joint KL $= N\times\text{per-element} \approx 22{,}000\times0.0032
> \approx 73$ nats. The familiar `target_kl = 0.015` refers to *joint* KL over small action
> spaces and does **not** transfer here. Monitor the per-element number.

### 6.5 Constraint thresholds

$s_{\min}=0.70$ sits between two measured populations  -  cross-speaker similarity
$0.16$-$0.56$, clone-vs-own-reference $0.836$. Well placed.

$w_{\max}=0.10$ is **arguably mis-set**: the riven *reference itself* scores $13.3\%$ WER
while the clone scores $5.3\%$. The bar currently holds the clone to a stricter standard
than the human source it is copying. Either raise it to $\approx0.15$, or define it
relative to the reference:

$$w_{\max} = \max\big(0.10,\ \text{wer}_{\text{reference}} + 0.02\big)$$

---

## 7. Compute budget

### 7.1 Wall-clock model

**General.**

$$t_{\text{iter}}(B) = B\,(t_{\text{gen}} + t_{\text{asr}} + t_{\text{spk}}) + t_{\text{det}}^{\text{batch}}$$

Detector scoring is batched over the whole rollout because **model loading dominates it**:
one invocation costs ~13 s cold regardless of clip count, so per-episode calls would burn
$B\times13$ s on loading alone.

**Applied** with measured $t_{\text{gen}}\approx10.5$ s, $t_{\text{asr}}+t_{\text{spk}}\approx2.5$ s,
$t_{\text{det}}^{\text{batch}}\approx 5 + 1.5B$ s:

$$t_{\text{iter}}(B) \approx 13B + 17\ \text{seconds}$$

| $B$ | $t_{\text{iter}}$ | iterations in 2 h | total episodes |
|---|---|---|---|
| 8 | 121 s | 58 | 464 |
| 16 | 225 s | 31 | 496 |

Same sample budget either way  -  it is the same compute. **Prefer $B=8$:** equal detection
power (section 7.2) but 58 policy updates instead of 31.

### 7.2 What a 2-hour run can resolve

**General.** For $n$ evenly spaced points, the OLS slope standard error is

$$\operatorname{SE}(\text{slope}) = \frac{\operatorname{SE}_{\text{batch}}}{\sqrt{\dfrac{n(n^2-1)}{12}}},
\qquad \operatorname{SE}_{\text{batch}} = \frac{\sigma_{ep}}{\sqrt{B}}$$

and the smallest total change resolvable at $2\sigma$ over the run is

$$\Delta_{\text{resolvable}} = 2\,\operatorname{SE}(\text{slope})\times n$$

**Applied** with the **measured** $\sigma_{ep} = 0.1304$ (184 episodes of the
`designA_budget4h` run  -  see the correction note below), $B=8$, $n=130$:

$$\operatorname{SE}_{\text{batch}} = \frac{0.1304}{\sqrt8} = 0.0461,\qquad
\sqrt{\frac{130(130^2-1)}{12}} = 427.9$$

$$\operatorname{SE}(\text{slope}) = \frac{0.0461}{427.9} = 1.077\times10^{-4}$$

$$\Delta_{\text{resolvable}} = 2\times1.077\times10^{-4}\times130 = \mathbf{0.028}$$

Against a baseline softmin of $0.087$, that is a **~32% relative improvement**  -  the
smallest effect this budget can distinguish from noise.

> **Correction.** Earlier revisions of this file used $\sigma_{ep}\approx0.09$, an
> estimate taken from between-iteration spread before anything had been measured
> directly. The real value is $0.1304$  -  **45% higher**  -  which inflates every
> standard error by the same factor. A 130-iteration run therefore resolves
> $\pm0.028$, not the $\pm0.014$ that the optimistic figure implied. The lesson is
> the one in section 8: measure $\sigma_{ep}$, do not estimate it. Two-hour and six-hour
> budgets should be re-derived from $0.1304$ before being quoted.

### 7.3 The honest caveat

464 episodes is roughly **three orders of magnitude** below what continuous-control PPO
normally consumes ($10^5$-$10^6$ steps). A converged policy is not on the table. What two
hours *can* answer is narrower and still worth knowing:

> Is there a detectable upward trend of at least $0.029$, with the constraints intact?

**Go / no-go.** Continue only if **both** hold:

1. $\lvert\text{slope}\times n\rvert > \Delta_{\text{resolvable}}$ and the slope is positive
2. mean similarity $\ge s_{\min}$ **and** mean WER $\le w_{\max}$

Reward rising while a constraint breaks is reward hacking, not progress.

---

## 8. Measure $\sigma_{ep}$ before trusting section 7

Every number in section 7.2 rests on $\sigma_{ep}$, the per-episode reward standard deviation,
which is currently a rough estimate from between-iteration spread. It is cheap to measure
directly: run with `--epochs 0` so the policy never updates, and take the spread of

$$\sigma_{ep} = \operatorname{std}\big(\{r_1,\dots,r_{Bn}\}\big)$$

Note this floor is not only sampling noise in the reward  -  CosyVoice2's LLM token
sampling **and** HiFT's source-filter noise both vary run to run, so a frozen policy still
produces varying audio. That is genuine environment stochasticity, which policy-gradient
methods handle, but it sets the noise floor everything else is measured against.
