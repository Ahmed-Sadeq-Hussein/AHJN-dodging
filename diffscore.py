r"""
diffscore.py -- the differentiable path from a mel to detector scores.

Design A treated the reward as a black box: write a wav, shell out to the
detectors in another conda env, read back a number. probe_pgd.py showed what
that threw away -- 60 gradient steps moved softmin +0.581 on one utterance,
while 1,040 black-box PPO episodes moved it backwards. This module makes that
gradient available to the training loop.

Two things had to be true for this to work, both established empirically:

1. The detectors import and run correctly under the "voiceclone" env
   (torch 2.3.1 / numpy 1.26), matching the "voicefake" env's scores to ~5e-5.
   detectors.py imports the model files by path and loads plain state_dicts,
   so it never touches the conflicting repo requirements that forced the env
   split in the first place. No subprocess needed.

2. HiFTGenerator.inference is decorated @torch.inference_mode(), which
   silently severs autograd -- the first PGD probe optimised nothing for 60
   steps while reporting |delta| = 0.0000. torch's decorator preserves the
   original under __wrapped__, so calling that gives a differentiable
   vocoder pass.

Also note: RawNet2 contains a GRU, and cuDNN refuses RNN backward in eval()
mode. Switching to train() would flip its BatchNorm layers to batch
statistics and change the very scores being measured, so the fix is to
disable cuDNN and use native kernels. Use the `differentiable()` context
manager below, which handles that.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

import detectors as det

SAMPLE_RATE_GEN = 24000     # CosyVoice2 output
SAMPLE_RATE_DET = 16000     # what the detectors expect


@contextmanager
def differentiable():
    """cuDNN off, so RawNet2's GRU can backprop while staying in eval()."""
    with torch.backends.cudnn.flags(enabled=False):
        yield


def hift_forward_fn(hift):
    """A differentiable stand-in for hift.inference (see module docstring)."""
    fn = type(hift).inference
    fn = getattr(fn, "__wrapped__", fn)

    def call(speech_feat, cache_source=None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0, device=speech_feat.device)
        return fn(hift, speech_feat=speech_feat, cache_source=cache_source)

    return call


def detector_scores(wav24: torch.Tensor, device, names=None,
                    max_windows: int | None = None) -> dict[str, torch.Tensor]:
    """P(human) per detector as grad-carrying tensors, from a 24 kHz waveform.

    Mirrors detectors.score_file -- same resampling, same 64600-sample
    windowing, same tile-pad for short clips, same "index 1 is bonafide" --
    but keeps everything in the autograd graph.
    """
    import torchaudio.functional as AF

    names = names or list(det.MODEL_ORDER)
    wav16 = AF.resample(wav24, SAMPLE_RATE_GEN, SAMPLE_RATE_DET)
    n = wav16.shape[-1]
    NB = det.NB_SAMP

    if n < NB:
        reps = int(NB / n) + 1
        wav16 = wav16.repeat(reps)[:NB]
        wins = wav16.unsqueeze(0)
    else:
        starts = list(range(0, n - NB + 1, NB))
        if starts[-1] + NB < n:
            starts.append(n - NB)
        wins = torch.stack([wav16[s:s + NB] for s in starts])

    # A 76 s clip yields ~19 windows, and RawGAT-ST's conv2d stack runs the
    # whole batch at once -- enough to exhaust a 10 GB card. Capping bounds
    # peak memory; for anything up to ~30 s this is a no-op, so it does not
    # change any result measured on training-length audio.
    if max_windows is not None and wins.shape[0] > max_windows:
        wins = wins[:max_windows]

    out = {}
    for name in names:
        spec = det._SPECS[name]
        net = det.get_model(name, device)
        raw = spec.unpack(net(wins, **spec.forward_kwargs))
        p = torch.softmax(raw, dim=1) if spec.activation == "logits" else torch.exp(raw)
        out[name] = p[:, 1].mean()
    return out


def soft_min(vals, k: float = 8.0) -> torch.Tensor:
    """Smooth minimum. Same formula as reward.soft_min, in torch."""
    x = torch.stack(list(vals))
    return -(torch.logsumexp(-k * x, dim=0) - np.log(len(x))) / k


def detector_objective(scores: dict[str, torch.Tensor], k: float = 8.0,
                       softmin_weight: float = 0.5) -> torch.Tensor:
    """Blend of soft-min and mean over the detector ensemble.

    Design A used pure soft-min, and the post-mortem showed why that failed:
    at k=8 on the measured scores, soft-min hands 50.2% / 49.8% of its
    gradient to RawNet2 / RawGAT-ST and 0.06% / 0.03% to AASIST / AASIST-L.
    The two detectors that were already passing were therefore undefended,
    and the learned policy degraded them significantly (AASIST t = -3.54,
    AASIST-L t = -2.47) while never improving the two it was optimising.

    Blending in the mean puts a floor under every detector's weight. At the
    default 0.5 the same four scores give roughly 12.5% / 12.5% / 37.6% /
    37.4% -- still concentrated on the failing pair, but AASIST now carries
    ~200x the weight it had, so damaging it costs something.
    """
    vals = list(scores.values())
    sm = soft_min(vals, k=k)
    mean = torch.stack(vals).mean()
    return softmin_weight * sm + (1.0 - softmin_weight) * mean


def detector_margins(wav24: torch.Tensor, device, names=None,
                     max_windows: int | None = None) -> dict[str, torch.Tensor]:
    """Logit margin (bonafide - spoof) per detector, instead of P(human).

    THE reason soft-min stalls on hard clips. On david_t00 the two detectors
    that block that voice report P(human) = 9e-06 and 1.6e-06; their softmax
    is completely flat there, and the gradient w.r.t. the mel is ~6e-08 and
    ~1.7e-07 -- roughly five thousand times smaller than AASIST's. PGD gets no
    signal at all, which is why every objective in the sweep returned an
    identical RawNet2 score of 0.292: it was never a weighting problem, it was
    a vanishing gradient.

    The margin is the pre-softmax difference. At P = 1e-6 it sits at -11.6 and
    still moves linearly, so gradient descent can work on it. This is the
    standard fix for saturated classifiers in adversarial attacks (Carlini &
    Wagner 2017, arXiv:1608.04644 §V-A).

    For 'log_softmax' detectors the difference of log-probabilities equals the
    difference of logits, so the same expression is correct for both spec
    types.
    """
    import torchaudio.functional as AF

    names = names or list(det.MODEL_ORDER)
    wav16 = AF.resample(wav24, SAMPLE_RATE_GEN, SAMPLE_RATE_DET)
    n = wav16.shape[-1]
    NB = det.NB_SAMP
    if n < NB:
        reps = int(NB / n) + 1
        wins = wav16.repeat(reps)[:NB].unsqueeze(0)
    else:
        starts = list(range(0, n - NB + 1, NB))
        if starts[-1] + NB < n:
            starts.append(n - NB)
        wins = torch.stack([wav16[s:s + NB] for s in starts])
    if max_windows is not None and wins.shape[0] > max_windows:
        wins = wins[:max_windows]

    out = {}
    for name in names:
        spec = det._SPECS[name]
        raw = spec.unpack(det.get_model(name, device)(wins, **spec.forward_kwargs))
        out[name] = (raw[:, 1] - raw[:, 0]).mean()
    return out


def margin_objective(margins: dict[str, torch.Tensor], cap: float = 4.0,
                     mode: str = "capped", k: float = 1.0) -> torch.Tensor:
    """Objective over logit margins.

    `cap` is where a detector stops earning credit. A margin of +4 is
    P(human) ~ 0.982, comfortably past any 0.5 threshold, so pushing further
    wastes edit budget that soft-min will never reward -- the overshoot to
    0.9997 seen in section 7l.

      capped   mean of min(margin_i, cap): every detector is pulled toward cap
               and no further
      softmin  smooth minimum over margins at temperature k (margins are in
               logit units, so k ~ 1 is the natural scale, not the k = 8 used
               for probabilities)
    """
    vals = torch.stack(list(margins.values()))
    if mode == "capped":
        return torch.clamp(vals, max=cap).mean()
    if mode == "softmin":
        return -(torch.logsumexp(-k * vals, dim=0) - np.log(len(vals))) / k
    raise ValueError(f"unknown margin mode {mode!r}")


def detector_objective_v2(scores, mode: str = "blend", k: float = 8.0,
                          softmin_weight: float = 0.5, tau: float = 0.9,
                          weights=None):
    """Objectives aimed at the WORST detector rather than the average.

    The default "blend" (0.5 soft-min + 0.5 mean) was chosen in section 4 to
    stop Design A abandoning detectors that were already passing. It succeeded,
    and then overshot: after the edit AASIST and AASIST-L sit at ~1.00 on
    nearly every voice while 13 of 15 are held back by RawNet2 or RawGAT-ST
    (section 7l). PGD drives AASIST to 0.9997 -- every unit of edit budget
    spent above ~0.9 buys nothing, because soft-min only ever reports the
    minimum.

    modes:
      blend    current: softmin_weight * soft-min(k) + (1-w) * mean
      softmin  pure soft-min at k (larger k -> closer to a true min)
      capped   mean of min(s_i, tau) -- a detector stops paying once it clears
               tau, so the gradient moves to whoever is still behind
      hinge    -mean(max(0, tau - s_i)^2) -- minimise the squared shortfall;
               quadratic weighting pushes hardest on the furthest-behind
      weighted fixed per-detector weights (pass `weights` as a dict)

    `capped` and `hinge` both bound how much a saturated detector can
    contribute, which is the property the other two lack.
    """
    names = list(scores)
    vals = torch.stack([scores[n] for n in names])

    if mode == "blend":
        return softmin_weight * soft_min(list(vals), k=k) + (1 - softmin_weight) * vals.mean()
    if mode == "softmin":
        return soft_min(list(vals), k=k)
    if mode == "capped":
        return torch.clamp(vals, max=tau).mean()
    if mode == "hinge":
        return -(torch.clamp(tau - vals, min=0.0) ** 2).mean()
    if mode == "weighted":
        w = torch.tensor([float(weights.get(n, 1.0)) for n in names],
                         device=vals.device, dtype=vals.dtype)
        return (w * vals).sum() / w.sum()
    raise ValueError(f"unknown objective mode {mode!r}")


def mel_gradient_hint(mel: torch.Tensor, hift_call, device, k: float = 8.0,
                      softmin_weight: float = 0.5, names=None,
                      max_windows: int | None = 8, space: str = "prob",
                      cap: float = 4.0, margin_mode: str = "softmin",
                      margin_k: float = 1.0) -> torch.Tensor:
    """d(detector objective) / d(mel) -- the actor's extra input channels.

    This is the signal Design A discarded. It is returned detached and
    normalised per-utterance (divided by its own RMS) so that its scale does
    not depend on how confidently the detectors happen to be firing on this
    particular clip -- the actor should read the direction, not the
    magnitude.

    `names` restricts which detectors contribute. Passing a subset is what
    makes a leave-one-detector-out experiment honest: the held-out detector
    must influence neither the target nor the hint the actor conditions on,
    or it is not held out at all.
    """
    m = mel.detach().clone().requires_grad_(True)
    with differentiable():
        wav, _ = hift_call(m)
        if space == "margin":
            # the probability-space hint is ~zero exactly where a detector is
            # saturated (see detector_margins), so a model trained on
            # margin-space targets must get a margin-space hint at inference
            obj = margin_objective(detector_margins(wav.squeeze(0), device,
                                                    names=names,
                                                    max_windows=max_windows),
                                   cap=cap, mode=margin_mode, k=margin_k)
        else:
            sc = detector_scores(wav.squeeze(0), device, names=names,
                                 max_windows=max_windows)
            obj = detector_objective(sc, k=k, softmin_weight=softmin_weight)
        g, = torch.autograd.grad(obj, m)
    rms = g.pow(2).mean().sqrt()
    return (g / (rms + 1e-8)).detach()
