# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
mel_policy.py -- Design A's actor: a bounded residual on CosyVoice2's mel,
and the hook that gets it applied without touching the vendored repo.

    from mel_policy import MelResidualPolicy, mel_edit_hook
    import cloners

    policy = MelResidualPolicy()
    net = cloners.load_model("CosyVoice2")

    with mel_edit_hook(net.model, policy) as record:
        cloners.generate("some text", ref_wav, model="CosyVoice2", out_path=out)
    # record.deltas / record.log_probs / record.mels now hold what happened
    # inside token2wav() during that call, for the PPO update.

Where this hooks in
--------------------
Read from cosyvoice/cli/model.py directly (CosyVoiceModel.token2wav, the
shared base class CosyVoice2Model inherits):

    tts_mel, ... = self.flow.inference(...)                      # (1, 80, T)
    ...
    tts_speech, tts_source = self.hift.inference(
        speech_feat=tts_mel, cache_source=hift_cache_source)

So `model.hift.inference` is the one call that always receives the finished
mel right before vocoding, streaming or not, with no other call site to miss.
Monkeypatching it (save the bound method, install a wrapper, restore on exit)
needs no fork of the vendored repo and stays correct if CosyVoice2's own
caching/fade logic around token2wav changes later -- the wrapper only ever
sees the mel it's handed.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

MEL_CHANNELS = 80          # CosyVoice2's mel is (batch, 80, time) -- fixed by its own config


class MelResidualPolicy(nn.Module):
    """Outputs a bounded, stochastic additive edit for a (B, 80, T) mel.

    Architecture: a small causal-ish 1D conv stack over time (kernel 5,
    padding 2 -- not literally causal, but local; there's no reason to hide
    future frames from an edit that's applied to the whole utterance at
    once) producing a per-bin mean. Log-std is a learned (80,) parameter,
    not conditioned on the input -- standard for continuous-action PPO,
    and one fewer thing for a first version to get wrong.

    The raw Gaussian sample is squashed through tanh and scaled by
    `max_delta`, so the edit is bounded in [-max_delta, max_delta] by
    construction rather than by a penalty the policy has to learn to
    respect. log_prob() carries the tanh-squash correction (same
    construction as SAC's, e.g. Haarnoja et al. 2018 appendix C) --
    skipping that correction silently biases the PPO ratio.
    """

    def __init__(self, hidden: int = 96, max_delta: float = 0.6,
                 init_log_std: float = -2.0, min_log_std: float = -3.0):
        super().__init__()
        self.max_delta = max_delta
        # Exploration in a continuous action space is sigma, not an epsilon-
        # greedy rate, and sigma here is a LEARNED parameter -- which means
        # PPO will happily drive it toward zero once it finds a decent mean,
        # ending exploration for the rest of the run. This floor is the cheap
        # insurance: at the default, sigma >= exp(-3) = 0.0498, giving a
        # typical |delta| of max_delta * sigma * sqrt(2/pi) = 0.024, about 4%
        # of the mel's own measured mean frame-to-frame movement (0.637), so
        # the policy keeps probing even at the floor.
        self.min_log_std = min_log_std

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.SiLU(),
            )

        self.trunk = nn.Sequential(
            block(MEL_CHANNELS, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
        )
        self.mean_head = nn.Conv1d(hidden, MEL_CHANNELS, kernel_size=1)
        # zero-init the last layer: a fresh policy starts as a near-identity
        # edit (mean ~= 0 -> tanh(noise) small), not an arbitrary large delta
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

        self.log_std = nn.Parameter(torch.full((MEL_CHANNELS,), init_log_std))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Mean pre-tanh action, shape (B, 80, T)."""
        return self.mean_head(self.trunk(mel))

    def sample(self, mel: torch.Tensor, deterministic: bool = False):
        """Return (delta, log_prob, mean_delta).

        delta, mean_delta : (B, 80, T), same shape as `mel`.
        log_prob          : (B,) -- summed over (channel, time), then
                             divided by numel so utterances of very
                             different length (this dataset runs 5-19 s)
                             contribute comparably-scaled ratios to PPO.
        """
        mean = self.forward(mel)                                   # (B, 80, T)
        std = self.log_std.clamp(min=self.min_log_std).exp().view(1, -1, 1)   # (1, 80, 1)

        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + std * noise

        squashed = torch.tanh(pre_tanh)
        delta = self.max_delta * squashed
        mean_delta = self.max_delta * torch.tanh(mean)

        # log N(pre_tanh; mean, std), elementwise
        var = std ** 2
        log_prob_gauss = -0.5 * (((pre_tanh - mean) ** 2) / var + 2 * std.log() + math.log(2 * math.pi))
        # tanh-squash correction: log|d tanh/dx| = log(1 - tanh(x)^2)
        log_prob_tanh = torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = (log_prob_gauss - log_prob_tanh)                 # (B, 80, T)

        numel = log_prob.shape[1] * log_prob.shape[2]
        log_prob = log_prob.sum(dim=(1, 2)) / numel                 # (B,)

        return delta, log_prob, mean_delta

    def log_prob_of(self, mel: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """log_prob of a delta produced earlier by this policy (or another
        rollout's), for PPO's ratio -- pi_new(a|s) needs re-evaluating on
        the actions the OLD policy actually took, not fresh samples."""
        mean = self.forward(mel)
        std = self.log_std.clamp(min=self.min_log_std).exp().view(1, -1, 1)

        squashed = (delta / self.max_delta).clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(squashed)

        var = std ** 2
        log_prob_gauss = -0.5 * (((pre_tanh - mean) ** 2) / var + 2 * std.log() + math.log(2 * math.pi))
        log_prob_tanh = torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = (log_prob_gauss - log_prob_tanh)

        numel = log_prob.shape[1] * log_prob.shape[2]
        return log_prob.sum(dim=(1, 2)) / numel


class MelValueNet(nn.Module):
    """State-value baseline V(mel) for PPO's advantage.

    Design A's episode is a single step -- one utterance, one edit, one
    terminal reward -- so there is no bootstrapping, no discounting and no
    GAE: the return IS the reward, and this network's only job is to predict
    it so the advantage (r - V) has lower variance than r alone. A plain
    running-mean baseline would also work, but different voices and
    sentences have genuinely different baseline difficulty (this repo's own
    detector sweep showed per-voice scores spanning 0.0 to 1.0), so
    conditioning the baseline on the actual mel is worth the small net.

    Global average pooling over time keeps it length-agnostic, which
    matters because utterances here run 5-19 s.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(MEL_CHANNELS, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, 80, T) -> (B,) predicted return."""
        h = self.trunk(mel)              # (B, hidden, T)
        h = h.mean(dim=2)                # (B, hidden) -- length-agnostic
        return self.head(h).squeeze(-1)  # (B,)


@dataclass
class HookRecord:
    """What happened inside token2wav() during one hooked generate() call.

    A single call to cloners.generate() can invoke hift.inference more than
    once (cloners chunks long text into sentences, each a separate
    frontend/tts call), so every field is a list, one entry per invocation.
    """
    mels: list[torch.Tensor] = field(default_factory=list)          # pre-edit, detached
    deltas: list[torch.Tensor] = field(default_factory=list)        # NOT detached -- needed for backward
    log_probs: list[torch.Tensor] = field(default_factory=list)     # NOT detached
    n_calls: int = 0


@contextmanager
def mel_edit_hook(model, policy: MelResidualPolicy, *, deterministic: bool = False,
                  device: torch.device | None = None):
    """Monkeypatch model.hift.inference for the duration of the `with` block.

    `model` is a CosyVoice2Model instance (e.g. `cloners.load_model("CosyVoice2").model`).
    Every call to hift.inference during the block gets its speech_feat edited
    by `policy` first; the edit, its log-prob, and the pre-edit mel are
    collected into the returned HookRecord.

    deterministic=True samples the policy's mean action (no noise) -- use
    this for eval/inference passes, not for PPO rollouts (which need the
    stochastic action and its log-prob).
    """
    record = HookRecord()
    original_inference = model.hift.inference

    def patched(*, speech_feat: torch.Tensor, cache_source=None, **kw):
        dev = device or speech_feat.device
        pol = policy.to(dev)

        # CosyVoice2's generation runs under torch.inference_mode(), and a
        # tensor created there can never be used in autograd again -- not
        # even outside that context (RuntimeError: "Inference tensors
        # cannot be saved for backward"). log_prob is what PPO's loss
        # actually differentiates, so its computation graph has to be built
        # from ordinary tensors: clone the incoming mel and run the policy
        # inside a nested inference_mode(False) block. The clone + the
        # policy's own parameters (created at construction time, never
        # under inference_mode) are both ordinary tensors, so log_prob
        # keeps a real grad_fn even after this function returns into the
        # caller's inference_mode(True) region.
        with torch.inference_mode(False):
            mel = speech_feat.clone()
            delta, log_prob, _mean = pol.sample(mel, deterministic=deterministic)
            edited = mel + delta

        record.mels.append(mel.detach())
        record.deltas.append(delta)
        record.log_probs.append(log_prob)
        record.n_calls += 1

        call_kw = {"speech_feat": edited}
        if cache_source is not None:
            call_kw["cache_source"] = cache_source
        call_kw.update(kw)
        return original_inference(**call_kw)

    model.hift.inference = patched
    try:
        yield record
    finally:
        model.hift.inference = original_inference
