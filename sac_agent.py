# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
sac_agent.py -- Design B's networks: a gradient-informed actor and twin critics.

Differences from Design A's MelResidualPolicy, and why each exists:

  * The actor sees [mel, grad_hint] -- 160 input channels, not 80. The hint
    is d(detector objective)/d(mel), the signal that took PGD from softmin
    0.085 to 0.666 in 60 steps while Design A's 1,040 black-box episodes went
    backwards. It is an INPUT, not a target: the actor is free to ignore it,
    but it no longer has to rediscover the direction from reward noise alone.

  * Twin Q critics, Q(mel, delta) -> predicted reward. Design A's critic was
    V(mel), a state-only baseline; it could say "this utterance is hard" but
    nothing about whether a particular edit was good. SAC needs an
    action-conditioned Q to push the actor toward better edits. Twin critics
    with a min() take the standard clipped-double-Q remedy for the
    overestimation bias a single Q develops when the actor maximises it.

  * Because an episode is a single step (observe mel, one edit, terminal
    reward), Q has no bootstrapping term: the regression target is simply r.
    No discount, no target network for the next state -- there is no next
    state. Target networks are therefore unnecessary here and omitted; the
    usual reason for them (a moving bootstrap target) does not apply.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

MEL_CHANNELS = 80


def _block(c_in, c_out, k=5, dilation=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k,
                  padding=dilation * (k // 2), dilation=dilation),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
    )


class GradInformedActor(nn.Module):
    """Bounded, stochastic mel edit conditioned on (mel, grad_hint)."""

    def __init__(self, hidden: int = 96, max_delta: float = 0.6,
                 init_log_std: float = -2.0, min_log_std: float = -3.0,
                 use_grad_hint: bool = True, dilations=(1, 1, 1),
                 spk_dim: int = 0, profile_bias: bool = False):
        super().__init__()
        self.max_delta = max_delta
        self.min_log_std = min_log_std
        self.use_grad_hint = use_grad_hint
        self.hidden = hidden
        self.dilations = tuple(dilations)
        in_ch = MEL_CHANNELS * (2 if use_grad_hint else 1)

        # dilations defaults to (1,1,1), which reproduces the original three
        # kernel-5 blocks exactly -- same modules, same state_dict keys, so
        # every Design B checkpoint still loads. Widening it is for
        # distillation, where the target is a specific edit pattern rather
        # than a direction to nudge along, and a 13-frame receptive field
        # (0.5 s) may simply be too narrow to express it.
        chans = [in_ch] + [hidden] * len(self.dilations)
        self.trunk = nn.Sequential(*[
            _block(chans[i], chans[i + 1], dilation=d)
            for i, d in enumerate(self.dilations)
        ])
        self.mean_head = nn.Conv1d(hidden, MEL_CHANNELS, kernel_size=1)
        # start at identity: a fresh actor makes no edit, so training begins
        # from the known-good point rather than a random perturbation
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((MEL_CHANNELS,), init_log_std))

        # A per-channel pre-tanh bias, so the network can be initialised AT a
        # fixed edit and learn only the residual. On the margin dataset the
        # average PGD edit applied blind recovers 34.5% of the gain while the
        # trained network recovers 25% (section 7n) -- the universal component
        # is real and the regression was not finding it. Starting there makes
        # that component free and leaves the network the part it is good at.
        self.profile_bias = profile_bias
        if profile_bias:
            self.base_bias = nn.Parameter(torch.zeros(MEL_CHANNELS, 1))

        # Optional voice conditioning: a CAM++ speaker embedding (192-d, the
        # one CosyVoice2 itself conditions on) produces a per-channel scale and
        # shift (FiLM) on the trunk output. Zero-initialised, so a conditioned
        # model starts out computing exactly what the unconditioned one does,
        # and spk_dim = 0 adds no modules so old checkpoints load unchanged.
        # Measured on the margin dataset: no benefit (25.3% vs 25.2%).
        self.spk_dim = spk_dim
        if spk_dim:
            self.film = nn.Linear(spk_dim, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            self.register_buffer("spk_emb", torch.zeros(spk_dim), persistent=False)

    def set_speaker(self, emb) -> None:
        """Fix the voice used when forward() is called without `spk`."""
        if self.spk_dim:
            self.spk_emb = torch.as_tensor(emb, dtype=self.film.weight.dtype,
                                           device=self.film.weight.device).flatten()

    def _inp(self, mel, grad):
        if not self.use_grad_hint:
            return mel
        if grad is None:
            grad = torch.zeros_like(mel)
        return torch.cat([mel, grad], dim=1)

    def set_profile(self, profile) -> None:
        """Initialise the bias so the net starts by emitting `profile`.

        delta = max_delta * tanh(pre), so the pre-tanh value that produces the
        profile is atanh(profile / max_delta); clamped because atanh diverges
        at the bound.
        """
        if not self.profile_bias:
            return
        pr = torch.as_tensor(profile, dtype=self.base_bias.dtype,
                             device=self.base_bias.device).flatten()
        ratio = (pr / self.max_delta).clamp(-0.999, 0.999)
        with torch.no_grad():
            self.base_bias.copy_(torch.atanh(ratio).unsqueeze(-1))

    def forward(self, mel, grad=None, spk=None):
        h = self.trunk(self._inp(mel, grad))
        if self.spk_dim:
            if spk is None:
                spk = self.spk_emb.unsqueeze(0).expand(mel.shape[0], -1)
            gamma, beta = self.film(spk).chunk(2, dim=1)
            h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        out = self.mean_head(h)
        if self.profile_bias:
            out = out + self.base_bias
        return out

    def sample(self, mel, grad=None, deterministic: bool = False):
        """-> (delta, log_prob, mean_delta); log_prob is per-element mean."""
        mean = self.forward(mel, grad)
        std = self.log_std.clamp(min=self.min_log_std).exp().view(1, -1, 1)

        if deterministic:
            pre = mean
        else:
            pre = mean + std * torch.randn_like(mean)

        squashed = torch.tanh(pre)
        delta = self.max_delta * squashed

        var = std ** 2
        lp = -0.5 * (((pre - mean) ** 2) / var + 2 * std.log() + math.log(2 * math.pi))
        lp = lp - torch.log(1 - squashed.pow(2) + 1e-6)      # tanh-squash correction
        n = lp.shape[1] * lp.shape[2]
        lp = lp.sum(dim=(1, 2)) / n                           # per-element mean

        return delta, lp, self.max_delta * torch.tanh(mean)

    def log_prob_of(self, mel, delta, grad=None):
        mean = self.forward(mel, grad)
        std = self.log_std.clamp(min=self.min_log_std).exp().view(1, -1, 1)
        squashed = (delta / self.max_delta).clamp(-1 + 1e-6, 1 - 1e-6)
        pre = torch.atanh(squashed)
        var = std ** 2
        lp = -0.5 * (((pre - mean) ** 2) / var + 2 * std.log() + math.log(2 * math.pi))
        lp = lp - torch.log(1 - squashed.pow(2) + 1e-6)
        n = lp.shape[1] * lp.shape[2]
        return lp.sum(dim=(1, 2)) / n


class QCritic(nn.Module):
    """Q(mel, delta) -> scalar predicted reward. Length-agnostic via mean-pool."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            _block(MEL_CHANNELS * 2, hidden), _block(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, mel, delta):
        h = self.trunk(torch.cat([mel, delta], dim=1))
        return self.head(h.mean(dim=2)).squeeze(-1)


class TwinQ(nn.Module):
    """Two independent critics; min() of the pair is the conservative estimate."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.q1 = QCritic(hidden)
        self.q2 = QCritic(hidden)

    def forward(self, mel, delta):
        return self.q1(mel, delta), self.q2(mel, delta)

    def min_q(self, mel, delta):
        a, b = self.forward(mel, delta)
        return torch.min(a, b)


class Temperature(nn.Module):
    """SAC's auto-tuned entropy coefficient alpha.

    Learned in log space and driven toward a target entropy, so exploration
    is regulated automatically instead of via a hand-set coefficient. This
    is the fix for what Design A could not express: there, entropy_coef was
    0 and log_std barely moved (-2.000 -> -2.019 over 130 iterations), so
    the exploration scale was effectively frozen at its initial value for
    the whole run regardless of whether that was appropriate.
    """

    def __init__(self, init_alpha: float = 0.05):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

    @property
    def alpha(self):
        return self.log_alpha.exp()


class ReplayBuffer:
    """Stores (mel, grad, delta, reward). Mels vary in length, so this is a
    list of tensors rather than a stacked array; batches are processed one
    item at a time by the update, which is cheap for nets this small.

    The point of off-policy replay here is purely economic: one reward
    evaluation costs a CosyVoice2 generation plus four detectors plus
    Whisper plus CAM++. Design A's on-policy PPO threw each of those away
    after a single gradient step. Reusing them is the main reason Design B
    should extract more from the same wall-clock budget.
    """

    def __init__(self, capacity: int = 4000):
        self.capacity = capacity
        self.data: list[dict] = []

    def add(self, mel, grad, delta, reward):
        self.data.append({
            "mel": mel.detach().cpu(),
            "grad": None if grad is None else grad.detach().cpu(),
            "delta": delta.detach().cpu(),
            "reward": float(reward),
        })
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, n, rng):
        idx = rng.choice(len(self.data), size=min(n, len(self.data)), replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)
