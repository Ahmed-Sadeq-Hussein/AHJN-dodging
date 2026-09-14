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
                 use_grad_hint: bool = True, dilations=(1, 1, 1)):
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

    def _inp(self, mel, grad):
        if not self.use_grad_hint:
            return mel
        if grad is None:
            grad = torch.zeros_like(mel)
        return torch.cat([mel, grad], dim=1)

    def forward(self, mel, grad=None):
        return self.mean_head(self.trunk(self._inp(mel, grad)))

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
