# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
perceptual.py -- penalties that know what "audible" means.

`mean(delta^2)` failed as a quality control (section 7o): raising its weight
tenfold made PESQ *worse* (-0.78 -> -0.89) while the edit magnitude still
grew. Two reasons, both visible in the data:

1. The mel is logarithmic (measured range -11.4 to +2.0), so a fixed delta is
   already a *relative* magnitude change. Penalising delta^2 uniformly charges
   the same price for a 10% change under loud speech -- where the ear cannot
   hear it -- as for the same change in near-silence, where it is obvious hiss.
2. Nothing in delta^2 refers to the waveform that actually reaches the ear.
   The optimiser is free to find edits that are cheap in mel space and loud in
   perceptual space.

Two replacements, both differentiable:

  masked_l2      delta^2 weighted by how QUIET the surrounding mel is, so
                 edits hide under loud content and are expensive in silence.
                 This is the cheap approximation of the masking-threshold idea
                 in Qin et al. 2019 (arXiv:1903.10346), which produced
                 imperceptible adversarial audio against speech recognition.

  mrstft         multi-resolution STFT distance between the edited and the
                 unedited waveform: spectral convergence plus log-magnitude
                 L1 at three window sizes. This is the standard perceptual
                 reconstruction loss for neural vocoders (Parallel WaveGAN,
                 arXiv:1910.11480), and unlike delta^2 it is computed on the
                 signal that is heard.
"""

from __future__ import annotations

import torch

FFT_SIZES = (512, 1024, 2048)


def masking_weights(mel: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    """Per-bin cost multiplier: ~1 where the mel is quiet, ~floor where loud.

    Normalised to mean 1 so the penalty weight stays comparable to the plain
    delta^2 version and `beta` keeps its meaning across penalties.
    """
    lo = mel.amin(dim=(1, 2), keepdim=True)
    hi = mel.amax(dim=(1, 2), keepdim=True)
    rel = (mel - lo) / (hi - lo + 1e-6)          # 0 = quietest, 1 = loudest
    w = (1.0 - rel).clamp(min=floor)
    return w / w.mean(dim=(1, 2), keepdim=True).clamp(min=1e-6)


def masked_l2(delta: torch.Tensor, mel: torch.Tensor, floor: float = 0.05):
    return (delta.pow(2) * masking_weights(mel, floor)).mean()


def _stft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                      window=win, return_complex=True, center=True)
    # abs() of a complex tensor has a non-finite gradient at exactly zero, so
    # go through the squared magnitude with a floor instead
    return (spec.real.pow(2) + spec.imag.pow(2) + 1e-9).sqrt()


def mrstft(edited: torch.Tensor, clean: torch.Tensor,
           fft_sizes=FFT_SIZES) -> torch.Tensor:
    """Spectral convergence + log-magnitude L1, averaged over resolutions."""
    e = edited.reshape(-1)
    c = clean.reshape(-1)
    n = min(e.shape[-1], c.shape[-1])
    e, c = e[:n], c[:n]
    total = e.new_zeros(())
    for n_fft in fft_sizes:
        if n < n_fft:
            continue
        me, mc = _stft_mag(e, n_fft), _stft_mag(c, n_fft)
        sc = torch.linalg.norm(me - mc) / (torch.linalg.norm(mc) + 1e-8)
        mag = (torch.log(me + 1e-7) - torch.log(mc + 1e-7)).abs().mean()
        total = total + sc + mag
    return total / len(fft_sizes)


def penalty(kind: str, delta: torch.Tensor, mel: torch.Tensor,
            edited_wav=None, clean_wav=None, floor: float = 0.05):
    """Dispatch. `mrstft` variants need both waveforms."""
    if kind == "l2":
        return delta.pow(2).mean()
    if kind == "mask":
        return masked_l2(delta, mel, floor)
    if kind == "mrstft":
        return mrstft(edited_wav, clean_wav)
    if kind == "mrstft+mask":
        return mrstft(edited_wav, clean_wav) + masked_l2(delta, mel, floor)
    raise ValueError(f"unknown penalty {kind!r}")
