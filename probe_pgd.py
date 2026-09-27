r"""
probe_pgd.py -- is a bounded mel edit even CAPABLE of fooling the detectors?
RUN IN THE "voiceclone" ENV (which, it turns out, can host the detectors too).

    python probe_pgd.py --voice riven --steps 60

Why this exists
---------------
RL is an expensive way to answer a question that gradient descent answers in
minutes. PPO has to discover a good edit through ~1000 noisy samples over a
~22,000-dimensional action space. But the entire chain -- mel -> HiFT ->
waveform -> detector -- is differentiable and we hold every weight locally.
So we can just ASK, directly, with full white-box knowledge:

    what is the highest softmin score any edit with |delta| <= eps_max can reach?

That number is an UPPER BOUND on what any policy could learn. It is the
feasibility question that should be settled before spending more GPU-hours:

  * If PGD cannot move softmin with a bounded edit, no amount of RL will --
    the intervention point is simply too weak, and the honest move is to
    change the action space, not to train longer.
  * If PGD trivially reaches ~1.0, the ceiling is high and RL's remaining job
    is purely GENERALISATION: learning one edit function that works on unseen
    utterances, rather than re-optimising per utterance. That is a real and
    much better-posed goal, and it reframes what the training run is for.

What PGD does NOT give you is that generalisation -- it overfits one
utterance by construction. The two approaches answer different questions;
this one is just far cheaper, so it should come first.

Reported alongside the score: similarity and WER of the final adversarial
audio, because an edit that reaches softmin 1.0 by destroying the voice has
answered the wrong question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import detectors as det
import reward as reward_mod
import Tester
from env_config import RESULTS_ROOT
from mel_policy import MelResidualPolicy, mel_edit_hook
from train_ppo_mel import DEFAULT_TEXTS, load_voice

SAMPLE_RATE_GEN = 24000     # CosyVoice2 output rate
SAMPLE_RATE_DET = 16000     # what the detectors expect


def differentiable_scores(wav24: torch.Tensor, device, names) -> dict[str, torch.Tensor]:
    """P(human) per detector, as tensors with grad, from a 24 kHz waveform tensor.

    Mirrors detectors.score_file but keeps everything in the autograd graph:
    it takes a tensor instead of a path, and averages over windows rather than
    calling .numpy() anywhere.
    """
    import torchaudio.functional as AF

    wav16 = AF.resample(wav24, SAMPLE_RATE_GEN, SAMPLE_RATE_DET)
    n = wav16.shape[-1]
    NB = det.NB_SAMP
    if n < NB:                                   # tile-pad, as detectors.py does
        reps = int(NB / n) + 1
        wav16 = wav16.repeat(reps)[:NB]
        wins = wav16.unsqueeze(0)
    else:
        starts = list(range(0, n - NB + 1, NB))
        if starts[-1] + NB < n:
            starts.append(n - NB)
        wins = torch.stack([wav16[s:s + NB] for s in starts])

    out = {}
    for name in names:
        spec = det._SPECS[name]
        net = det.get_model(name, device)
        raw = spec.unpack(net(wins, **spec.forward_kwargs))
        p = torch.softmax(raw, dim=1) if spec.activation == "logits" else torch.exp(raw)
        out[name] = p[:, 1].mean()               # index 1 = bonafide
    return out


def soft_min_t(vals: list[torch.Tensor], k: float = 8.0) -> torch.Tensor:
    x = torch.stack(vals)
    return -(torch.logsumexp(-k * x, dim=0) - np.log(len(vals))) / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="riven")
    ap.add_argument("--text", default=None)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-delta", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    text = args.text or DEFAULT_TEXTS[0]
    ref, prompt = load_voice(args.voice)
    names = list(det.MODEL_ORDER)

    torch.manual_seed(args.seed)
    cosy = cloners.load_model("CosyVoice2", device=str(device))
    cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=args.voice, device=str(device))

    # --- capture one real mel by generating with a zero-init (identity) policy ---
    out_dir = RESULTS_ROOT / "rl_runs" / "pgd_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = MelResidualPolicy(max_delta=args.max_delta).to(device)   # zero-init -> no-op
    base_wav_path = out_dir / f"{args.voice}_base.wav"
    with mel_edit_hook(cosy.model, identity, deterministic=True, device=device) as rec:
        cloners.generate(text, ref, model="CosyVoice2",
                         out_path=base_wav_path, device=str(device))
    mel = rec.mels[0].clone().to(device)
    print(f"captured mel {tuple(mel.shape)}  (N = {mel.shape[1]*mel.shape[2]} action dims)")

    hift = cosy.model.hift
    cache = torch.zeros(1, 1, 0).to(device)

    # HiFTGenerator.inference is decorated @torch.inference_mode(), which
    # severs the autograd graph: every tensor it produces is an inference
    # tensor, the delta receives no gradient, and PGD silently optimises
    # nothing while reporting |delta| = 0.0000 forever. torch's decorator
    # keeps the original under __wrapped__, so call that (unbound -> pass
    # the module explicitly) to get a differentiable vocoder pass.
    _hift_fwd = type(hift).inference
    _hift_fwd = getattr(_hift_fwd, "__wrapped__", _hift_fwd)

    def hift_diff(speech_feat, cache_source):
        return _hift_fwd(hift, speech_feat=speech_feat, cache_source=cache_source)

    # --- baseline: unedited mel through the vocoder ---
    with torch.no_grad():
        wav0, _ = hift.inference(speech_feat=mel, cache_source=cache)
        s0 = differentiable_scores(wav0.squeeze(0), device, names)
        base_soft = float(soft_min_t([s0[n] for n in names]))
    print(f"baseline softmin: {base_soft:.4f}   " +
          "  ".join(f"{n}={float(s0[n]):.4f}" for n in names))

    # --- PGD: ascend softmin w.r.t. a bounded delta on the mel ---
    raw = torch.zeros_like(mel, requires_grad=True)     # pre-tanh, so the bound is structural
    opt = torch.optim.Adam([raw], lr=args.lr)

    print(f"\nPGD: {args.steps} steps, |delta| <= {args.max_delta}, lr {args.lr}")
    history = []
    # cuDNN refuses to backprop through an RNN in eval() mode ("cudnn RNN
    # backward can only be called in training mode"), and RawNet2 has a GRU.
    # Switching the net to train() would let cuDNN run but would also switch
    # its BatchNorm layers to batch statistics and start updating running
    # means -- i.e. it would silently change the very scores being probed.
    # Disabling cuDNN falls back to native RNN kernels, which support
    # eval-mode backward. Slower, but the numbers stay honest.
    with torch.backends.cudnn.flags(enabled=False):
      for step in range(args.steps):
        opt.zero_grad()
        delta = args.max_delta * torch.tanh(raw)
        wav, _ = hift_diff(mel + delta, cache)
        sc = differentiable_scores(wav.squeeze(0), device, names)
        soft = soft_min_t([sc[n] for n in names])
        (-soft).backward()                              # ascend
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            with torch.no_grad():
                print(f"  step {step:3d}  softmin {float(soft):.4f}  "
                      f"|delta| mean {float(delta.abs().mean()):.4f} max {float(delta.abs().max()):.4f}   "
                      + "  ".join(f"{n}={float(sc[n]):.3f}" for n in names))
        history.append({"step": step, "softmin": float(soft),
                        "delta_abs_mean": float(delta.abs().mean())})

    # --- write the adversarial audio and score it honestly (sim + WER) ---
    import soundfile as sf
    with torch.no_grad():
        delta = args.max_delta * torch.tanh(raw)
        wav_adv, _ = hift.inference(speech_feat=mel + delta, cache_source=cache)
    adv_path = out_dir / f"{args.voice}_pgd.wav"
    sf.write(str(adv_path), wav_adv.squeeze(0).detach().cpu().numpy(), cosy.sample_rate)

    lag = reward_mod.LagrangianMultipliers()
    _, info_base = reward_mod.compute_reward(base_wav_path, ref, text, lagrange=lag,
                                             wer_max=0.15, device=str(device))
    _, info_adv = reward_mod.compute_reward(adv_path, ref, text, lagrange=lag,
                                            wer_max=0.15, device=str(device))

    print("\n" + "=" * 70)
    print(f"{'':<14}{'softmin':>10}{'similarity':>13}{'WER':>9}")
    print("-" * 70)
    print(f"{'baseline':<14}{info_base['detect_reward_softmin']:>10.4f}"
          f"{info_base['similarity']:>13.3f}{info_base['wer']*100:>8.1f}%")
    print(f"{'PGD edited':<14}{info_adv['detect_reward_softmin']:>10.4f}"
          f"{info_adv['similarity']:>13.3f}{info_adv['wer']*100:>8.1f}%")
    print("=" * 70)

    gain = info_adv["detect_reward_softmin"] - info_base["detect_reward_softmin"]
    sim_cost = info_base["similarity"] - info_adv["similarity"]
    print(f"\nsoftmin gain from a white-box, per-utterance, bounded mel edit: {gain:+.4f}")
    print(f"similarity cost: {sim_cost:+.4f}   WER cost: "
          f"{(info_adv['wer']-info_base['wer'])*100:+.1f} points")
    print("\nINTERPRETATION")
    if gain < 0.05:
        print("  The intervention point is too weak. Full white-box gradient access")
        print("  cannot move the detectors with a bounded mel edit, so no RL policy")
        print("  will either -- this is an upper bound. Change the action space")
        print("  (bigger max_delta, or edit the waveform) rather than training longer.")
    elif sim_cost > 0.15:
        print("  Evasion is reachable but only by damaging the voice. The constraint,")
        print("  not the detector, is the binding limit -- which is what the")
        print("  Lagrangian is there to enforce.")
    else:
        print("  Evasion IS reachable within the bound at acceptable cost. The ceiling")
        print("  is high, so RL's remaining job is generalisation: learning one edit")
        print("  function that transfers to unseen utterances, rather than")
        print("  re-optimising per utterance as PGD does.")

    (out_dir / f"pgd_{args.voice}.json").write_text(json.dumps({
        "voice": args.voice, "text": text, "steps": args.steps,
        "max_delta": args.max_delta, "baseline": info_base["detector_scores"],
        "adversarial": info_adv["detector_scores"],
        "softmin_baseline": info_base["detect_reward_softmin"],
        "softmin_adversarial": info_adv["detect_reward_softmin"],
        "similarity_baseline": info_base["similarity"],
        "similarity_adversarial": info_adv["similarity"],
        "wer_baseline": info_base["wer"], "wer_adversarial": info_adv["wer"],
        "history": history,
    }, indent=2), encoding="utf8")
    print(f"\nwrote {out_dir / f'pgd_{args.voice}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
