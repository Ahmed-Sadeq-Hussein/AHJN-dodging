# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
train_ppo_mel.py -- Design A's training loop. RUN IN THE "voiceclone" ENV.

    conda activate voiceclone
    python train_ppo_mel.py --voices riven ahmed --iterations 20 --batch-size 8

Puts together the three pieces built alongside it:

    mel_policy.MelResidualPolicy   the actor: a bounded edit on the mel
    mel_policy.MelValueNet         the critic: V(mel), the advantage baseline
    reward.compute_reward          soft-min(detectors) - hinge penalties
    reward.LagrangianMultipliers   self-tuning lambda on those penalties

Why there is no GAE / discounting / bootstrapping
--------------------------------------------------
An episode here is ONE utterance: observe the mel, apply one edit, hear one
terminal reward. There is no next state to bootstrap from, so the return is
just the reward and the advantage is (r - V(mel)). Multi-step machinery
would be dead code -- the iterative-refinement variant of Design A (edit,
re-score, edit again) is the version that would need it, and this is not
that variant.

Cost per iteration
------------------
Each episode costs one CosyVoice2 generation (~10-15 s), plus Whisper and
CAM++ on the result. Detector scoring is batched across the whole rollout in
one subprocess (see reward.score_detectors_batch) because model loading, not
inference, dominates it. Budget roughly batch_size x 25 s + 15 s per
iteration; a 20-iteration run at batch 8 is a couple of hours.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners
import reward as reward_mod
from env_config import RESULTS_ROOT, VOICE_DATA_ROOT
from mel_policy import MelResidualPolicy, MelValueNet, mel_edit_hook

# Sentences the policy is trained to say. Kept short and varied; the point is
# that the policy must generalise across content, not memorise one utterance.
DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog while the morning light spills across the valley.",
    "She sold seashells by the seashore every summer for as long as anyone could remember.",
    "There is nothing more deceptive than an obvious fact when you have not looked closely at it.",
    "The train arrives at platform nine just before midnight, and the station is almost empty.",
    "Rain fell steadily on the tin roof, and the whole house smelled of woodsmoke and coffee.",
]


# --------------------------------------------------------------------------
# save state -- so a killed run is resumable and a diverged run is recoverable
# --------------------------------------------------------------------------

def save_checkpoint(path: Path, *, iteration, policy, critic, optimizer,
                    lagrange, args, best_reward, rng: bool = True) -> None:
    """Write a checkpoint atomically, RNG included.

    Atomic because the naive `torch.save(..., path)` truncates the existing
    file first: a crash (or a Ctrl-C, or an OOM) partway through the write
    destroys the only good checkpoint along with the new one. Writing to a
    temp file in the same directory and then os.replace()-ing it is atomic on
    both NTFS and POSIX, so the previous checkpoint survives until the new
    one is complete.

    RNG state is saved because otherwise a resumed run draws different
    actions and different CosyVoice2 samples than an uninterrupted one would
    have -- the run stops being reproducible at exactly the point you most
    want to reason about it.
    """
    blob = {
        "iteration": iteration,
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lambda_sim": lagrange.lambda_sim,
        "lambda_wer": lagrange.lambda_wer,
        "lambda_history": lagrange.history,
        "best_reward": best_reward,
        "args": vars(args) if not isinstance(args, dict) else args,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if rng:
        blob["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)          # atomic swap; old file intact until now


def load_checkpoint(path: Path, *, policy, critic, optimizer, lagrange,
                    restore_rng: bool = True) -> dict:
    """Restore everything save_checkpoint wrote. Returns the raw blob."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(blob["policy"])
    critic.load_state_dict(blob["critic"])
    optimizer.load_state_dict(blob["optimizer"])
    lagrange.lambda_sim = blob["lambda_sim"]
    lagrange.lambda_wer = blob["lambda_wer"]
    lagrange.history = blob.get("lambda_history", [])

    if restore_rng and blob.get("rng"):
        rng = blob["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            except (RuntimeError, ValueError) as exc:
                # e.g. resumed on a box with a different GPU count -- not fatal
                print(f"  (could not restore CUDA RNG: {exc})")
    return blob


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """`latest.pt` if present, else the highest-numbered periodic snapshot."""
    latest = run_dir / "latest.pt"
    if latest.is_file():
        return latest
    snaps = sorted(run_dir.glob("iter_*.pt"))
    return snaps[-1] if snaps else None


def load_voice(voice: str) -> tuple[Path, str | None]:
    folder = VOICE_DATA_ROOT / voice
    wav = folder / "reference.wav"
    if not wav.is_file():
        raise FileNotFoundError(f"no reference.wav in {folder}")
    txt = folder / "reference.txt"
    return wav, (txt.read_text(encoding="utf8").strip() if txt.is_file() else None)


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def collect_rollout(cosy, policy, voices, texts, run_dir, iteration, batch_size,
                    device, whisper_model, sim_min, wer_max, lagrange, keep_wavs):
    """Generate `batch_size` episodes, then score them all in one batch.

    Returns a list of episode dicts, each carrying the mel/delta/log_prob the
    PPO update needs plus the reward breakdown for logging.
    """
    episodes = []
    roll_dir = run_dir / "rollouts" / f"iter{iteration:04d}"
    roll_dir.mkdir(parents=True, exist_ok=True)

    # --- generate ---
    for ep in range(batch_size):
        voice = random.choice(voices)
        text = random.choice(texts)
        ref_wav, _prompt = load_voice(voice)
        out_path = roll_dir / f"ep{ep:03d}_{voice}.wav"

        t0 = time.time()
        with mel_edit_hook(cosy.model, policy, deterministic=False, device=device) as rec:
            cloners.generate(text, ref_wav, model="CosyVoice2",
                             out_path=out_path, device=device)
        if rec.n_calls == 0:
            print(f"  [ep {ep}] WARNING: hook never fired, skipping episode")
            continue

        episodes.append({
            "voice": voice,
            "text": text,
            "reference_wav": str(ref_wav),
            "wav": out_path,
            # one generate() can vocode in several chunks; treat the whole set
            # of edits as one composite action -- its log-prob is the sum
            "mels": [m for m in rec.mels],
            "deltas": [d.detach() for d in rec.deltas],
            "old_log_prob": torch.stack(rec.log_probs).sum().detach(),
            "gen_seconds": time.time() - t0,
        })
        print(f"  [ep {ep}] {voice:<10} {rec.n_calls} vocode call(s)  "
              f"{episodes[-1]['gen_seconds']:.1f}s")

    if not episodes:
        return []

    # --- score every clip in ONE detector subprocess ---
    t0 = time.time()
    det_batch = reward_mod.score_detectors_batch([e["wav"] for e in episodes])
    print(f"  detectors: {len(episodes)} clip(s) in one call, {time.time()-t0:.1f}s")

    # --- per-episode reward (whisper + CAM++ in-process) ---
    for e in episodes:
        det = det_batch[str(e["wav"])]
        r, info = reward_mod.compute_reward(
            e["wav"], e["reference_wav"], e["text"],
            lagrange=lagrange, sim_min=sim_min, wer_max=wer_max,
            whisper_model=whisper_model, device=device, det_scores=det,
        )
        e["reward"] = r
        e["info"] = info
        if not keep_wavs:
            Path(e["wav"]).unlink(missing_ok=True)

    return episodes


# --------------------------------------------------------------------------
# PPO update
# --------------------------------------------------------------------------

def ppo_update(policy, critic, optimizer, episodes, *, clip_eps, epochs,
               entropy_coef, value_coef, max_grad_norm, device):
    """Clipped-surrogate PPO over a batch of single-step episodes.

    Episodes are processed one at a time rather than as a stacked batch: the
    mel is (1, 80, T) with T differing per utterance, and padding to a common
    T would feed the conv net silence it never sees at generation time. The
    nets are tiny, so the loop costs little.
    """
    rewards = torch.tensor([e["reward"] for e in episodes], dtype=torch.float32, device=device)
    old_log_probs = torch.stack([e["old_log_prob"] for e in episodes]).to(device).flatten()

    # Advantages are computed ONCE, from the value function as it stood when
    # the rollout was collected, and held fixed across all K epochs -- the
    # standard PPO construction. Recomputing them each epoch (with the critic
    # that this same update is training on this same batch) actively breaks
    # the update: the critic fits the rewards, V -> r, so the advantages decay
    # toward zero and each successive epoch contributes less policy gradient
    # than the last. It also makes the advantage non-stationary with respect
    # to the old_log_probs the importance ratio is measured against.
    with torch.no_grad():
        values = torch.stack([
            critic(e["mels"][0].to(device)).squeeze() for e in episodes
        ])
    advantages = rewards - values
    if advantages.numel() > 1 and advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "clip_frac": 0.0, "approx_kl": 0.0, "n_updates": 0}

    for _epoch in range(epochs):
        optimizer.zero_grad()
        policy_losses, value_losses, entropies = [], [], []
        clipped, kls = [], []

        for i, e in enumerate(episodes):
            # re-evaluate the OLD actions under the CURRENT policy
            new_lp = torch.stack([
                policy.log_prob_of(m.to(device), d.to(device))
                for m, d in zip(e["mels"], e["deltas"])
            ]).sum()

            ratio = torch.exp(new_lp - old_log_probs[i])
            adv = advantages[i]
            unclipped = ratio * adv
            clipped_term = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_losses.append(-torch.min(unclipped, clipped_term))

            v_pred = critic(e["mels"][0].to(device)).squeeze()
            value_losses.append((v_pred - rewards[i]) ** 2)

            # Pre-squash Gaussian entropy, PER DIMENSION, used only as a small
            # exploration bonus. The tanh-squashed distribution has no
            # closed-form entropy; this proxy is standard and is not part of
            # the objective's correctness, only its exploration pressure.
            #
            # .mean(), not .sum(): log_prob is averaged over all 80xT elements,
            # so a summed entropy (80 channels -> about -46) sits four orders
            # of magnitude above the policy loss (about 0.003) and no sane
            # entropy_coef can balance the two. Per-dimension (about -0.58)
            # puts them on comparable footing, which is what makes
            # --entropy-coef a usable knob rather than a decorative one.
            entropies.append((policy.log_std.clamp(min=policy.min_log_std)
                              + 0.5 * np.log(2 * np.pi * np.e)).mean())

            with torch.no_grad():
                clipped.append(float((ratio < 1 - clip_eps) or (ratio > 1 + clip_eps)))
                kls.append(float(old_log_probs[i] - new_lp))

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + list(critic.parameters()), max_grad_norm)
        optimizer.step()

        stats["policy_loss"] += float(policy_loss)
        stats["value_loss"] += float(value_loss)
        stats["entropy"] += float(entropy)
        stats["clip_frac"] += float(np.mean(clipped))
        stats["approx_kl"] += float(np.mean(kls))
        stats["n_updates"] += 1

    for k in ("policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"):
        stats[k] /= max(stats["n_updates"], 1)
    return stats


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Design A: PPO-Lagrangian mel-edit policy")
    ap.add_argument("--voices", nargs="+", default=["riven"])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8, help="episodes per iteration")
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--max-delta", type=float, default=0.6, help="mel edit bound (+/-)")
    ap.add_argument("--min-log-std", type=float, default=-3.0,
                    help="floor on the policy log-std; this is the exploration "
                         "control in a continuous action space (there is no "
                         "epsilon-greedy rate). -3.0 -> sigma >= 0.0498")
    ap.add_argument("--sim-min", type=float, default=0.70)
    ap.add_argument("--wer-max", type=float, default=0.10)
    ap.add_argument("--lambda-lr", type=float, default=0.05)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--keep-wavs", action="store_true",
                    help="keep every rollout clip (default: delete after scoring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="'auto' to continue the newest checkpoint in this run dir "
                         "(needs --run-id), or an explicit path to a .pt file")
    ap.add_argument("--snapshot-every", type=int, default=5,
                    help="also keep a permanent iter_NNNN.pt every N iterations")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / "rl_runs" / f"designA_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    print(f"run dir : {run_dir}")
    print(f"device  : {device}")
    print(f"voices  : {args.voices}\n")

    # one model load for the whole run; enrol every voice once up front
    cosy = cloners.load_model("CosyVoice2", device=device)
    for v in args.voices:
        ref, prompt = load_voice(v)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=v, device=device)
        print(f"enrolled {v}")

    policy = MelResidualPolicy(max_delta=args.max_delta,
                               min_log_std=args.min_log_std).to(device)
    critic = MelValueNet().to(device)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()), lr=args.lr)
    lagrange = reward_mod.LagrangianMultipliers(lr=args.lambda_lr)

    # ---- resume, if asked ----
    start_iteration = 0
    best_reward = float("-inf")
    if args.resume:
        ckpt_path = (find_latest_checkpoint(run_dir) if args.resume == "auto"
                     else Path(args.resume))
        if ckpt_path is None or not Path(ckpt_path).is_file():
            print(f"--resume {args.resume!r}: no checkpoint found "
                  f"(looked in {run_dir}); starting from scratch")
        else:
            blob = load_checkpoint(Path(ckpt_path), policy=policy, critic=critic,
                                   optimizer=optimizer, lagrange=lagrange)
            start_iteration = blob["iteration"] + 1
            best_reward = blob.get("best_reward", float("-inf"))
            print(f"resumed {ckpt_path} @ iteration {blob['iteration']} "
                  f"(saved {blob.get('saved_at')})")
            print(f"  continuing at iteration {start_iteration}, "
                  f"lambda_sim={lagrange.lambda_sim:.4f} lambda_wer={lagrange.lambda_wer:.4f}, "
                  f"best_reward={best_reward:+.4f}")

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")

    for iteration in range(start_iteration, args.iterations):
        print(f"\n=== iteration {iteration + 1}/{args.iterations} ===")
        t0 = time.time()

        episodes = collect_rollout(
            cosy, policy, args.voices, DEFAULT_TEXTS, run_dir, iteration,
            args.batch_size, device, args.whisper, args.sim_min, args.wer_max,
            lagrange, args.keep_wavs,
        )
        if not episodes:
            print("  no usable episodes this iteration, skipping update")
            continue

        stats = ppo_update(
            policy, critic, optimizer, episodes,
            clip_eps=args.clip_eps, epochs=args.epochs,
            entropy_coef=args.entropy_coef, value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm, device=device,
        )

        # one dual-ascent step per batch, on the SIGNED slack so lambda can relax
        lagrange.update(
            [e["info"]["sim_slack"] for e in episodes],
            [e["info"]["wer_slack"] for e in episodes],
        )

        rec = {
            "iteration": iteration,
            "n_episodes": len(episodes),
            "mean_reward": float(np.mean([e["reward"] for e in episodes])),
            # per-episode rewards and their spread: the noise floor. Every
            # batch-size and trend-detection number in MATH.md depends on
            # sigma_ep, and a mean alone cannot give it. Run with --epochs 0
            # to hold the policy fixed and measure this directly.
            "episode_rewards": [float(e["reward"]) for e in episodes],
            # which voice each episode used. The four training voices have very
            # different unedited baselines (measured: swain 0.087, riven 0.089,
            # samira 0.170, jhin 0.529), so batch composition alone moves
            # mean_reward by more than the policy plausibly does. The PPO update
            # is immune -- V(mel) is state-conditioned and absorbs it -- but a
            # trend fit on raw mean_reward is not, so record it and stratify.
            "episode_voices": [e["voice"] for e in episodes],
            "reward_std": float(np.std([e["reward"] for e in episodes], ddof=1))
                          if len(episodes) > 1 else 0.0,
            "mean_detect_softmin": float(np.mean([e["info"]["detect_reward_softmin"] for e in episodes])),
            "mean_similarity": float(np.mean([e["info"]["similarity"] for e in episodes])),
            "mean_wer": float(np.mean([e["info"]["wer"] for e in episodes])),
            "sim_violations": int(sum(e["info"]["sim_violation"] > 0 for e in episodes)),
            "wer_violations": int(sum(e["info"]["wer_violation"] > 0 for e in episodes)),
            "lambda_sim": lagrange.lambda_sim,
            "lambda_wer": lagrange.lambda_wer,
            "seconds": time.time() - t0,
            **stats,
        }
        with log_path.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")

        print(f"  reward {rec['mean_reward']:+.4f} | softmin(det) {rec['mean_detect_softmin']:.4f} "
              f"| sim {rec['mean_similarity']:.3f} | wer {rec['mean_wer']*100:.1f}% "
              f"| lam_sim {rec['lambda_sim']:.3f} lam_wer {rec['lambda_wer']:.3f}")
        print(f"  policy_loss {stats['policy_loss']:+.4f} | value_loss {stats['value_loss']:.4f} "
              f"| clip_frac {stats['clip_frac']:.2f} | approx_kl {stats['approx_kl']:+.4f} "
              f"| {rec['seconds']:.0f}s")

        # Three tiers, so you can always work backwards:
        #   latest.pt      -- resume point, overwritten atomically each iteration
        #   iter_NNNN.pt   -- permanent periodic snapshots, never overwritten
        #   best.pt        -- highest mean reward seen, survives later divergence
        save_kw = dict(iteration=iteration, policy=policy, critic=critic,
                       optimizer=optimizer, lagrange=lagrange, args=args,
                       best_reward=max(best_reward, rec["mean_reward"]))
        save_checkpoint(run_dir / "latest.pt", **save_kw)
        if args.snapshot_every and (iteration + 1) % args.snapshot_every == 0:
            save_checkpoint(run_dir / f"iter_{iteration:04d}.pt", **save_kw)
        if rec["mean_reward"] > best_reward:
            best_reward = rec["mean_reward"]
            save_checkpoint(run_dir / "best.pt", **save_kw)
            print(f"  new best mean_reward {best_reward:+.4f} -> best.pt")

    print(f"\ndone. log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
