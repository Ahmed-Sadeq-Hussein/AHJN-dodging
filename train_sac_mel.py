# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
train_sac_mel.py -- Design B. RUN IN THE "voiceclone" ENV.

    python train_sac_mel.py --voices riven swain samira jhin --iterations 400

Four things change from Design A (train_ppo_mel.py), each aimed at a
specific diagnosed failure of that run:

  1. GRADIENT HINT. The actor sees d(detector objective)/d(mel) alongside the
     mel. Design A had to find the edit direction from reward noise alone and
     could not: 1,040 episodes moved softmin backwards, while PGD with the
     same gradient moved it +0.581 in 60 steps.

  2. BLENDED DETECTOR OBJECTIVE. Design A's pure soft-min gave AASIST and
     AASIST-L 0.06% / 0.03% of gradient weight, leaving them undefended --
     and the policy duly degraded both (t = -3.54, -2.47) while improving
     neither of the two it was optimising. The 0.5 soft-min / 0.5 mean blend
     raises their weight ~200x. See diffscore.detector_objective.

  3. OFF-POLICY REPLAY. One reward costs a generation + 4 detectors +
     Whisper + CAM++. PPO discarded each after one gradient step; SAC reuses
     them, which is where the extra sample efficiency comes from.

  4. NO SUBPROCESS. Detectors run in-process (they work fine under this
     env's torch 2.3.1 / numpy 1.26, matching the other env to ~5e-5),
     removing 5-46 s per rollout AND making the gradient reachable at all.

Constraint handling (similarity, WER) is unchanged from Design A: quadratic
hinge with dual-ascent lambdas. Those worked -- the A run held similarity
0.83-0.85 and WER 4-6% throughout. The objective was what failed, not the
constraints.
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
import diffscore
import reward as reward_mod
import Tester
from env_config import RESULTS_ROOT, VOICE_DATA_ROOT
from mel_policy import mel_edit_hook
from sac_agent import GradInformedActor, ReplayBuffer, Temperature, TwinQ
from train_ppo_mel import DEFAULT_TEXTS, load_voice


# --------------------------------------------------------------------------
# reward, computed in-process
# --------------------------------------------------------------------------

def score_clip(wav_path, ref_wav, text, *, lagrange, sim_min, wer_max,
               whisper_model, device, k, softmin_weight):
    """Full constrained reward for one generated clip, no subprocess."""
    import detectors as det
    arr = det.score_file(wav_path, device=device)
    names = list(det.MODEL_ORDER)
    scores = {n: float(v) for n, v in zip(names, arr)}

    t = torch.tensor([scores[n] for n in names], dtype=torch.float64)
    sm = float(diffscore.soft_min(list(t), k=k))
    detect = softmin_weight * sm + (1.0 - softmin_weight) * float(t.mean())

    gen_emb = Tester.speaker_embedding(Path(wav_path))
    ref_emb = Tester.speaker_embedding(Path(ref_wav))
    sim = float(np.dot(gen_emb, ref_emb))

    hyp = Tester.transcribe(Path(wav_path), whisper_model, device)
    wer_val, *_ = Tester.wer(text, hyp)

    sim_slack, wer_slack = sim_min - sim, wer_val - wer_max
    sim_v, wer_v = max(0.0, sim_slack), max(0.0, wer_slack)
    r = detect - lagrange.lambda_sim * sim_v ** 2 - lagrange.lambda_wer * wer_v ** 2
    if sim_v > 0.25 or wer_v > 0.20:
        r = min(r, -5.0)

    return r, {"detector_scores": scores, "detect_objective": detect, "softmin": sm,
               "similarity": sim, "sim_slack": sim_slack, "sim_violation": sim_v,
               "wer": wer_val, "wer_slack": wer_slack, "wer_violation": wer_v,
               "transcript": hyp, "reward": r}


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def collect(cosy, actor, voices, texts, run_dir, it, batch, device, args, lagrange, buf):
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    roll = run_dir / "rollouts" / f"iter{it:04d}"
    roll.mkdir(parents=True, exist_ok=True)
    eps = []

    for i in range(batch):
        voice = random.choice(voices)
        text = random.choice(texts)
        ref, _ = load_voice(voice)
        out = roll / f"ep{i:03d}_{voice}.wav"

        # The hook needs the gradient hint for the mel it is about to edit,
        # but that mel only exists inside the vocoder call. So compute the
        # hint on the fly, from the very tensor being edited.
        holder = {}

        class _HintedActor(torch.nn.Module):
            """Thin adapter so mel_edit_hook can drive the grad-informed actor."""
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.max_delta = inner.max_delta
                self.min_log_std = inner.min_log_std
                self.log_std = inner.log_std

            def sample(self, mel, deterministic=False):
                g = (diffscore.mel_gradient_hint(
                        mel, hift_call, device, k=args.softmin_k,
                        softmin_weight=args.softmin_weight)
                     if self.inner.use_grad_hint else None)
                holder["grad"] = g
                holder["mel"] = mel.detach()
                return self.inner.sample(mel, g, deterministic=deterministic)

            def to(self, *a, **kw):
                self.inner.to(*a, **kw)
                return self

        t0 = time.time()
        with mel_edit_hook(cosy.model, _HintedActor(actor), deterministic=False,
                           device=device) as rec:
            cloners.generate(text, ref, model="CosyVoice2", out_path=out, device=device)
        if rec.n_calls == 0:
            continue

        eps.append({"voice": voice, "text": text, "ref": str(ref), "wav": out,
                    "mel": holder.get("mel"), "grad": holder.get("grad"),
                    "delta": rec.deltas[0].detach(), "gen_s": time.time() - t0})
        print(f"  [ep {i}] {voice:<8} {eps[-1]['gen_s']:.1f}s", flush=True)

    for e in eps:
        r, info = score_clip(e["wav"], e["ref"], e["text"], lagrange=lagrange,
                             sim_min=args.sim_min, wer_max=args.wer_max,
                             whisper_model=args.whisper, device=device,
                             k=args.softmin_k, softmin_weight=args.softmin_weight)
        e["reward"], e["info"] = r, info
        buf.add(e["mel"], e["grad"], e["delta"], r)
        if not args.keep_wavs:
            Path(e["wav"]).unlink(missing_ok=True)
    return eps


# --------------------------------------------------------------------------
# SAC update (single-step: Q regresses straight onto r, no bootstrapping)
# --------------------------------------------------------------------------

def sac_update(actor, critics, temp, opt_a, opt_c, opt_t, buf, args, device, rng):
    if len(buf) < args.min_buffer:
        return {"skipped": 1}

    stats = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0, "q_mean": 0.0, "n": 0}
    target_entropy = -args.target_entropy_scale     # per-element, matches log_prob scale

    for _ in range(args.updates_per_iter):
        batch = buf.sample(args.update_batch, rng)

        # --- critics: regress Q(mel, delta) onto the observed reward ---
        q_losses = []
        for b in batch:
            mel = b["mel"].to(device)
            delta = b["delta"].to(device)
            r = torch.tensor(b["reward"], dtype=torch.float32, device=device)
            q1, q2 = critics(mel, delta)
            q_losses.append((q1 - r) ** 2 + (q2 - r) ** 2)
        q_loss = torch.stack(q_losses).mean()
        opt_c.zero_grad(); q_loss.backward()
        torch.nn.utils.clip_grad_norm_(critics.parameters(), args.max_grad_norm)
        opt_c.step()

        # --- actor: maximise Q(mel, pi(mel)) - alpha * log pi ---
        a_losses, lps, qs = [], [], []
        for b in batch:
            mel = b["mel"].to(device)
            grad = None if b["grad"] is None else b["grad"].to(device)
            delta, lp, _ = actor.sample(mel, grad)
            q = critics.min_q(mel, delta)
            a_losses.append(temp.alpha.detach() * lp - q)
            lps.append(lp.detach())
            qs.append(q.detach())
        actor_loss = torch.stack(a_losses).mean()
        opt_a.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt_a.step()

        # --- temperature: drive entropy toward the target ---
        lp_mean = torch.stack(lps).mean()
        alpha_loss = -(temp.log_alpha * (lp_mean + target_entropy).detach())
        opt_t.zero_grad(); alpha_loss.backward(); opt_t.step()

        stats["q_loss"] += float(q_loss); stats["actor_loss"] += float(actor_loss)
        stats["alpha"] += float(temp.alpha); stats["q_mean"] += float(torch.stack(qs).mean())
        stats["n"] += 1

    for k in ("q_loss", "actor_loss", "alpha", "q_mean"):
        stats[k] /= max(stats["n"], 1)
    stats["skipped"] = 0
    return stats


def save_ckpt(path, *, it, actor, critics, temp, opt_a, opt_c, opt_t, lagrange,
              args, best):
    blob = {"iteration": it, "actor": actor.state_dict(),
            "critics": critics.state_dict(), "temp": temp.state_dict(),
            "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
            "opt_t": opt_t.state_dict(),
            "lambda_sim": lagrange.lambda_sim, "lambda_wer": lagrange.lambda_wer,
            "best_reward": best, "args": vars(args) if not isinstance(args, dict) else args,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp); os.replace(tmp, path)


def save_buffer(path: Path, buf) -> None:
    """Persist the replay buffer separately from the model checkpoint.

    Separate because it is large -- ~288 kB per transition at fp32 (mel, grad
    and delta are each 80 x T floats) -- so writing it every iteration
    alongside a 2 MB model checkpoint would dominate the run's I/O. Stored as
    fp16, which halves it at no meaningful cost: these tensors are inputs to a
    small conv net, not precision-critical accumulators.

    This matters for multi-session training: without it, --resume restarts SAC
    with an empty buffer and discards every transition bought with GPU time,
    which is precisely the advantage off-policy learning is supposed to have
    over the on-policy Design A.
    """
    payload = [{"mel": d["mel"].half(),
                "grad": None if d["grad"] is None else d["grad"].half(),
                "delta": d["delta"].half(),
                "reward": d["reward"]} for d in buf.data]
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_buffer(path: Path, buf) -> int:
    if not path.is_file():
        return 0
    for d in torch.load(path, map_location="cpu", weights_only=False):
        buf.data.append({"mel": d["mel"].float(),
                         "grad": None if d["grad"] is None else d["grad"].float(),
                         "delta": d["delta"].float(),
                         "reward": d["reward"]})
    while len(buf.data) > buf.capacity:
        buf.data.pop(0)
    return len(buf.data)


def build_parser() -> argparse.ArgumentParser:
    """Every knob Design B takes, in one list."""
    ap = argparse.ArgumentParser(description="Design B: gradient-informed SAC mel policy")
    ap.add_argument("--voices", nargs="+", default=["riven", "swain", "samira", "jhin"])
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--updates-per-iter", type=int, default=8)
    ap.add_argument("--update-batch", type=int, default=16)
    ap.add_argument("--min-buffer", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=4000)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--lr-temp", type=float, default=3e-4)
    ap.add_argument("--target-entropy-scale", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-delta", type=float, default=0.6,
                    help="edit BOUND. Kept at 0.6 so the policy can still reach "
                         "PGD-scale solutions (|delta| ~0.199), which are known good.")
    # Exploration scale, measured rather than guessed. Stepping along the
    # gradient hint on a real utterance: +0.005 took the detector objective
    # 0.458 -> 0.925, while +0.020 collapsed it to 0.143. The useful scale is
    # therefore ~0.005. Design A explored at |delta| = max_delta*sigma*sqrt(2/pi)
    # = 0.6*0.135*0.798 = 0.065 -- 13x too large, i.e. entirely inside the
    # harmful regime. init_log_std -4.5 gives sigma 0.011 -> |delta| ~0.0053,
    # which puts random exploration where the signal actually lives.
    ap.add_argument("--init-log-std", type=float, default=-4.5)
    ap.add_argument("--min-log-std", type=float, default=-6.0)
    ap.add_argument("--no-grad-hint", action="store_true",
                    help="ablation: run Design B without the gradient hint")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5,
                    help="1.0 = Design A's pure soft-min; 0.5 = blend with mean")
    ap.add_argument("--sim-min", type=float, default=0.70)
    ap.add_argument("--wer-max", type=float, default=0.15)
    ap.add_argument("--lambda-lr", type=float, default=0.05)
    ap.add_argument("--lambda-min", type=float, default=0.25,
                    help="floor on lambda; Design A's collapsed to 0 by iter ~36")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--keep-wavs", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # A wall-clock cap is more reliable than guessing an iteration count:
    # measured per-iteration time in this project has ranged from 78 s to
    # 25,876 s depending on GPU contention, so any fixed --iterations guess
    # either stops early or overruns. Set --iterations high and let this stop it.
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this many hours (checkpoint first)")
    ap.add_argument("--save-buffer", action="store_true", default=True,
                    help="persist the replay buffer so --resume keeps its experience")
    ap.add_argument("--resume", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / "rl_runs" / f"designB_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    print(f"run dir : {run_dir}\ndevice  : {device}\nvoices  : {args.voices}")
    print(f"grad hint: {not args.no_grad_hint}   softmin_weight: {args.softmin_weight}\n")

    cosy = cloners.load_model("CosyVoice2", device=device)
    for v in args.voices:
        ref, prompt = load_voice(v)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=v, device=device)
        print(f"enrolled {v}")

    actor = GradInformedActor(max_delta=args.max_delta,
                              init_log_std=args.init_log_std,
                              min_log_std=args.min_log_std,
                              use_grad_hint=not args.no_grad_hint).to(device)
    critics = TwinQ().to(device)
    temp = Temperature().to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(critics.parameters(), lr=args.lr_critic)
    opt_t = torch.optim.Adam(temp.parameters(), lr=args.lr_temp)
    lagrange = reward_mod.LagrangianMultipliers(lr=args.lambda_lr)
    buf = ReplayBuffer(args.buffer)

    start_it, best = 0, float("-inf")
    if args.resume:
        ck = (run_dir / "latest.pt") if args.resume == "auto" else Path(args.resume)
        if ck.is_file():
            b = torch.load(ck, map_location="cpu", weights_only=False)
            actor.load_state_dict(b["actor"]); critics.load_state_dict(b["critics"])
            temp.load_state_dict(b["temp"]); opt_a.load_state_dict(b["opt_a"])
            opt_c.load_state_dict(b["opt_c"]); opt_t.load_state_dict(b["opt_t"])
            lagrange.lambda_sim, lagrange.lambda_wer = b["lambda_sim"], b["lambda_wer"]
            start_it, best = b["iteration"] + 1, b.get("best_reward", float("-inf"))
            print(f"resumed {ck.name} @ iter {b['iteration']} -> continuing at {start_it}")
            n_restored = load_buffer(run_dir / "buffer.pt", buf)
            print(f"  replay buffer restored: {n_restored} transitions" if n_restored
                  else "  no buffer file - SAC would restart with an empty buffer")

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")

    wall_start = time.time()
    stopped_on_time = False

    for it in range(start_it, args.iterations):
        if args.max_hours is not None:
            elapsed_h = (time.time() - wall_start) / 3600.0
            if elapsed_h >= args.max_hours:
                print(f"\nreached --max-hours {args.max_hours} "
                      f"({elapsed_h:.2f} h elapsed) -- stopping cleanly at iteration {it}")
                stopped_on_time = True
                break
        print(f"\n=== iteration {it+1}/{args.iterations} ===", flush=True)
        t0 = time.time()
        eps = collect(cosy, actor, args.voices, DEFAULT_TEXTS, run_dir, it,
                      args.batch_size, device, args, lagrange, buf)
        if not eps:
            print("  no usable episodes"); continue

        stats = sac_update(actor, critics, temp, opt_a, opt_c, opt_t, buf, args, device, rng)
        lagrange.update([e["info"]["sim_slack"] for e in eps],
                        [e["info"]["wer_slack"] for e in eps])
        # Design A's lambdas hit exactly 0 by iteration ~36 and stayed there,
        # leaving the constraints unenforced for 70% of the run. Floor them.
        lagrange.lambda_sim = max(lagrange.lambda_sim, args.lambda_min)
        lagrange.lambda_wer = max(lagrange.lambda_wer, args.lambda_min)

        rec = {"iteration": it, "n_episodes": len(eps),
               "mean_reward": float(np.mean([e["reward"] for e in eps])),
               "episode_rewards": [float(e["reward"]) for e in eps],
               "episode_voices": [e["voice"] for e in eps],
               "reward_std": float(np.std([e["reward"] for e in eps], ddof=1)) if len(eps) > 1 else 0.0,
               "mean_detect": float(np.mean([e["info"]["detect_objective"] for e in eps])),
               "mean_softmin": float(np.mean([e["info"]["softmin"] for e in eps])),
               "mean_similarity": float(np.mean([e["info"]["similarity"] for e in eps])),
               "mean_wer": float(np.mean([e["info"]["wer"] for e in eps])),
               "per_detector": {n: float(np.mean([e["info"]["detector_scores"][n] for e in eps]))
                                for n in eps[0]["info"]["detector_scores"]},
               "lambda_sim": lagrange.lambda_sim, "lambda_wer": lagrange.lambda_wer,
               "buffer": len(buf), "seconds": time.time() - t0,
               "delta_abs_mean": float(np.mean([e["delta"].abs().mean().item() for e in eps])),
               "log_std_mean": float(actor.log_std.mean()), **stats}
        with log_path.open("a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")

        print(f"  reward {rec['mean_reward']:+.4f} | detect {rec['mean_detect']:.4f} "
              f"| softmin {rec['mean_softmin']:.4f} | sim {rec['mean_similarity']:.3f} "
              f"| wer {rec['mean_wer']*100:.1f}%", flush=True)
        print(f"  q_loss {stats.get('q_loss',0):.4f} | actor_loss {stats.get('actor_loss',0):+.4f} "
              f"| alpha {stats.get('alpha',0):.4f} | |delta| {rec['delta_abs_mean']:.4f} "
              f"| buf {len(buf)} | {rec['seconds']:.0f}s", flush=True)

        kw = dict(it=it, actor=actor, critics=critics, temp=temp, opt_a=opt_a,
                  opt_c=opt_c, opt_t=opt_t, lagrange=lagrange, args=args,
                  best=max(best, rec["mean_reward"]))
        save_ckpt(run_dir / "latest.pt", **kw)
        if args.snapshot_every and (it + 1) % args.snapshot_every == 0:
            save_ckpt(run_dir / f"iter_{it:04d}.pt", **kw)
            if args.save_buffer:
                save_buffer(run_dir / "buffer.pt", buf)
        if rec["mean_reward"] > best:
            best = rec["mean_reward"]
            save_ckpt(run_dir / "best.pt", **kw)
            print(f"  new best {best:+.4f} -> best.pt", flush=True)

    if args.save_buffer:
        save_buffer(run_dir / "buffer.pt", buf)
        print(f"replay buffer saved: {len(buf)} transitions")
    print(f"\ndone{' (hit time limit)' if stopped_on_time else ''}. log: {log_path}")
    print(f"continue later with:  --run-id {run_id} --resume auto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
