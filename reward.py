r"""
reward.py -- the constrained reward for Design A (PPO-Lagrangian), and the
one place detector scores cross from the "voiceclone" env into this process.

    from reward import compute_reward, LagrangianMultipliers

    lag = LagrangianMultipliers()
    r, info = compute_reward(wav_path, reference_wav, target_text, lagrange=lag)
    lag.update([info["sim_slack"]], [info["wer_slack"]])   # after a batch -- signed, so lambda can relax

Reward shape
------------
    R = soft_min(detector scores)  -  lambda_sim * hinge(sim_min - sim)^2
                                    -  lambda_wer * hinge(wer - wer_max)^2

- soft_min, not mean, across the four detectors: a mean lets the policy max
  out on whichever two detectors are already easy (RawNet2/RawGAT-ST were
  already near-1.0 on unedited CosyVoice2 clones in this repo's own runs)
  while ignoring the one that still catches it. soft_min forces the worst
  detector to actually move. This is the same objective structure as
  ensemble-transferable adversarial attacks (Liu et al. 2017) -- fooling a
  committee, not a single classifier.
- Quadratic hinge, not linear: a small constraint violation costs little, a
  large one costs a lot, so the policy can't trade identity/intelligibility
  for detector score at a constant rate.
- lambda_sim / lambda_wer are NOT fixed constants -- see LagrangianMultipliers.
  They rise via dual ascent whenever a training batch violates its floor on
  average, and relax when it doesn't, so the "penalize harder" requirement
  is enforced by the optimizer, not by a hand-picked weight.
- A hard safety floor sits under all of it: if either violation is more than
  2x its threshold, the reward is clipped to a large negative value
  regardless of detector score, so no amount of evasion can pay for
  wrecking the voice.

Detector scores are the one thing this module cannot compute in-process:
they need the "voicefake" env's torch/numpy pins, incompatible with the
"voiceclone" env this module and CosyVoice2 run under (see requirements.txt
and requirements-voiceclone.txt's header comments). So they're fetched with
one subprocess call to `detectors.py --json` under VOICEFAKE_PYTHON. CAM++
similarity and Whisper WER both already run fine in this env (Tester.py
uses them directly), so those stay in-process.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import VOICEFAKE_PYTHON

# Tester.py's speaker_embedding / transcribe / wer are already built and
# verified (Tester.py's own runs this session) -- reused rather than
# reimplemented so there is exactly one WER/similarity implementation in
# this repo, not two that could quietly drift apart.
import Tester


# --------------------------------------------------------------------------
# detector scores (cross-env subprocess)
# --------------------------------------------------------------------------

def score_detectors_batch(wav_paths: list[str | Path], *, device: str | None = None,
                          timeout: float = 900.0) -> dict[str, dict[str, float]]:
    """{path: {"AASIST": 0.95, ...}} for many clips in ONE subprocess.

    Batching is the whole point: loading the four detector models costs
    ~13 s and dominates a single-file call, so scoring a PPO rollout of N
    episodes one-at-a-time would burn N x 13 s on nothing but model loads.
    One invocation per rollout amortises that to 13 s per *batch*.

    Raises RuntimeError with the subprocess's stderr on failure -- a silent
    fallback (e.g. treating a crashed detector as score 0) would corrupt the
    reward without any visible sign of why.
    """
    if not VOICEFAKE_PYTHON:
        raise RuntimeError(
            "VOICEFAKE_PYTHON is not set in .env -- reward.py needs it to reach "
            "the four anti-spoofing detectors, which only import under the "
            "voicefake env's torch/numpy pins. Add e.g. "
            r"VOICEFAKE_PYTHON=E:\envs\voicefake\python.exe to .env."
        )
    if not wav_paths:
        return {}

    detectors_py = Path(__file__).resolve().parent / "detectors.py"
    cmd = [VOICEFAKE_PYTHON, str(detectors_py)] + [str(p) for p in wav_paths] + ["--json"]
    if device:
        cmd += ["--device", device]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"detector scoring failed (exit {proc.returncode}) for "
            f"{len(wav_paths)} clip(s):\n{proc.stderr[-2000:]}"
        )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"could not parse detector JSON. stdout was:\n{proc.stdout}"
        ) from exc


def score_detectors(wav_path: str | Path, *, device: str | None = None,
                    timeout: float = 120.0) -> dict[str, float]:
    """{"AASIST": 0.95, ...} for one clip. Prefer score_detectors_batch in a
    training loop -- this pays the full model-load cost every call."""
    out = score_detectors_batch([wav_path], device=device, timeout=timeout)
    # detectors.py keys by the path string exactly as passed on the command line
    return out[str(wav_path)]


def soft_min(values, k: float = 8.0) -> float:
    """Smooth approximation to min(values); k -> inf recovers the hard min.

    -log(mean(exp(-k*x))) / k. Bounded, differentiable in spirit (used here
    on plain floats, since PPO's loss only needs to differentiate log_prob,
    not the reward itself), and much less noisy than a hard min when scores
    are close together.
    """
    x = np.asarray(list(values), dtype=np.float64)
    # subtract max(-k*x) for numerical stability before exponentiating
    z = -k * x
    m = z.max()
    return float(-(m + math.log(np.mean(np.exp(z - m)))) / k)


# --------------------------------------------------------------------------
# constrained reward
# --------------------------------------------------------------------------

@dataclass
class LagrangianMultipliers:
    """Dual variables for the similarity/WER constraints.

    update() takes this batch's *signed* slack -- (threshold - value),
    positive when violated, negative when comfortably satisfied -- and does
    one projected-gradient-ascent step on each lambda: lambda grows while
    the constraint is violated, and genuinely relaxes back down once it's
    comfortably satisfied, rather than only ever holding flat.

    That relaxation is deliberate, not the textbook-minimal version: a dual
    ascent step driven by the hinge-clamped violation alone (clamped at 0,
    same value compute_reward's penalty term uses) is monotonically
    non-decreasing -- correct in the classical sense (it still converges to
    whatever tension level keeps the constraint satisfied), but it means
    lambda can never soften even after the policy has clearly moved well
    past the threshold, which makes the penalty needlessly conservative
    for the rest of training. Using the signed slack here (sometimes called
    a "soft" Lagrangian relaxation, e.g. in reward-constrained policy
    optimization / RCPO-style updates) fixes that, while compute_reward's
    actual penalty term stays hinge-clamped at zero -- a satisfied episode
    is never penalized, only lambda's *trajectory* is allowed to ease.
    """
    lambda_sim: float = 1.0
    lambda_wer: float = 1.0
    lr: float = 0.05
    lambda_max: float = 50.0
    history: list[dict] = field(default_factory=list)

    def update(self, sim_slacks: list[float], wer_slacks: list[float]) -> None:
        mean_sim_slack = float(np.mean(sim_slacks)) if sim_slacks else 0.0
        mean_wer_slack = float(np.mean(wer_slacks)) if wer_slacks else 0.0
        self.lambda_sim = float(np.clip(self.lambda_sim + self.lr * mean_sim_slack, 0.0, self.lambda_max))
        self.lambda_wer = float(np.clip(self.lambda_wer + self.lr * mean_wer_slack, 0.0, self.lambda_max))
        self.history.append({
            "lambda_sim": self.lambda_sim, "lambda_wer": self.lambda_wer,
            "mean_sim_slack": mean_sim_slack, "mean_wer_slack": mean_wer_slack,
        })


def compute_reward(
    wav_path: str | Path,
    reference_wav: str | Path,
    target_text: str,
    *,
    lagrange: LagrangianMultipliers,
    sim_min: float = 0.70,
    wer_max: float = 0.10,
    softmin_k: float = 8.0,
    # Absolute margins past the threshold, NOT multiples of it. The earlier
    # "violation > 2 * sim_min" form was unreachable: sim_violation is
    # max(0, sim_min - sim), so with sim_min=0.70 it is bounded above by 0.70
    # for any non-negative cosine (and CAM++ similarities measured in this
    # repo run 0.16-1.0), while the trigger needed 1.40. The similarity floor
    # could therefore never fire. WER is unbounded above so its version did
    # work -- but the two need different treatment, hence two margins.
    #   sim_hard_margin=0.25 -> fires below sim 0.45, which is under every
    #   cross-speaker pair measured in this repo's own similarity matrix.
    #   wer_hard_margin=0.20 -> fires above 30% WER, i.e. barely intelligible.
    sim_hard_margin: float = 0.25,
    wer_hard_margin: float = 0.20,
    hard_floor_value: float = -5.0,
    whisper_model: str = "base",
    device: str | None = None,
    det_scores: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """One (reward, breakdown) pair for one generated clip.

    sim_min=0.70 and wer_max=0.10 both come from thresholds already used
    elsewhere in this repo: cloners.py's own OpenVoiceV2 similarity verdict
    calls >=0.70 "same speaker", and 10% WER is a reasonable comprehension
    floor given this repo's own riven baseline sits around 13% and its
    CosyVoice2 clone at 5%.
    """
    # det_scores can be passed in when a caller already batch-scored the whole
    # rollout (see score_detectors_batch) -- otherwise fetch this one clip.
    if det_scores is None:
        det_scores = score_detectors(wav_path)
    detect_reward = soft_min(det_scores.values(), k=softmin_k)

    gen_emb = Tester.speaker_embedding(Path(wav_path))
    ref_emb = Tester.speaker_embedding(Path(reference_wav))
    sim = float(np.dot(gen_emb, ref_emb))     # both L2-normalised -> cosine

    hyp = Tester.transcribe(Path(wav_path), whisper_model, device)
    wer_val, sub, dele, ins, n_words = Tester.wer(target_text, hyp)

    # signed slack -- positive means violated, negative means comfortably
    # satisfied. Feed *this* to LagrangianMultipliers.update() so lambda can
    # relax; the reward's own penalty term uses the hinge-clamped version
    # below, so a satisfied episode is never penalized either way.
    sim_slack = sim_min - sim
    wer_slack = wer_val - wer_max

    sim_violation = max(0.0, sim_slack)
    wer_violation = max(0.0, wer_slack)

    penalty_sim = lagrange.lambda_sim * sim_violation ** 2
    penalty_wer = lagrange.lambda_wer * wer_violation ** 2

    reward = detect_reward - penalty_sim - penalty_wer

    hard_violation = (
        sim_violation > sim_hard_margin
        or wer_violation > wer_hard_margin
    )
    if hard_violation:
        reward = min(reward, hard_floor_value)

    info = {
        "detector_scores": det_scores,
        "detect_reward_softmin": detect_reward,
        "similarity": sim,
        "sim_slack": sim_slack,
        "sim_violation": sim_violation,
        "penalty_sim": penalty_sim,
        "wer": wer_val,
        "wer_slack": wer_slack,
        "wer_violation": wer_violation,
        "penalty_wer": penalty_wer,
        "transcript": hyp,
        "hard_floor_applied": hard_violation,
        "lambda_sim": lagrange.lambda_sim,
        "lambda_wer": lagrange.lambda_wer,
        "reward": reward,
    }
    return reward, info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute one PPO-Lagrangian reward for a clip")
    ap.add_argument("wav")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--text", required=True, help="target text the clip is supposed to say")
    ap.add_argument("--sim-min", type=float, default=0.70)
    ap.add_argument("--wer-max", type=float, default=0.10)
    args = ap.parse_args()

    lag = LagrangianMultipliers()
    r, info = compute_reward(args.wav, args.reference, args.text,
                             lagrange=lag, sim_min=args.sim_min, wer_max=args.wer_max)
    print(json.dumps(info, indent=2))
    print(f"\nreward = {r:.4f}")
