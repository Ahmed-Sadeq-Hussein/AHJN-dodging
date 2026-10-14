# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
rl.common -- the parts both reinforcement-learning designs share.

Checkpoint discovery, training-log reading, policy loading and generation are
identical for Design A (PPO-Lagrangian) and Design B (SAC with a gradient
hint); only the learner differs. Those live in rl/design_a.py and
rl/design_b.py, which both import from here.

Nothing here retrains or re-implements anything. The two training loops that
produced the recorded results are train_ppo_mel.py and train_sac_mel.py, and
the design modules drive those directly so a run launched through this package
is the same run, with the same checkpoint format and the same resume
behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cloners
import diffscore
from env_config import RESULTS_ROOT
from eval_generalization import _EvalAdapter, build_policy
from mel_policy import mel_edit_hook
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"

__all__ = ["RL_RUNS", "run_dir", "checkpoints", "latest_checkpoint", "history",
           "summarise", "load_policy", "generate", "load_voice"]


# --------------------------------------------------------------------------
# runs and checkpoints
# --------------------------------------------------------------------------

def run_dir(run_id: str) -> Path:
    """Where a run keeps its checkpoints and log. Accepts a bare id or a path."""
    p = Path(run_id)
    return p if p.is_dir() else RL_RUNS / run_id


def checkpoints(run_id: str) -> list[Path]:
    """Every checkpoint in a run, oldest first. `latest.pt` sorts last."""
    d = run_dir(run_id)
    if not d.is_dir():
        return []
    iters = sorted(d.glob("iter_*.pt"))
    tail = [p for p in (d / "latest.pt", d / "best.pt") if p.is_file()]
    return iters + tail


def latest_checkpoint(run_id: str) -> Path | None:
    """The checkpoint a resume would pick up: latest.pt, else the last snapshot."""
    d = run_dir(run_id)
    if (d / "latest.pt").is_file():
        return d / "latest.pt"
    snaps = sorted(d.glob("iter_*.pt"))
    return snaps[-1] if snaps else None


# --------------------------------------------------------------------------
# training logs
# --------------------------------------------------------------------------

def history(run_id: str) -> list[dict]:
    """The per-iteration training log as a list of dicts, in order.

    Both designs append one JSON object per iteration to train_log.jsonl. A
    truncated final line (a run killed mid-write) is skipped rather than
    raising, because a log is usually read while the run is still going.
    """
    log = run_dir(run_id) / "train_log.jsonl"
    if not log.is_file():
        return []
    rows = []
    for line in log.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarise(run_id: str, key: str = "mean_reward") -> dict:
    """Trend of one logged quantity: slope, t statistic, first and last value.

    The t statistic is what decides whether a run moved at all. Design B's
    reward trend came out at t = +0.46 against a threshold the run was powered
    to resolve, which is a well-powered null rather than an inconclusive one;
    reading the endpoints alone would have made it look like a small gain.
    """
    rows = [r for r in history(run_id) if isinstance(r.get(key), (int, float))]
    if len(rows) < 3:
        return {"n": len(rows)}
    import numpy as np
    y = np.array([float(r[key]) for r in rows])
    x = np.arange(len(y), dtype=float)
    n = len(y)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()))
    return {"n": n, "key": key, "slope": float(slope),
            "t": float(slope / se) if se > 0 else float("nan"),
            "first": float(y[0]), "last": float(y[-1]),
            "best": float(y.max()), "mean": float(y.mean())}


# --------------------------------------------------------------------------
# using a trained policy
# --------------------------------------------------------------------------

def load_policy(run_id: str, checkpoint: str | Path = "latest.pt", *,
                device: str | None = None, cosy=None):
    """Load a trained policy and wrap it so mel_edit_hook can drive it.

    Works for either design: build_policy reads the architecture out of the
    checkpoint, and _EvalAdapter computes the gradient hint on the fly only
    when the network expects one (Design B does, Design A does not).

    Returns (adapter, kind, cosy). `cosy` is returned so a caller generating
    several clips can pass it back in and avoid reloading CosyVoice2, which
    is by far the slowest part.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(checkpoint)
    if not ck.is_file():
        ck = run_dir(run_id) / checkpoint
    if not ck.is_file():
        raise FileNotFoundError(f"no checkpoint at {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    net, kind, _ = build_policy(blob, device, None)
    net.eval()
    if cosy is None:
        cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    adapter = _EvalAdapter(net, hift_call, device, 8.0, 0.5)
    return adapter, kind, cosy


def generate(run_id: str, voice: str, text: str, out_path, *,
             checkpoint: str | Path = "latest.pt", device: str | None = None,
             seed: int = 0, cosy=None, adapter=None, no_edit: bool = False):
    """Speak `text` in `voice` with a trained policy, writing a wav.

    Pass no_edit=True for the unedited clone under the same seed. The policy
    is deterministic, so it consumes no RNG, and the edited and unedited arms
    replay the same speech tokens and the same vocoder noise. Without that the
    two clips are different takes and nothing about them is comparable.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if adapter is None and not no_edit:
        adapter, _, cosy = load_policy(run_id, checkpoint, device=device, cosy=cosy)
    if cosy is None:
        cosy = cloners.load_model("CosyVoice2", device=device)

    ref, prompt = load_voice(voice)
    cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice, device=device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if no_edit:
        cloners.generate(text, ref, model="CosyVoice2", out_path=out_path,
                         device=device)
    else:
        with mel_edit_hook(cosy.model, adapter, deterministic=True, device=device):
            cloners.generate(text, ref, model="CosyVoice2", out_path=out_path,
                             device=device)
    return out_path
