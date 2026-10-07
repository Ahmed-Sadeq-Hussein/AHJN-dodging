r"""
rl.design_b -- Design B, Soft Actor Critic with a gradient hint.

    from rl import design_b
    design_b.train(run_id="b1", iterations=400)
    design_b.train(run_id="b1", resume="auto")          # keeps the replay buffer
    design_b.history("designB_B9h")
    design_b.summarise("designB_B9h")                   # slope and t statistic
    design_b.generate("designB_B9h", "riven", "some text", "out.wav")

Three changes from Design A, each aimed at one diagnosed failure:

  1. a blended objective, half soft-min and half mean, so no detector can
     carry a near-zero weight and be abandoned;
  2. the gradient of the objective with respect to the Mel, fed to the actor
     as extra input channels -- the signal Design A discarded entirely;
  3. twin Q critics conditioned on the action, since Design A's state-value
     critic could say an utterance was hard but nothing about whether a
     particular edit was good.

RESULT: flat. Reward slope +0.000033 at t = +0.46, over 396 iterations and
2,520 episodes in the main nine-hour run, against a threshold of +/-0.045 the
run was powered to resolve. That is a well-powered null, not an inconclusive
one. It did fix what it aimed at: the degradation disappeared (t = -7.26 ->
+0.46) and the two previously undefended detectors improved. It did not move
the objective.

The diagnosis is in the internals rather than the reward curve. The entropy
coefficient climbed from 0.060 to 0.115 across nine hours while the Q loss
stayed near 0.03: a critic that predicts reward accurately but cannot RANK two
candidate edits leaves entropy as the only term the objective can still
improve. See rl/ANALYSIS.md.

Resuming restores the replay buffer as well as the weights. Without that, a
resumed off-policy run starts with an empty buffer and spends its first
iterations re-collecting experience it already had.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.common import (checkpoints, generate, history, latest_checkpoint,  # noqa: F401
                       load_policy, run_dir, summarise)

__all__ = ["train", "generate", "history", "summarise", "load_policy",
           "checkpoints", "latest_checkpoint", "run_dir", "DEFAULTS"]

DEFAULTS = {
    "voices": ["riven", "swain", "samira", "jhin"],
    "iterations": 400,
    "batch": 8,
    "wer-max": 0.15,
    "snapshot-every": 20,
    "save-buffer": True,
}


def train(run_id: str, *, resume: str | None = None, **overrides) -> Path:
    """Run (or continue) a Design B training job. Returns the run directory.

    resume="auto" restores weights, both critics, the temperature, the
    optimisers, the Lagrange multipliers AND the replay buffer.
    """
    argv = ["train_sac_mel.py", "--run-id", str(run_id)]
    opts = dict(DEFAULTS)
    opts.update({k.replace("_", "-"): v for k, v in overrides.items()})
    for key, value in opts.items():
        if value is None or value is False:
            continue
        if value is True:
            argv.append(f"--{key}")
            continue
        argv.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            argv.extend(str(v) for v in value)
        else:
            argv.append(str(value))
    if resume:
        argv += ["--resume", str(resume)]

    root = Path(__file__).resolve().parent.parent
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(root / "train_sac_mel.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = old
    return run_dir(f"designB_{run_id}")
