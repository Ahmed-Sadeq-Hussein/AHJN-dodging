r"""
rl.design_a -- Design A, PPO with Lagrangian constraints.

    from rl import design_a
    design_a.train(voices=["riven", "swain"], iterations=130, run_id="a1")
    design_a.train(run_id="a1", resume="auto")          # pick up where it stopped
    design_a.history("designA_budget4h")                # per-iteration log
    design_a.summarise("designA_budget4h")              # slope and t statistic
    design_a.generate("designA_budget4h", "riven", "some text", "out.wav")

A stochastic policy proposes a residual on the Mel spectrogram, the clip is
rendered and scored, and the scalar reward updates the policy by clipped
policy gradient. Similarity and word error rate enter as dual-adapted
multipliers rather than as fixed penalties.

RESULT: this design made detection significantly EASIER than doing nothing,
with a reward slope of -0.000727 at t = -7.26 over 130 iterations and 1,040
episodes. It is kept because the reason is instructive and is not a property
of PPO. Pure soft-min at k = 8 hands about half its gradient to each of the
two detectors that are already failing and roughly 0.06 and 0.03 percent to
the two that are already passing, so the passing pair was undefended and the
policy degraded them while never improving the others. The aggregator chose
that outcome, not the algorithm. See rl/ANALYSIS.md.

Note that an episode here is a single step: one Mel in, one edit out, one
reward. There are no state transitions, so this is formally a contextual
bandit rather than a Markov decision process.
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

# The settings the recorded run used. Anything passed to train() overrides one.
DEFAULTS = {
    "voices": ["riven", "swain", "samira", "jhin"],
    "iterations": 130,
    "batch-size": 8,
    "epochs": 4,
    "lr": 6e-4,
    "lambda-lr": 0.2,
    "wer-max": 0.15,
    "snapshot-every": 10,
}


def train(run_id: str, *, resume: str | None = None, **overrides) -> Path:
    """Run (or continue) a Design A training job. Returns the run directory.

    resume="auto" picks up the newest checkpoint in the run directory and
    carries the optimiser state and the Lagrange multipliers with it, so a
    budget split across sessions is equivalent to one long run.

    Everything is written as it goes: a checkpoint every `snapshot-every`
    iterations plus latest.pt, and one JSON line per iteration in
    train_log.jsonl. Read the log with history() while the job is still
    running; nothing here waits for it to finish.
    """
    argv = ["train_ppo_mel.py", "--run-id", str(run_id)]
    opts = dict(DEFAULTS)
    opts.update({k.replace("_", "-"): v for k, v in overrides.items()})
    for key, value in opts.items():
        if value is None or value is False:
            continue
        argv.append(f"--{key}")
        if value is True:
            argv.pop()
            argv.append(f"--{key}")
            continue
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
        runpy.run_path(str(root / "train_ppo_mel.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = old
    return run_dir(f"designA_{run_id}")
