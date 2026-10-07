r"""
rl -- the two reinforcement-learning designs, with one module each.

    rl/design_a.py   PPO with Lagrangian constraints
    rl/design_b.py   Soft Actor Critic with a gradient hint
    rl/common.py     checkpoints, training logs, loading and generation
    rl/mos_corpus.py renders a listening-test corpus from trained policies
    rl/ANALYSIS.md   what each design did, what it did not do, and why

Both modules expose the same four things:

    train(run_id, resume=..., **hyperparameters)   run or continue a job
    history(run_id)                                the per-iteration log
    summarise(run_id)                              slope and t statistic
    generate(run_id, voice, text, out_path)        speak with a trained policy

Training writes a checkpoint every few iterations plus latest.pt, and one JSON
line per iteration, so a job can be read while it runs and resumed after it
stops. Neither module re-implements a learner: they drive train_ppo_mel.py and
train_sac_mel.py, which are the loops that produced the recorded results.
"""

from . import common, design_a, design_b  # noqa: F401
from .common import (RL_RUNS, checkpoints, generate, history,  # noqa: F401
                     latest_checkpoint, load_policy, run_dir, summarise)
