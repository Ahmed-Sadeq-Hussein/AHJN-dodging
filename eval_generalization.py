r"""
eval_generalization.py -- does the learned edit help ONLY the voices it was
trained on, or does it transfer? RUN IN THE "voiceclone" ENV.

    python eval_generalization.py --run designB_B_main --pairs-per-voice 4

The question this answers
------------------------
Design B trains on four voices (riven, swain, samira, jhin). A policy that
improves only those has memorised them and is worthless -- PGD already
re-optimises per utterance far more cheaply (softmin 0.085 -> 0.666 in 60
steps). The ONLY thing an RL policy offers over PGD is generalisation: one
edit function that works on speech it has never seen. So the metric that
matters is the gap between trained and held-out voices, not the raw gain.

Method
------
Same paired design as eval_policy.py, per voice: generate the same utterance
with and without the learned edit under an identical seed. A deterministic
policy consumes no RNG (verified), so both arms replay the same LLM tokens
and the same HiFT vocoder noise, leaving the mel edit as the only
difference. That removes the three nuisance variance sources -- voice
choice, token sampling, vocoder noise -- which together gave Design A a
per-episode sigma of 0.121 and made its trend fits nearly useless.

Reported per voice, then grouped:

    delta_softmin = softmin(policy) - softmin(baseline)

  * TRAINED    the four voices in the run's config
  * HELD-OUT   every other voice with a reference.wav
  * gap        mean(TRAINED) - mean(HELD-OUT); near zero means it transfers

Also reported: whether improvement correlates with baseline difficulty. A
policy that only helps voices already near 1.0 has found nothing useful --
the interesting case is improving the ones the detectors currently catch.
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
import diffscore
import reward as reward_mod
import Tester
from env_config import RESULTS_ROOT, VOICE_DATA_ROOT
from mel_policy import MelResidualPolicy, mel_edit_hook
from train_ppo_mel import DEFAULT_TEXTS, load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"


def all_voices() -> list[str]:
    return sorted(p.name for p in VOICE_DATA_ROOT.iterdir()
                  if p.is_dir() and (p / "reference.wav").is_file())


def paired_stats(d: np.ndarray):
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else 0.0), float("nan"), float("nan")
    m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n))
    return m, se, (m / se if se > 0 else float("nan"))


def build_policy(blob, device, args):
    """Load either a Design A ('policy') or Design B ('actor') checkpoint."""
    targs = blob.get("args", {})
    if "actor" in blob:
        from sac_agent import GradInformedActor
        net = GradInformedActor(
            hidden=targs.get("hidden", 96),
            max_delta=targs.get("max_delta", 0.6),
            init_log_std=targs.get("init_log_std", -4.5),
            min_log_std=targs.get("min_log_std", -6.0),
            use_grad_hint=not targs.get("no_grad_hint", False),
            dilations=tuple(targs.get("dilations", (1, 1, 1))),
            spk_dim=targs.get("spk_dim", 0),
        ).to(device)
        net.load_state_dict(blob["actor"])
        if blob.get("distilled"):
            kind = "D (distilled from PGD)"
        else:
            kind = "B (SAC" + (" + grad hint)" if net.use_grad_hint else ", no hint)")
    else:
        net = MelResidualPolicy(
            max_delta=targs.get("max_delta", 0.6),
            min_log_std=targs.get("min_log_std", -3.0),
        ).to(device)
        net.load_state_dict(blob["policy"])
        kind = "A (PPO)"
    net.eval()
    return net, kind, targs


class _EvalAdapter(torch.nn.Module):
    """Drives either policy type through mel_edit_hook, computing the
    gradient hint on the fly when the actor expects one."""

    def __init__(self, inner, hift_call, device, k, w):
        super().__init__()
        self.inner = inner
        self.max_delta = inner.max_delta
        self.min_log_std = getattr(inner, "min_log_std", -3.0)
        self.log_std = inner.log_std
        self.hift_call, self.device, self.k, self.w = hift_call, device, k, w

    def sample(self, mel, deterministic=False):
        if getattr(self.inner, "use_grad_hint", False):
            g = diffscore.mel_gradient_hint(mel, self.hift_call, self.device,
                                            k=self.k, softmin_weight=self.w)
            return self.inner.sample(mel, g, deterministic=deterministic)
        return self.inner.sample(mel, deterministic=deterministic)

    def to(self, *a, **kw):
        self.inner.to(*a, **kw)
        return self


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--voices", nargs="*", default=None, help="default: every voice")
    ap.add_argument("--pairs-per-voice", type=int, default=4)
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--sim-min", type=float, default=0.70)
    ap.add_argument("--wer-max", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = RL_RUNS / args.run
    ck = run_dir / args.checkpoint
    if not ck.is_file():
        print(f"no checkpoint at {ck}"); return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    net, kind, targs = build_policy(blob, device, args)
    trained = set(targs.get("voices", []))
    voices = args.voices or all_voices()

    print(f"checkpoint : {args.run}/{args.checkpoint}  (iteration {blob.get('iteration')})")
    print(f"design     : {kind}")
    print(f"trained on : {sorted(trained)}")
    print(f"evaluating : {len(voices)} voices x {args.pairs_per_voice} pairs\n")

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    adapter = _EvalAdapter(net, hift_call, device, args.softmin_k, args.softmin_weight)

    eval_dir = run_dir / "eval_generalization"
    eval_dir.mkdir(parents=True, exist_ok=True)
    import detectors as det
    names = list(det.MODEL_ORDER)

    rows = []
    for vi, voice in enumerate(voices):
        try:
            ref, prompt = load_voice(voice)
        except FileNotFoundError:
            print(f"  {voice}: no reference.wav, skipping"); continue
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice, device=device)

        for pi in range(args.pairs_per_voice):
            text = DEFAULT_TEXTS[pi % len(DEFAULT_TEXTS)]
            seed = args.seed + vi * 100 + pi

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            base = eval_dir / f"{voice}_{pi}_base.wav"
            cloners.generate(text, ref, model="CosyVoice2", out_path=base, device=device)

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            pol = eval_dir / f"{voice}_{pi}_pol.wav"
            with mel_edit_hook(cosy.model, adapter, deterministic=True, device=device) as rec:
                cloners.generate(text, ref, model="CosyVoice2", out_path=pol, device=device)
            dmag = float(np.mean([d.abs().mean().item() for d in rec.deltas])) if rec.deltas else 0.0

            rows.append({"voice": voice, "pair": pi, "text": text, "ref": str(ref),
                         "base": base, "pol": pol, "delta_abs": dmag,
                         "trained": voice in trained})
        print(f"  [{vi+1}/{len(voices)}] {voice:<12} {args.pairs_per_voice} pairs generated", flush=True)

    print("\nscoring every clip...")
    def score(p):
        arr = det.score_file(p, device=device)
        s = {n: float(v) for n, v in zip(names, arr)}
        t = torch.tensor([s[n] for n in names], dtype=torch.float64)
        sm = float(diffscore.soft_min(list(t), k=args.softmin_k))
        obj = args.softmin_weight * sm + (1 - args.softmin_weight) * float(t.mean())
        return s, sm, obj

    for r in rows:
        for arm in ("base", "pol"):
            s, sm, obj = score(r[arm])
            r[f"{arm}_det"], r[f"{arm}_softmin"], r[f"{arm}_obj"] = s, sm, obj
        ge = Tester.speaker_embedding(Path(r["pol"])); re_ = Tester.speaker_embedding(Path(r["ref"]))
        be = Tester.speaker_embedding(Path(r["base"]))
        r["pol_sim"] = float(np.dot(ge, re_)); r["base_sim"] = float(np.dot(be, re_))
        r["pol_wer"] = Tester.wer(r["text"], Tester.transcribe(Path(r["pol"]), "base", device))[0]
        r["base_wer"] = Tester.wer(r["text"], Tester.transcribe(Path(r["base"]), "base", device))[0]
        Path(r["base"]).unlink(missing_ok=True); Path(r["pol"]).unlink(missing_ok=True)

    # ---------------- per-voice ----------------
    print("\n" + "=" * 96)
    print(f"{'voice':<13}{'set':<10}{'base softmin':>13}{'policy':>9}{'delta':>9}{'t':>7}"
          f"{'sim d':>8}{'wer d':>8}{'|delta|':>9}")
    print("-" * 96)
    per_voice = []
    for voice in voices:
        rs = [r for r in rows if r["voice"] == voice]
        if not rs:
            continue
        d = np.array([r["pol_softmin"] - r["base_softmin"] for r in rs])
        m, se, t = paired_stats(d)
        rec = {"voice": voice, "trained": rs[0]["trained"],
               "base_softmin": float(np.mean([r["base_softmin"] for r in rs])),
               "pol_softmin": float(np.mean([r["pol_softmin"] for r in rs])),
               "delta_softmin": m, "se": se, "t": t,
               "d_sim": float(np.mean([r["pol_sim"] - r["base_sim"] for r in rs])),
               "d_wer": float(np.mean([r["pol_wer"] - r["base_wer"] for r in rs])),
               "delta_abs": float(np.mean([r["delta_abs"] for r in rs])),
               "n": len(rs)}
        per_voice.append(rec)
        print(f"{voice:<13}{'TRAINED' if rec['trained'] else 'held-out':<10}"
              f"{rec['base_softmin']:>13.4f}{rec['pol_softmin']:>9.4f}{m:>+9.4f}"
              f"{t:>7.2f}{rec['d_sim']:>+8.3f}{rec['d_wer']*100:>+7.1f}%{rec['delta_abs']:>9.4f}")

    # ---------------- generalisation ----------------
    tr = np.array([r["delta_softmin"] for r in per_voice if r["trained"]])
    ho = np.array([r["delta_softmin"] for r in per_voice if not r["trained"]])
    all_d = np.array([r["pol_softmin"] - r["base_softmin"] for r in rows])
    tr_d = np.array([r["pol_softmin"] - r["base_softmin"] for r in rows if r["trained"]])
    ho_d = np.array([r["pol_softmin"] - r["base_softmin"] for r in rows if not r["trained"]])

    print("\n" + "=" * 96)
    print("GENERALISATION")
    print("=" * 96)
    for label, d in (("TRAINED voices", tr_d), ("HELD-OUT voices", ho_d), ("ALL", all_d)):
        if len(d) == 0:
            continue
        m, se, t = paired_stats(d)
        sig = "significant" if abs(t) > 2 else "not distinguishable from 0"
        print(f"  {label:<18} n={len(d):>3}  mean delta softmin {m:+.4f}  SE {se:.4f}  t={t:+.2f}   {sig}")
    gap = (float(tr_d.mean()) - float(ho_d.mean())) if len(tr_d) and len(ho_d) else float("nan")
    print(f"\n  generalisation gap (trained - held-out): {gap:+.4f}")

    # does it help the HARD voices, or only the easy ones?
    if len(per_voice) > 2:
        b = np.array([r["base_softmin"] for r in per_voice])
        dd = np.array([r["delta_softmin"] for r in per_voice])
        if b.std() > 1e-6 and dd.std() > 1e-6:
            corr = float(np.corrcoef(b, dd)[0, 1])
            print(f"  corr(baseline difficulty, improvement) = {corr:+.3f}")
            print("    negative => helps the voices the detectors currently CATCH (what we want)")
            print("    positive => only helps voices that were already passing")

    print("\n" + "=" * 96)
    m_all, se_all, t_all = paired_stats(all_d)
    m_ho, se_ho, t_ho = paired_stats(ho_d) if len(ho_d) else (0, 0, 0)
    if t_all > 2 and t_ho > 2 and abs(gap) < 0.05:
        print("VERDICT: improves detection evasion and TRANSFERS to unseen voices.")
    elif t_all > 2 and t_ho <= 2:
        print("VERDICT: improves only the voices it trained on -- memorisation, not")
        print("         generalisation. PGD does per-utterance optimisation better.")
    elif abs(t_all) <= 2:
        print("VERDICT: no effect on evasion anywhere, trained or held-out.")
    else:
        print("VERDICT: the policy makes evasion WORSE.")
    print("=" * 96)

    out = Path(args.out) if args.out else run_dir / f"generalization_{args.checkpoint.replace('.pt','')}.json"
    out.write_text(json.dumps({
        "run": args.run, "checkpoint": args.checkpoint, "design": kind,
        "trained_voices": sorted(trained), "pairs_per_voice": args.pairs_per_voice,
        "per_voice": per_voice,
        "trained_mean": float(tr_d.mean()) if len(tr_d) else None,
        "heldout_mean": float(ho_d.mean()) if len(ho_d) else None,
        "gap": gap,
        "all": {"mean": m_all, "se": se_all, "t": t_all, "n": len(all_d)},
    }, indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
