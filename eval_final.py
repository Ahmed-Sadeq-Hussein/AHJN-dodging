r"""
eval_final.py -- the decisive comparison. RUN IN "voiceclone", alone on the GPU.

    python eval_final.py --distill-run distill_distill1

Arms, all from one shared control with identical seeds:
    baseline    unedited CosyVoice2
    designB     best SAC checkpoint (the RL result)
    distilled   the network trained by regression on PGD solutions

Evaluated on voices that appear in NO training set and sentences that appear
in no collection run, so a gain here cannot be memorisation on either axis.
The training voices are included as a labelled reference group, because "no
gain on unseen data" and "no gain anywhere" are different findings.

Metrics: the four detectors and their soft-min; similarity to the target
speaker; WER; and SQUIM's PESQ/STOI estimates, which is the perceptual axis
this project tracked for the first time only after the PGD audit showed a
0.67 PESQ cost that similarity and WER were both blind to.
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
import diffscore
import quality
import Tester
from env_config import RESULTS_ROOT
from eval_generalization import build_policy, _EvalAdapter
from mel_policy import mel_edit_hook
from textbank import EVAL_TEXTS, EVAL_VOICES, TRAIN_VOICES
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"


class AmplifiedAdapter(torch.nn.Module):
    """Wraps a policy so its edit can be scaled and applied more than once.

    Regression against noisy targets shrinks predictions toward the mean, so
    a distilled edit can hold the right direction at the wrong magnitude; a
    scalar multiplier costs nothing to test. Iterating re-computes the
    gradient hint on the already-edited mel and predicts again, which turns
    one forward pass into an amortised multi-step attack -- meaningful only
    if the network learned a direction rather than a fixed displacement.
    """

    def __init__(self, inner, hift_call, device, k, w, scale=1.0, iters=1,
                 names=None, space="prob", cap=4.0):
        super().__init__()
        self.inner = inner
        self.max_delta = inner.max_delta
        self.min_log_std = getattr(inner, "min_log_std", -3.0)
        self.log_std = inner.log_std
        self.hift_call, self.device, self.k, self.w = hift_call, device, k, w
        self.scale, self.iters = scale, iters
        # A leave-one-detector-out policy must not see the held-out detector
        # at inference either. The hint is computed from detector gradients,
        # so passing the full set here would quietly re-introduce it and make
        # the holdout meaningless.
        self.names = names
        # a model trained on margin-space targets needs a margin-space hint;
        # feeding it the probability hint would hand it ~zero signal on
        # exactly the saturated detectors it was trained to move
        self.space, self.cap = space, cap

    def sample(self, mel, deterministic=False):
        cur, total = mel, torch.zeros_like(mel)
        for _ in range(self.iters):
            g = diffscore.mel_gradient_hint(cur, self.hift_call, self.device,
                                            k=self.k, softmin_weight=self.w,
                                            names=self.names, space=self.space,
                                            cap=self.cap)
            d = self.scale * self.inner.max_delta * torch.tanh(self.inner(cur, g))
            total = total + d
            cur = mel + total
        return total, torch.zeros(mel.shape[0], device=mel.device), total

    def set_speaker(self, emb):
        """Voice-conditioned models need the speaker fixed before generating."""
        if getattr(self.inner, "spk_dim", 0):
            self.inner.set_speaker(emb)

    def to(self, *a, **kw):
        self.inner.to(*a, **kw)
        return self


def agg(d: np.ndarray):
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else 0.0), float("nan"), float("nan")
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return m, se, (m / se if se > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill-run", default="distill_distill1")
    ap.add_argument("--distill-ckpt", default="best.pt")
    ap.add_argument("--b-run", default="designB_B9h")
    ap.add_argument("--b-ckpt", default="best.pt")
    ap.add_argument("--texts", type=int, default=None)
    ap.add_argument("--ref-voices", type=int, default=4,
                    help="how many TRAIN_VOICES to include as a reference group")
    ap.add_argument("--softmin-k", type=float, default=8.0)
    ap.add_argument("--softmin-weight", type=float, default=0.5)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--edit-scale", type=float, default=1.0,
                    help="multiply the distilled edit by this before applying it")
    ap.add_argument("--edit-iters", type=int, default=1,
                    help="apply the distilled edit this many times, re-hinting each pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(det.MODEL_ORDER)
    texts = EVAL_TEXTS[:args.texts] if args.texts else EVAL_TEXTS
    held = list(EVAL_VOICES)
    ref_voices = TRAIN_VOICES[:args.ref_voices]
    voices = held + ref_voices

    arms = {}
    for tag, run, ck in (("designB", args.b_run, args.b_ckpt),
                         ("distilled", args.distill_run, args.distill_ckpt)):
        p = RL_RUNS / run / ck
        if not p.is_file():
            print(f"no checkpoint at {p}")
            return 1
        blob = torch.load(p, map_location="cpu", weights_only=False)
        arms[tag] = {"blob": blob, "run": run, "ckpt": ck,
                     "iteration": blob.get("iteration"),
                     "trained_on": sorted(blob.get("args", {}).get("voices", []))}
        print(f"{tag:<10} {run}/{ck}  iteration {blob.get('iteration')}")

    cosy = cloners.load_model("CosyVoice2", device=device)
    hift_call = diffscore.hift_forward_fn(cosy.model.hift)
    for tag in arms:
        net, kind, _ = build_policy(arms[tag]["blob"], device, args)
        arms[tag]["kind"] = kind
        if tag == "distilled" and (args.edit_scale != 1.0 or args.edit_iters != 1):
            ck_names = arms[tag]["blob"].get("args", {}).get("detectors")
            ck = arms[tag]["blob"].get("args", {})
            arms[tag]["adapter"] = AmplifiedAdapter(
                net, hift_call, device, args.softmin_k, args.softmin_weight,
                scale=args.edit_scale, iters=args.edit_iters, names=ck_names,
                space=ck.get("objective", "prob"), cap=ck.get("margin_cap", 4.0))
            print(f"{tag:<10} amplified: scale {args.edit_scale}, "
                  f"iters {args.edit_iters}"
                  + (f", hint restricted to {ck_names}" if ck_names else ""))
        else:
            arms[tag]["adapter"] = _EvalAdapter(net, hift_call, device,
                                                args.softmin_k, args.softmin_weight)
        print(f"{tag:<10} design {kind}")

    out_dir = RL_RUNS / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_arms = ["baseline"] + list(arms)

    print(f"\n{len(voices)} voices ({len(held)} held-out + {len(ref_voices)} reference) "
          f"x {len(texts)} unseen sentences x {len(all_arms)} arms "
          f"= {len(voices) * len(texts) * len(all_arms)} clips\n", flush=True)

    rows = []
    for vi, voice in enumerate(voices):
        ref, prompt = load_voice(voice)
        cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice, device=device)
        for ti, text in enumerate(texts):
            seed = args.seed + vi * 100 + ti
            row = {"voice": voice, "text_idx": ti, "text": text, "ref": str(ref),
                   "held_out": voice in held}
            for arm in all_arms:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                p = out_dir / f"{voice}_{ti}_{arm}.wav"
                if arm == "baseline":
                    cloners.generate(text, ref, model="CosyVoice2",
                                     out_path=p, device=device)
                    row[f"{arm}_delta_abs"] = 0.0
                else:
                    with mel_edit_hook(cosy.model, arms[arm]["adapter"],
                                       deterministic=True, device=device) as rec:
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=p, device=device)
                    row[f"{arm}_delta_abs"] = (
                        float(np.mean([d.abs().mean().item() for d in rec.deltas]))
                        if rec.deltas else 0.0)
                row[f"{arm}_path"] = p
            rows.append(row)
        print(f"  [{vi + 1}/{len(voices)}] {voice:<13} {len(texts)} sentences", flush=True)

    print("\nscoring...", flush=True)
    for ri, r in enumerate(rows):
        ref_emb = Tester.speaker_embedding(Path(r["ref"]))
        for arm in all_arms:
            p = Path(r[f"{arm}_path"])
            arr = det.score_file(p, device=device)
            s = {n: float(v) for n, v in zip(names, arr)}
            t = torch.tensor([s[n] for n in names], dtype=torch.float64)
            r[f"{arm}_det"] = s
            r[f"{arm}_softmin"] = float(diffscore.soft_min(list(t), k=args.softmin_k))
            r[f"{arm}_meandet"] = float(t.mean())
            r[f"{arm}_sim"] = float(np.dot(Tester.speaker_embedding(p), ref_emb))
            r[f"{arm}_wer"] = Tester.wer(
                r["text"], Tester.transcribe(p, args.whisper, device))[0]
            if not args.no_quality:
                q = quality.measure(p, Path(r["ref"]), device)
                r[f"{arm}_pesq"] = q["pesq"]
                r[f"{arm}_stoi"] = q["stoi"]
            p.unlink(missing_ok=True)
        if (ri + 1) % 5 == 0:
            print(f"  scored {ri + 1}/{len(rows)}", flush=True)

    keys = ["meandet", "softmin", "sim", "wer"] + ([] if args.no_quality
                                                   else ["pesq", "stoi"])

    def group(rs, label):
        if not rs:
            return None
        print(f"\n{label}  ({len({r['voice'] for r in rs})} voices, {len(rs)} clips)")
        print(f"  {'metric':<12}" + "".join(f"{a:>13}" for a in all_arms)
              + f"{'dB vs base':>13}{'t':>8}")
        rec = {"label": label, "n": len(rs)}
        for k in keys:
            vals = {a: float(np.mean([r[f"{a}_{k}"] for r in rs])) for a in all_arms}
            scale = 100.0 if k == "wer" else 1.0
            line = f"  {k:<12}" + "".join(f"{vals[a] * scale:>13.4f}" for a in all_arms)
            d = np.array([r["distilled_" + k] - r["baseline_" + k] for r in rs])
            m, se, t = agg(d)
            print(line + f"{m * scale:>+13.4f}{t:>8.2f}")
            rec[k] = vals
            rec[f"d_distilled_{k}"] = m
            rec[f"t_distilled_{k}"] = t
        for a in all_arms[1:]:
            rec[f"{a}_delta_abs"] = float(np.mean([r[f"{a}_delta_abs"] for r in rs]))
        return rec

    print("\n" + "=" * 92)
    print("the last two columns are the DISTILLED arm's paired change vs baseline")
    print("=" * 92)
    g_held = group([r for r in rows if r["held_out"]], "HELD-OUT voices, unseen sentences")
    g_ref = group([r for r in rows if not r["held_out"]],
                  "REFERENCE voices (seen in collection), unseen sentences")

    print("\n" + "=" * 92)
    print("PER-VOICE soft-min")
    print(f"{'voice':<14}{'set':<11}" + "".join(f"{a:>13}" for a in all_arms))
    print("-" * 92)
    per_voice = []
    for voice in voices:
        rs = [r for r in rows if r["voice"] == voice]
        rec = {"voice": voice, "held_out": rs[0]["held_out"]}
        for a in all_arms:
            for k in keys:
                rec[f"{a}_{k}"] = float(np.mean([r[f"{a}_{k}"] for r in rs]))
        per_voice.append(rec)
        print(f"{voice:<14}{'held-out' if rec['held_out'] else 'reference':<11}"
              + "".join(f"{rec[a + '_softmin']:>13.4f}" for a in all_arms))

    out = Path(args.out) if args.out else out_dir / "final_eval.json"
    out.write_text(json.dumps({
        "seed": args.seed, "texts": texts, "arms": all_arms,
        "checkpoints": {t: {k: arms[t][k] for k in
                            ("run", "ckpt", "iteration", "kind", "trained_on")}
                        for t in arms},
        "held_out_voices": held, "reference_voices": ref_voices,
        "detector_order": names,
        "groups": [g for g in (g_held, g_ref) if g],
        "per_voice": per_voice,
        "clips": [{k: v for k, v in r.items() if not k.endswith("_path")}
                  for r in rows],
    }, indent=2), encoding="utf8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
