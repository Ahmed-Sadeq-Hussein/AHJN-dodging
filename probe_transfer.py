# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
probe_transfer.py -- does the edit fool detectors it never saw? RUN IN "voiceclone".

    python probe_transfer.py                  # generate + score
    python probe_transfer.py --rescore        # score the kept clips again,
                                              # e.g. after adding a detector

Why this matters more than any further training
-----------------------------------------------
Every number in Plan_X was measured on the same four checkpoints the edit was
optimised against with full gradient access. Leave-one-detector-out (section
7d) showed transfer to a held-out detector is real but small: +0.07 against
+0.15 to +0.29 on the optimised ones. If the effect does not survive contact
with detectors outside the four, a stronger teacher for those same four buys
nothing outside this experiment.

Detectors scored here that influence NOTHING upstream -- not the PGD targets,
not the gradient hint, not checkpoint selection:

    RawGAT-ST-add   same architecture family as the attacked RawGAT-ST (mul),
    RawGAT-ST-cat   same training data, different fusion and separately trained
                    weights. A NEAR transfer test: success here is expected
                    and would not by itself show generality.
    LFCC-LCNN       (only if its official ASVspoof 2021 weights are present)
                    hand-crafted LFCC front end + light CNN. A genuinely
                    different architecture from all four attacked models --
                    the FAR transfer test.

Clips are kept on disk with a manifest, so adding a detector later needs only
--rescore, not regeneration.
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
import Tester
from env_config import GENERATORS_DIR, RESULTS_ROOT
from eval_final import AmplifiedAdapter
from eval_generalization import build_policy
from mel_policy import mel_edit_hook
from textbank import EVAL_TEXTS
from train_ppo_mel import load_voice

RL_RUNS = RESULTS_ROOT / "rl_runs"
WORK = GENERATORS_DIR / "transfer_probe"
ATTACKED = list(det.MODEL_ORDER)


# --------------------------------------------------------------------------
# detectors outside the attacked four
# --------------------------------------------------------------------------

def _unseen_specs() -> dict[str, "det._Spec"]:
    specs = {
        "RawGAT-ST-add": det._Spec("RawGAT-ST-add",
                                   lambda d: det._build_rawgat(d, "add"),
                                   activation="logits",
                                   forward_kwargs={"Freq_aug": False}),
        "RawGAT-ST-cat": det._Spec("RawGAT-ST-cat",
                                   lambda d: det._build_rawgat(d, "cat"),
                                   activation="logits",
                                   forward_kwargs={"Freq_aug": False}),
    }
    try:
        import lcnn_detector
        if lcnn_detector.available():
            specs["LFCC-LCNN"] = lcnn_detector.spec()
    except ImportError:
        pass
    return specs


_UNSEEN_CACHE: dict[str, torch.nn.Module] = {}


def score_unseen(path: Path, specs, device) -> dict[str, float]:
    """Same windowing and mean aggregation as detectors.score_file."""
    wav = det.load_audio(path)
    wins = torch.from_numpy(det._windows(wav, det.NB_SAMP, None)).float()
    out = {}
    for name, spec in specs.items():
        if name not in _UNSEEN_CACHE:
            net = spec.build(torch.device(device))
            _UNSEEN_CACHE[name] = net.to(device).eval()
        net = _UNSEEN_CACHE[name]
        probs = []
        with torch.no_grad():
            for s in range(0, wins.shape[0], 8):
                raw = spec.unpack(net(wins[s:s + 8].to(device), **spec.forward_kwargs))
                p = torch.softmax(raw, dim=1) if spec.activation == "logits" else torch.exp(raw)
                probs.append(p[:, 1].cpu().numpy())
        out[name] = float(np.mean(np.concatenate(probs)))
    return out


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(d.mean() / se) if se and se > 0 else float("nan"), \
        float((d > 0).mean())


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="+",
                    default=["Tywin_2", "evelynn", "thresh", "mordekaiser", "riven", "Ego"])
    ap.add_argument("--texts", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--run", default="distill_final")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--rescore", action="store_true",
                    help="skip generation; score the clips in the manifest")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=2718)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    manifest_p = WORK / "manifest.json"
    specs = _unseen_specs()
    print(f"attacked (white-box): {ATTACKED}")
    print(f"never seen          : {list(specs)}\n", flush=True)

    if args.rescore:
        manifest = json.loads(manifest_p.read_text(encoding="utf8"))
    else:
        cosy = cloners.load_model("CosyVoice2", device=device)
        blob = torch.load(RL_RUNS / args.run / args.checkpoint,
                          map_location="cpu", weights_only=False)
        net, kind, _ = build_policy(blob, device, args)
        net.eval()
        adapter = AmplifiedAdapter(net, diffscore.hift_forward_fn(cosy.model.hift),
                                   device, 8.0, 0.5, scale=1.0, iters=args.iters,
                                   names=blob.get("args", {}).get("detectors"))
        manifest = {"run": args.run, "checkpoint": args.checkpoint,
                    "iters": args.iters, "clips": []}
        texts = EVAL_TEXTS[:args.texts]
        for vi, voice in enumerate(args.voices):
            ref, prompt = load_voice(voice)
            cloners.enroll(ref, "CosyVoice2", prompt_text=prompt, alias=voice,
                           device=device)
            for ti, text in enumerate(texts):
                seed = args.seed + vi * 100 + ti
                row = {"voice": voice, "text_idx": ti, "text": text, "ref": str(ref)}
                for arm in ("base", "edit"):
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    p = WORK / f"{voice}_{ti}_{arm}.wav"
                    if arm == "base":
                        cloners.generate(text, ref, model="CosyVoice2",
                                         out_path=p, device=device)
                    else:
                        with mel_edit_hook(cosy.model, adapter, deterministic=True,
                                           device=device):
                            cloners.generate(text, ref, model="CosyVoice2",
                                             out_path=p, device=device)
                    row[f"{arm}_path"] = str(p)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                manifest["clips"].append(row)
            print(f"  [{vi + 1}/{len(args.voices)}] {voice} generated", flush=True)
        manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    print("\nscoring...", flush=True)
    for i, row in enumerate(manifest["clips"]):
        emb = Tester.speaker_embedding(Path(row["ref"]))
        for arm in ("base", "edit"):
            p = Path(row[f"{arm}_path"])
            arr = det.score_file(p, device=device)
            row[f"{arm}_det"] = {n: float(v) for n, v in zip(ATTACKED, arr)}
            row[f"{arm}_det"].update(score_unseen(p, specs, device))
            row[f"{arm}_sim"] = float(np.dot(Tester.speaker_embedding(p), emb))
        if (i + 1) % 6 == 0:
            print(f"  scored {i + 1}/{len(manifest['clips'])}", flush=True)
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf8")

    clips = manifest["clips"]
    names = ATTACKED + list(specs)
    print("\n" + "=" * 88)
    print(f"{len(clips)} paired clips, {len({c['voice'] for c in clips})} voices, "
          f"edit = {manifest['run']} x{manifest['iters']}")
    print("=" * 88)
    print(f"{'detector':<16}{'role':<13}{'baseline':>10}{'edited':>10}{'gain':>9}"
          f"{'t':>8}{'win rate':>10}")
    print("-" * 88)
    summary = []
    for n in names:
        b = [c["base_det"][n] for c in clips]
        e = [c["edit_det"][n] for c in clips]
        g, t, w = paired(e, b)
        role = "attacked" if n in ATTACKED else "NEVER SEEN"
        summary.append({"detector": n, "role": role, "baseline": float(np.mean(b)),
                        "edited": float(np.mean(e)), "gain": g, "t": t, "win_rate": w})
        print(f"{n:<16}{role:<13}{np.mean(b):>10.4f}{np.mean(e):>10.4f}{g:>+9.4f}"
              f"{t:>8.2f}{w * 100:>9.0f}%")
    ds, ts, _ = paired([c["edit_sim"] for c in clips], [c["base_sim"] for c in clips])
    print("-" * 88)
    print(f"{'similarity':<29}{np.mean([c['base_sim'] for c in clips]):>10.4f}"
          f"{np.mean([c['edit_sim'] for c in clips]):>10.4f}{ds:>+9.4f}{ts:>8.2f}")

    seen = [s["gain"] for s in summary if s["role"] == "attacked"]
    unseen = [s for s in summary if s["role"] != "attacked"]
    if unseen:
        print(f"\nmean gain on attacked detectors: {np.mean(seen):+.4f}")
        for s in unseen:
            ratio = s["gain"] / np.mean(seen) if np.mean(seen) else float("nan")
            print(f"  {s['detector']:<14} gain {s['gain']:+.4f}  "
                  f"= {ratio * 100:.0f}% of the attacked mean  (t = {s['t']:+.2f})")

    (RL_RUNS / "transfer_probe.json").write_text(
        json.dumps({"summary": summary, "similarity_gain": ds,
                    "n_clips": len(clips), "run": manifest["run"],
                    "iters": manifest["iters"]}, indent=2), encoding="utf8")
    print(f"\nwrote {RL_RUNS / 'transfer_probe.json'}  (clips kept in {WORK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
