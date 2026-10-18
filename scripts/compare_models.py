# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
Compare the unedited clone against the trained policies, on the same lines.

    python scripts/compare_models.py                       # quick, 3 voices x 1 line
    python scripts/compare_models.py --voices all --texts 2
    python scripts/compare_models.py --arms baseline b --voices riven jhin
    python scripts/compare_models.py --from-json out/comparison.json

Every arm of an item is rendered under ONE seed. The policies are
deterministic and consume no RNG, so the arms replay the same speech tokens
and the same vocoder noise and the mel edit is the only difference between
them. That is what makes the comparison paired, and paired is what this
needs: per-clip variance across voices is far larger than any effect being
looked for, with baseline soft-min spanning roughly 0.24 to 0.96 in this
project.

Reported per arm: the soft-min over the four detectors (1 = human to all of
them), speaker similarity, WER, and SQUIM PESQ. Each edited arm is then
tested against the unedited one with a paired t-test and a Wilcoxon signed
rank test, so the conclusion does not rest on treating the measures as
normal.

Sentences come from textbank's EVAL split, which no policy was trained on.

Runs under the voiceclone environment. Rendering is the slow part, about
three clips per item; --from-json re-reports a previous run instantly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rl import common

ARMS = {
    "baseline": (None, "unedited"),
    "a": ("designA_budget4h", "Design A"),
    "b": ("designB_B9h", "Design B"),
}
METRICS = [("softmin", "soft-min", 1, True),
           ("sim", "similarity", 1, True),
           ("wer", "WER %", 100, False),
           ("pesq", "PESQ", 1, True)]


def eval_texts(n: int) -> list[str]:
    """Sentences no policy has been trained on."""
    try:
        import textbank
        pool = list(getattr(textbank, "EVAL_TEXTS", []))
    except Exception:
        pool = []
    if not pool:
        pool = ["Whatever happens tomorrow, remember that the tide comes in "
                "twice and asks nobody first."]
    return pool[:n]


def render_and_score(voices, texts, arms, seed0, device, out_dir):
    import cloners
    import torch
    from scorecard import score_clip

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cosy = cloners.load_model("CosyVoice2", device=device)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(voices) * len(texts)
    for vi, voice in enumerate(voices):
        ref, _ = common.load_voice(voice)
        for ti, text in enumerate(texts):
            # one seed per (voice, line): the arms of an item must match
            seed = seed0 + vi * 10 + ti
            item = f"{voice} line {ti + 1}"
            line = f"  [{vi * len(texts) + ti + 1}/{total}] {item:<24}"
            for arm in arms:
                run_id, label = ARMS[arm]
                path = out_dir / f"{voice}_L{ti + 1}_{arm}.wav"
                common.generate(run_id, voice, text, path, device=device,
                                seed=seed, cosy=cosy, no_edit=run_id is None)
                s = score_clip(path, reference=ref, text=text, device=device)
                rows.append({"item": item, "voice": voice, "line": ti + 1,
                             "arm": arm, "file": path.name,
                             **{k: s[k] for k in s if k != "problems"}})
                line += f"  {arm} {s.get('softmin', float('nan')):.3f}"
            print(line, flush=True)
    return rows


def report(rows, arms):
    from scipy import stats

    items = sorted({r["item"] for r in rows})
    by = {(r["arm"], r["item"]): r for r in rows}
    complete = [i for i in items if all((a, i) in by for a in arms)]
    if len(complete) < len(items):
        print(f"\nnote: {len(items) - len(complete)} item(s) missing an arm, "
              f"excluded from the paired tests")
    if not complete:
        print("nothing to compare")
        return

    def col(arm, key):
        return np.array([by[(arm, i)].get(key, np.nan) for i in complete],
                        dtype=float)

    print("\n" + "=" * 70)
    print(f"{len(complete)} items, {len(arms)} arms, paired by voice and line")
    print("=" * 70)
    head = f"{'arm':<12}" + "".join(f"{lbl:>13}" for _, lbl, _, _ in METRICS)
    print(head)
    print("-" * len(head))
    for arm in arms:
        out = f"{ARMS[arm][1]:<12}"
        for key, _, scale, _ in METRICS:
            out += f"{np.nanmean(col(arm, key)) * scale:>13.3f}"
        print(out)

    if "baseline" not in arms:
        print("\n(no baseline arm selected, so no paired tests)")
        return

    for arm in [a for a in arms if a != "baseline"]:
        print(f"\n{ARMS[arm][1]} against the unedited clone")
        print(f"  {'metric':<12}{'change':>10}{'t':>9}{'p':>11}"
              f"{'wilcoxon':>11}{'better':>9}")
        for key, label, scale, higher_better in METRICS:
            b, x = col("baseline", key), col(arm, key)
            ok = ~(np.isnan(b) | np.isnan(x))
            if ok.sum() < 2:
                continue
            b, x = b[ok], x[ok]
            d = x - b
            wins = int((d > 0).sum() if higher_better else (d < 0).sum())
            # with one item, or with every difference identical, a t-test is
            # undefined and reports +/-inf at p = 0. Say so rather than print
            # a number someone might quote.
            if len(d) < 2 or d.std(ddof=1) == 0:
                print(f"  {label:<12}{d.mean() * scale:>+10.3f}"
                      f"{'n/a':>9}{'n/a':>11}{'n/a':>11}{wins:>6}/{len(d)}")
                continue
            t, p = stats.ttest_rel(x, b)
            try:
                w = stats.wilcoxon(x, b).pvalue
            except ValueError:
                w = float("nan")
            print(f"  {label:<12}{d.mean() * scale:>+10.3f}{t:>9.2f}"
                  f"{p:>11.2e}{w:>11.2e}{wins:>6}/{len(d)}")
        print("  (better counts items that moved the right way for that metric)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired comparison of the unedited clone and the policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=["baseline", "a", "b"],
                    choices=list(ARMS), help="which arms to render")
    ap.add_argument("--voices", nargs="*", default=["riven", "jhin", "swain"],
                    help='voice names, or "all" for every voice')
    ap.add_argument("--texts", type=int, default=1,
                    help="how many held-out sentences per voice")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="out/comparison",
                    help="where the rendered wavs and the json go")
    ap.add_argument("--from-json", default=None,
                    help="skip rendering and re-report a previous run")
    args = ap.parse_args()

    if args.from_json:
        blob = json.loads(Path(args.from_json).read_text(encoding="utf8"))
        print(f"re-reporting {args.from_json}")
        report(blob["rows"], blob.get("arms", args.arms))
        return 0

    if args.voices == ["all"]:
        from eval_generalization import all_voices
        voices = all_voices()
    else:
        voices = args.voices

    texts = eval_texts(args.texts)
    out_dir = Path(args.out_dir)

    print(f"{len(voices)} voices x {len(texts)} lines x {len(args.arms)} arms "
          f"= {len(voices) * len(texts) * len(args.arms)} clips")
    for i, t in enumerate(texts, 1):
        print(f'  line {i}: "{t}"')
    print()

    rows = render_and_score(voices, texts, args.arms, args.seed, args.device,
                            out_dir)

    out_json = out_dir / "comparison.json"
    out_json.write_text(json.dumps(
        {"arms": args.arms, "voices": voices, "texts": texts,
         "seed": args.seed, "rows": rows}, indent=1, default=float),
        encoding="utf8")

    report(rows, args.arms)
    print(f"\nwrote {out_json}")
    print(f"re-report it without re-rendering: "
          f"python scripts/compare_models.py --from-json {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
