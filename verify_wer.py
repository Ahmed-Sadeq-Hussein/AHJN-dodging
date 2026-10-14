# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
verify_wer.py -- check the jiwer-backed Tester.wer against the hand-written
Levenshtein pass it replaced.

    python verify_wer.py
    python verify_wer.py --cases 50000

`Tester.wer` used to carry its own Levenshtein DP. It now calls jiwer. The two
should agree exactly, and this script is the evidence: the old routine is kept
below, unchanged, and both are run over

  * every reference/transcript pair stored in the listening-test corpus,
  * tens of thousands of randomly generated pairs chosen to force deletions,
    insertions, substitutions and repeated words,
  * the degenerate cases (empty reference, empty hypothesis, identical input).

The rate and the total edit count must match everywhere. The split into
S / D / I need not: when several alignments share the same edit distance the
two implementations may break the tie differently, which changes the labels
without changing the rate. The script reports how often that happens rather
than failing on it, because the reported WER does not depend on it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Tester


def wer_levenshtein(reference: str, hypothesis: str):
    """The implementation Tester.wer used before jiwer. Kept verbatim."""
    ref, hyp = Tester.normalise_text(reference), Tester.normalise_text(hypothesis)
    n, m = len(ref), len(hyp)
    if n == 0:
        return (float(m > 0), 0, 0, m, 0)

    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])

    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            S += 1; i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return ((S + D + I) / n, S, D, I, n)


WORDS = ("the tide comes in twice and asks nobody first beyond heaven earth "
         "he alone was honored one to stand above all other men a an of it "
         "whatever happens tomorrow remember that").split()


def mutate(words, rng):
    """Damage a word list the way an ASR transcript is damaged."""
    out = list(words)
    for _ in range(rng.randint(0, 4)):
        if not out:
            break
        op = rng.choice(("sub", "del", "ins", "swap", "dup"))
        i = rng.randrange(len(out))
        if op == "sub":
            out[i] = rng.choice(WORDS)
        elif op == "del":
            out.pop(i)
        elif op == "ins":
            out.insert(i, rng.choice(WORDS))
        elif op == "dup":
            out.insert(i, out[i])
        elif op == "swap" and len(out) > 1:
            j = rng.randrange(len(out))
            out[i], out[j] = out[j], out[i]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--corpus", default="MOS TEST RL/mos_ratings.json")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    pairs: list[tuple[str, str, str]] = []

    # 1. the real transcripts
    corpus = Path(args.corpus)
    if corpus.is_file():
        d = json.loads(corpus.read_text(encoding="utf8"))
        texts = d.get("texts") or [d.get("text", "")]
        for c in d.get("clips", []):
            if c.get("transcript") and c.get("line"):
                pairs.append((texts[c["line"] - 1], c["transcript"], "corpus"))
        print(f"{len(pairs)} reference/transcript pairs from {corpus}")
    else:
        print(f"note: {corpus} not found, skipping the real transcripts")

    # 2. the degenerate cases
    edge = [("", "", "edge"), ("", "hello there", "edge"),
            ("hello there", "", "edge"), ("one", "one", "edge"),
            ("a a a a", "a a", "edge"), ("a b c", "c b a", "edge"),
            ("...", "!!!", "edge"), ("word", "word word word", "edge")]
    pairs.extend(edge)

    # 3. fuzz
    rng = random.Random(args.seed)
    for _ in range(args.cases):
        ref = [rng.choice(WORDS) for _ in range(rng.randint(1, 18))]
        pairs.append((" ".join(ref), " ".join(mutate(ref, rng)), "fuzz"))

    print(f"{len(pairs)} cases in total\n")

    bad_rate, bad_total, tie_split = [], [], 0
    for ref, hyp, kind in pairs:
        a = Tester.wer(ref, hyp)
        b = wer_levenshtein(ref, hyp)
        if abs(a[0] - b[0]) > 1e-12 or a[4] != b[4]:
            bad_rate.append((kind, ref, hyp, a, b))
        elif sum(a[1:4]) != sum(b[1:4]):
            bad_total.append((kind, ref, hyp, a, b))
        elif a[1:4] != b[1:4]:
            tie_split += 1

    print(f"rate and reference length disagree : {len(bad_rate)}")
    print(f"total edit count disagrees         : {len(bad_total)}")
    print(f"same total, different S/D/I split  : {tie_split}"
          f"  ({tie_split / len(pairs) * 100:.2f}% -- alignment ties)")

    for label, rows in (("RATE", bad_rate), ("TOTAL", bad_total)):
        for kind, ref, hyp, a, b in rows[:5]:
            print(f"\n  {label} [{kind}]\n    ref {ref!r}\n    hyp {hyp!r}"
                  f"\n    jiwer {a}\n    old   {b}")

    ok = not bad_rate and not bad_total
    print("\n" + ("PASS -- jiwer reproduces the previous WER exactly"
                  if ok else "FAIL -- the two implementations differ"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
