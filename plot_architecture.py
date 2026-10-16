# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
plot_architecture.py -- figure 21: how the pieces of this project fit together.

    python plot_architecture.py

Writes graphs/21_architecture.svg and .png. SVG is the one to put in the
report: it stays sharp at any size and prints cleanly.

The diagram is generated rather than drawn by hand so it can be corrected
when the code moves. Everything in it was read off the source: the mel is the
intervention point because mel_edit_hook patches model.hift.inference; the
reward is soft-min plus two Lagrangian constraints because that is what
reward.compute_reward builds; and the detectors appear twice, in-process and
behind a subprocess, because both paths genuinely exist.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_config import GRAPHS_DIR

W, H = 1480, 1020

# one palette, used by role rather than by position
INK = "#1b1f27"
INK_2 = "#5a6070"
LINE = "#9aa1ad"
CLONE_BG, CLONE_EDGE = "#eef2fb", "#4a5fc1"      # the voiceclone environment
FAKE_BG, FAKE_EDGE = "#fdf1ec", "#c2413b"        # the voicefake environment
MEL_BG, MEL_EDGE = "#fff6d8", "#b9881f"          # the intervention point
POLICY_BG, POLICY_EDGE = "#e8f5f2", "#2a9d8f"    # anything learned
OUT_BG, OUT_EDGE = "#f2f3f5", "#8a8f98"          # results and reports

parts: list[str] = []


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def box(x, y, w, h, title, lines=(), *, fill="#ffffff", edge=LINE,
        title_size=15, mono=False, rx=9):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    family = "ui-monospace, Consolas, monospace" if mono else "Inter, Segoe UI, sans-serif"
    parts.append(
        f'<text x="{x + w / 2}" y="{y + 23}" text-anchor="middle" '
        f'font-family="{family}" font-size="{title_size}" font-weight="600" '
        f'fill="{INK}">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        parts.append(
            f'<text x="{x + w / 2}" y="{y + 44 + i * 17}" text-anchor="middle" '
            f'font-family="Inter, Segoe UI, sans-serif" font-size="12.5" '
            f'fill="{INK_2}">{esc(ln)}</text>')


def band(x, y, w, h, label, fill, edge):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" '
        f'stroke="{edge}" stroke-width="1.4" stroke-dasharray="7 5" opacity="0.85"/>')
    parts.append(
        f'<text x="{x + 16}" y="{y + 24}" font-family="Inter, Segoe UI, sans-serif" '
        f'font-size="13" font-weight="700" fill="{edge}" '
        f'letter-spacing="0.6">{esc(label)}</text>')


def arrow(x1, y1, x2, y2, *, label="", colour=LINE, dashed=False, width=2.0,
          label_dx=0, label_dy=-7, anchor="middle"):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{colour}" '
        f'stroke-width="{width}"{dash} marker-end="url(#head)"/>')
    if label:
        parts.append(
            f'<text x="{(x1 + x2) / 2 + label_dx}" y="{(y1 + y2) / 2 + label_dy}" '
            f'text-anchor="{anchor}" font-family="Inter, Segoe UI, sans-serif" '
            f'font-size="12" fill="{colour}">{esc(label)}</text>')


def elbow(points, *, label="", colour=LINE, dashed=False, width=2.0,
          label_at=0, label_dx=0, label_dy=-8, anchor="middle"):
    """A right-angled path through the given points, arrow on the last leg."""
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"'
        f'{dash} marker-end="url(#head)" stroke-linejoin="round"/>')
    if label:
        (x1, y1), (x2, y2) = points[label_at], points[label_at + 1]
        parts.append(
            f'<text x="{(x1 + x2) / 2 + label_dx}" y="{(y1 + y2) / 2 + label_dy}" '
            f'text-anchor="{anchor}" font-family="Inter, Segoe UI, sans-serif" '
            f'font-size="12" fill="{colour}">{esc(label)}</text>')


def build() -> str:
    parts.clear()
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="Inter, Segoe UI, sans-serif">')
    parts.append(
        '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/></marker></defs>')
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    parts.append(
        f'<text x="40" y="46" font-size="25" font-weight="700" fill="{INK}">'
        'AHJN architecture: where the edit happens and what judges it</text>')
    parts.append(
        f'<text x="40" y="70" font-size="14" fill="{INK_2}">'
        'A clone is made, one tensor inside it is edited, and four detectors '
        'plus three quality measures decide whether that was worth it.</text>')

    # ---------------- environments ---------------------------------------
    band(28, 92, 1000, 712, "VOICECLONE ENVIRONMENT   torch 2.3.1+cu121 / numpy 1.26.4",
         CLONE_BG, CLONE_EDGE)
    band(1052, 92, 400, 300, "VOICEFAKE ENVIRONMENT   torch 2.6 / numpy 2.2",
         FAKE_BG, FAKE_EDGE)

    # ---------------- inputs ---------------------------------------------
    box(52, 128, 210, 72, "reference.wav + prompt",
        ["VOICE_DATA_ROOT, 15 voices"], fill="#ffffff")
    box(52, 214, 210, 72, "target text",
        ["textbank.py, train / eval split"], fill="#ffffff")

    # ---------------- CosyVoice2 chain -----------------------------------
    box(300, 140, 700, 44, "CosyVoice2   (cloners.py)", [], fill="#ffffff",
        edge=CLONE_EDGE)
    chain = [
        (312, "LLM", "text to tokens"),
        (452, "speech tokens", "discrete"),
        (592, "flow decoder", "flow matching"),
        (732, "mel  80 x T", "THE TENSOR"),
        (872, "HiFT vocoder", "to waveform"),
    ]
    for x, title, sub in chain:
        is_mel = title.startswith("mel")
        box(x, 196, 116, 62, title, [sub],
            fill=MEL_BG if is_mel else "#ffffff",
            edge=MEL_EDGE if is_mel else LINE,
            title_size=13.5)
    for x in (428, 568, 708, 848):
        arrow(x, 227, x + 22, 227)
    arrow(264, 168, 306, 212)
    arrow(264, 248, 306, 240)
    box(872, 288, 116, 44, "wav 24 kHz", [], title_size=13)
    arrow(930, 258, 930, 286)

    # ---------------- the edit -------------------------------------------
    box(560, 330, 360, 104, "mel_edit_hook   (mel_policy.py)",
        ["patches model.hift.inference for a with-block",
         "mel <- mel + delta,  delta = max_delta * tanh(raw)",
         "records the edit, its log-prob and the pre-edit mel"],
        fill=POLICY_BG, edge=POLICY_EDGE)
    elbow([(790, 258), (790, 326)], label="the only intervention point",
          colour=MEL_EDGE, label_at=0, label_dx=-12, label_dy=4, anchor="end")
    elbow([(920, 356), (1012, 356), (1012, 227), (990, 227)],
          colour=POLICY_EDGE, dashed=True, label="edited mel", label_at=0,
          label_dx=0, label_dy=-8)

    # ---------------- where a policy comes from ---------------------------
    box(52, 330, 230, 104, "MelResidualPolicy",
        ["+ MelValueNet critic", "5-frame receptive field"],
        fill=POLICY_BG, edge=POLICY_EDGE)
    arrow(282, 382, 556, 382, colour=POLICY_EDGE)

    box(52, 470, 230, 96, "Design A   PPO-Lagrangian",
        ["train_ppo_mel.py", "clipped surrogate, dual ascent"],
        fill="#ffffff", edge=POLICY_EDGE)
    box(300, 470, 230, 96, "Design B   SAC + hint",
        ["train_sac_mel.py", "twin critics, gradient channels"],
        fill="#ffffff", edge=POLICY_EDGE)
    box(548, 470, 230, 96, "PGD teacher",
        ["collect_pgd.py, diffscore.py", "per-clip, expensive, strong"],
        fill="#ffffff", edge=MEL_EDGE)
    box(796, 470, 204, 96, "distilled student",
        ["distill.py", "one pass, imitates PGD"],
        fill="#ffffff", edge=MEL_EDGE)
    arrow(782, 518, 792, 518, colour=MEL_EDGE)
    parts.append(
        f'<text x="787" y="459" text-anchor="middle" font-size="11.5" '
        f'fill="{MEL_EDGE}">regress onto its edits</text>')
    arrow(167, 466, 167, 440, colour=POLICY_EDGE)
    elbow([(415, 466), (415, 452), (242, 452), (242, 440)],
          colour=POLICY_EDGE)
    arrow(880, 466, 880, 440, colour=MEL_EDGE)

    # ---------------- judging --------------------------------------------
    box(1072, 128, 360, 150, "detectors.py   four countermeasures",
        ["AASIST       AASIST-L",
         "RawNet2      RawGAT-ST (mul)",
         "each returns P(human), 1 = human",
         "soft-min over the four is the objective"],
        fill="#ffffff", edge=FAKE_EDGE)
    box(1072, 296, 360, 76, "cross-environment bridge",
        ["reward.py shells out to VOICEFAKE_PYTHON",
         "detectors.py --json, whole batch in one call"],
        fill=FAKE_BG, edge=FAKE_EDGE)

    elbow([(988, 310), (1032, 310), (1032, 203), (1068, 203)],
          colour=FAKE_EDGE, label="generated wav", label_at=1, label_dx=-8,
          label_dy=0, anchor="end")

    box(300, 636, 330, 104, "constraints   (Tester.py)",
        ["CAM++ 192-d embedding, cosine >= 0.70",
         "Whisper 'base' + jiwer, WER <= 0.10",
         "the same embedding CosyVoice2 conditions on"],
        fill="#ffffff", edge=LINE)
    box(660, 636, 340, 104, "audio quality   (quality.py)",
        ["TorchAudio-SQUIM, reference-free",
         "PESQ, STOI, SI-SDR, estimated MOS",
         "reports the cost; never optimised against"],
        fill="#ffffff", edge=LINE)

    # ---------------- reward ---------------------------------------------
    box(52, 636, 220, 104, "reward.py",
        ["soft-min  +  two",
         "Lagrange multipliers,",
         "hard floor on collapse"],
        fill=POLICY_BG, edge=POLICY_EDGE)
    arrow(296, 688, 278, 688, colour=POLICY_EDGE)
    elbow([(1252, 372), (1252, 592), (162, 592), (162, 632)],
          colour=FAKE_EDGE, dashed=True, label="detector scores", label_at=1,
          label_dx=-330, label_dy=-9)
    elbow([(52, 688), (30, 688), (30, 518), (48, 518)],
          colour=POLICY_EDGE, label="reward", label_at=1, label_dx=10,
          label_dy=-6, anchor="start")

    # the gradient path, which skips the reward entirely
    parts.append(
        f'<text x="340" y="614" font-size="12" fill="{MEL_EDGE}">'
        'diffscore.py backpropagates through HiFT and the detectors, '
        'so the teacher never uses the reward at all</text>')

    # ---------------- outputs --------------------------------------------
    band(28, 832, 1424, 150, "WHAT COMES OUT", "#fafafa", OUT_EDGE)

    box(52, 868, 300, 92, "ahjn.py   the command line",
        ["check / detect / score / rate",
         "train / status / speak / figures / verify"],
        fill="#ffffff", edge=OUT_EDGE)
    box(372, 868, 250, 92, "scorecard.py",
        ["every metric for a clip,", "one call, never raises"],
        fill=OUT_BG, edge=OUT_EDGE)
    box(642, 868, 270, 92, "listening test",
        ["rl/mos_corpus.py + the rating page",
         "MOS and PMOS from a person"],
        fill=OUT_BG, edge=OUT_EDGE)
    box(932, 868, 230, 92, "figures",
        ["plot_*.py -> graphs/", "21 of them"],
        fill=OUT_BG, edge=OUT_EDGE)
    box(1182, 868, 250, 92, "the write-up",
        ["Plan_X/RESULTS.md, SOURCES.md", "rl/ANALYSIS.md, MATH.md"],
        fill=OUT_BG, edge=OUT_EDGE)

    parts.append(
        f'<text x="40" y="1004" font-size="12" fill="{INK_2}">'
        'Solid arrows carry audio or tensors. Dashed arrows carry scores. '
        'The yellow path is the white-box attack, which has gradients; '
        'the green path is reinforcement learning, which has only a scalar.</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = GRAPHS_DIR / "21_architecture.svg"
    svg_path.write_text(build(), encoding="utf8")
    print(f"wrote {svg_path}")

    png_path = GRAPHS_DIR / "21_architecture.png"
    try:
        import cairosvg
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=1.5)
        print(f"wrote {png_path}")
    except Exception:
        # no cairosvg in either env; a browser or Inkscape will convert it,
        # and the SVG is the better thing to put in a document anyway
        print(f"(no PNG: install cairosvg, or open the SVG and export) "
              f"-> {png_path} not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
