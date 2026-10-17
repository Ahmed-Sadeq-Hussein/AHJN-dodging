# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
ahjn.py -- one command line for the whole project.

    python ahjn.py --help
    python ahjn.py check
    python ahjn.py detect "MOS TEST RL/design_a/Ego_L1.wav"
    python ahjn.py score  "MOS TEST RL/design_a/Ego_L1.wav" --reference ...
    python ahjn.py rate serve
    python ahjn.py rate results --split
    python ahjn.py train b --iters 600 --resume
    python ahjn.py speak --design b --voice riven --text "hello there"

The project grew one script per experiment, which is right for experiments and
wrong for using the thing afterwards. This gathers the parts anyone actually
re-runs behind verbs, and leaves the probe_*/eval_* scripts alone.

TWO ENVIRONMENTS. The detectors import only under "voicefake" (torch 2.6,
numpy 2.2) and CosyVoice2 only under "voiceclone" (torch 2.3.1, numpy 1.26).
Commands that need the other side shell out to VOICEFAKE_PYTHON from .env
rather than asking you to remember which interpreter you are in. `check` tells
you where you stand.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CORPUS = HERE / "MOS TEST RL"
RATINGS = CORPUS / "mos_ratings.json"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _say(msg: str = "") -> None:
    print(msg, flush=True)


def _rule(title: str) -> None:
    _say("\n" + title)
    _say("-" * max(len(title), 20))


def _voicefake_python() -> str | None:
    """The interpreter that can import the detectors, or None if unset."""
    from env_config import VOICEFAKE_PYTHON
    return VOICEFAKE_PYTHON


def _run_there(python: str, args: list[str]) -> int:
    """Run a project script under another interpreter, inheriting stdio."""
    return subprocess.call([python, *args], cwd=str(HERE))


def _have_detectors() -> bool:
    try:
        import detectors  # noqa: F401
        import torch      # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def cmd_check(args) -> int:
    """Say what this machine can and cannot do, before anything slow runs."""
    _rule("environment")
    _say(f"interpreter        {sys.executable}")
    try:
        import torch
        _say(f"torch              {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as exc:
        _say(f"torch              not importable here ({exc})")
    try:
        import numpy
        _say(f"numpy              {numpy.__version__}")
    except Exception:
        pass

    vf = _voicefake_python()
    _say(f"VOICEFAKE_PYTHON   {vf or 'not set in .env'}")
    if vf and not Path(vf).is_file():
        _say("                   ^ set, but that file does not exist")

    _rule("paths")
    import env_config as cfg
    for name in ("VOICE_DETECTION_ROOT", "VOICE_CLONING_ROOT", "VOICE_DATA_ROOT",
                 "RESULTS_ROOT", "GRAPHS_DIR"):
        p = getattr(cfg, name, None)
        mark = "ok " if p and Path(p).exists() else "MISSING"
        _say(f"{mark:<8} {name:<22} {p}")

    _rule("detectors")
    if _have_detectors():
        import detectors
        missing = detectors.check_install(verbose=True)
        if any(missing.values()):
            return 1
    elif vf:
        _say("not importable in this interpreter; asking the voicefake one")
        return _run_there(vf, ["detectors.py", "--check"])
    else:
        _say("not importable here, and VOICEFAKE_PYTHON is not set")
        return 1

    _rule("listening test")
    _say(f"corpus     {CORPUS}  {'present' if CORPUS.is_dir() else 'MISSING'}")
    _say(f"ratings    {RATINGS}  "
         f"{'present' if RATINGS.is_file() else 'not exported yet'}")
    return 0


# --------------------------------------------------------------------------
# detect / score
# --------------------------------------------------------------------------

def cmd_detect(args) -> int:
    """Detector scores only. 1 = human to that detector."""
    if not _have_detectors():
        vf = _voicefake_python()
        if not vf:
            _say("the detectors do not import here and VOICEFAKE_PYTHON is unset")
            return 1
        argv = ["detectors.py", *args.files]
        if args.models:
            argv += ["--models", *args.models]
        if args.json:
            argv.append("--json")
        return _run_there(vf, argv)

    import detectors
    results = {f: detectors.score_file_dict(f, args.models or None) for f in args.files}
    if args.json:
        print(json.dumps(results, indent=1))
        return 0
    for path, row in results.items():
        _say(f"\n{path}")
        for name, value in row.items():
            bar = "#" * int(round(value * 30))
            verdict = "human" if value >= 0.5 else "AI"
            _say(f"  {name:<10} {value:.4f}  {bar:<30} {verdict}")
        _say(f"  {'soft-min':<10} {min(row.values()):.4f}   (the worst of them)")
    return 0


def cmd_score(args) -> int:
    """Every metric for a clip: detectors, similarity, WER and quality."""
    from scorecard import score_clip
    rows = []
    for f in args.files:
        row = score_clip(f, reference=args.reference, text=args.text,
                         device=args.device, quality=not args.no_quality)
        rows.append((f, row))

    if args.json:
        print(json.dumps({f: r for f, r in rows}, indent=1, default=float))
        return 0

    for f, r in rows:
        _say(f"\n{f}")
        if r.get("det"):
            for name, value in r["det"].items():
                _say(f"  {name:<10} {value:.4f}")
        for key, label, scale in (("softmin", "soft-min", 1), ("sim", "similarity", 1),
                                  ("wer", "WER", 100), ("pesq", "PESQ", 1),
                                  ("stoi", "STOI", 1), ("mos", "SQUIM MOS", 1)):
            if key in r:
                unit = "%" if scale == 100 else ""
                _say(f"  {label:<10} {r[key] * scale:.3f}{unit}")
        for problem in r.get("problems", []):
            _say(f"  note: {problem}")
    return 0


# --------------------------------------------------------------------------
# rate -- the listening test
# --------------------------------------------------------------------------

def cmd_rate(args) -> int:
    if args.what == "build":
        argv = [sys.executable, "-m", "rl.mos_corpus", "--out-dir", str(CORPUS)]
        if args.voices:
            argv += ["--voices", *args.voices]
        return subprocess.call(argv, cwd=str(HERE))

    if args.what == "page":
        return subprocess.call(
            [sys.executable, "mos_build_app.py", "--dir", str(CORPUS)], cwd=str(HERE))

    if args.what == "serve":
        # Opened as a file:// page the rating app has no database and no
        # download, so ratings go to browser storage and Export used to go
        # nowhere. Served over http both work, and the audio resolves.
        import http.server
        import socketserver
        import webbrowser
        page = CORPUS / "listening_test.html"
        if not page.is_file():
            _say(f"no {page}; run `ahjn.py rate page` first")
            return 1
        os.chdir(CORPUS)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as srv:
            url = f"http://127.0.0.1:{args.port}/listening_test.html"
            _say(f"serving {CORPUS}")
            _say(f"open     {url}")
            _say("Export writes mos_ratings.json to your downloads. Ctrl-C to stop.")
            if not args.no_open:
                webbrowser.open(url)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                _say("\nstopped")
        return 0

    if args.what == "import":
        src = Path(args.path)
        if not src.is_file():
            _say(f"no such file: {src}")
            return 1
        data = json.loads(src.read_text(encoding="utf8"))
        rated = [r for r in data.get("ratings", [])
                 if r.get("mos") is not None or r.get("pmos") is not None]
        RATINGS.write_text(json.dumps(data, indent=2), encoding="utf8")
        _say(f"copied {src} -> {RATINGS}")
        _say(f"{len(rated)} rated items of {len(data.get('ratings', []))}")
        return 0

    if args.what == "results":
        argv = [sys.executable, "plot_mos.py", "--ratings", str(RATINGS)]
        if args.split:
            argv.append("--split")
        return subprocess.call(argv, cwd=str(HERE))

    _say(f"unknown: {args.what}")
    return 1


# --------------------------------------------------------------------------
# train / status / speak
# --------------------------------------------------------------------------

DESIGNS = {"a": "designA_budget4h", "b": "designB_B9h"}


def cmd_train(args) -> int:
    import rl
    mod = rl.design_a if args.design == "a" else rl.design_b
    run_id = args.run or DESIGNS[args.design]
    overrides = {}
    if args.iters:
        overrides["iterations"] = args.iters
    if args.voices:
        overrides["voices"] = args.voices
    _say(f"Design {args.design.upper()}  run={run_id}  "
         f"resume={'auto' if args.resume else 'no'}  overrides={overrides or 'none'}")
    out = mod.train(run_id, resume="auto" if args.resume else None, **overrides)
    _say(f"run directory: {out}")
    return 0


def cmd_status(args) -> int:
    from rl import common
    for design, run_id in DESIGNS.items():
        _rule(f"Design {design.upper()}  ({run_id})")
        ckpts = common.checkpoints(run_id)
        if not ckpts:
            _say("no checkpoints")
            continue
        latest = common.latest_checkpoint(run_id)
        hist = common.history(run_id)
        _say(f"checkpoints  {len(ckpts)}")
        _say(f"latest       {latest.name if latest else '-'}")
        # the log does not always start at iteration 0 -- Design B's begins at
        # 81 -- so the row count is not the iteration reached. Report both.
        steps = [r["iteration"] for r in hist if "iteration" in r]
        if steps:
            _say(f"reached      iteration {max(steps) + 1} "
                 f"(log covers {min(steps)}..{max(steps)})")
        _say(f"logged rows  {len(hist)}")
        if hist:
            summary = common.summarise(run_id)
            for key in ("mean", "slope", "t", "first", "last"):
                if key in summary:
                    _say(f"{key:<12} {summary[key]}")
    return 0


def cmd_speak(args) -> int:
    from rl import common
    run_id = args.run or DESIGNS[args.design] if args.design else None
    out = Path(args.out)
    if args.design is None:
        common.generate(None, args.voice, args.text, out, seed=args.seed,
                        no_edit=True, device=args.device)
        _say(f"unedited clone -> {out}")
    else:
        common.generate(run_id, args.voice, args.text, out, seed=args.seed,
                        device=args.device)
        _say(f"Design {args.design.upper()} ({run_id}) -> {out}")
    return 0


# --------------------------------------------------------------------------
# figures / verify
# --------------------------------------------------------------------------

def cmd_figures(args) -> int:
    scripts = ["plot_mos.py --split", "plot_plan_x.py", "plot_distill.py",
               "plot_qc_frontier.py", "plot_frontier.py", "plot_margin.py",
               "plot_ab_voices.py", "plot_longform.py", "plot_architecture.py"]
    rc = 0
    for s in scripts:
        parts = s.split()
        if not (HERE / parts[0]).is_file():
            continue
        _rule(parts[0])
        got = subprocess.call([sys.executable, *parts], cwd=str(HERE))
        rc = rc or got
    return rc


def cmd_verify(args) -> int:
    rc = 0
    for script in ("verify_claims.py", "verify_wer.py"):
        if (HERE / script).is_file():
            _rule(script)
            rc = subprocess.call([sys.executable, script], cwd=str(HERE)) or rc
    return rc


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ahjn",
        description="AHJN voice-cloning evasion project: one entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="run `ahjn.py check` first on a new machine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="environment, paths, model files")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("detect", help="detector scores for one or more clips")
    p.add_argument("files", nargs="+")
    p.add_argument("--models", nargs="*", default=None,
                   help="subset of AASIST AASIST-L RawNet2 RawGAT-ST")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("score", help="every metric for a clip")
    p.add_argument("files", nargs="+")
    p.add_argument("--reference", default=None, help="the real recording, for similarity")
    p.add_argument("--text", default=None, help="what was said, for WER")
    p.add_argument("--device", default=None)
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("rate", help="the listening test: build, serve, import, results")
    p.add_argument("what", choices=["build", "page", "serve", "import", "results"])
    p.add_argument("path", nargs="?", help="for `import`: the exported mos_ratings.json")
    p.add_argument("--voices", nargs="*", default=None)
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--split", action="store_true", help="for `results`: one figure per panel")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("train", help="train or continue Design A or B")
    p.add_argument("design", choices=["a", "b"])
    p.add_argument("--run", default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--voices", nargs="*", default=None)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("status", help="checkpoints and training trend per design")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("speak", help="say a line with a trained policy")
    p.add_argument("--design", choices=["a", "b"], default=None,
                   help="omit for the unedited clone")
    p.add_argument("--run", default=None)
    p.add_argument("--voice", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_speak)

    p = sub.add_parser("figures", help="redraw every figure")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("verify", help="re-derive the published numbers")
    p.set_defaults(func=cmd_verify)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
