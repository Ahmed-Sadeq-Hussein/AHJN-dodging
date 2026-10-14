# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
mos_build_app.py -- assemble the listening-test page from the rendered corpus.

    python mos_build_app.py                     # reads "MOS TEST/scores.json"
    python mos_build_app.py --stage <dir>       # also copy the corpus for publishing

Injects scores.json into mos_app_template.html and writes
"MOS TEST/listening_test.html". The page references its audio by the same
relative paths scores.json records ("baseline/riven.wav"), so the folder is
self-contained: open the HTML from inside "MOS TEST" and every clip plays.

--stage copies the whole corpus somewhere the Artifact tool can publish from,
and prints the file list to hand it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="MOS TEST")
    ap.add_argument("--template", default="mos_app_template.html")
    ap.add_argument("--stage", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    scores = root / "scores.json"
    if not scores.is_file():
        print(f"no {scores} -- run mos_test.py first")
        return 1

    data = json.loads(scores.read_text(encoding="utf8"))
    tpl = Path(args.template).read_text(encoding="utf8")
    if "/*__DATA__*/" not in tpl:
        print("template has no /*__DATA__*/ placeholder")
        return 1

    # Clip paths are what the page feeds to new Audio(), so they have to be
    # relative to the page with forward slashes. Repair what we can -- older
    # scores.json files carry OS paths with the corpus directory on the front
    # -- and refuse to build if anything still does not resolve.
    missing = []
    for c in data.get("clips", []):
        f = str(c.get("file", "")).replace("\\", "/")
        # the stored path may carry the corpus directory either as it was
        # spelled on the command line or just by name, so try both
        for pre in (root.as_posix().rstrip("/") + "/", root.name + "/"):
            if f.startswith(pre):
                f = f[len(pre):]
                break
        f = f.lstrip("/")
        c["file"] = f
        if not f or not (root / f).is_file():
            missing.append(f"{c.get('arm')} {c.get('voice')}: {c.get('file')!r}")
    if missing:
        print(f"{len(missing)} clip(s) do not resolve under {root}:")
        for m in missing[:10]:
            print(f"  {m}")
        return 1

    # Every arm in the corpus has to reach the page; the arm table in the
    # template is presentation only and must not decide what gets shown.
    arms = sorted({c["arm"] for c in data.get("clips", [])} - {"reference"})
    declared = sorted(data.get("arms", {}))
    if declared and declared != arms:
        print(f"arms in clips {arms} do not match declared {declared}")
        return 1
    print(f"{len(arms)} arms: {', '.join(arms)}")

    page = tpl.replace("/*__DATA__*/{}", json.dumps(data, separators=(",", ":")))
    out = root / "listening_test.html"
    out.write_text(page, encoding="utf8")

    wavs = sorted(p for p in root.rglob("*.wav"))
    total = sum(p.stat().st_size for p in wavs)
    print(f"wrote {out}  ({len(page):,} bytes)")
    print(f"{len(wavs)} clips, {total / 1e6:.1f} MB")

    if args.stage:
        stage = Path(args.stage)
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(root, stage)
        rel = sorted(str(p.relative_to(root)).replace("\\", "/") for p in wavs)
        (stage / "files.json").write_text(json.dumps(rel, indent=1), encoding="utf8")
        print(f"staged to {stage}  ({len(rel)} audio files listed in files.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
