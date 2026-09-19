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
