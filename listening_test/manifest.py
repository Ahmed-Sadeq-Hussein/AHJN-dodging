r"""
listening_test.manifest -- describe a folder of audio for the listening-test app.

    python -m listening_test "D:\some audio"                  # scan, write manifest.json
    python -m listening_test "D:\some audio" --app            # ...and copy the app in
    python -m listening_test "D:\some audio" --print          # show it, write nothing
    python -m listening_test DIR --title "Pitch test" --prompt "The line every clip says"

The app does the same scan in the browser when it is given a folder with no
manifest.json, following the same rules, so either route gives the same file.
Use this one when you want the manifest before opening the app, or when you
want to script it.

Folder convention
-----------------
    my test/
      references/      alice.wav  bob.wav       optional: the original for each item
      model_a/         alice.wav  bob.wav       one folder per configuration
      model_b/         alice.wav  bob.wav

* Each subfolder is one configuration (condition). Files with the same name
  across subfolders are the same item and are rated side by side.
* A subfolder called references / reference / refs / ref / originals /
  original supplies the original recording for each item, matched by name.
* Audio files directly in the folder, with no subfolders, become a single
  configuration named after the folder.
* Hidden files and folders (leading dot) are skipped; nesting deeper than
  four levels is ignored.

Everything in the manifest can be edited by hand afterwards: labels, notes,
colours, the questions asked, and per-clip metrics to show beside each clip.
See listening_test/README.md for the schema.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "listening-test/1"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".webm"}
REFERENCE_DIRS = {"references", "reference", "refs", "ref", "originals", "original"}
FIRST = ["baseline", "control", "unedited", "anchor"]   # listed first when present
MAX_DEPTH = 4
ROOT_COND = "_root"
PALETTE = ["#8a8f98", "#2a9d8f", "#4a5fc1", "#e07a1f",
           "#b5487a", "#6a994e", "#c9a227", "#3d7ea6"]
APP = Path(__file__).resolve().parent / "app.html"

# Kept identical to DEFAULT_QUESTIONS in app.html.
DEFAULT_QUESTIONS = [
    {"id": "mos", "type": "scale", "label": "MOS",
     "prompt": "How human does it sound?", "min": 1, "max": 5, "reference": True,
     "anchors": {"1": "clearly synthetic", "2": "noticeably off", "3": "acceptable",
                 "4": "natural", "5": "indistinguishable from a person"}},
    {"id": "pmos", "type": "scale", "label": "PMOS",
     "prompt": "Does it still sound like the character?", "min": 1, "max": 5,
     "reference": True,
     "anchors": {"1": "a different person", "2": "barely them", "3": "somewhat them",
                 "4": "clearly them", "5": "the same performer"}},
    {"id": "glitch", "type": "boolean", "label": "Glitch",
     "prompt": "Is there static or audible glitches in the voice?", "yes": "bad"},
]


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "listening-test"


def prettify(s: str) -> str:
    return re.sub(r"[_-]+", " ", s).strip()


def _strip_ext(path: str) -> str:
    head, _, name = path.rpartition("/")
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return f"{head}/{stem}" if head else stem


def scan(root, title: str | None = None, prompt: str = "") -> tuple[dict, list[str]]:
    """Build a manifest from a folder of audio. Returns (manifest, problems)."""
    root = Path(root)
    title = title or root.name
    by_cond: dict[str, dict[str, str]] = {}
    refs: dict[str, str] = {}
    problems: list[str] = []

    rels = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            rel = p.relative_to(root).as_posix()
            parts = rel.split("/")
            if len(parts) <= MAX_DEPTH and not any(s.startswith(".") for s in parts):
                rels.append(rel)

    for rel in sorted(rels, key=natural_key):
        parts = rel.split("/")
        cond = ROOT_COND if len(parts) == 1 else parts[0]
        item = _strip_ext(parts[0] if len(parts) == 1 else "/".join(parts[1:]))
        if cond != ROOT_COND and cond.lower() in REFERENCE_DIRS:
            refs.setdefault(item, rel)
            continue
        bucket = by_cond.setdefault(cond, {})
        if item in bucket:
            problems.append(f'{cond}: two files for "{item}", kept {bucket[item]}')
        else:
            bucket[item] = rel

    def cond_order(c):
        low = c.lower()
        return (FIRST.index(low) if low in FIRST else len(FIRST), natural_key(c))

    conds = sorted(by_cond, key=cond_order)
    items = sorted({i for c in conds for i in by_cond[c]}, key=natural_key)
    ref_lower = {k.lower(): v for k, v in refs.items()}
    for r in refs:
        if r.lower() not in {i.lower() for i in items}:
            problems.append(f'reference "{r}" matches no clip')

    manifest = {
        "schema": SCHEMA,
        "id": slug(title),
        "title": title,
        "prompt": prompt,
        "blind": True,
        "generated": {"by": "listening_test.manifest",
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "questions": DEFAULT_QUESTIONS,
        "conditions": [{"id": c, "label": title if c == ROOT_COND else prettify(c),
                        "color": PALETTE[i % len(PALETTE)]}
                       for i, c in enumerate(conds)],
        "items": [{"id": it, "label": it,
                   "reference": refs.get(it) or ref_lower.get(it.lower()),
                   "clips": {c: by_cond[c][it] for c in conds if it in by_cond[c]}}
                  for it in items],
    }
    return manifest, problems


def validate(manifest: dict, root=None) -> list[str]:
    """Problems that would stop the app using this manifest, or lose clips."""
    out = []
    if manifest.get("schema") not in (None, SCHEMA):
        out.append(f'schema is "{manifest.get("schema")}", expected "{SCHEMA}"')
    conds = {c.get("id") for c in manifest.get("conditions", []) if isinstance(c, dict)}
    if not conds:
        out.append("no configurations")
    items = manifest.get("items", [])
    if not items:
        out.append("no items")
    for it in items:
        for c, path in (it.get("clips") or {}).items():
            if c not in conds:
                out.append(f'{it.get("id")}: clip for unknown configuration "{c}"')
            elif root is not None and not (Path(root) / path).is_file():
                out.append(f'{it.get("id")}: missing file {path}')
        ref = it.get("reference")
        if ref and root is not None and not (Path(root) / ref).is_file():
            out.append(f'{it.get("id")}: missing reference {ref}')
    return out


def write(manifest: dict, root, name: str = "manifest.json") -> Path:
    out = Path(root) / name
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return out


def install_app(root, name: str = "listening_test.html") -> Path:
    """Copy the app into the folder so it travels with the audio."""
    out = Path(root) / name
    shutil.copy2(APP, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m listening_test",
                                 description="Write manifest.json for a folder of audio.")
    ap.add_argument("folder")
    ap.add_argument("--title", default=None, help="defaults to the folder name")
    ap.add_argument("--prompt", default="", help="the line every clip speaks, if any")
    ap.add_argument("--print", action="store_true", help="print, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest.json (hand edits are lost)")
    ap.add_argument("--app", action="store_true",
                    help="also copy the app into the folder as listening_test.html")
    args = ap.parse_args(argv)

    root = Path(args.folder)
    if not root.is_dir():
        print(f"not a folder: {root}")
        return 1
    manifest, problems = scan(root, args.title, args.prompt)
    for p in problems:
        print("  note:", p)
    if not manifest["items"]:
        print("no audio found -- see the folder convention in --help")
        return 1

    n = sum(len(i["clips"]) for i in manifest["items"])
    refs = sum(1 for i in manifest["items"] if i["reference"])
    print(f'{len(manifest["items"])} items x {len(manifest["conditions"])} configurations '
          f"({n} clips, {refs} with an original)")
    print("configurations:", ", ".join(c["id"] for c in manifest["conditions"]))

    if args.print:
        print(json.dumps(manifest, indent=2))
        return 0
    target = root / "manifest.json"
    if target.exists() and not args.force:
        print(f"{target} already exists -- pass --force to replace it")
    else:
        print("wrote", write(manifest, root))
    if args.app:
        print("wrote", install_app(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
