r"""
generate_all.py -- phase 1 of the benchmark. RUN IN THE "voiceclone" ENV.

Clones one reference voice with every cloning model and writes the results to
D:\results and benchmarks\generators\<Model>\.

    conda activate voiceclone
    python generate_all.py --voice riven

Phase 2 (score_all.py) runs in the "voicefake" env instead, because the two
stacks pin incompatible torch/numpy versions. The two phases talk to each
other through the .wav files on disk and a manifest.json, not in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cloners

from env_config import VOICE_DATA_ROOT as VOICE_DATA, GENERATORS_DIR as GENERATORS

# One fixed sentence for every model, so outputs are directly comparable.
# Long enough to fill more than one 64600-sample detector window.
TEST_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning light "
    "spills across the quiet valley below."
)


def load_reference(voice: str) -> tuple[Path, str | None]:
    """Return (reference.wav, transcript or None) for a voice folder."""
    folder = VOICE_DATA / voice
    wav = folder / "reference.wav"
    if not wav.is_file():
        raise FileNotFoundError(f"no reference.wav in {folder}")
    txt = folder / "reference.txt"
    prompt = txt.read_text(encoding="utf8").strip() if txt.is_file() else None
    return wav, prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="riven")
    ap.add_argument("--text", default=TEST_TEXT)
    ap.add_argument("--models", nargs="*", default=list(cloners.ENABLED_MODELS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ref_wav, prompt_text = load_reference(args.voice)
    print(f"reference : {ref_wav}")
    print(f"transcript: {prompt_text!r}")
    print(f"text      : {args.text!r}\n")

    # Merge into any existing manifest rather than replacing it. Running this
    # script for one model at a time is the normal way to debug, and a plain
    # overwrite would silently drop the other model's already-good output.
    man_path = GENERATORS / f"manifest_{args.voice}.json"
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "voice": args.voice,
        "reference_wav": str(ref_wav),
        "reference_transcript": prompt_text,
        "text": args.text,
        "outputs": [],
        "failures": [],
    }
    if man_path.is_file():
        try:
            prev = json.loads(man_path.read_text(encoding="utf8"))
            if prev.get("text") == args.text:
                # keep entries for models we are not re-running now
                rerunning = {cloners._canon_model(m) for m in args.models}
                manifest["outputs"] = [
                    o for o in prev.get("outputs", [])
                    if o.get("model") not in rerunning and Path(o.get("path", "")).is_file()
                ]
                if manifest["outputs"]:
                    print(f"carrying over {len(manifest['outputs'])} existing output(s): "
                          f"{[o['model'] for o in manifest['outputs']]}\n")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"could not read existing manifest ({exc}); starting fresh\n")

    for model in args.models:
        name = cloners._canon_model(model)
        out_dir = GENERATORS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.voice}.wav"
        print(f"=== {name} ===")
        try:
            v = cloners.enroll(
                ref_wav, name, prompt_text=prompt_text,
                alias=args.voice, device=args.device,
            )
            print(f"  enrolled {v.alias!r}"
                  + (f" (spk_id {v.payload.get('spk_id')})" if v.payload.get("spk_id") else ""))
            written = cloners.generate(
                args.text, ref_wav, model=name,
                out_path=out_path, device=args.device,
            )
            import soundfile as sf
            info = sf.info(str(written))
            print(f"  wrote {written}  ({info.duration:.2f}s @ {info.samplerate} Hz)\n")
            manifest["outputs"].append({
                "model": name,
                "path": str(written),
                "duration_s": round(info.duration, 3),
                "samplerate": info.samplerate,
                "used_transcript": bool(prompt_text),
            })
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
            traceback.print_exc()
            manifest["failures"].append({
                "model": name, "error": f"{type(exc).__name__}: {exc}",
            })

    GENERATORS.mkdir(parents=True, exist_ok=True)
    manifest["outputs"].sort(key=lambda o: o["model"])
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(f"manifest -> {man_path}")
    print(f"{len(manifest['outputs'])} generated, {len(manifest['failures'])} failed")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
