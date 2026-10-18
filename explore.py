# Project made by AHJN . Ahmed Hussein and Julius Norén.  All rights reserved
r"""
explore.py -- browse every trained run and every generated clip in one place.

    python explore.py
    python explore.py --port 8800 --no-open
    python explore.py --scan-only          # print what it found, serve nothing

What it answers, without you opening five files:

  * which runs exist, how far each got, how long it actually took, and
    whether its reward went anywhere
  * every clip any of them produced, with the model that made it
  * each clip's objective scores (four detectors, soft-min, speaker
    similarity, WER, PESQ, STOI) and its human MOS and PMOS where a person
    rated it
  * the audio itself, playable, next to those numbers

Standard library only, so it cannot disturb the pinned environments. It reads
stored JSON and never recomputes anything: if a number is missing it is
because nothing has measured that clip yet, not because this tool failed.

This is the LISTENING AND BROWSING tool. It is deliberately not the rating
page: blind MOS collection lives in mos_app_template.html, which is a single
file a rater can open with no Python at all. See the note at the bottom of
this docstring.

Why the two are separate:
    A rating page has to run on someone else's machine, blind, with the arm
    labels hidden and the order shuffled. A browsing tool is the opposite:
    it is for you, it shows everything, and it needs to read the results tree
    live. Merging them would mean either a rater has to install this project,
    or you lose the labels you came to see.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import socketserver
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from env_config import RESULTS_ROOT

RL_RUNS = RESULTS_ROOT / "rl_runs"

# run id -> how it is described in the write-up
KNOWN_RUNS = {
    "designA_budget4h": "Design A, PPO-Lagrangian (published)",
    "designB_B9h": "Design B, SAC + gradient hint (published, iters 81-399)",
    "designB_B3h": "Design B, the first 81 iterations of the same run",
    "designB_B_ablation": "Design B without the gradient hint",
    "designB_B_main": "Design B, abandoned early attempt",
}
ARM_LABEL = {"baseline": "unedited clone", "design_a": "Design A",
             "design_b": "Design B", "reference": "real recording",
             "a": "Design A", "b": "Design B"}


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def scan_runs() -> list[dict]:
    """Every training run, with how far it got and what it cost."""
    out = []
    if not RL_RUNS.is_dir():
        return out
    for d in sorted(RL_RUNS.glob("design*")):
        if not d.is_dir():
            continue
        log = d / "train_log.jsonl"
        rows = []
        if log.is_file():
            for line in log.read_text(encoding="utf8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        steps = [r["iteration"] for r in rows if "iteration" in r]
        rewards = [r["mean_reward"] for r in rows if "mean_reward" in r]
        seconds = sum(r.get("seconds", 0) for r in rows)
        cfg = {}
        cfg_path = d / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf8"))
            except json.JSONDecodeError:
                pass

        trend = None
        if len(rewards) > 2:
            n = len(rewards)
            xs = list(range(n))
            mx = sum(xs) / n
            my = sum(rewards) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, rewards))
            slope = sxy / sxx if sxx else 0.0
            resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, rewards)]
            dof = max(n - 2, 1)
            se2 = sum(r * r for r in resid) / dof / sxx if sxx else 0.0
            se = se2 ** 0.5
            trend = {"slope": slope, "t": (slope / se) if se else 0.0,
                     "first": rewards[0], "last": rewards[-1],
                     "mean": my}

        out.append({
            "run": d.name,
            "what": KNOWN_RUNS.get(d.name, "unlabelled run"),
            "design": "A" if d.name.startswith("designA") else "B",
            "rows": len(rows),
            "reached": (max(steps) + 1) if steps else 0,
            "covers": [min(steps), max(steps)] if steps else None,
            "hours": seconds / 3600.0,
            "checkpoints": len(list(d.glob("*.pt"))),
            "voices": cfg.get("voices"),
            "trend": trend,
        })
    return out


def _clip_key(row: dict) -> tuple:
    return (str(row.get("voice", "")), str(row.get("arm", "")))


def scan_corpus(root: Path, label: str) -> list[dict]:
    """Clips from one rendered corpus, with whatever has measured them."""
    scores = root / "scores.json"
    ratings = root / "mos_ratings.json"
    if not scores.is_file() and not ratings.is_file():
        return []

    blob = {}
    for candidate in (ratings, scores):          # ratings carry clips too
        if candidate.is_file():
            try:
                blob = json.loads(candidate.read_text(encoding="utf8"))
                break
            except json.JSONDecodeError:
                continue

    human = {}
    if ratings.is_file():
        try:
            rb = json.loads(ratings.read_text(encoding="utf8"))
            for r in rb.get("ratings", []):
                human[(r.get("voice"), r.get("arm"))] = r
        except json.JSONDecodeError:
            pass

    clips = []
    for c in blob.get("clips", []):
        rel = str(c.get("file", "")).replace("\\", "/")
        path = root / rel
        if not path.is_file():
            continue
        h = human.get(_clip_key(c), {})
        clips.append({
            "corpus": label,
            "voice": c.get("speaker") or c.get("voice"),
            "item": c.get("voice"),
            "line": c.get("line"),
            "arm": c.get("arm"),
            "arm_label": ARM_LABEL.get(str(c.get("arm")), str(c.get("arm"))),
            "rel": rel,
            "det": c.get("det") or {},
            "softmin": c.get("softmin"),
            "sim": c.get("sim"),
            "wer": c.get("wer"),
            "pesq": c.get("pesq"),
            "stoi": c.get("stoi"),
            "transcript": c.get("transcript"),
            "mos": h.get("mos"),
            "pmos": h.get("pmos"),
        })
    return clips


def scan_comparisons(root: Path) -> list[dict]:
    """Clips written by scripts/compare_models.py."""
    clips = []
    for js in sorted(root.glob("**/comparison.json")):
        try:
            blob = json.loads(js.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            continue
        for r in blob.get("rows", []):
            path = js.parent / r.get("file", "")
            if not path.is_file():
                continue
            clips.append({
                "corpus": f"comparison/{js.parent.name}",
                "voice": r.get("voice"), "item": r.get("item"),
                "line": r.get("line"), "arm": r.get("arm"),
                "arm_label": ARM_LABEL.get(str(r.get("arm")), str(r.get("arm"))),
                "rel": r.get("file"),
                "det": r.get("det") or {}, "softmin": r.get("softmin"),
                "sim": r.get("sim"), "wer": r.get("wer"),
                "pesq": r.get("pesq"), "stoi": r.get("stoi"),
                "transcript": r.get("transcript"),
                "mos": None, "pmos": None,
            })
    return clips


def build_index() -> dict:
    sources: dict[str, Path] = {}
    clips: list[dict] = []

    for name, label in (("MOS TEST RL", "RL listening test"),
                        ("MOS TEST", "earlier non-RL test")):
        root = REPO / name
        if root.is_dir():
            got = scan_corpus(root, label)
            if got:
                sources[label] = root
                clips.extend(got)

    out_root = REPO / "out"
    if out_root.is_dir():
        got = scan_comparisons(out_root)
        for c in got:
            sources.setdefault(c["corpus"], (out_root / c["corpus"].split("/")[-1]))
        clips.extend(got)

    for i, c in enumerate(clips):
        c["id"] = i
    return {"runs": scan_runs(), "clips": clips,
            "sources": {k: str(v) for k, v in sources.items()},
            "_roots": sources}


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AHJN explorer</title>
<style>
 :root{--bg:#f6f6f4;--card:#fff;--ink:#1b1f27;--ink2:#5a6070;--line:#e2e4e8;
       --grey:#8a8f98;--red:#c2413b;--blue:#4a5fc1;--green:#2a9d8f;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.5 "Inter","Segoe UI",system-ui,sans-serif}
 header{padding:20px 24px 8px}
 h1{margin:0 0 4px;font-size:22px}
 .sub{color:var(--ink2);font-size:13px}
 .wrap{padding:0 24px 40px;max-width:1500px}
 h2{font-size:15px;margin:26px 0 10px;text-transform:uppercase;
    letter-spacing:.6px;color:var(--ink2)}
 .runs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(310px,1fr))}
 .run{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
 .run h3{margin:0 0 2px;font-size:15px}
 .run .what{color:var(--ink2);font-size:12.5px;margin-bottom:10px}
 .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:13px}
 .kv dt{color:var(--ink2)} .kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
 .tagA{border-left:4px solid var(--red)} .tagB{border-left:4px solid var(--blue)}
 .bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 10px}
 input,select{font:inherit;padding:6px 9px;border:1px solid var(--line);
   border-radius:7px;background:var(--card);color:var(--ink)}
 table{width:100%;border-collapse:collapse;background:var(--card);
   border:1px solid var(--line);border-radius:10px;overflow:hidden}
 th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);
   font-variant-numeric:tabular-nums;white-space:nowrap}
 th{background:#fafafa;text-align:right;font-size:12px;color:var(--ink2);
    cursor:pointer;user-select:none;position:sticky;top:0}
 th:first-child,td:first-child,th.l,td.l{text-align:left}
 tr:hover td{background:#fbfbfa}
 .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;
   color:#fff;background:var(--ink2)}   /* default: an arm this tool has no
   colour for still has to be readable, white on nothing is invisible */
 .p-baseline{background:var(--grey)} .p-design_a,.p-a{background:var(--red)}
 .p-design_b,.p-b{background:var(--blue)} .p-reference{background:#3a3f4a}
 audio{height:30px;vertical-align:middle}
 .muted{color:var(--ink2)}
 .note{background:#fff8e6;border:1px solid #efdca8;border-radius:9px;
   padding:10px 13px;font-size:13px;margin:14px 0}
 @media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){--bg:#15171b;--card:#1d2026;--ink:#e9eaec;
    --ink2:#9aa1ad;--line:#2b2f37}
  :root:not([data-theme="light"]) th{background:#22262d}
  :root:not([data-theme="light"]) tr:hover td{background:#22262d}
  :root:not([data-theme="light"]) .note{background:#2a2518;border-color:#4a412a}
 }
</style></head><body>
<header>
  <h1>AHJN explorer</h1>
  <div class="sub" id="sub">loading</div>
</header>
<div class="wrap">
  <h2>Training runs</h2>
  <div class="runs" id="runs"></div>

  <h2>Clips</h2>
  <div class="bar">
    <input id="q" placeholder="filter: voice, arm, corpus" size="28">
    <select id="arm"><option value="">every arm</option></select>
    <select id="corpus"><option value="">every corpus</option></select>
    <label class="muted"><input type="checkbox" id="rated"> only human-rated</label>
    <span class="muted" id="count"></span>
  </div>
  <div class="note" id="hint"></div>
  <table id="tbl"><thead><tr>
    <th class="l" data-k="item">item</th>
    <th class="l" data-k="arm">arm</th>
    <th class="l">listen</th>
    <th data-k="softmin">soft-min</th>
    <th data-k="mos">MOS</th>
    <th data-k="pmos">PMOS</th>
    <th data-k="sim">sim</th>
    <th data-k="wer">WER</th>
    <th data-k="pesq">PESQ</th>
    <th data-k="stoi">STOI</th>
    <th class="l" data-k="corpus">corpus</th>
  </tr></thead><tbody id="rows"></tbody></table>
</div>
<script>
let DATA={runs:[],clips:[]}, sortKey="item", sortDir=1;
const $=s=>document.querySelector(s);
const n=(v,d=3)=>v==null||Number.isNaN(v)?'<span class="muted">-</span>':Number(v).toFixed(d);

function runCard(r){
  const t=r.trend;
  const verdict = !t ? "no trend" :
    (Math.abs(t.t)<2 ? "flat (t="+t.t.toFixed(2)+")"
     : (t.slope>0?"rising":"falling")+" (t="+t.t.toFixed(2)+")");
  return `<div class="run tag${r.design}">
    <h3>${r.run}</h3><div class="what">${r.what}</div>
    <dl class="kv">
      <dt>reached</dt><dd>iteration ${r.reached}</dd>
      <dt>log covers</dt><dd>${r.covers? r.covers[0]+" to "+r.covers[1] : "-"}</dd>
      <dt>training time</dt><dd>${r.hours.toFixed(2)} h</dd>
      <dt>checkpoints</dt><dd>${r.checkpoints}</dd>
      <dt>mean reward</dt><dd>${t? t.mean.toFixed(4):"-"}</dd>
      <dt>reward trend</dt><dd>${verdict}</dd>
      <dt>voices</dt><dd>${r.voices? r.voices.join(", "):"-"}</dd>
    </dl></div>`;
}

function draw(){
  const q=$("#q").value.toLowerCase(), arm=$("#arm").value,
        corp=$("#corpus").value, onlyRated=$("#rated").checked;
  let rows=DATA.clips.filter(c=>{
    if(arm && c.arm!==arm) return false;
    if(corp && c.corpus!==corp) return false;
    if(onlyRated && c.mos==null) return false;
    if(q && !((c.item||"")+" "+(c.arm||"")+" "+(c.corpus||"")).toLowerCase().includes(q))
      return false;
    return true;
  });
  rows.sort((a,b)=>{
    const x=a[sortKey], y=b[sortKey];
    if(x==null&&y==null) return 0;
    if(x==null) return 1; if(y==null) return -1;
    return (typeof x==="number"? x-y : String(x).localeCompare(String(y)))*sortDir;
  });
  $("#count").textContent = rows.length+" of "+DATA.clips.length+" clips";
  $("#rows").innerHTML = rows.map(c=>`<tr>
    <td class="l">${c.item||""}</td>
    <td class="l"><span class="pill p-${c.arm}">${c.arm_label||c.arm}</span></td>
    <td class="l"><audio controls preload="none" src="/audio/${c.id}"></audio></td>
    <td>${n(c.softmin)}</td><td>${n(c.mos,0)}</td><td>${n(c.pmos,0)}</td>
    <td>${n(c.sim)}</td><td>${c.wer==null?'<span class="muted">-</span>':(c.wer*100).toFixed(1)+"%"}</td>
    <td>${n(c.pesq,2)}</td><td>${n(c.stoi)}</td>
    <td class="l muted">${c.corpus||""}</td></tr>`).join("");
}

fetch("/api/index").then(r=>r.json()).then(d=>{
  DATA=d;
  $("#runs").innerHTML = d.runs.map(runCard).join("") ||
    '<div class="muted">no runs found under rl_runs</div>';
  const arms=[...new Set(d.clips.map(c=>c.arm))].filter(Boolean).sort();
  const corps=[...new Set(d.clips.map(c=>c.corpus))].filter(Boolean).sort();
  $("#arm").innerHTML += arms.map(a=>`<option value="${a}">${a}</option>`).join("");
  $("#corpus").innerHTML += corps.map(a=>`<option value="${a}">${a}</option>`).join("");
  const rated=d.clips.filter(c=>c.mos!=null).length;
  $("#sub").textContent = `${d.runs.length} runs, ${d.clips.length} clips, `
    + `${rated} of them rated by a person`;
  $("#hint").textContent =
    "MOS and PMOS are a person's 1 to 5 ratings and exist only where someone "
    + "rated that clip. Soft-min is the worst of the four detectors, 1 = human. "
    + "Click a column to sort. Nothing here is recomputed: blanks mean nothing "
    + "has measured that clip yet.";
  draw();
});

for(const el of ["q","arm","corpus","rated"]) $("#"+el).addEventListener("input",draw);
document.querySelectorAll("th[data-k]").forEach(th=>th.addEventListener("click",()=>{
  const k=th.dataset.k;
  sortDir = (k===sortKey) ? -sortDir : 1;
  sortKey = k; draw();
}));
</script></body></html>
"""


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    index: dict = {}

    def log_message(self, *a):            # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = unquote(urlparse(self.path).path)
        if route in ("/", "/index.html"):
            return self._send(200, PAGE.encode("utf8"), "text/html; charset=utf-8")

        if route == "/api/index":
            public = {k: v for k, v in self.index.items() if not k.startswith("_")}
            return self._send(200, json.dumps(public).encode("utf8"),
                              "application/json")

        if route.startswith("/audio/"):
            # served by index id, never by a path from the request, so a
            # crafted URL cannot walk out of the corpus directories
            try:
                clip = self.index["clips"][int(route.split("/")[2])]
            except (ValueError, IndexError, KeyError):
                return self._send(404, b"no such clip", "text/plain")
            root = self.index["_roots"].get(clip["corpus"])
            if root is None:
                return self._send(404, b"corpus not mounted", "text/plain")
            path = Path(root) / clip["rel"]
            if not path.is_file():
                return self._send(404, b"file missing", "text/plain")
            ctype = mimetypes.guess_type(path.name)[0] or "audio/wav"
            return self._send(200, path.read_bytes(), ctype)

        self._send(404, b"not found", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser(description="Browse runs, clips and scores.")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--scan-only", action="store_true",
                    help="print what was found and exit")
    args = ap.parse_args()

    index = build_index()

    print(f"{len(index['runs'])} training runs")
    for r in index["runs"]:
        t = r["trend"]
        print(f"  {r['run']:<22} iter {r['reached']:>4}  {r['hours']:>6.2f} h  "
              f"{r['checkpoints']:>3} ckpt  "
              f"{'trend t=%+.2f' % t['t'] if t else 'no trend'}")
    print(f"\n{len(index['clips'])} clips")
    by_corpus: dict[str, int] = {}
    for c in index["clips"]:
        by_corpus[c["corpus"]] = by_corpus.get(c["corpus"], 0) + 1
    for k, v in sorted(by_corpus.items()):
        print(f"  {k:<28} {v}")
    rated = sum(1 for c in index["clips"] if c.get("mos") is not None)
    print(f"\n{rated} clips carry a human rating")

    if args.scan_only:
        return 0
    if not index["clips"] and not index["runs"]:
        print("\nnothing to show. Render a corpus or train a run first.")
        return 1

    Handler.index = index
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as srv:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"\nserving {url}   (ctrl-C to stop)")
        if not args.no_open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
