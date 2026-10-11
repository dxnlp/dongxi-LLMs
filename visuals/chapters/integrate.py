"""Embed/check the approved 28 placements and generate a local review gallery.

uv run --project visuals/animations python visuals/chapters/integrate.py --apply
uv run --project visuals/animations python visuals/chapters/integrate.py --verify
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visuals.chapters.catalog import CHAPTERS, PLACEMENTS
from visuals.chapters import fixtures
from visuals.chapters import render

HERE = Path(__file__).resolve().parent
BASE = "4e5a5ccda6327a7a8a7d88ebfc5b73c2280a778e"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def relative(chapter, target):
    return Path(os.path.relpath(ROOT / target, (ROOT / chapter).parent)).as_posix()


def block(p):
    start = f"\n<!-- chapter-gif: {p['id']} -->\n"
    image = "" if p["mode"] == "retain" else f"![{p['caption']}]({relative(p['chapter'], p['gif'])})\n\n"
    return (start + f"**Predict:** {p['question']}\n\n"
            + image + f"*{p['caption']}*\n\n"
            + f"[Try the notebook]({relative(p['chapter'], p['notebook'])}) · "
            + f"[View a still image]({relative(p['chapter'], p['poster'])})\n"
            + f"<!-- /chapter-gif: {p['id']} -->\n\n")


def ensure_poster():
    p = next(p for p in PLACEMENTS if p["mode"] == "retain")
    poster = ROOT / p["poster"]
    if not poster.exists():
        poster.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(ROOT / p["gif"]) as im:
            im.seek(round(im.n_frames * .7))
            im.convert("RGB").save(poster)


def integrate():
    ensure_poster()
    for p in PLACEMENTS:
        assert all((ROOT / p[k]).is_file() for k in ["gif", "poster", "notebook"]), p
        chapter = ROOT / p["chapter"]
        source = chapter.read_text()
        if f"<!-- chapter-gif: {p['id']} -->" in source:
            continue
        if p["mode"] == "retain":
            found = re.search(r"^!\[.*?\]\(.*?bpe-byte-merges\.gif\).*$", source, re.M)
            assert found
            at = found.end() + 1
        else:
            heading = re.search(rf"^## {re.escape(p['section'])} [^\n]+\n", source, re.M)
            assert heading, p
            after = re.search(r"^## ", source[heading.end():], re.M)
            at = heading.end() + after.start() if after else len(source)
        source = source[:at] + block(p) + source[at:]
        chapter.write_text(source)


def verify():
    counts, records = Counter(), []
    for p in PLACEMENTS:
        source = (ROOT / p["chapter"]).read_text()
        assert source.count(block(p)) == 1, p["id"]
        for key in ["gif", "poster", "notebook"]:
            assert (ROOT / p[key]).is_file(), (p["id"], key)
        assert p["question"] and p["caption"]
        gif = ROOT / p["gif"]
        poster = ROOT / p["poster"]
        with Image.open(gif) as im:
            unique, delays = set(), []
            assert im.info.get("loop") == 0 and im.n_frames > 1
            dimensions = list(im.size)
            for i in range(im.n_frames):
                im.seek(i)
                unique.add(sha(im.convert("RGB").tobytes()))
                delays.append(im.info.get("duration", 0))
            assert len(unique) > 10 and all(d > 0 for d in delays)
        if p["mode"] in ["reuse", "retain"]:
            assert gif.read_bytes() == original(p["gif"]), p["gif"]
        else:
            render.decoded_info(gif)
        with Image.open(poster) as still:
            poster_dimensions = list(still.size)
            if p["mode"] == "new":
                assert poster_dimensions == dimensions
            else:
                assert still.width * dimensions[1] == still.height * dimensions[0]
        records.append({**p, "gif_sha256": sha(gif.read_bytes()), "bytes": gif.stat().st_size,
                        "poster_sha256": sha(poster.read_bytes()), "dimensions": dimensions,
                        "poster_dimensions": poster_dimensions,
                        "duration_seconds": sum(delays) / 1000, "unique_frames": len(unique)})
        counts[p["chapter"]] += 1

    assert len(PLACEMENTS) == 28 and len(counts) == 15
    preserved, links = {}, 0
    for chapter in CHAPTERS:
        source = chapter.read_text()
        stripped = source
        for p in PLACEMENTS:
            if ROOT / p["chapter"] == chapter:
                stripped = stripped.replace(block(p), "", 1)
        rel = str(chapter.relative_to(ROOT))
        baseline = original(rel)
        assert stripped.encode() == baseline, f"Unexpected chapter edit: {rel}"
        expected = counts[rel]
        assert len(re.findall(r"!\[[^\]]*\]\([^\n)]*\.gif\)", source)) == expected, rel
        preserved[rel] = {"baseline_sha256": sha(baseline), "current_sha256": sha(source.encode()),
                          "placements": expected, "existing_prose_and_math_unchanged": True}
        for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", source):
            parsed = urlsplit(target)
            if parsed.scheme or target.startswith("#"):
                continue
            path = unquote(parsed.path)
            if path:
                assert (chapter.parent / path).exists(), (rel, target)
                links += 1

    manifest = json.loads((HERE / "manifest.json").read_text())
    for path, checksum in manifest["sources"].items():
        assert sha((ROOT / path).read_bytes()) == checksum, path
    assert set(item["scene"] for item in manifest["outputs"]) == set(render.SCENES)
    for item in manifest["outputs"]:
        assert item["sha256"] == sha((ROOT / item["gif"]).read_bytes())
        assert item["poster_sha256"] == sha((ROOT / item["poster"]).read_bytes())

    report = ROOT / "experiments/reports/2026-10-04-grpo-diagnostics-distillation.json"
    assert report.read_bytes() == original(str(report.relative_to(ROOT)))
    archived = json.loads(report.read_text())["reward_hacking"]
    error = 0.
    for arm in archived:
        replay = fixtures.hacking_history(arm["repair"])
        assert len(replay) == len(arm["history"]) == 61
        for actual, saved in zip(replay, arm["history"]):
            assert actual["update"] == saved["update"]
            for key in ["proxy_reward", "strict_accuracy"]:
                error = max(error, abs(actual[key] - saved[key]))
            for actual_p, saved_p in zip(actual["probabilities"], saved["probabilities"]):
                error = max(error, abs(actual_p - saved_p))
    assert error < 1e-12
    assert not subprocess.check_output(["git", "diff", BASE, "--name-only", "--", "notebooks",
                                       "src/dongxi_llms", "experiments/reports"], cwd=ROOT)
    proposals = "visuals/animations/PROPOSALS.md"
    assert (ROOT / proposals).read_bytes().startswith(original(proposals))

    # No new tracked/untracked video files in this deliverable.
    assert not list(HERE.rglob("*.mp4"))
    receipt = {"base_commit": BASE, "placements": records, "chapter_counts": dict(counts),
               "new_gifs": sum(p["mode"] == "new" for p in PLACEMENTS),
               "reused_gifs": sum(p["mode"] == "reuse" for p in PLACEMENTS),
               "retained_gifs": sum(p["mode"] == "retain" for p in PLACEMENTS),
               "new_gif_bytes": sum(r["bytes"] for r in records if r["mode"] == "new"),
               "all_embedded_gif_bytes": sum(r["bytes"] for r in records),
               "local_chapter_links_checked": links, "chapter_preservation": preserved,
               "historical_proposal_prefix_preserved": True,
               "production_source_hashes": {str(path.relative_to(ROOT)): sha(path.read_bytes())
                                           for path in [HERE / "catalog.py", HERE / "fixtures.py",
                                                        HERE / "render.py", Path(__file__)]},
               "numeric_checks": fixtures.verify(),
               "historical_policy_replay": {"report_sha256": sha(report.read_bytes()),
                                            "arms": 2, "integer_updates_per_arm": 61,
                                            "maximum_absolute_error": error},
               "media_verified": True, "no_video_outputs": True,
               "notebook_source_changes": False, "new_model_experiments": False}
    (HERE / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def gallery():
    cards = []
    for p in PLACEMENTS:
        title = p["id"].replace("-", " ").capitalize()
        for plain, acronym in [("Bpe", "BPE"), ("Kv", "KV"), ("Rope", "RoPE"), ("Lora", "LoRA"), ("Dpo", "DPO"), ("Ppo", "PPO")]:
            title = title.replace(plain, acronym)
        title = html.escape(title)
        src = html.escape(os.path.relpath(ROOT / p["gif"], HERE))
        poster = html.escape(os.path.relpath(ROOT / p["poster"], HERE))
        chapter = html.escape(os.path.relpath(ROOT / p["chapter"], HERE))
        notebook = html.escape(os.path.relpath(ROOT / p["notebook"], HERE))
        cards.append(f'<article><p class="route">Chapter {p["section"].split(".")[0]} · §{p["section"]}</p><h2>{title}</h2>'
                     f'<img src="{src}" data-gif="{src}" data-still="{poster}" alt="{html.escape(p["caption"])}">'
                     f'<p>{html.escape(p["caption"])}</p><p><b>Predict:</b> {html.escape(p["question"])}</p>'
                     f'<p><a href="{chapter}">Read the chapter</a> · <a href="{notebook}">Try the notebook</a> · <a href="{poster}">Still image</a></p></article>')
    markup = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Language-model mechanisms in motion</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#111827;font:17px/1.55 Arial,sans-serif}
main{max-width:1160px;margin:36px auto;padding:0 24px}h1{font-size:32px;font-weight:400}h2{font-size:24px;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}article{border:1px solid #e2e8f0;border-radius:16px;padding:20px}
img{width:100%;height:auto;display:block}a{color:#2563eb}.route{color:#64748b;font-size:15px}button{font:inherit;background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:8px 16px;cursor:pointer;margin-bottom:24px}
@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:0 14px}article{padding:12px}}
</style><main><h1>Language-model mechanisms in motion</h1><p>Follow one computation, predict the change, then test it in the chapter's notebook.</p>
<button id="motion">Pause animations</button><div class="grid">''' + "\n".join(cards) + '''</div></main><script>
let paused=matchMedia('(prefers-reduced-motion: reduce)').matches;
function update(){document.querySelectorAll('img').forEach(i=>i.src=paused?i.dataset.still:i.dataset.gif);document.querySelector('#motion').textContent=paused?'Play animations':'Pause animations'}
document.querySelector('#motion').onclick=()=>{paused=!paused;update()};update();
</script></html>'''
    (HERE / "review.html").write_text(markup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.apply:
        integrate()
    receipt = verify()
    gallery()
    print(f"Verified {len(receipt['placements'])} placements / {len(receipt['chapter_counts'])} chapters; "
          f"{receipt['local_chapter_links_checked']} links; {receipt['new_gif_bytes'] / 1024 / 1024:.2f} MiB new GIFs")


if __name__ == "__main__":
    main()
