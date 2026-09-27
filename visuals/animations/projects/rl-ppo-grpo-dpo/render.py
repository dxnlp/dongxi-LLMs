"""Render the film, check it against the fixture, and sample frames for review.

    python render.py --preview     # 960x540 @ 15 fps  -> media/preview/ (git-ignored)
    python render.py               # 1920x1080 @ 30 fps -> out/ppo-grpo-dpo.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FRAMES = HERE / "media" / "frames"
EXPECTED = ["title", "shared_goal", "families",
            "ppo_sampled", "ppo_scored", "ppo_values", "ppo_advantages", "ppo_clipped", "ppo_done",
            "grpo_no_critic", "grpo_group", "grpo_mean", "grpo_advantages", "grpo_broadcast",
            "grpo_done",
            "dpo_offline", "dpo_pair", "dpo_implicit_reward", "dpo_start", "dpo_trained",
            "dpo_done", "summary"]


def sh(*args, capture=False):
    return subprocess.run(args, check=True, text=True, cwd=HERE,
                          stdout=subprocess.PIPE if capture else None).stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(tl: dict, fx: dict) -> dict:
    ev = {e["name"]: e for e in tl["events"]}
    times = [e["t"] for e in tl["events"]]
    checks = {"event_order": [e["name"] for e in tl["events"]] == EXPECTED,
              "events_monotonic": all(b > a for a, b in zip(times, times[1:]))}
    adv = fx["ppo"]["advantages"]
    checks["ppo_bar_signs_match_fixture"] = ev["ppo_advantages"]["signs"] == [
        int(math.copysign(1, a)) for a in adv]
    checks["ppo_bar_heights_proportional"] = all(
        math.isclose(h, 3.6 * a, abs_tol=2e-3) for h, a in zip(ev["ppo_advantages"]["heights"], adv))
    checks["ppo_ratios_end_on_clip_bounds"] = all(
        math.isclose(r, e, abs_tol=1e-6)
        for r, e in zip(ev["ppo_clipped"]["final_ratio"], fx["ppo"]["ratio_path"][-1]))
    checks["ppo_uses_four_models"] = ev["ppo_done"]["models"] == ["policy", "critic", "reward",
                                                                  "reference"]
    checks["grpo_drops_only_critic"] = ev["grpo_no_critic"]["models"] == ["policy", "reward",
                                                                          "reference"]
    checks["grpo_bar_widths_proportional"] = all(
        math.isclose(w, 0.95 * a, abs_tol=2e-3)
        for w, a in zip(ev["grpo_advantages"]["widths"], fx["grpo"]["advantages"]))
    checks["grpo_mean_matches"] = math.isclose(ev["grpo_mean"]["mean"], fx["grpo"]["mean"])
    checks["dpo_two_models"] = ev["dpo_offline"]["models"] == ["policy", "reference"]
    last = fx["dpo"]["trajectory"][-1]
    checks["dpo_starts_at_zero_margin"] = ev["dpo_start"]["margin"] == 0.0 and math.isclose(
        ev["dpo_start"]["loss"], math.log(2), abs_tol=1e-4)
    checks["dpo_ends_at_fixture_state"] = all(
        math.isclose(ev["dpo_trained"][k], last[k], abs_tol=1e-4) for k in ("margin", "loss", "weight"))
    return checks


def sample_times(tl: dict, duration: float, every: float) -> list[tuple[float, str]]:
    ts = [(e["t"] - 0.05, e["name"]) for e in tl["events"]]
    k = every
    while k < duration:
        ts.append((k, ""))
        k += every
    return sorted((min(max(t, 0.0), duration - 0.05), n) for t, n in ts)


def frames(video: Path, samples, dest: Path, width: int) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (t, _) in enumerate(samples):
        f = dest / f"{i:03d}_{t:06.2f}.png"
        sh("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
           "-frames:v", "1", "-vf", f"scale={width}:-1", str(f))
        out.append(f)
    return out


def contact_sheets(paths, samples, prefix: str, per_sheet: int = 12) -> list[Path]:
    sheets = []
    font = ImageFont.load_default(size=16)
    for s in range(0, len(paths), per_sheet):
        chunk = list(zip(paths, samples))[s:s + per_sheet]
        cols, tw, th = 3, 640, 360
        rows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 28)), "#1f2937")
        draw = ImageDraw.Draw(sheet)
        for i, (p, (t, name)) in enumerate(chunk):
            x, y = i % cols * tw, i // cols * (th + 28)
            sheet.paste(Image.open(p).convert("RGB").resize((tw, th)), (x, y))
            draw.text((x + 8, y + th + 5), f"{t:6.2f}s  {name}", fill="#e5e7eb", font=font)
        path = OUT / f"{prefix}-sheet-{s // per_sheet + 1}.png"
        sheet.save(path)
        sheets.append(path)
    return sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--fps", type=int)
    args = ap.parse_args()
    global OUT
    res, fps = ("960,540", 15) if args.preview else ("1920,1080", 30)
    fps = args.fps or fps
    name = "preview" if args.preview else "ppo-grpo-dpo"
    if args.preview:
        OUT = HERE / "media" / "preview"

    sh(sys.executable, "fixture.py")
    fx = json.loads((HERE / "fixture.json").read_text())
    cmd = [sys.executable, "-m", "manim", "render", "-r", res, "--fps", str(fps),
           "--renderer", "cairo", "--progress_bar", "none", "--disable_caching",
           "--verbosity", "WARNING", "--media_dir", str(HERE / "media"), "-o", name,
           "scene.py", "FeedbackFamilies"]
    sh(*cmd)
    found = list((HERE / "media" / "videos").glob(f"**/{name}.mp4"))
    assert len(found) == 1, found
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{name}.mp4"
    shutil.copy2(found[0], video)

    tl = json.loads((HERE / "timeline.json").read_text())
    checks = validate(tl, fx)
    info = json.loads(sh("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                         str(video), capture=True))
    st = info["streams"][0]
    w, h = map(int, res.split(","))
    checks["video_dimensions"] = (st["width"], st["height"]) == (w, h)
    checks["video_h264_yuv420p"] = st["codec_name"] == "h264" and st["pix_fmt"] == "yuv420p"
    duration = float(info["format"]["duration"])

    samples = sample_times(tl, duration, every=2.0 if args.preview else 4.0)
    shutil.rmtree(FRAMES / name, ignore_errors=True)
    paths = frames(video, samples, FRAMES / name, 960)
    sheets = contact_sheets(paths, samples, name)
    poster = OUT / f"{name}-poster.png"
    summary_t = next(e["t"] for e in tl["events"] if e["name"] == "summary")
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", f"{summary_t - 0.05:.3f}", "-i", str(video),
       "-frames:v", "1", str(poster))

    report = dict(
        video=str(video.relative_to(HERE)), bytes=video.stat().st_size, sha256=sha(video),
        events=tl["events"],
        duration_seconds=round(duration, 3), resolution=[st["width"], st["height"]],
        fps=st["r_frame_rate"], codec=st["codec_name"], render_command=cmd[1:],
        environment=dict(host=platform.node(), platform=platform.platform(),
                         python=platform.python_version(), manim=version("manim"),
                         matplotlib=version("matplotlib"), numpy=version("numpy"),
                         ffmpeg=sh("ffmpeg", "-version", capture=True).splitlines()[0]),
        checks=checks, fixture_checks=fx["checks"],
        sources={p: sha(HERE / p) for p in ("scene.py", "kit.py", "mathsvg.py", "fixture.py",
                                             "render.py")},
        contact_sheets=[str(s.relative_to(HERE)) for s in sheets], poster=str(poster.relative_to(HERE)),
    )
    (OUT / f"{name}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(duration=report["duration_seconds"], checks=checks), indent=2))
    assert all(checks.values()), [k for k, v in checks.items() if not v]


if __name__ == "__main__":
    main()
