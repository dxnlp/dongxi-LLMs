"""Render the beginner film, check it against the fixture, and sample frames for review.

    python render.py --check              # no video: run the story, bounds checks, fixture checks
    python render.py --preview            # 960x540 @ 15 fps  -> media/preview/ (git-ignored)
    python render.py --preview --only ppo # one section, no validation
    python render.py                      # 1920x1080 @ 30 fps -> out/ppo-grpo-dpo-beginner.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
SCENE = "BeginnerFeedback"
EXPECTED = ["title", "words", "wheel", "spun", "mistake", "soup", "judge", "hacking", "leash",
            "pulled_back", "families",
            "ppo_scored", "ppo_guesses", "ppo_credit", "ppo_clip", "ppo_rounds", "ppo_done",
            "grpo_tries", "grpo_no_coach", "grpo_scored", "grpo_credit", "grpo_shared",
            "grpo_done",
            "dpo_eye_test", "dpo_pair", "dpo_why", "dpo_start", "dpo_trained", "dpo_leash",
            "dpo_done", "summary"]


def sh(*args, capture=False, env=None):
    return subprocess.run(args, check=True, text=True, cwd=HERE, env=env,
                          stdout=subprocess.PIPE if capture else None).stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b, tol=1e-4):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b)) and len(a) == len(b)


def validate(tl: dict, fx: dict) -> dict:
    ev = {e["name"]: e for e in tl["events"]}
    times = [e["t"] for e in tl["events"]]
    p, g, d = fx["ppo"], fx["grpo"], fx["dpo"]["path"]
    guesses = p["coach_guesses"] + [p["judge_score"]]
    checks = {
        "event_order": [e["name"] for e in tl["events"]] == EXPECTED,
        "events_monotonic": all(b > a for a, b in zip(times, times[1:])),
        "wheel_matches_fixture": close(ev["wheel"]["shares"], fx["wheel"]["shares"]),
        "spin_lands_on_biggest_slice": ev["spun"]["landed"] == fx["wheel"]["words"][
            max(range(4), key=lambda i: fx["wheel"]["shares"][i])],
        "judge_fooled_by_length": ev["hacking"]["scores"] == sorted(ev["hacking"]["scores"]),
        "ppo_score_matches": ev["ppo_scored"]["score"] == p["judge_score"],
        "ppo_guess_line_matches": close(ev["ppo_guesses"]["guesses"], guesses),
        "ppo_credit_is_change_in_guess": close(
            ev["ppo_credit"]["credits"], [b - a for a, b in zip(guesses, guesses[1:])]),
        "ppo_clip_stops_at_fence": math.isclose(
            ev["ppo_clip"]["share"], fx["wheel"]["shares"][0] * (1 - fx["wheel"]["eps"])),
        "ppo_rounds_end_on_fixture": close(ev["ppo_rounds"]["final"],
                                           fx["wheel"]["round_shares"][-1]),
        "ppo_four_models": ev["ppo_done"]["models"] == ["student", "coach", "judge", "original"],
        "grpo_drops_only_coach": ev["grpo_no_coach"]["models"] == ev["grpo_done"]["models"] == [
            "student", "judge", "original"],
        "grpo_scores_match_key": ev["grpo_scored"]["scores"] == g["scores"],
        "grpo_credit_is_score_minus_mean": math.isclose(
            ev["grpo_credit"]["mean"], sum(g["scores"]) / len(g["scores"])) and close(
            ev["grpo_credit"]["credits"], [s - ev["grpo_credit"]["mean"] for s in g["scores"]]),
        "dpo_two_models": ev["dpo_done"]["models"] == ev["dpo_why"]["models"] == [
            "student", "original"],
        "dpo_starts_even": ev["dpo_start"]["margin"] == 0.0 and math.isclose(
            ev["dpo_start"]["push"], 0.5),
        "dpo_ends_on_fixture": math.isclose(ev["dpo_trained"]["ratio_a"], d[-1]["ratio_a"],
                                            abs_tol=1e-5)
        and math.isclose(ev["dpo_trained"]["ratio_b"], d[-1]["ratio_b"], abs_tol=1e-5)
        and math.isclose(ev["dpo_trained"]["push"], d[-1]["push"], abs_tol=1e-5),
        "dpo_push_is_one_over_one_plus_lead": math.isclose(
            ev["dpo_trained"]["lead"], ev["dpo_trained"]["ratio_a"] / ev["dpo_trained"]["ratio_b"],
            rel_tol=1e-5) and math.isclose(ev["dpo_trained"]["push"],
                                           1 / (1 + ev["dpo_trained"]["lead"]), abs_tol=1e-5),
    }
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
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--fps", type=int)
    args = ap.parse_args()
    global OUT
    res, fps = ("960,540", 15) if args.preview or args.check else ("1920,1080", 30)
    fps = args.fps or fps
    name = "preview" if args.preview else "ppo-grpo-dpo-beginner"
    if args.only:
        name = f"preview-{args.only.replace(',', '-')}"
    if args.preview or args.only or args.check:
        OUT = HERE / "media" / "preview"

    sh(sys.executable, "fixture.py")
    fx = json.loads((HERE / "fixture.json").read_text())
    env = dict(os.environ)
    if args.only:
        env["ONLY"] = args.only
    cmd = [sys.executable, "-m", "manim", "render", "-r", res, "--fps", str(fps),
           "--renderer", "cairo", "--progress_bar", "none", "--disable_caching",
           "--verbosity", "WARNING", "--media_dir", str(HERE / "media"), "-o", name,
           "scene.py", SCENE]
    if args.check:
        cmd.insert(4, "-s")
    sh(*cmd, env=env)
    if args.only:
        found = list((HERE / "media" / "videos").glob(f"**/{name}.mp4"))
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found[0], OUT / f"{name}.mp4")
        print(OUT / f"{name}.mp4")
        return

    tl = json.loads((HERE / "media" / "timeline.json").read_text())
    checks = validate(tl, fx)
    if args.check:
        print(json.dumps(dict(duration=tl["duration"], checks=checks), indent=2))
        assert all(checks.values()), [k for k, v in checks.items() if not v]
        return

    found = list((HERE / "media" / "videos").glob(f"**/{name}.mp4"))
    assert len(found) == 1, found
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"{name}.mp4"
    shutil.copy2(found[0], video)
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
                         numpy=version("numpy"),
                         ffmpeg=sh("ffmpeg", "-version", capture=True).splitlines()[0]),
        checks=checks, fixture_checks=fx["checks"],
        sources={p: sha(HERE / p) for p in ("scene.py", "kit.py", "fixture.py", "render.py")},
        contact_sheets=[str(s.relative_to(HERE)) for s in sheets],
        poster=str(poster.relative_to(HERE)),
    )
    (OUT / f"{name}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(duration=report["duration_seconds"], checks=checks), indent=2))
    assert all(checks.values()), [k for k, v in checks.items() if not v]


if __name__ == "__main__":
    main()
