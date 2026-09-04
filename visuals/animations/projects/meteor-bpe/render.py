"""Rebuild the measured trace, Manim media, and local ANIM-BPE-002 review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
import manim
import manimpango

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[3]
ANIM = PROJECT.parents[1]
OUT = PROJECT / "rendered"


def run(command: list[str], *, capture: bool = False) -> str:
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout or ""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_review(video: Path, still: Path) -> None:
    page = f'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>流星雨 · BPE 动画审阅</title>
<style>
html {{ background:#fff; color:#111827; }}
body {{ margin:0 auto; width:min(1160px,94%); padding:40px 0 70px;
  font:20px/1.7 Arial,"Songti SC",serif; }}
h1 {{ font-size:36px; font-weight:400; margin:0 0 8px; }}
p {{ color:#64748b; margin:10px 0 20px; }}
video {{ display:block; width:100%; aspect-ratio:16/9; background:white;
  border:1px solid #e2e8f0; border-radius:12px; }}
.note {{ font-size:16px; }}
a {{ color:#2563eb; text-underline-offset:4px; }}
details {{ margin-top:28px; font-size:17px; }}
li {{ margin:10px 0; }}
code {{ font-family:Arial,"Songti SC",serif; color:#047857; }}
</style>
<h1>流星雨：从语料到 Token</h1>
<p>字符级 BPE · 两轮实际计数 · 1–5 为教学编号</p>
<video controls playsinline preload="metadata" poster="rendered/{still.name}">
  <source src="rendered/{video.name}" type="video/mp4">
</video>
<p class="note">本地审阅版，尚未发布。<a href="rendered/{video.name}">打开 MP4</a> ·
<a href="rendered/meteor-bpe.gif">GIF 预览</a> · <a href="trace.json">完整计数与词表</a></p>
<details><summary>实验边界与原语料</summary>
<p>流星蝴蝶剑，带你去看流星雨，流水，星辰大海，雨一直下。五条语料各出现一次。</p>
<ul>
<li><code>流 + 星</code> 出现两次，是第一轮唯一最高频相邻对。</li>
<li>第二轮 15 个候选都只出现一次。本例预先声明并列时优先选择
<code>流星 + 雨</code>；这条合并无法由频次单独确定。</li>
<li>完整教学词表有 19 项。动画展示其中五项：流 1、星 2、雨 3、流星 4、流星雨 5。
其他字符占用 6–19，基础字符始终保留。</li>
<li>这是字符级 BPE 的可执行教学示例；生产 Tokenizer 的基础单位、边界和编号可能不同。</li>
<li>本片展示分词与编号，不模拟神经网络训练。原文关于人的体验与 AI 的“苍白感”是文学判断，
无法仅凭 Tokenization 得到验证。</li>
</ul></details>
</html>'''
    (PROJECT / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    available_fonts = set(manimpango.list_fonts())
    assert {"Arial", "Songti SC"}.issubset(available_fonts), "Required fonts are missing"
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from dongxi_llms.meteor_bpe import build_trace

    (PROJECT / "trace.json").write_text(json.dumps(build_trace(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    size, fps = ("960,540", "15") if args.preview else ("1920,1080", "30")
    video = OUT / ("preview.mp4" if args.preview else "meteor-bpe.mp4")
    with tempfile.TemporaryDirectory(prefix="meteor-bpe-media-") as temporary:
        command = [sys.executable, "-m", "manim", "render", "-r", size, "--fps", fps,
                   "--renderer", "cairo", "--progress_bar", "none", "--disable_caching",
                   "--verbosity", "WARNING",
                   "--media_dir", temporary, "-o", "meteor-bpe", str(PROJECT / "scene.py"), "MeteorBPE"]
        run(command)
        candidates = list(Path(temporary).glob("videos/**/meteor-bpe.mp4"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one rendered movie, found {candidates}")
        shutil.copy2(candidates[0], video)

    timeline = json.loads((PROJECT / "timeline.json").read_text(encoding="utf-8"))
    probe = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture=True))
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    expected_w, expected_h = map(int, size.split(","))
    assert (stream["width"], stream["height"]) == (expected_w, expected_h)
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    still = OUT / "meteor-bpe-still.png"
    checkpoint_frames = []
    with tempfile.TemporaryDirectory(prefix="meteor-bpe-frames-") as temporary:
        for i, checkpoint in enumerate(timeline["checkpoints"]):
            frame = Path(temporary) / f"{i:02d}.png"
            seconds = max(0.0, checkpoint["seconds"] - .20)
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(seconds), "-i", str(video),
                 "-frames:v", "1", "-update", "1", str(frame)])
            checkpoint_frames.append((checkpoint, Image.open(frame).convert("RGB")))
        final = checkpoint_frames[-1][1]
        final.save(still)
        # A diagnostic contact sheet is not a reader-facing explanatory figure.
        thumb_w, thumb_h, caption_h = 480, 270, 30
        columns = 3
        rows = (len(checkpoint_frames) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + caption_h)), "#e2e8f0")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        for i, (checkpoint, frame) in enumerate(checkpoint_frames):
            x, y = (i % columns) * thumb_w, (i // columns) * (thumb_h + caption_h)
            sheet.paste(frame.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 10, y + thumb_h + 6), f"{checkpoint['seconds']:.1f}s  {checkpoint['name']}", fill="#111827", font=font)
        sheet.save(OUT / "contact-sheet.png")
        final.resize((960, 540), Image.Resampling.LANCZOS).save(OUT / "meteor-bpe-still-half.png")

    gif = OUT / "meteor-bpe.gif"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex",
         "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
         "-loop", "0", str(gif)])
    write_review(video, still)
    source_files = [PROJECT / "scene.py", PROJECT / "render.py", PROJECT / "trace.json",
                    ROOT / "src/dongxi_llms/meteor_bpe.py", ROOT / "src/dongxi_llms/tiny_bpe.py",
                    ANIM / "manim_style.py", ANIM / "STYLE_GUIDE.md", ANIM / "uv.lock"]
    outputs = [video, gif, still, OUT / "contact-sheet.png", OUT / "meteor-bpe-still-half.png"]
    metadata = {
        "task_id": "ANIM-BPE-002", "status": "preview" if args.preview else "review",
        "rendered_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": run(["git", "rev-parse", "HEAD"], capture=True).strip(),
        "branch": run(["git", "branch", "--show-current"], capture=True).strip(),
        "render_command": "uv run --project visuals/animations python visuals/animations/projects/meteor-bpe/render.py" + (" --preview" if args.preview else ""),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "manim": manim.__version__, "renderer": "cairo",
                        "manimpango": version("manimpango"), "pycairo": version("pycairo"),
                        "pillow": version("pillow"), "numpy": version("numpy"),
                        "ffmpeg": run(["ffmpeg", "-version"], capture=True).splitlines()[0],
                        "fonts": {"chinese": "Songti SC", "english": "Arial"}},
        "video": {"width": stream["width"], "height": stream["height"], "codec": stream["codec_name"],
                  "pixel_format": stream["pix_fmt"], "frame_rate": stream["r_frame_rate"],
                  "duration_seconds": float(probe["format"]["duration"])},
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in source_files],
        "outputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size} for p in outputs],
        "timeline": timeline,
        "limitations": build_trace()["limitations"],
    }
    (PROJECT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Review ready: {PROJECT / 'review.html'}", flush=True)


if __name__ == "__main__":
    main()
