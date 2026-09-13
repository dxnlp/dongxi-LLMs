"""Render, inspect and export the approved article loop without touching other films."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageChops

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
OUT = P/'rendered'


def run(args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true', help='key-frame/geometry review before final export')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = 'preview' if args.preview else 'prefill-decode'
    size, fps = ('960,540', '15') if args.preview else ('1920,1080', '30')
    video = OUT/(stem+'.mp4')
    with tempfile.TemporaryDirectory(prefix='kv-prefill-render-') as tmp:
        command = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps,
                   '--renderer', 'cairo', '--progress_bar', 'none', '--disable_caching',
                   '--verbosity', 'WARNING', '--media_dir', tmp, '-o', stem,
                   str(P/'scenes.py'), 'PrefillDecode']
        run(command)
        found = list(Path(tmp).glob(f'videos/**/{stem}.mp4'))
        assert len(found) == 1
        shutil.copy2(found[0], video)
    timeline = json.loads((P/'timeline.json').read_text())
    events = timeline['events']
    assert [e['cached_positions'] for e in events] == [0, 3, 3, 3, 3, 4, 4, 4]
    assert [e['name'] for e in events] == ['prompt_input', 'prompt_kv_ready', 'prefill_read',
        'blue_selected', 'blue_fed_back', 'new_kv_appended', 'decode_read', 'period_selected']
    assert all(not e['selected_token_cached'] for e in events if 'selected_token_cached' in e)
    assert events[2]['visible_positions'] == [1, 2, 3] and events[6]['visible_positions'] == [1, 2, 3, 4]
    info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], True))
    stream = info['streams'][0]
    assert (stream['width'], stream['height']) == tuple(map(int, size.split(',')))
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    with tempfile.TemporaryDirectory(prefix='kv-prefill-frames-') as tmp:
        frames = []
        for i, e in enumerate(events):
            f = Path(tmp)/f'{i}.png'
            run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(max(0, e['seconds']-.03)),
                 '-i', str(video), '-frames:v', '1', '-update', '1', str(f)])
            frames.append(Image.open(f).convert('RGB'))
        frames[-1].save(OUT/(stem+'-still.png'))
        frames[-1].resize((960, 540)).save(OUT/(stem+'-still-half.png'))
        sheet = Image.new('RGB', (1920, 1200), '#e2e8f0')
        d = ImageDraw.Draw(sheet)
        for i, (im, e) in enumerate(zip(frames, events)):
            x, y = i%2*960, i//2*300
            sheet.paste(im.resize((480, 270)), (x, y))
            d.text((x+490, y+110), f"{e['seconds']:.2f}s\n{e['name']}\ncache: {e['cached_positions']}",
                   font=ImageFont.load_default(size=20), fill='#111827')
        sheet.save(OUT/(stem+'-contact-sheet.png'))
    outputs = [video, OUT/(stem+'-still.png'), OUT/(stem+'-still-half.png'), OUT/(stem+'-contact-sheet.png')]
    gif_info = None
    if not args.preview:
        gif = OUT/'prefill-decode.gif'
        run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video), '-filter_complex',
             'fps=15,scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer',
             '-loop', '0', str(gif)])
        with Image.open(gif) as im:
            assert im.size == (960, 540) and im.info['loop'] == 0 and im.n_frames > 30
            duration = 0
            hashes = set()
            for i in range(im.n_frames):
                im.seek(i)
                duration += im.info['duration']
                hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
            assert len(hashes) > 30 and gif.stat().st_size < 5_000_000
            gif_info = dict(width=960, height=540, frames=im.n_frames, duration_ms=duration,
                            loop=0, unique_frames=len(hashes), bytes=gif.stat().st_size)
        outputs.append(gif)
    metadata = dict(preview=args.preview, base_commit=run(['git','rev-parse','HEAD'],True).strip(),
        environment=dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                         manim=version('manim'), pillow=version('pillow'),
                         ffmpeg=run(['ffmpeg','-version'],True).splitlines()[0]),
        render_command=command, timeline=timeline, duration_seconds=float(info['format']['duration']),
        video=dict(width=stream['width'],height=stream['height'],codec=stream['codec_name'],fps=stream['r_frame_rate']),
        gif=gif_info, checks=dict(event_order=True, past_geometry_unchanged=True, media_valid=True),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in
                 [P/'scenes.py',P/'render.py',P.parents[1]/'manim_style.py',P.parents[1]/'uv.lock']],
        outputs=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in outputs],
        limitations=['Schematic token pieces; no tokenizer/model run.',
                     'Only final prompt query shown; prefill computes all prompt queries.',
                     'Vector glyphs and paths have no numeric activation/weight meaning.',
                     'Model tail summarizes remaining layers and vocabulary projection.',
                     'End fade is an editorial loop reset, not cache eviction.'])
    (P/('preview-metadata.json' if args.preview else 'metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(video=str(video),gif=gif_info,duration=metadata['duration_seconds']),indent=2),flush=True)


if __name__ == '__main__':
    main()
