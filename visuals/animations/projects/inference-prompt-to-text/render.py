"""Render CAND-ANIM-040: inference from a prompt to text, seven GIFs.

uv run --project visuals/animations python visuals/animations/projects/inference-prompt-to-text/render.py [--preview] [--only STEM]
"""
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

from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
CLIPS = [('01-overview', 'Overview'), ('02-chat-template', 'ChatTemplate'), ('03-tokenize', 'Tokenize'),
         ('04-prefill', 'Prefill'), ('05-pick-token', 'PickToken'), ('06-decode-loop', 'DecodeLoop'),
         ('07-detokenize', 'Detokenize')]
PROMPT_IDS = [151644, 872, 198, 3872, 5264, 15469, 1787, 30, 151645, 198, 151644, 77091, 198]
ANSWER_IDS = [5800, 398, 13, 4329, 14324, 525, 1787, 13, 151645]
REPLY = 'Partly. Some weights are open.'
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps):
    with tempfile.TemporaryDirectory(prefix='inference-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene]
        run(cmd)
        found = list(Path(tmp).glob(f'videos/**/{stem}.mp4')); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='inference-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def validate(stem, tl):
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 6 < tl['duration_seconds'] < 30
    c = e['complete']
    k = stem[:2]
    if k == '01':
        assert c['text'] == REPLY and c['forward_passes'] == 9 and c['prompt_tokens'] == 13
    elif k == '02':
        assert c['tokens'] == 13 and e['wrapped']['user_tokens'] == 5
    elif k == '03':
        assert c['ids'] == PROMPT_IDS and c['vocabulary'] == 151669
    elif k == '04':
        assert c['positions'] == 13 and c['layers'] == 28 and c['logits'] == 151936
    elif k == '05':
        assert c['keep'] == 3 and c['greedy'] == ['Part'] * 3 and c['sampled'] == ['Part', 'Open', 'Part', 'No', 'Part']
        assert abs(sum(e['top_p']['p'].values()) - 1) < 2e-3
    elif k == '06':
        assert c['text'] == REPLY and c['passes'] == 8 and c['cached'] == 21
    elif k == '07':
        assert c['text'] == REPLY and e['ids']['ids'] == ANSWER_IDS


def gif(scene, stem, out):
    """Encode the GIF from a lossless render (see post-training-sft/render.py for why)."""
    w, h = GIF_W, GIF_W * 9 // 16
    with tempfile.TemporaryDirectory(prefix='inference-mov-') as tmp:
        run([sys.executable, '-m', 'manim', 'render', '-t', '-r', f'{w},{h}', '--fps', str(GIF_FPS),
             '--renderer', 'cairo', '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING',
             '--media_dir', tmp, '-o', stem, str(P / 'scenes.py'), scene])
        found = list(Path(tmp).glob(f'videos/**/{stem}.mov')); assert len(found) == 1, found
        run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(found[0]), '-filter_complex',
             f'color=white:s={w}x{h}:r={GIF_FPS}[bg];[bg][0:v]overlay=shortest=1,format=rgb24,split[a][b];'
             '[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];'
             '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', str(out)])
    with Image.open(out) as im:
        assert im.size == (w, h) and im.info['loop'] == 0
        hashes = set(); dur = 0
        for i in range(im.n_frames):
            im.seek(i); dur += im.info['duration']
            hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    size = out.stat().st_size
    assert len(hashes) > 30 and size < 5_000_000, (len(hashes), size)
    return dict(width=w, height=h, fps=GIF_FPS, frames=im.n_frames, duration_ms=dur,
                unique_frames=len(hashes), loop=0, bytes=size, source='lossless qtrle render composited on white')


def frames_at(video, seconds, tmp):
    out = []
    for i, s in enumerate(seconds):
        f = Path(tmp) / f'{i}.png'
        run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(max(0, s)), '-i', str(video),
             '-frames:v', '1', '-update', '1', str(f)])
        out.append(Image.open(f).convert('RGB'))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--preview', action='store_true')
    ap.add_argument('--only', action='append'); args = ap.parse_args()
    size, fps = ('960,540', '15') if args.preview else ('1920,1080', '30')
    report = dict(task='CAND-ANIM-040', preview=args.preview, clips=[])
    for stem, scene in CLIPS:
        if args.only and stem not in args.only:
            continue
        out = P / 'rendered' / stem; out.mkdir(parents=True, exist_ok=True)
        tmpvid, cmd = manim(scene, stem, size, fps)
        video = out / ('preview.mp4' if args.preview else stem + '.mp4'); shutil.move(tmpvid, video)
        tl = json.loads((P / (stem + '-timeline.json')).read_text()); validate(stem, tl)
        info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], True))
        st = info['streams'][0]
        assert (st['width'], st['height']) == tuple(map(int, size.split(','))) and st['codec_name'] == 'h264'
        prefix = 'preview-' if args.preview else ''
        ev = tl['events']; samples = []
        for i, x in enumerate(ev):
            if i and x['seconds'] - ev[i - 1]['seconds'] > .5:
                samples.append(dict(seconds=(x['seconds'] + ev[i - 1]['seconds']) / 2, name='motion'))
            samples.append(x)
        with tempfile.TemporaryDirectory() as tmp:
            fr = frames_at(video, [s['seconds'] + .25 for s in samples], tmp)
            poster = fr[-1]
            poster.save(out / (prefix + 'poster.png'))
            sheet = Image.new('RGB', (1440, 310 * ((len(fr) + 2) // 3)), '#e2e8f0'); d = ImageDraw.Draw(sheet)
            for i, (f, s) in enumerate(zip(fr, samples)):
                x, y = i % 3 * 480, i // 3 * 310
                sheet.paste(f.resize((480, 270)), (x, y))
                d.text((x + 8, y + 279), f"{s['seconds']:.2f}s · {s['name']}", font=ImageFont.load_default(size=14), fill='#111827')
            sheet.save(out / (prefix + 'contact-sheet.png'))
        entry = dict(id=stem, scene=scene, render_command=cmd, duration_seconds=float(info['format']['duration']),
                     video=dict(width=st['width'], height=st['height'], codec=st['codec_name'], fps=st['r_frame_rate']),
                     timeline=tl)
        if not args.preview:
            g = out / (stem + '.gif'); entry['gif'] = gif(scene, stem, g)
            entry['outputs'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
                                for p in [g, out / 'poster.png', out / 'contact-sheet.png']]
        report['clips'].append(entry)
        print(json.dumps(dict(clip=stem, duration=entry['duration_seconds'], gif=entry.get('gif'))), flush=True)
    if not args.preview:
        report['environment'] = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                                     manim=version('manim'), pillow=version('pillow'),
                                     ffmpeg=run(['ffmpeg', '-version'], True).splitlines()[0],
                                     base_commit=run(['git', 'rev-parse', 'HEAD'], True).strip())
        report['sources'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
                             [P / 'scenes.py', P / 'render.py', P.parents[1] / 'manim_style.py', P.parents[1] / 'uv.lock']]
        report['limitations'] = [
            'Token strings and IDs are real (Qwen3 tokenizer, Qwen/Qwen3-0.6B); the answer text is chosen, not generated.',
            'The template is the ChatML core; Qwen3 with thinking disabled also adds an empty think block, left out here.',
            'Logits, probabilities and hidden-vector shades are illustrative; softmax, temperature and top-p are exact on them.',
            'Three layers stand for 28; frames show order of operations, not measured time.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _ in CLIPS if s in keep]
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
