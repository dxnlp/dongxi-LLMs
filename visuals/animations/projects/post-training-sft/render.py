"""Render ANIM-POST-002: six X-POST-002 article loops (MP4 + high-quality GIF).

uv run --project visuals/animations python visuals/animations/projects/post-training-sft/render.py [--preview] [--only STEM]
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
ARTICLE = ROOT / 'publications/x-articles/x-post-training-002'
CLIPS = [('01-alpaca-recipe', 'AlpacaRecipe'), ('02-lima', 'Lima'), ('03-flan', 'Flan'), ('04-recipe', 'Recipe'),
         ('05-three-forms', 'ThreeForms'), ('07-what-gets-loss', 'WhatGetsLoss'),
         ('08-masked-not-untouched', 'MaskedNotUntouched'), ('09-one-position-shift', 'OnePositionShift'),
         ('10-token-share', 'TokenShare')]
STILL = ('06-three-forms-summary', 'ThreeFormsSummary')
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, capture=False, env=None):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, env=env,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps, still=False, env=None):
    with tempfile.TemporaryDirectory(prefix='post-sft-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene] + (['-s'] if still else [])
        run(cmd, env=env)
        pattern = f'images/**/{stem}*.png' if still else f'videos/**/{stem}.mp4'
        found = list(Path(tmp).glob(pattern)); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='post-sft-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def validate(stem, tl):
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 6 < tl['duration_seconds'] < 30
    k = stem[:2]
    if k == '01':
        assert e['stats']['examples'] == 52000 and e['stats']['blind'] == '90 vs 89'
    elif k == '02':
        assert (e['complete']['examples'], e['complete']['pct']) == (1000, 43)
    elif k == '03':
        assert (e['complete']['wins'], e['complete']['total']) == (20, 25)
    elif k == '04':
        assert e['complete']['steps'] == ['Collect', 'Format', 'Mask', 'Train', 'Evaluate']
    elif k == '05':
        assert e['mask']['graded'] == ['Elated.', '<end>']
    elif k == '07':
        assert e['mask']['graded'] == ['Elated.', '<end>', 'Joyful.', '<end>']
    elif k == '08':
        assert e['gradient_reaches_prompt']['prompt_gradient_nonzero']
    elif k == '09':
        assert (e['shifted']['alignment'], e['unshifted']['alignment'], e['unshifted']['error_raised']) == ('next token', 'same token', False)
    else:
        assert e['two_examples']['tokens'] == [2, 500] and e['shares']['ratio'] == 250


def gif(scene, stem, out):
    """Encode the GIF from a lossless render at GIF size and rate.

    Encoding from the H.264 MP4 let compression noise change nearly every pixel near
    text in every frame (one GIF reached 14.9 MB). Lossless frames keep static areas
    byte-identical, so the GIF stores only what actually moves.
    """
    w, h = GIF_W, GIF_W * 9 // 16
    with tempfile.TemporaryDirectory(prefix='post-sft-mov-') as tmp:
        # -t: transparent background, which Manim writes losslessly (qtrle .mov) with exact timing.
        run([sys.executable, '-m', 'manim', 'render', '-t', '-r', f'{w},{h}', '--fps', str(GIF_FPS),
             '--renderer', 'cairo', '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING',
             '--media_dir', tmp, '-o', stem, str(P / 'scenes.py'), scene])
        found = list(Path(tmp).glob(f'videos/**/{stem}.mov')); assert len(found) == 1, found
        # Composite onto the white canvas; one global palette, no dithering.
        run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(found[0]), '-filter_complex',
             f'color=white:s={w}x{h}:r={GIF_FPS}[bg];[bg][0:v]overlay=shortest=1,format=rgb24,split[a][b];'
             '[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];'
             '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', str(out)])
    with Image.open(out) as im:
        assert im.size == (GIF_W, GIF_W * 9 // 16) and im.info['loop'] == 0
        hashes = set(); dur = 0
        for i in range(im.n_frames):
            im.seek(i); dur += im.info['duration']
            hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    size = out.stat().st_size
    assert len(hashes) > 30 and size < 8_000_000, (len(hashes), size)
    return dict(width=im.size[0], height=im.size[1], fps=GIF_FPS, frames=im.n_frames,
                duration_ms=dur, unique_frames=len(hashes), loop=0, bytes=size, source='lossless qtrle render composited on white')


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
    assets = ARTICLE / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    report = dict(task='ANIM-POST-002', preview=args.preview, clips=[], stills=[])
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
            g = assets / (stem + '.gif'); entry['gif'] = gif(scene, stem, g)  # GIF lives only with the article
            poster.save(assets / (stem + '-poster.png'))
            entry['outputs'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
                                for p in [g, assets / (stem + '-poster.png'), out / 'contact-sheet.png']]
        report['clips'].append(entry)
        print(json.dumps(dict(clip=stem, duration=entry['duration_seconds'], gif=entry.get('gif'))), flush=True)
    if not args.only or STILL[0] in args.only:
        report['stills'] = []
        for lang in ['en']:  # English-only figures for both article versions (learner, 2026-10-22)
            env = dict(os.environ, SUMMARY_LANG=lang)
            png, cmd = manim(STILL[1], f'{STILL[0]}-{lang}', '1920,1080', '30', still=True, env=env)
            dest = assets if lang == 'en' else ARTICLE / 'zh/assets'; dest.mkdir(parents=True, exist_ok=True)
            final = dest / f'{STILL[0]}.png'
            shutil.copy2(png, final)
            with Image.open(final) as im:
                assert im.size == (1920, 1080)
            report['stills'].append(dict(lang=lang, path=str(final.relative_to(ROOT)), sha256=sha(final)))
            print(json.dumps(dict(still=str(final.relative_to(ROOT)))), flush=True)
    if not args.preview:
        report['environment'] = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                                     manim=version('manim'), pillow=version('pillow'),
                                     ffmpeg=run(['ffmpeg', '-version'], True).splitlines()[0],
                                     base_commit=run(['git', 'rev-parse', 'HEAD'], True).strip())
        report['sources'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
                             [P / 'scenes.py', P / 'render.py', P.parents[1] / 'manim_style.py', P.parents[1] / 'uv.lock']]
        report['limitations'] = [
            '01 shows the Stanford Alpaca team\'s published numbers; 02 LIMA (1,000 examples, 43% vs GPT-4) and 03 FLAN (60+ tasks, 20 of 25 vs zero-shot GPT-3 175B) show the authors\' published results; which 5 of 25 FLAN tiles stay grey is illustrative.',
            '04, 05, 07 use simplified template markers (<user>, <assistant>, <end>) standing for Qwen\'s <|im_start|>role / <|im_end|>; the static summary uses the real markers.',
            '08 and 09 are schematic.',
            '10 token counts are illustrative.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _ in CLIPS if s in keep]
            if not report['stills']: report['stills'] = old.get('stills', [])
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
