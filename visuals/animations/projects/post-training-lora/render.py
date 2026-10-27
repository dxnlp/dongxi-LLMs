"""Render ANIM-POST-003: X-POST-003 (LoRA and QLoRA) GIFs and the static pattern-lens card.

uv run --project visuals/animations python visuals/animations/projects/post-training-lora/render.py [--preview] [--only STEM]
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
ARTICLE = ROOT / 'publications/x-articles/x-post-training-003'
CLIPS = [('01-one-gpu', 'OneGpu'), ('02-pattern', 'Pattern'), ('03-strongest-patterns', 'StrongestPatterns'),
         ('04-low-rank', 'LowRank'), ('05-lora', 'Lora'), ('06-quantize-block', 'QuantizeBlock'),
         ('07-nf4-levels', 'Nf4Levels'), ('08-lowrank-vs-quant', 'LowRankVsQuant'), ('09-qlora', 'Qlora'),
         ('10-memory-7b', 'Memory7b')]
STILL = ('11-pattern-lens', 'PatternLens')
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, capture=False, env=None):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, env=env,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps, still=False, env=None):
    with tempfile.TemporaryDirectory(prefix='post-lora-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene] + (['-s'] if still else [])
        run(cmd, env=env)
        pattern = f'images/**/{stem}*.png' if still else f'videos/**/{stem}.mp4'
        found = list(Path(tmp).glob(pattern)); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='post-lora-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def validate(stem, tl):
    import numpy as np
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 6 < tl['duration_seconds'] < 30
    k = stem[:2]
    c = e['complete']
    if k == '01':
        assert (c['pct'], c['hours'], c['gpu_gb']) == (99.3, 24, 48)
    elif k == '02':
        r1 = np.outer([10, 20, 30], [1, 2, 2, 1]); r2 = r1 + np.outer([1, 0, 2], [0, 0, 5, 5])
        assert c['rank1'] == r1.tolist() and c['rank2'] == r2.tolist()
        assert np.linalg.matrix_rank(r1) == 1 and np.linalg.matrix_rank(r2) == 2
    elif k == '03':
        assert c['ranks'] == [1, 5, 20, 50]
    elif k == '04':
        assert (c['full'], c['low']) == (16777216, 65536)
    elif k == '05':
        assert e['training']['trained'] == ['A', 'B'] and e['training']['frozen'] == ['W'] and e['setup']['B_zero']
    elif k == '06':
        assert c['codes'] == [9, 2, 14, 6, 7, 12, 0, 11], c['codes']
    elif k == '07':
        assert c['levels'] == 16
    elif k == '08':
        assert (c['low_rank_kept'], c['quant_kept']) == (8, 30)
    elif k == '09':
        assert e['backward']['trained'] == ['A', 'B']
    else:
        assert c['gb'] == [100, 15, 5]


def gif(scene, stem, out):
    """Encode the GIF from a lossless render at GIF size and rate.

    Encoding from the H.264 MP4 let compression noise change nearly every pixel near
    text in every frame (one GIF reached 14.9 MB). Lossless frames keep static areas
    byte-identical, so the GIF stores only what actually moves.
    """
    w, h = GIF_W, GIF_W * 9 // 16
    with tempfile.TemporaryDirectory(prefix='post-lora-mov-') as tmp:
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
    report = dict(task='ANIM-POST-003', preview=args.preview, clips=[], stills=[])
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
        for lang in ['en']:  # English-only figures for both article versions
            env = dict(os.environ)
            png, cmd = manim(STILL[1], STILL[0], '1920,1080', '30', still=True, env=env)
            dest = assets; dest.mkdir(parents=True, exist_ok=True)
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
            '01 shows QLoRA abstract claims (65B on one 48 GB GPU, 24 h, 99.3% of ChatGPT on Vicuna); the ~1 TB full fine-tuning bar is a byte-count estimate and not to scale.',
            '03 uses the public-domain Grace Hopper portrait from matplotlib sample data; bar heights are square-rooted singular values.',
            '06 uses the learning note worked example (8 weights, absmax 0.062, NF4 codes); 07 histogram is synthetic Gaussian data.',
            '08 spectra are illustrative, not measured; 10 bars are byte-count estimates (weights, gradients, optimizer; no activations).',
            '05 and 09 are schematics of LoRA and QLoRA data flow.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _ in CLIPS if s in keep]
            if not report['stills']: report['stills'] = old.get('stills', [])
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
