"""Render the KL article GIF set: twelve clips, each GIF and MP4 with the DONGXI LABS mark.

uv run --project visuals/animations python visuals/animations/projects/post-training-kl/render.py [--preview] [--only STEM]
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
sys.path.insert(0, str(P.parents[1]))
from brand import write_mark, gif_filter, brand_mp4  # noqa: E402

CLIPS = [('01-two-splits', 'TwoSplits', 'CAND-ANIM-058'), ('02-why-log', 'WhyLog', 'CAND-ANIM-059'),
         ('03-kl-term-by-term', 'TermByTerm', 'CAND-ANIM-058'), ('04-evidence', 'Evidence', 'CAND-ANIM-065'),
         ('05-never-negative', 'NeverNegative', 'CAND-ANIM-060'), ('06-no-direction', 'NoDirection', 'CAND-ANIM-061'),
         ('07-rare-moves', 'RareMoves', 'CAND-ANIM-054'), ('08-which-way', 'WhichWay', 'CAND-ANIM-055'),
         ('09-from-samples', 'FromSamples', 'CAND-ANIM-063'), ('10-tokens', 'Tokens', 'CAND-ANIM-037'),
         ('11-reading-a-run', 'ReadingRun', 'CAND-ANIM-064'), ('12-implicit-reward', 'ImplicitReward', 'CAND-ANIM-066')]
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(x): return 1 / (1 + math.exp(-x))


def run(args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps):
    with tempfile.TemporaryDirectory(prefix='kl-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene]
        run(cmd)
        found = list(Path(tmp).glob(f'videos/**/{stem}.mp4')); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='kl-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def close(a, b, tol=2e-3):
    return abs(a - b) < tol


def validate(stem, tl):
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 8 < tl['duration_seconds'] < 36, tl['duration_seconds']
    c, k = e['complete'], stem[:2]
    if k == '01':
        assert c['shares'] == [.30, .25, .15, .22, .08] and close(sum(e['reference']['shares']), 1)
    elif k == '02':
        assert c['double'] == .693 and c['halve'] == -.693 and c['four'] == 1.386 and c['tenth'] == -2.303
    elif k == '03':
        assert c['kl'] == .047 and c['terms'] == [-.086, .133]
    elif k == '04':
        assert c['kl'] == .047 and c['replies'] == 63.7 and c['always'] == 2.12 and c['odds'] == 8.33
    elif k == '05':
        assert e['terms']['total'] == .047 and c['log_ratio'] == -3.689 and c['term'] == -.037 and c['kl'] == .701
    elif k == '06':
        assert e['both_cost']['kl'] == [.047, .076] and c['score'] == [.638, .209]
        assert c['best'] == [1.0, .71, .4, .24, .16, .12]
    elif k == '07':
        assert c['kl'] == [.031, .233] and c['times'] == 7.4 and e['moved']['total_variation'] == [.1, .1]
    elif k == '08':
        assert (c['forward'], c['reverse'], c['drop']) == (.047, .042, .102)
    elif k == '09':
        assert c['values'] == [.096, .061, .048] and c['exact'] == .047 and e['negative_reply']['value'] < 0
    elif k == '10':
        assert c['logs'] == [.223, 0, .182] and c['total'] == .405
    elif k == '11':
        assert c['peak_kl'] == .81 and c['peak_gold'] == 1.1 and c['peak_reward'] == 1.57 and c['last_reward_rise']
        assert c['end'][1:3] == [2.67, -.76] and abs(c['start'][0] - .505) < .011 and c['start'][1] == .48
    elif k == '12':
        assert c['constant'] == .808 and c['p'] == .731 and c['loss'] == [.693, .313]
        assert all(close(d, -.808, 1e-3) for d in e['match']['diffs'])


def gif(scene, stem, out):
    """Encode the GIF from a lossless render, composited on white, with the DONGXI LABS mark."""
    w, h = GIF_W, GIF_W * 9 // 16
    with tempfile.TemporaryDirectory(prefix='kl-mov-') as tmp:
        run([sys.executable, '-m', 'manim', 'render', '-t', '-r', f'{w},{h}', '--fps', str(GIF_FPS),
             '--renderer', 'cairo', '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING',
             '--media_dir', tmp, '-o', stem, str(P / 'scenes.py'), scene])
        found = list(Path(tmp).glob(f'videos/**/{stem}.mov')); assert len(found) == 1, found
        mark = write_mark(w, Path(tmp) / 'mark.png')
        run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(found[0]), '-i', str(mark), '-filter_complex',
             gif_filter(w, h, GIF_FPS), '-loop', '0', str(out)])
    with Image.open(out) as im:
        assert im.size == (w, h) and im.info['loop'] == 0
        hashes = set(); dur = 0
        for i in range(im.n_frames):
            im.seek(i); dur += im.info['duration']
            hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    size = out.stat().st_size
    assert len(hashes) > 30 and size < 5_000_000, (len(hashes), size)
    return dict(width=w, height=h, fps=GIF_FPS, frames=im.n_frames, duration_ms=dur, unique_frames=len(hashes),
                loop=0, bytes=size, brand_mark=True, source='lossless qtrle render composited on white')


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
    report = dict(task='KL article GIF set (12 clips)', preview=args.preview, clips=[])
    for stem, scene, cand in CLIPS:
        if args.only and stem not in args.only:
            continue
        out = P / 'rendered' / stem; out.mkdir(parents=True, exist_ok=True)
        tmpvid, cmd = manim(scene, stem, size, fps)
        video = out / ('preview.mp4' if args.preview else stem + '.mp4')
        if args.preview:
            shutil.move(tmpvid, video)
        else:
            brand_mp4(tmpvid, video); tmpvid.unlink()
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
            fr[-1].save(out / (prefix + 'poster.png'))
            sheet = Image.new('RGB', (1440, 310 * ((len(fr) + 2) // 3)), '#e2e8f0'); d = ImageDraw.Draw(sheet)
            for i, (f, s) in enumerate(zip(fr, samples)):
                x, y = i % 3 * 480, i // 3 * 310
                sheet.paste(f.resize((480, 270)), (x, y))
                d.text((x + 8, y + 279), f"{s['seconds']:.2f}s · {s['name']}", font=ImageFont.load_default(size=14), fill='#111827')
            sheet.save(out / (prefix + 'contact-sheet.png'))
        entry = dict(id=stem, scene=scene, candidate=cand, render_command=cmd,
                     duration_seconds=float(info['format']['duration']),
                     video=dict(width=st['width'], height=st['height'], codec=st['codec_name'], fps=st['r_frame_rate'],
                                brand_mark=not args.preview), timeline=tl)
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
                             [P / 'scenes.py', P / 'render.py', P.parents[1] / 'manim_style.py', P.parents[1] / 'brand.py',
                              P.parents[1] / 'uv.lock']]
        report['limitations'] = [
            'Shares, rewards and gold values of the egg example are illustrative; every ratio, log, sum and optimum on them is exact.',
            'GIF 04 and 09 use one seeded draw (NumPy seed 0), the same as check_numbers.py in the article package.',
            'GIF 10 token probabilities are illustrative; the two beta values are published (InstructGPT, DeepSeekMath).',
            'GIF 11 is a toy: each point is the best compromise at a smaller beta, standing in for longer training.',
            'No numbers from our own notebooks or runs appear.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _, _ in CLIPS if s in keep]
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
