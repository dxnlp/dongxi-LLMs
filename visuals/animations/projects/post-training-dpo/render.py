"""Render the DPO article GIF set: thirteen clips, each GIF and MP4 with the DONGXI LABS mark.

uv run --project visuals/animations python visuals/animations/projects/post-training-dpo/render.py [--preview] [--only STEM]
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

CLIPS = [('00a-two-routes', 'TwoRoutes', 'CAND-ANIM-079'), ('00b-change-is-the-score', 'ChangeIsScore', 'CAND-ANIM-079'),
         ('01-secretly-a-reward-model', 'Secretly', 'CAND-ANIM-067'), ('02-three-signals', 'ThreeSignals', 'CAND-ANIM-057'),
         ('03-why-pay-for-distance', 'WhyPay', 'CAND-ANIM-052'), ('04-no-move-helps', 'NoMoveHelps', 'CAND-ANIM-069, 053'),
         ('05-implicit-reward', 'ImplicitReward', 'CAND-ANIM-066'),
         ('06-loss-on-one-pair', 'LossOnPair', 'CAND-ANIM-070'), ('07-the-push', 'ThePush', 'CAND-ANIM-071'),
         ('08-where-probability-comes-from', 'WhereFrom', 'CAND-ANIM-073'), ('09-ideal-test', 'IdealTest', 'CAND-ANIM-074'),
         ('10-answers-to-tokens', 'AnswersToTokens', 'CAND-ANIM-077'), ('11-what-to-watch', 'WhatToWatch', 'CAND-ANIM-078')]
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(x): return 1 / (1 + math.exp(-x))


def run(args, capture=False):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps):
    with tempfile.TemporaryDirectory(prefix='dpo-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene]
        run(cmd)
        found = list(Path(tmp).glob(f'videos/**/{stem}.mp4')); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='dpo-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def close(a, b, tol=2e-3):
    return abs(a - b) < tol


def validate(stem, tl):
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 8 < tl['duration_seconds'] < 36, tl['duration_seconds']
    c, k = e['complete'], stem[:2]
    if stem.startswith('00a'):
        assert c['stages'] == [4, 2]
    elif stem.startswith('00b'):
        assert e['small_move']['scores'] == [-.288, 0, 0, .606, 0] and e['start']['scores'] == [0] * 5
        assert c['scores'] == [-.808, .192, -.308, 1.192, -1.808]
    elif k == '01':
        assert c['scores'] == [-.288, 0, 0, .606, 0] and e['score']['ratio'] == 1.83 and e['score']['log'] == .606
    elif k == '02':
        assert c['in_update'] == ['reward', 'KL'] and c['outside'] == ['gold']
    elif k == '03':
        assert (e['reference']['reward'], e['reference']['gold']) == (.51, .48) or (e['reference']['reward'], e['reference']['gold']) == (.5, .48)
        assert (e['reward_alone']['reward'], e['reward_alone']['gold']) == (3, -2)
        assert (c['reward'], c['gold'], c['trick'], c['best']) == (1.22, 1.0, .043, .379)
    elif k == '04':
        assert e['unequal']['net'] == [.288, 1, .5, 1.394, -1] and e['unequal']['score'] == .638
        assert e['chip']['after'] == .649 and e['chip']['gain'] == .011
        assert e['level']['constant'] == .808 and e['level']['score'] == .808 and e['level']['net'] == [.808] * 5
        assert e['no_move']['forward'] == -.0004 and e['no_move']['back'] == -.0004
        assert e['multiply']['weights'] == [.4, .68, .247, .887, .029]
        assert c['z'] == 2.243 and c['shares'] == [.178, .303, .11, .395, .013]
    elif k == '05':
        assert c['constant'] == .808 and c['p'] == .731 and c['loss'] == [.693, .313]
        assert all(close(d, -.808, 1e-3) for d in e['match']['diffs'])
    elif k == '06':
        assert e['three_points']['loss'] == [.693, .435, .313] and e['three_points']['gap'] == [0, .606, 1]
        assert c['chance'] == [.5, .647, .731] and c['ends'] == [3.049, .049]
    elif k == '07':
        assert e['nudge']['before'] == .6931 and e['nudge']['after'] == .6882
        assert e['sizes']['push'] == [.953, .5, .269, .047]
        assert c['gaps'] == [0, .5, .878, 1.171, 1.408] and c['losses'] == [.693, .474, .348, .27, .219]
        assert c['pushes'] == [.5, .378, .294, .237, .197]
    elif k == '08':
        assert c['shares'] == [.408, .155, .153, .202, .082] and c['gap'] == 1 and e['wrong_rose']['after'] == .082
    elif k == '09':
        assert c['distance'] == [.33, .05] and c['average'] == [.38, .15, .04, .01] and c['pairs_shown'] == 8
        assert e['after_run']['shares'] == [.189, .325, .128, .343, .014] and e['records']['pairs'] == 1000
    elif k == '10':
        assert c['changes'] == [2, 1] and e['raw']['totals'] == [-40, -60] and c['total'] == -3.0
    elif k == '11':
        assert c['measured_values'] == 0


def gif(scene, stem, out):
    """Encode the GIF from a lossless render, composited on white, with the DONGXI LABS mark."""
    w, h = GIF_W, GIF_W * 9 // 16
    with tempfile.TemporaryDirectory(prefix='dpo-mov-') as tmp:
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
    report = dict(task='DPO article GIF set (13 clips)', preview=args.preview, clips=[])
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
            'Shares, rewards and gold values of the egg example are illustrative; every ratio, log, loss and optimum on them is exact.',
            'GIF 09 replays one simulated run from check_numbers.py (notebook code, seed 0): the bars are its shares at 16 training steps. Its averages come from eight simulated datasets.',
            'GIF 10 token values and totals are illustrative.',
            'GIF 11 curves are invented and carry no measured values.',
            'No numbers from our own notebooks or runs appear, apart from the simulated toy in GIF 09.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _, _ in CLIPS if s in keep]
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
