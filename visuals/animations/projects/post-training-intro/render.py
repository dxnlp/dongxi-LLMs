"""Render ANIM-POST-001: five article loops (MP4 + high-quality GIF) and comparison stills.

uv run --project visuals/animations python visuals/animations/projects/post-training-intro/render.py [--preview] [--only STEM]
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
ARTICLE = ROOT / 'publications/x-articles/x-post-training-001'
CLIPS = [('01-same-prompt-two-models', 'SamePrompt'), ('02-two-stages', 'TwoStages'),
         ('03-instruction-tuning', 'InstructionTuning'), ('04-preference-alignment', 'PreferenceAlignment'),
         ('05-rlvr', 'RLVR')]
GIF_W, GIF_FPS = 1280, 20


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, capture=False, env=None):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, env=env,
                          stdout=subprocess.PIPE if capture else None).stdout


def manim(scene, stem, size, fps, still=False, env=None):
    with tempfile.TemporaryDirectory(prefix='post-intro-') as tmp:
        cmd = [sys.executable, '-m', 'manim', 'render', '-r', size, '--fps', fps, '--renderer', 'cairo',
               '--progress_bar', 'none', '--disable_caching', '--verbosity', 'WARNING', '--media_dir', tmp,
               '-o', stem, str(P / 'scenes.py'), scene] + (['-s'] if still else [])
        run(cmd, env=env)
        pattern = f'images/**/{stem}*.png' if still else f'videos/**/{stem}.mp4'
        found = list(Path(tmp).glob(pattern)); assert len(found) == 1, found
        dest = Path(tempfile.mkdtemp(prefix='post-intro-out-')) / found[0].name
        shutil.copy2(found[0], dest)
        return dest, cmd


def validate(stem, tl):
    e = {v['name']: v for v in tl['events']}
    assert all(tl['checks'].values()) and 'complete' in e and 6 < tl['duration_seconds'] < 30
    if stem.startswith('01'):
        assert e['distributions']['base_top'] == 'Explain' and e['distributions']['post_top'] == 'People'
        assert e['distributions']['base_question_mass'] > .5 and e['distributions']['post_answer_mass'] > .5
    elif stem.startswith('02'):
        order = [v['name'] for v in tl['events']]
        assert order.index('pretrained') < order.index('post_examples') < order.index('post_comparisons') < order.index('post_checked')
        assert e['complete']['same_model']
    elif stem.startswith('03'):
        assert e['instruction_is_context']['graded'] == [] and e['complete']['instruction_graded'] == []
        assert e['complete']['graded'] == ['I', "didn't", 'have', 'a', 'laundry.']
    elif stem.startswith('04'):
        h0 = e['start_equals_original']['heights']
        hA, hB = e['nudged']['heights']
        assert h0[0] == h0[1] and hA > h0[0] > hB
    else:
        for r in (1, 2):
            ans = e[f'round_{r}_sampled']['answers']
            assert e[f'round_{r}_checked']['correct'] == [a == 17 * 3 for a in ans]
        c = [e['setup']['chances'], e['round_1_update']['chances'], e['round_2_update']['chances']]
        assert all(abs(sum(x.values()) - 1) < 1e-9 for x in c)
        assert c[0]['51'] < c[1]['51'] < c[2]['51']


def gif(video, out):
    # One global palette from the whole clip, no dithering: crisp flat fills, stable colors between frames.
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video), '-filter_complex',
         f'fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,split[a][b];'
         '[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];'
         '[b][p]paletteuse=dither=none:diff_mode=rectangle', '-loop', '0', str(out)])
    with Image.open(out) as im:
        assert im.size == (GIF_W, GIF_W * 9 // 16) and im.info['loop'] == 0
        hashes = set(); dur = 0
        for i in range(im.n_frames):
            im.seek(i); dur += im.info['duration']
            hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    size = out.stat().st_size
    assert len(hashes) > 30 and size < 15_000_000, (len(hashes), size)
    return dict(width=im.size[0], height=im.size[1], fps=GIF_FPS, frames=im.n_frames,
                duration_ms=dur, unique_frames=len(hashes), loop=0, bytes=size)


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
    report = dict(task='ANIM-POST-001', preview=args.preview, clips=[], stills=[])
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
            g = assets / (stem + '.gif'); entry['gif'] = gif(video, g)  # GIF lives only with the article
            poster.save(assets / (stem + '-poster.png'))
            entry['outputs'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
                                for p in [g, assets / (stem + '-poster.png'), out / 'contact-sheet.png']]
        report['clips'].append(entry)
        print(json.dumps(dict(clip=stem, duration=entry['duration_seconds'], gif=entry.get('gif'))), flush=True)
    if not args.only or 'comparison' in args.only:
        for lang in ['en', 'zh']:
            env = dict(os.environ, COMPARE_LANG=lang)
            png, cmd = manim('Comparison', f'comparison-{lang}', '1920,1080', '30', still=True, env=env)
            dest = (assets if lang == 'en' else ARTICLE / 'zh/assets'); dest.mkdir(parents=True, exist_ok=True)
            final = dest / '03-pretraining-vs-post-training.png'
            shutil.copy2(png, final)
            with Image.open(final) as im: assert im.size == (1920, 1080)
            report['stills'].append(dict(lang=lang, path=str(final.relative_to(ROOT)), sha256=sha(final),
                                         render_command=cmd + ['# COMPARE_LANG=' + lang]))
            print(json.dumps(dict(still=str(final.relative_to(ROOT)))), flush=True)
    if not args.preview:
        report['environment'] = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                                     manim=version('manim'), pillow=version('pillow'),
                                     ffmpeg=run(['ffmpeg', '-version'], True).splitlines()[0],
                                     base_commit=run(['git', 'rev-parse', 'HEAD'], True).strip())
        report['sources'] = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
                             [P / 'scenes.py', P / 'render.py', P.parents[1] / 'manim_style.py', P.parents[1] / 'uv.lock']]
        report['limitations'] = [
            'A1 next-word probabilities and replies are authored illustrations, not measured from any model.',
            'A2 stream volumes are not to scale; stage order shown is common, real pipelines vary.',
            'A3 reveals response words one at a time for readability; training grades all response positions in one pass.',
            'A4 merges the reward-model (RLHF/PPO) and DPO routes into one picture; the leash stands for the KL penalty.',
            'A5 checker reads only the final answer; chance updates are illustrative, not a GRPO computation.']
        mpath = P / 'metadata.json'
        old = json.loads(mpath.read_text()) if mpath.exists() and args.only else None
        if old:
            keep = {c['id']: c for c in old.get('clips', [])}
            keep.update({c['id']: c for c in report['clips']}); report['clips'] = [keep[s] for s, _ in CLIPS if s in keep]
            if not report['stills']: report['stills'] = old.get('stills', [])
        mpath.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
