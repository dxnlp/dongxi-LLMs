"""Reproducible local Manim render, media validation and visual-review artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont
from mechanism import validate

P=Path(__file__).resolve().parent
ROOT=P.parents[3]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args,capture=False):
    return subprocess.run(args,cwd=ROOT,check=True,text=True,stdout=subprocess.PIPE if capture else None).stdout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    fixture=validate()
    out=P/'rendered';out.mkdir(exist_ok=True)
    prefix='preview-' if args.preview else ''
    name='prefix-abc';video=out/(prefix+name+'.mp4')
    size,fps=('960,540','15') if args.preview else ('1920,1080','30')
    with tempfile.TemporaryDirectory(prefix='prefix-abc-manim-') as tmp:
        command=[sys.executable,'-m','manim','render','-r',size,'--fps',fps,'--renderer','cairo',
            '--progress_bar','none','--disable_caching','--verbosity','WARNING','--media_dir',tmp,
            '-o',name,str(P/'scene.py'),'PrefixABC']
        run(command)
        files=list(Path(tmp).glob(f'videos/**/{name}.mp4'));assert len(files)==1
        shutil.copy2(files[0],video)
    timeline=json.loads((P/'timeline.json').read_text())
    events=timeline['events'];names=[e['name'] for e in events]
    assert names==['A_input','A_cache','B_instruction_only','B_new_instruction_computed',
                   'C_document_edited','C_suffix_invalidated','C_suffix_recomputed','summary']
    assert all(timeline['checks'].values())
    assert events[2]['reused']==events[3]['reused']==[0,1,2,3]
    assert events[4]['shared_prefix']==1 and events[5]['invalidated']==[1,2,3]
    assert events[6]['computed']==[1,2,3,4] and events[6]['unchanged_text_recomputed']==[2,3]
    info=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],True))
    stream=info['streams'][0];duration=float(info['format']['duration'])
    assert (stream['width'],stream['height'])==tuple(map(int,size.split(',')))
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert 18<=duration<=25
    samples=[]
    for i,event in enumerate(events):
        if i:samples.append(dict(seconds=(events[i-1]['seconds']+event['seconds'])/2,name='transition'))
        samples.append(event)
    outputs=[video]
    with tempfile.TemporaryDirectory(prefix='prefix-abc-stills-') as tmp:
        frames=[]
        for i,s in enumerate(samples):
            f=Path(tmp)/f'{i}.png'
            run(['ffmpeg','-y','-loglevel','error','-ss',str(max(.01,s['seconds']-.03)),'-i',str(video),'-frames:v','1','-update','1',str(f)])
            frames.append(Image.open(f).convert('RGB'))
        # C's completed cache is the poster; the summary is also preserved.
        ci=next(i for i,s in enumerate(samples) if s['name']=='C_suffix_recomputed')
        for file,im in [('still.png',frames[ci]),('still-half.png',frames[ci].resize((960,540))),('summary.png',frames[-1])]:
            dest=out/(prefix+file);im.save(dest);outputs.append(dest)
        sheet=Image.new('RGB',(1440,310*((len(frames)+2)//3)),'#E2E8F0');draw=ImageDraw.Draw(sheet)
        for i,(f,s) in enumerate(zip(frames,samples)):
            x,y=i%3*480,i//3*310;sheet.paste(f.resize((480,270)),(x,y))
            draw.text((x+8,y+279),f"{s['seconds']:.2f}s {s['name']}",font=ImageFont.load_default(size=14),fill='#111827')
        dest=out/(prefix+'contact-sheet.png');sheet.save(dest);outputs.append(dest)
    gif_info=None
    if not args.preview:
        gif=out/(name+'.gif')
        run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
            'fps=15,scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer','-loop','0',str(gif)])
        with Image.open(gif) as im:
            assert im.size==(960,540) and im.info['loop']==0
            hashes=set();ms=0
            for i in range(im.n_frames):
                im.seek(i);ms+=im.info['duration'];hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
            assert len(hashes)>80 and gif.stat().st_size<8_000_000
            gif_info=dict(width=960,height=540,frames=im.n_frames,duration_ms=ms,unique_frames=len(hashes),loop=0,bytes=gif.stat().st_size)
        outputs.append(gif)
    sources=[P/'scene.py',P/'mechanism.py',P/'render.py',P.parents[1]/'manim_style.py',P.parents[1]/'STYLE_GUIDE.md',P.parents[1]/'uv.lock']
    report=dict(preview=args.preview,task='ANIM-KV-007',base_commit=run(['git','rev-parse','HEAD'],True).strip(),
        duration_seconds=duration,fixture=fixture,timeline=timeline,checks=dict(event_order=True,geometry=True,media=True),gif=gif_info,
        environment=dict(host=platform.node(),platform=platform.platform(),python=platform.python_version(),
            manim=importlib.metadata.version('manim'),pillow=importlib.metadata.version('pillow'),ffmpeg=run(['ffmpeg','-version'],True).splitlines()[0]),
        command=command,video=dict(codec=stream['codec_name'],width=stream['width'],height=stream['height'],fps=stream['r_frame_rate']),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sources],
        outputs=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in outputs],
        limitations=['Illustrative input groups and fingerprint glyphs; not actual tokens or activations.',
            'Recomputing suffix is a conservative validity rule, not proof every activation changes.',
            'Same weights, position handling, execution settings and available prefix cache assumed.',
            'Only prompt processing shown; answer generation still needs computation.',
            'Visual sweep is explanatory, not a prefill scheduling or timing measurement.',
            'Changing requests replaces the displayed suffix, not a global cache eviction.',
            'Fade/reset is editorial, not reverse inference.'])
    (P/(prefix+'metadata.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(duration=duration,gif=gif_info,checks=report['checks'])),flush=True)


if __name__=='__main__':main()
