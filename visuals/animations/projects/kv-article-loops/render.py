"""Render the two approved article loops, preserving the first loop byte-for-byte."""
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

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
CLIPS=[('append-edit','AppendEdit'),('global-sharing','GlobalSharing')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args,capture=False):
    return subprocess.run(args,cwd=ROOT,check=True,text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def validate(stem,timeline):
    assert all(timeline['checks'].values())
    e=timeline['events']
    if stem=='append-edit':
        assert [x['name'] for x in e]==['two_prefixes','append_input','append_kv','edit_input','suffix_invalidated','suffix_recomputed']
        assert e[2]['reusable']==[1,2,3,4] and e[2]['appended']==5
        assert e[3]['changed_position']==2
        assert e[4]['invalidated']==e[5]['recomputed']==[2,3,4]
        assert e[4]['reusable']==e[5]['reusable']==[1]
    else:
        assert [x['name'] for x in e]==['encoder_output','global_bank_created','layer_local_state',
            'layer_21_read','layer_22_read','layer_40_read','selected_entry_detail','same_entry_scores','same_entry_accumulates']
        assert e[1]['source']=='encoder_output' and e[1]['producer_layer']==21
        assert all(x['global_banks']==1 for x in e[1:])
        assert [x['reader'] for x in e if 'reader' in x]==[21,22,40]
        assert e[-2]['entry']==e[-1]['entry']==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    protected={str(p):sha(p) for p in (P.parent/'kv-prefill-decode').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    size,fps=('960,540','15') if args.preview else ('1920,1080','30')
    clips=[]
    for stem,scene in CLIPS:
        out=P/'rendered'/stem;out.mkdir(parents=True,exist_ok=True)
        video=out/('preview.mp4' if args.preview else stem+'.mp4')
        with tempfile.TemporaryDirectory(prefix='kv-article-loop-') as tmp:
            command=[sys.executable,'-m','manim','render','-r',size,'--fps',fps,'--renderer','cairo',
                '--progress_bar','none','--disable_caching','--verbosity','WARNING','--media_dir',tmp,
                '-o',stem,str(P/'scenes.py'),scene]
            run(command)
            found=list(Path(tmp).glob(f'videos/**/{stem}.mp4'));assert len(found)==1
            shutil.copy2(found[0],video)
        timeline=json.loads((P/(stem+'-timeline.json')).read_text());validate(stem,timeline)
        info=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],True))
        stream=info['streams'][0]
        assert (stream['width'],stream['height'])==tuple(map(int,size.split(',')))
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        prefix='preview-' if args.preview else ''
        # Include mid-motion samples, not just the pauses at event boundaries.
        events=timeline['events'];samples=[]
        for i,event in enumerate(events):
            if i and event['seconds']-events[i-1]['seconds']>.45:
                samples.append(dict(seconds=(event['seconds']+events[i-1]['seconds'])/2,name='motion: '+event['name']))
            samples.append(event)
        with tempfile.TemporaryDirectory(prefix='kv-article-frames-') as tmp:
            frames=[]
            for i,sample in enumerate(samples):
                f=Path(tmp)/f'{i}.png'
                run(['ffmpeg','-y','-loglevel','error','-ss',str(max(0,sample['seconds']-.03)),
                     '-i',str(video),'-frames:v','1','-update','1',str(f)])
                frames.append(Image.open(f).convert('RGB'))
            frames[-1].save(out/(prefix+'still.png'))
            frames[-1].resize((960,540)).save(out/(prefix+'still-half.png'))
            sheet=Image.new('RGB',(1440,310*((len(frames)+2)//3)),'#e2e8f0');draw=ImageDraw.Draw(sheet)
            for i,(frame,sample) in enumerate(zip(frames,samples)):
                x,y=i%3*480,i//3*310
                sheet.paste(frame.resize((480,270)),(x,y))
                draw.text((x+8,y+279),f"{sample['seconds']:.2f}s · {sample['name']}",
                          font=ImageFont.load_default(size=14),fill='#111827')
            sheet.save(out/(prefix+'contact-sheet.png'))
        outputs=[video,out/(prefix+'still.png'),out/(prefix+'still-half.png'),out/(prefix+'contact-sheet.png')]
        gif_info=None
        if not args.preview:
            gif=out/(stem+'.gif')
            run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
                'fps=15,scale=960:540:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer',
                '-loop','0',str(gif)])
            with Image.open(gif) as im:
                assert im.size==(960,540) and im.info['loop']==0
                hashes=set();duration=0
                for i in range(im.n_frames):
                    im.seek(i);duration+=im.info['duration'];hashes.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
                assert len(hashes)>30 and gif.stat().st_size<5_000_000
                gif_info=dict(width=960,height=540,frames=im.n_frames,duration_ms=duration,
                              unique_frames=len(hashes),loop=0,bytes=gif.stat().st_size)
            outputs.append(gif)
        clips.append(dict(id=stem,timeline=timeline,gif=gif_info,render_command=command,
            duration_seconds=float(info['format']['duration']),
            video=dict(width=stream['width'],height=stream['height'],codec=stream['codec_name'],fps=stream['r_frame_rate']),
            outputs=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in outputs]))
        print(json.dumps(dict(clip=stem,gif=gif_info,duration=clips[-1]['duration_seconds'])),flush=True)
    assert all(sha(Path(p))==h for p,h in protected.items())
    sources=[P/'scenes.py',P/'render.py',P.parents[1]/'manim_style.py',P.parents[1]/'uv.lock']
    metadata=dict(preview=args.preview,base_commit=run(['git','rev-parse','HEAD'],True).strip(),
        environment=dict(host=platform.node(),platform=platform.platform(),python=platform.python_version(),
                         manim=version('manim'),pillow=version('pillow'),ffmpeg=run(['ffmpeg','-version'],True).splitlines()[0]),
        checks=dict(event_order=True,first_loop_unchanged=True,media_valid=True),clips=clips,
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sources],
        limitations=['Dependency schematics; no new model or timing experiment.',
                     'Suffix recomputation does not claim every numerical value changes.',
                     'Global bank remains separate from distinct layer-local Q and KV.',
                     'Selected-vector detail is magnification, not another cache allocation.',
                     'Sparse selection/indexer mechanics and other model operations omitted.',
                     'Fade resets the illustration; it does not depict cache eviction.'])
    (P/('preview-metadata.json' if args.preview else 'metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
