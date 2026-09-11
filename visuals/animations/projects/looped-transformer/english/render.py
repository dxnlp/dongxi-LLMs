"""Render the three English clips and their learner-facing review players."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from PIL import Image,ImageDraw,ImageFont
import manimpango

P=Path(__file__).resolve().parent
ROOT=P.parents[4]
ANIM=P.parents[2]
CLIPS=[('01-one-block','OneBlockEnglish','Reuse one block',
        'One block, three applications. Watch the hidden state change while the weights stay shared.'),
       ('02-repeat-stack','RepeatedStack','Repeat a three-block stack',
        'A, B and C have distinct weights. Two passes execute A → B → C → A → B → C.'),
       ('03-unfold-sharing','UnrolledSharing','Unfold the shared computation',
        'Six block applications, paired by their shared weights. Unfolding adds views of the computation, not parameters.')]

def run(cmd,capture=False):
    r=subprocess.run(cmd,check=True,cwd=ROOT,text=True,stdout=subprocess.PIPE if capture else None)
    return r.stdout
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def pages(preview):
    data=[dict(id=i,title=t,description=d,video=f'rendered/{i}/{"preview" if preview else i}.mp4',poster=f'rendered/{i}/still.png') for i,_,t,d in CLIPS]
    style='''body{max-width:1200px;margin:38px auto;padding:0 24px;background:white;color:#111827;font:18px/1.65 Arial,sans-serif}h1{font-size:36px;font-weight:400;margin:0 0 18px}h2{font-size:25px;font-weight:400}p,details{color:#64748b}nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}button{font:17px Arial;padding:11px 17px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:#334155;cursor:pointer}button[aria-pressed=true]{color:#2563eb;border-color:#2563eb;background:#eff6ff}video{display:block;width:100%;aspect-ratio:16/9;border:1px solid #e2e8f0;border-radius:12px}a{color:#2563eb}details{margin-top:22px}'''
    notes='''<details><summary>Model and accounting details</summary><p>These are untrained CPU examples using a causal Transformer block with normalization, attention, residual connections and a feed-forward network. Internal drawings abbreviate those operations. More passes do not establish better prediction quality.</p><p>Each block contains 2,160 parameters. Three distinct blocks store 6,480. Reusing the stack twice gives six block applications without allocating another copy of the weights. Six independent copies would store 12,960 block parameters.</p><p>The heatmaps display one sample with 6 positions and 16 features. Blue/purple indicate negative/positive activations on a fixed scale. They do not measure reasoning quality.</p><p>Matrix work counts dense multiply-add arithmetic only, with two FLOPs per multiply-add. Embeddings, output head, normalization, activation functions and softmax are excluded. No latency or memory benchmark is implied.</p></details>'''
    header=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style>'
    for clip in data:
        (P/f'{clip["id"]}.html').write_text(header+f'<title>{clip["title"]}</title><h1>{clip["title"]}</h1><p>{clip["description"]}</p><video controls playsinline preload="metadata" poster="{clip["poster"]}"><source src="{clip["video"]}" type="video/mp4"></video><p><a href="review.html">All three animations</a></p>'+notes+'</html>')
    buttons=''.join(f'<button type="button" data-index="{i}" aria-pressed="{str(i==0).lower()}">{i+1}. {c["title"]}</button>' for i,c in enumerate(data))
    script='''const clips=CLIP_DATA;const video=document.querySelector('video');const title=document.querySelector('h2');const description=document.querySelector('#description');const direct=document.querySelector('#direct');document.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{const i=Number(button.dataset.index),c=clips[i];video.pause();video.poster=c.poster;video.src=c.video;video.load();title.textContent=c.title;description.textContent=c.description;direct.href=c.id+'.html';document.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));}));'''.replace('CLIP_DATA',json.dumps(data))
    c=data[0]
    (P/'review.html').write_text(header+f'<title>Looped Transformers</title><h1>Looped Transformers</h1><nav aria-label="Choose an animation">{buttons}</nav><h2>{c["title"]}</h2><p id="description">{c["description"]}</p><video controls playsinline preload="metadata" poster="{c["poster"]}" src="{c["video"]}"></video><p><a id="direct" href="{c["id"]}.html">Open this animation separately</a></p>'+notes+f'<script>{script}</script></html>')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true')
    parser.add_argument('--only',choices=[c[0] for c in CLIPS]);args=parser.parse_args()
    assert 'Arial' in manimpango.list_fonts()
    trace=json.loads((P/'trace.json').read_text());assert all(trace['checks'].values())
    assert sha(ROOT/'src/dongxi_llms/decoder_lab.py')==trace['source_sha256']
    original=[P.parent/n for n in ('scene.py','trace.json','timeline.json','review.html','metadata.json')]
    original+=list((P.parent/'rendered').glob('*'))
    old_hashes={str(p):sha(p) for p in original if p.is_file()}
    size,fps=('960,540','15') if args.preview else ('1920,1080','30')
    results=[];outputs=[]
    if args.only:
        previous=json.loads((P/'metadata.json').read_text())
        assert previous['status']==('preview' if args.preview else 'local-review')
        results=[c for c in previous['clips'] if c['id']!=args.only]
        outputs=[ROOT/e['path'] for e in previous['outputs'] if f'/{args.only}/' not in e['path']]
    for clip,cls,title,_ in CLIPS:
        if args.only and clip!=args.only:continue
        out=P/'rendered'/clip;out.mkdir(parents=True,exist_ok=True)
        video=out/('preview.mp4' if args.preview else f'{clip}.mp4')
        with tempfile.TemporaryDirectory(prefix=f'{clip}-') as tmp:
            run([sys.executable,'-m','manim','render','-r',size,'--fps',fps,'--renderer','cairo','--progress_bar','none','--disable_caching','--verbosity','WARNING','--media_dir',tmp,'-o',clip,str(P/'scenes.py'),cls])
            movies=list(Path(tmp).glob(f'videos/**/{clip}.mp4'));assert len(movies)==1
            shutil.copy2(movies[0],video)
        probe=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],True))
        s=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert (s['width'],s['height'])==tuple(map(int,size.split(',')))
        assert s['codec_name']=='h264' and s['pix_fmt']=='yuv420p'
        timeline=json.loads((P/f'{clip}-timeline.json').read_text())
        with tempfile.TemporaryDirectory(prefix='english-frames-') as tmp:
            frames=[]
            for i,c in enumerate(timeline['checkpoints']):
                path=Path(tmp)/f'{i}.png'
                run(['ffmpeg','-y','-loglevel','error','-ss',str(c['seconds']-.15),'-i',str(video),'-frames:v','1','-update','1',str(path)])
                frames.append(Image.open(path).convert('RGB'))
            frames[-1].save(out/'still.png');frames[-1].resize((960,540),Image.Resampling.LANCZOS).save(out/'still-half.png')
            sheet=Image.new('RGB',(1440,300*((len(frames)+2)//3)),'#e2e8f0');draw=ImageDraw.Draw(sheet)
            for i,(im,c) in enumerate(zip(frames,timeline['checkpoints'])):
                x,y=i%3*480,i//3*300;sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(x,y))
                draw.text((x+8,y+276),f"{c['seconds']:.1f}s · {c['name']}",font=ImageFont.load_default(size=14),fill='#111827')
            sheet.save(out/'contact-sheet.png')
        run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex','fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer','-loop','0',str(out/f'{clip}.gif')])
        outputs += [video,out/f'{clip}.gif',out/'still.png',out/'still-half.png',out/'contact-sheet.png']
        results.append(dict(id=clip,title=title,width=s['width'],height=s['height'],fps=s['r_frame_rate'],duration_seconds=float(probe['format']['duration']),timeline=timeline))
        print(f'Finished {clip}',flush=True)
    pages(args.preview)
    assert all(sha(Path(p))==h for p,h in old_hashes.items()),'Chinese artifacts changed'
    sources=[p for p in P.iterdir() if p.suffix in ('.py','.mjs','.html','.json') and p.name not in ('metadata.json','qa.json')]
    sources += list((P.parent/'assets').glob('*'))
    sources += [P.parent/n for n in ('trace.json','export.mjs','build_sketches.py','sketches.json','package.json','pnpm-lock.yaml')]
    sources += [ANIM/n for n in ('manim_style.py','STYLE_GUIDE.md','uv.lock')]+[ROOT/'src/dongxi_llms/decoder_lab.py']
    metadata=dict(task='ANIM-LOOP-001 English series',status='preview' if args.preview else 'local-review',
        rendered_at=datetime.now(timezone.utc).isoformat(),base_commit=run(['git','rev-parse','HEAD'],True).strip(),
        command='uv run --project visuals/animations python visuals/animations/projects/looped-transformer/english/render.py'+(' --preview' if args.preview else ''),
        environment=dict(python=platform.python_version(),platform=platform.platform(),packages={n:version(n) for n in ('manim','manimpango','numpy','pillow')},torch=trace['torch_version'],font='Arial'),
        clips=sorted(results,key=lambda c:c['id']),numerical_checks=trace['checks'],chinese_artifacts_unchanged=True,
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)],
        outputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in outputs])
    (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Review ready: {P/"review.html"}')

if __name__=='__main__':main()
