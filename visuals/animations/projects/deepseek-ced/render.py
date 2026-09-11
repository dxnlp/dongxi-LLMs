"""Reproducible two-film render, media checks, review pages and manifests."""
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
from PIL import Image,ImageDraw,ImageFont

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
ANIM=P.parents[1]
CLIPS=[('01-global-kv','GlobalKV','Causal Encoder–Decoder','Global KV flow.'),
       ('02-shared-roles','SharedRoles','Shared KV','Scoring and weighted accumulation.')]
POSTER='assets/deepseek-v41-architecture-poster.png'

def source_records():
    sources=[p for p in P.iterdir() if p.suffix in ('.py','.json','.html','.mjs') and p.name not in ('metadata.json','qa.json')]+[ANIM/'manim_style.py',ANIM/'uv.lock']
    sources += [p for p in (P/'assets').iterdir() if p.suffix in ('.json','.txt')]
    return [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)]

def output_record(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)

def run(args,capture=False):
    return subprocess.run(args,check=True,cwd=ROOT,text=True,stdout=subprocess.PIPE if capture else None).stdout

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def pages(preview):
    style='''body{max-width:1150px;margin:40px auto;padding:0 24px;background:#fff;color:#111827;font:18px/1.65 Arial,sans-serif}h1{font-size:36px;font-weight:400}h2{font-size:26px;font-weight:400}p,details{color:#64748b}nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}button{font:17px Arial;padding:12px 18px;border:1px solid #cbd5e1;border-radius:8px;background:white;color:#334155;cursor:pointer}button[aria-pressed=true]{background:#eff6ff;color:#2563eb;border-color:#2563eb}video{display:block;width:100%;aspect-ratio:16/9;border:1px solid #e2e8f0;border-radius:10px}a{color:#2563eb}details{margin-top:25px}'''
    header=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style>'
    notes='''<details><summary>Accuracy notes</summary><p>Causal Encoder–Decoder changes the source of decoder global KV. Cross-layer sharing is introduced separately. The intermediate separate-bank view isolates the source change; it is not the released cache configuration.</p><p>States and numerical mixtures come from small untrained examples, not DeepSeek activations. The first film colors activation magnitude. The second uses three 2D queries; the released model has 64 query heads.</p><p>The toy fixes selected entries and omits sinks, RoPE, quantization and output projection. Actual core attention reads selected global and local entries together. Other caches and indexer keys still exist.</p><p>No memory, speed or quality benchmark is shown. Bounded prefill replay is outside these films; its local-state reconstruction is approximate.</p></details>'''
    provenance=json.loads((P/'assets/provenance.json').read_text())
    attribution=f'<p style="font-size:14px">Cover: <a href="{provenance["source_url"]}">DeepSeek-V4.1-Flash architecture, Figure 3</a> · <a href="{POSTER}">View diagram</a> · <a href="assets/LICENSE.deepseek.txt">MIT license</a></p>'
    data=[dict(id=i,title=t,description=d,video=f'rendered/{i}/{"preview" if preview else i}.mp4',poster=POSTER) for i,_,t,d in CLIPS]
    for c in data:
        (P/f'{c["id"]}.html').write_text(header+f'<title>{c["title"]}</title><h1>{c["title"]}</h1><p>{c["description"]}</p><video controls playsinline preload="metadata" poster="{c["poster"]}" src="{c["video"]}"></video>'+attribution+'<p><a href="review.html">Both animations</a></p>'+notes+'</html>')
    buttons=''.join(f'<button type="button" data-index="{i}" aria-pressed="{str(i==0).lower()}">{c["title"]}</button>' for i,c in enumerate(data))
    js='''const clips=DATA;const v=document.querySelector('video');document.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{const c=clips[Number(b.dataset.index)];v.pause();v.src=c.video;v.poster=c.poster;v.load();document.querySelector('h2').textContent=c.title;document.querySelector('#description').textContent=c.description;document.querySelector('#direct').href=c.id+'.html';document.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));}));'''.replace('DATA',json.dumps(data))
    c=data[0]
    (P/'review.html').write_text(header+f'<title>DeepSeek attention</title><h1>DeepSeek attention</h1><nav aria-label="Choose an animation">{buttons}</nav><h2>{c["title"]}</h2><p id="description">{c["description"]}</p><video controls playsinline preload="metadata" poster="{c["poster"]}" src="{c["video"]}"></video>'+attribution+f'<p><a id="direct" href="{c["id"]}.html">Open separately</a></p>'+notes+f'<script>{js}</script></html>')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--only',choices=[c[0] for c in CLIPS]);parser.add_argument('--pages-only',action='store_true');args=parser.parse_args()
    if args.pages_only:
        if args.only or args.preview:parser.error('--pages-only uses the existing manifest mode; do not combine with --only or --preview')
        metadata=json.loads((P/'metadata.json').read_text())
        # Refuse to bless changed media when refreshing only the player and cover.
        for entry in metadata['outputs']:
            if entry['path']!=str((P/POSTER).relative_to(ROOT)):
                assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
        pages(metadata['preview'])
        metadata['sources']=source_records()
        metadata['outputs']=[e for e in metadata['outputs'] if e['path']!=str((P/POSTER).relative_to(ROOT))]+[output_record(P/POSTER)]
        metadata['poster_update']=dict(mp4_bytes_unchanged=True,provenance='assets/provenance.json')
        (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(f'Updated covers and pages; videos unchanged: {P/"review.html"}')
        return
    trace=json.loads((P/'trace.json').read_text());assert all(trace['checks'].values())
    assert sha(P/'build_trace.py')==trace['source_sha256']
    protected={str(p):sha(p) for p in (P.parent/'looped-transformer').rglob('*') if p.is_file() and 'node_modules' not in p.parts}
    size,fps=('960,540','15') if args.preview else ('1920,1080','30')
    clips=[];outputs=[]
    if args.only:
        old=json.loads((P/'metadata.json').read_text());assert old['preview']==args.preview
        clips=[c for c in old['clips'] if c['id']!=args.only]
        outputs=[ROOT/e['path'] for e in old['outputs'] if f'/{args.only}/' not in e['path']]
    for id,cls,title,_ in CLIPS:
        if args.only and args.only!=id:continue
        out=P/'rendered'/id;out.mkdir(parents=True,exist_ok=True)
        video=out/('preview.mp4' if args.preview else id+'.mp4')
        with tempfile.TemporaryDirectory(prefix='ced-render-') as tmp:
            run([sys.executable,'-m','manim','render','-r',size,'--fps',fps,'--renderer','cairo','--progress_bar','none','--disable_caching','--verbosity','WARNING','--media_dir',tmp,'-o',id,str(P/'scenes.py'),cls])
            found=list(Path(tmp).glob(f'videos/**/{id}.mp4'));assert len(found)==1;shutil.copy2(found[0],video)
        info=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],True));s=info['streams'][0]
        assert (s['width'],s['height'])==tuple(map(int,size.split(','))) and s['codec_name']=='h264' and s['pix_fmt']=='yuv420p'
        timeline=json.loads((P/f'{id}-timeline.json').read_text())
        with tempfile.TemporaryDirectory(prefix='ced-frames-') as tmp:
            frames=[]
            for i,c in enumerate(timeline['checkpoints']):
                f=Path(tmp)/f'{i}.png';run(['ffmpeg','-y','-loglevel','error','-ss',str(c['seconds']-.1),'-i',str(video),'-frames:v','1','-update','1',str(f)])
                frames.append(Image.open(f).convert('RGB'))
            frames[-1].save(out/'still.png');frames[-1].resize((960,540)).save(out/'still-half.png')
            sheet=Image.new('RGB',(1440,300*((len(frames)+2)//3)),'#e2e8f0');draw=ImageDraw.Draw(sheet)
            for i,(im,c) in enumerate(zip(frames,timeline['checkpoints'])):
                x,y=i%3*480,i//3*300;sheet.paste(im.resize((480,270)),(x,y));draw.text((x+8,y+276),f"{c['seconds']:.1f}s - {c['name']}",font=ImageFont.load_default(size=13),fill='#111827')
            sheet.save(out/'contact-sheet.png')
        outputs.extend([video,out/'still.png',out/'still-half.png',out/'contact-sheet.png'])
        if not args.preview:
            gif=out/f'{id}.gif';run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex','fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer','-loop','0',str(gif)]);outputs.append(gif)
        clips.append(dict(id=id,title=title,width=s['width'],height=s['height'],fps=s['r_frame_rate'],duration_seconds=float(info['format']['duration']),timeline=timeline));print(f'Finished {id}',flush=True)
    pages(args.preview)
    assert all(sha(Path(p))==h for p,h in protected.items()),'Other animation assets changed'
    outputs=list(dict.fromkeys(outputs+[P/POSTER]))
    metadata=dict(preview=args.preview,base_commit=run(['git','rev-parse','HEAD'],True).strip(),environment=dict(python=platform.python_version(),platform=platform.platform(),manim=version('manim'),numpy=version('numpy')),checks=trace['checks'],other_project_unchanged=True,clips=sorted(clips,key=lambda c:c['id']),sources=source_records(),outputs=[output_record(p) for p in outputs])
    (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Ready: {P/"review.html"}',flush=True)

if __name__=='__main__':main()
