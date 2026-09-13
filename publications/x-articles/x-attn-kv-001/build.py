"""Build original diagrams, bilingual review pages, and X transfer sources.

Run with a Python environment containing Pillow and mistune. No model execution.
"""
from pathlib import Path
import hashlib
import html
import json
import math
import re
import subprocess
import sys
import shutil
from io import BytesIO

import mistune
import matplotlib
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageChops, ImageDraw, ImageFont

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
ASSETS=P/'assets'
COVER='cover-kv-cache-v2.png'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
FG='#111827';MUTED='#64748B';BLUE='#2563EB';PURPLE='#7C3AED';GREEN='#047857';AMBER='#B45309';GRID='#CBD5E1'
FILES=['01-prefill-decode.png','02-causal-reuse.png','03-cache-evidence.png','04-memory.png','05-three-changes.png']
EQUATIONS={
    'math-attention.png': dict(size=(1600,400),title='One query. All available keys and values.',
        lines=[r'\mathrm{output}_{t} = \mathrm{softmax}\!\left(\frac{q_{t}K_{\leq t}^{\top}}{\sqrt{d}}\right)V_{\leq t}']),
    'math-projections.png': dict(size=(1600,500),title='Rows processed per K or V projection',
        labels=['Cached','Full prefix'],
        lines=[r'2\times(2+1+1+1+1)=12',
               r'2\times(2+3+4+5+6)=40']),
    'math-memory.png': dict(size=(1600,360),title='Separate K/V tensor payload',
        lines=[r'\mathrm{bytes}=2\times B\times L\times T\times H_{\mathrm{KV}}\times d\times b']),
}
# Experiment assets remain reproducible for the course, but are excluded from the article.
ARTICLE_FILES=['prefill-decode.gif','math-attention.png','append-edit.gif','decode-step.gif','math-memory.png','memory-growth.gif','global-sharing.gif']
LOOPS={
    'prefill-decode':dict(project='kv-prefill-decode',folder='',still='prefill-decode-still-half.png'),
    'append-edit':dict(project='kv-article-loops',folder='append-edit',still='still-half.png'),
    'global-sharing':dict(project='kv-article-loops',folder='global-sharing',still='still-half.png'),
    'memory-growth':dict(project='kv-memory-growth',folder='memory-growth',still='still-half.png'),
    'decode-step':dict(project='kv-decode-step',folder='decode-step',still='still-half.png'),
}


def animation_assets():
    records={}
    for stem,spec in LOOPS.items():
        project=ROOT/'visuals/animations/projects'/spec['project']
        manifest=json.loads((project/'metadata.json').read_text())
        assert not manifest['preview'] and manifest['checks']['event_order']
        clip=next(c for c in manifest['clips'] if c['id']==stem) if 'clips' in manifest else manifest
        expected={entry['path']:entry['sha256'] for entry in clip['outputs']}
        for source,name in [(stem+'.gif',stem+'.gif'),(spec['still'],stem+'-still.png')]:
            source_path=project/'rendered'/spec['folder']/source
            assert hashlib.sha256(source_path.read_bytes()).hexdigest()==expected[str(source_path.relative_to(ROOT))]
            shutil.copy2(source_path,ASSETS/name)
        records[stem]=dict(project=str(project.relative_to(ROOT)),gif=clip['gif'],
                          manifest_sha256=hashlib.sha256((project/'metadata.json').read_bytes()).hexdigest())
    return records


def equations():
    """Render exact mathematical notation locally; PNGs preserve it during X upload."""
    with matplotlib.rc_context({'mathtext.fontset':'dejavusans'}):
        for name,spec in EQUATIONS.items():
            im=Image.new('RGB',spec['size'],'white');draw=ImageDraw.Draw(im)
            text(draw,80,65,spec['title'],34,MUTED)
            rendered_lines=[]
            for formula in spec['lines']:
                buffer=BytesIO()
                mathtext.math_to_image('$'+formula+'$',buffer,prop=FontProperties(size=44),dpi=144,format='png',color=FG)
                buffer.seek(0)
                raw=Image.open(buffer).convert('RGBA')
                rendered=Image.new('RGB',raw.size,'white');rendered.paste(raw,mask=raw.getchannel('A'))
                bounds=ImageChops.difference(rendered,Image.new('RGB',rendered.size,'white')).getbbox()
                assert bounds, formula
                rendered_lines.append(rendered.crop(bounds))
            band=(im.height-140)//len(spec['lines'])
            available=1100 if 'labels' in spec else 1440
            scale=min(1,available/max(r.width for r in rendered_lines),(band-40)/max(r.height for r in rendered_lines))
            for i,rendered in enumerate(rendered_lines):
                if scale<1:
                    rendered=rendered.resize((round(rendered.width*scale),round(rendered.height*scale)),Image.Resampling.LANCZOS)
                x=420 if 'labels' in spec else (im.width-rendered.width)//2
                y=130+i*band+(band-rendered.height)//2
                assert x>=60 and y>=110 and y+rendered.height<=im.height-20
                im.paste(rendered,(x,y))
                if 'labels' in spec:
                    text(draw,80,130+i*band+band/2,spec['labels'][i],48,FG)
            save(im,name)


def text(d,x,y,s,size=32,color=FG,anchor='lm'):
    f=ImageFont.truetype(FONT,size)
    box=d.textbbox((x,y),s,font=f,anchor=anchor)
    assert box[0]>=0 and box[1]>=0 and box[2]<=d._image.width and box[3]<=d._image.height,(s,box)
    d.text((x,y),s,font=f,fill=color,anchor=anchor)


def rect(d,x,y,w,h,color=BLUE,fill=None):
    d.rounded_rectangle((x,y,x+w,y+h),radius=14,outline=color,fill=fill or 'white',width=3)


def chip(d,x,y,s,color=BLUE,w=115):
    rect(d,x,y,w,68,color)
    text(d,x+w/2,y+34,s,30,color,'mm')


def arrow(d,a,b,color=MUTED):
    d.line([a,b],fill=color,width=4)
    dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
    d.polygon([b,(b[0]-16*ux+8*uy,b[1]-16*uy-8*ux),(b[0]-16*ux-8*uy,b[1]-16*uy+8*ux)],fill=color)


def canvas(title,subtitle):
    im=Image.new('RGB',(1600,900),'white');d=ImageDraw.Draw(im)
    text(d,80,86,title,51);text(d,80,155,subtitle,27,MUTED)
    return im,d


def save(im,name):
    ASSETS.mkdir(exist_ok=True);im.save(ASSETS/name,optimize=True)


def figures():
    im=Image.new('RGB',(2000,800),'white');d=ImageDraw.Draw(im)
    text(d,1000,190,'WHY CACHE K AND V?',116,FG,'mm')
    text(d,1000,310,'Each new token asks. Earlier K/V remain available.',37,MUTED,'mm')
    for row,y in enumerate([460,575]):
        text(d,490,y+34,['K','V'][row],42,[PURPLE,GREEN][row])
        for j in range(5):
            x=590+j*145;rect(d,x,y,116,68,[PURPLE,GREEN][row])
            for k in range(3):d.rounded_rectangle((x+19+k*29,y+22,x+36+k*29,y+46),radius=3,fill=[PURPLE,GREEN][row])
    chip(d,1460,485,'Q',BLUE,120);arrow(d,(1430,519),(1320,519),BLUE)
    save(im,'cover.png')

    im,d=canvas('Prefill once. Extend the cache.','One layer shown · illustrative token pieces · chronological rows')
    text(d,80,236,'Compute now',29,MUTED);text(d,640,236,'Retained K/V positions',29,MUTED);text(d,1260,236,'Predict next',29,MUTED)
    rows=[(320,'Prefill: The sky is',['P1','P2','P3'],'blue'),(510,'Decode: blue',['P1','P2','P3','blue'],'.'),(700,'Decode: .',['P1','P2','P3','blue','.'],'next')]
    for idx,(y,caption,labels,next_) in enumerate(rows):
        text(d,80,y+34,caption,32,BLUE)
        arrow(d,(440,y+34),(590,y+34),BLUE)
        for j,s in enumerate(labels):chip(d,620+j*118,y,s,GREEN if idx and j==len(labels)-1 else PURPLE,100)
        arrow(d,(1230,y+34),(1310,y+34));chip(d,1330,y,next_,GREEN,155)
    text(d,80,835,'Selecting a token comes before computing its K/V.',29,MUTED)
    save(im,FILES[0])

    im,d=canvas('Reuse follows the causal boundary','Fixed weights and positions · ordinary exact inference')
    text(d,80,278,'Append',37,BLUE)
    for j in range(4):chip(d,410+j*180,245,f'p{j+1}',PURPLE,140)
    chip(d,1190,245,'new',GREEN,140)
    d.line([(400,345),(1090,345)],fill=PURPLE,width=4)
    text(d,750,400,'Earlier causal states stay valid',33,PURPLE,'mm')
    text(d,80,583,'Edit prefix',37,AMBER)
    for j in range(5):chip(d,410+j*180,550,'edit' if j==0 else f'p{j+1}',AMBER,140)
    for j in range(4):arrow(d,(552+j*180,584),(584+j*180,584),AMBER)
    text(d,850,700,'Dependent later states can change',33,AMBER,'mm')
    text(d,80,822,'Cache identity includes prefix, position, layer and model settings.',29,MUTED)
    save(im,FILES[1])

    im,d=canvas('Saved computation. Checked outputs.','Recorded CPU float64 toy · 2 layers · 6 positions · prefill length 2')
    text(d,90,255,'Rows processed per K or V projection',33)
    for y,name,n,col in [(345,'Cached',12,BLUE),(490,'Full-prefix',40,MUTED)]:
        text(d,90,y,name,32,col)
        d.rounded_rectangle((310,y-25,310+n*13,y+25),radius=5,fill=col)
        text(d,340+n*13,y,str(n),39,col)
    for y,title,value,col in [(275,'Unchanged prefix','0.0',GREEN),(505,'Stale prefix','1.80691',AMBER)]:
        rect(d,1040,y,470,165,col)
        text(d,1275,y+44,title,30,col,'mm');text(d,1275,y+106,value,53,col,'mm')
    text(d,1275,730,'Max. absolute logit difference',27,MUTED,'mm')
    text(d,90,818,'Measured toy outputs + counted projections. No timing benchmark.',29,MUTED)
    save(im,FILES[2])

    im,d=canvas('The memory bill grows with the cache','24 layers · batch 1 · head width 64 · 2 bytes per element · separate K/V')
    cases=[(310,'4,096 positions · 8 KV heads',192,BLUE),(495,'8,192 positions · 8 KV heads',384,PURPLE),(680,'4,096 positions · 4 KV heads',96,GREEN)]
    for y,s,n,col in cases:
        text(d,80,y-60,s,32)
        d.rounded_rectangle((80,y-17,80+n*3.3,y+30),radius=5,fill=col)
        text(d,108+n*3.3,y+7,f'{n} MiB',36,col)
    text(d,80,835,'Calculated tensor payload only. Model weights and runtime overhead excluded.',28,MUTED)
    save(im,FILES[3])

    im,d=canvas('Three different changes to KV','DeepSeek-V4.1-Flash · schematic, not a full architecture diagram')
    text(d,80,270,'Source',34,BLUE)
    chip(d,370,230,'Encoder output',BLUE,340);arrow(d,(740,264),(830,264));chip(d,865,230,'Global KV',PURPLE,250)
    text(d,1180,264,'Causal Encoder–Decoder',26,MUTED)
    text(d,80,495,'Across layers',34,PURPLE)
    chip(d,370,450,'Global KV',PURPLE,270)
    d.line([(640,484),(780,484),(780,390),(1395,390)],fill=PURPLE,width=4)
    for j in range(3):
        chip(d,920+j*200,450,['L21','L22','…L40'][j],BLUE,150)
        arrow(d,(995+j*200,390),(995+j*200,450),PURPLE)
    text(d,1070,583,'Distinct Q and local KV at each layer',28,MUTED,'mm')
    text(d,80,747,'K / V roles',34,GREEN)
    chip(d,370,710,'One vector',PURPLE,270)
    arrow(d,(665,744),(845,698),PURPLE);arrow(d,(665,744),(845,808),GREEN)
    text(d,875,698,'Q matching → score',33,PURPLE)
    text(d,875,808,'Weighted sum → output',33,GREEN)
    save(im,FILES[4])


STYLE='''*{box-sizing:border-box}body{margin:0;background:white;color:#111827;font:19px/1.75 Arial,"Songti SC",serif}main{width:min(900px,calc(100% - 36px));margin:28px auto 80px}nav{display:flex;justify-content:space-between;font-size:16px;margin-bottom:28px}a{color:#2563eb;text-underline-offset:3px}h1{font-size:clamp(34px,5vw,54px);font-weight:400;line-height:1.18;margin:48px 0 24px}h2{font-size:30px;font-weight:400;line-height:1.3;margin-top:56px}p{margin:0 0 23px}strong{font-weight:600}figure{margin:34px 0}img,video{width:100%;display:block;border:1px solid #e2e8f0;border-radius:9px}figcaption{font-size:15px;color:#64748b;line-height:1.55;margin-top:12px}blockquote{margin:26px 0;padding:18px 22px;border-left:3px solid #2563eb;background:#f8fafc;overflow-wrap:anywhere}blockquote p{margin:0}details{margin-top:48px;border-top:1px solid #cbd5e1;padding-top:24px}summary{cursor:pointer}video{margin:20px 0}li{margin:8px 0}@media(max-width:600px){body{font-size:18px}h2{font-size:27px}main{width:calc(100% - 28px)}}'''


def pages(languages=('en', 'zh')):
    md=mistune.create_markdown(escape=False)
    for lang in languages:
        folder=P if lang=='en' else P/'zh'
        prefix='assets/' if lang=='en' else '../assets/'
        source=(folder/'x-editor-draft-body-with-image-placeholders.md').read_text()
        assert not re.search(r'https?://|\[[^\]]+\]\([^)]+\)',source), 'Article must be self-contained and link-free'
        captions=re.findall(r'^\[IMAGE (\d{2})\] (.+)$',source,re.M)
        assert [int(n) for n,_ in captions]==list(range(1,len(ARTICLE_FILES)+1))
        transfer=re.sub(r'\A<h1>.*?</h1>\s*','',md(source),count=1,flags=re.S)
        # X paste handles quote-style rich text more reliably than native code blocks.
        def code_quote(match):
            lines=match[1].rstrip('\n').split('\n')
            return '<blockquote><p>'+ '<br>\n'.join(line.replace(' ', '&nbsp;') for line in lines) +'</p></blockquote>'
        transfer=re.sub(r'<pre><code[^>]*>(.*?)</code></pre>',code_quote,transfer,flags=re.S)
        (folder/'x-editor-draft-body-with-image-placeholders.html').write_text('<!doctype html>\n<meta charset="utf-8">\n'+transfer)
        def figure(match):
            i=int(match[1])-1;cap=html.escape(match[2]);name=ARTICLE_FILES[i];url=prefix+name
            kind='equation' if name in EQUATIONS else 'diagram'
            if name.endswith('.gif'):
                still=prefix+Path(name).stem+'-still.png'
                pause='Pause animation' if lang=='en' else '暂停动画'
                play='Play animation' if lang=='en' else '播放动画'
                return (f'<figure class="animation"><picture><source media="(prefers-reduced-motion: reduce)" srcset="{still}">'
                        f'<img src="{url}" alt="{cap}" data-motion="{url}" data-still="{still}"></picture>'
                        f'<button type="button" class="animation-toggle" aria-pressed="true" data-pause="{pause}" data-play="{play}">{pause}</button>'
                        f'<figcaption>{cap}</figcaption></figure>')
            return f'<figure class="{kind}"><img src="{url}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
        body=md(re.sub(r'^\[IMAGE (\d{2})\] (.+)$',figure,source,flags=re.M))
        videos='../../../visuals/animations/projects/deepseek-ced/' if lang=='en' else '../../../../visuals/animations/projects/deepseek-ced/'
        companion=''
        if lang=='en':
            companion='<details><summary>Companion animations</summary>'
            for clip,title in [('01-global-kv','Causal Encoder–Decoder'),('02-shared-roles','Shared KV')]:
                companion+=f'<h2>{title}</h2><video controls playsinline preload="none" poster="{videos}assets/deepseek-v41-architecture-poster.png" src="{videos}rendered/{clip}/{clip}.mp4"></video>'
            companion+='</details>'
        title=source.splitlines()[0][2:]
        alternate='zh/review.html' if lang=='en' else '../review.html'
        code_style='code{font:0.9em Menlo,monospace;overflow-wrap:anywhere}pre{background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;padding:20px;overflow-x:auto;font:16px/1.65 Menlo,monospace}pre code{font:inherit}.equation img{border:none;border-radius:0}.equation{margin:30px 0}@media(max-width:600px){pre{font-size:12px;padding:12px}}'
        controls_style='.animation-toggle{font:15px Arial,sans-serif;color:#475569;border:1px solid #cbd5e1;border-radius:6px;background:white;padding:8px 12px;margin-top:12px;cursor:pointer}'
        controls_script='''document.querySelectorAll('.animation-toggle').forEach(b=>{const f=b.closest('figure'),im=f.querySelector('img');const set=playing=>{f.querySelector('source')?.remove();im.src=playing?im.dataset.motion:im.dataset.still;b.setAttribute('aria-pressed',String(playing));b.textContent=playing?b.dataset.pause:b.dataset.play;};if(matchMedia('(prefers-reduced-motion: reduce)').matches)set(false);b.addEventListener('click',()=>set(b.getAttribute('aria-pressed')!=='true'));});'''
        page=f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}{code_style}{controls_style}h1,h2{{text-wrap:balance}}</style></head><body><main><nav><span>Dongxi · Attention</span><a href="{alternate}">'+('中文版' if lang=='en' else 'English')+f'</a></nav><img src="{prefix}{COVER}" alt="KV Cache — retained information and reuse"><article>{body}</article>{companion}</main><script>{controls_script}</script></body></html>'
        (folder/'review.html').write_text(page)
        order='# Image Upload Order\n\n'+'\n'.join(f'{n}. `{prefix}{ARTICLE_FILES[int(n)-1]}` — {cap}' for n,cap in captions)+'\n'
        (folder/'image-upload-order.md').write_text(order)
        subprocess.run([sys.executable,str(Path.home()/'.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py'),str(folder)],check=True)


def main():
    assert 2*24*4096*8*64*2==192*1024**2
    assert 2*24*8192*8*64*2==384*1024**2
    assert 2*24*4096*4*64*2==96*1024**2
    assert 2*(2+1+1+1+1)==12 and 2*sum([2,3,4,5,6])==40
    figures();equations();animation=animation_assets();pages()
    entries=[]
    for path in sorted(p for p in ASSETS.iterdir() if p.suffix in ('.png','.gif')):
        with Image.open(path) as im: dimensions=list(im.size)
        entries.append(dict(path=str(path.relative_to(P)),dimensions=dimensions,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    evidence=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        figures='Code-native diagrams and animations; AI-generated editorial cover; shared English visual labels',
        cover=dict(path='assets/'+COVER,method='built-in image generation',prompt='cover-concepts/kv-cache-v2.md'),
        font='Arial Regular; natural whole-string layout',cover_ratio='5:2',
        equations=EQUATIONS,math_renderer='Matplotlib MathText '+matplotlib.__version__,
        inline_image_order=ARTICLE_FILES,
        animations=animation,
        calculations=dict(cache_mib=[192,384,96],projected_rows_per_k_or_v=dict(cached=12,uncached=40)),
        experimental_values='Transcribed from the committed 2026-09-05 report, not rerun on this host',
        source_report_sha256=hashlib.sha256((ROOT/'experiments/reports/2026-09-05-attention-gradients-cache.md').read_bytes()).hexdigest(),
        files=entries)
    (P/'metadata.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
