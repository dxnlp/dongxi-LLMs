"""Original code-native static diagrams; no browser, model or animation job."""
from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import math
import re
import shutil
import sys

import mistune
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
A, Z = P / 'assets', P / 'zh'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
FG, MUTED, GRID = '#111827', '#64748B', '#CBD5E1'
BLUE, PURPLE, GREEN, ORANGE = '#2563EB', '#7C3AED', '#047857', '#B45309'
FILES = ['01-lifecycle.png', '02-first-token.png', '03-shapes.png',
         '04-scheduling.png', '05-handoff.png']
ORDER = FILES[:1] + ['07-parallel-positions.png'] + FILES[1:2] + ['06-decode-loop.png'] + FILES[2:]
CLIPS = ['01-lifecycle','02-parallel','03-first-token','04-decode-loop',
         '05-workloads','06-scheduling','07-handoff']
ANIMATION_ROOT = P.parents[2]/'visuals/animations/projects/prefill-decode-article'
ANIMATED = '--animations' in sys.argv
COVER = 'cover-prefill-decode-v2.png'
BOUNDS = []


def text(d, x, y, s, size=32, color=FG, anchor='lm'):
    f = ImageFont.truetype(FONT, size)
    b = d.textbbox((x, y), s, font=f, anchor=anchor)
    assert 0 <= b[0] <= b[2] <= d._image.width and 0 <= b[1] <= b[3] <= d._image.height, (s, b)
    BOUNDS.append({'text': s, 'bounds': b})
    d.text((x, y), s, font=f, fill=color, anchor=anchor)


def box(d, x, y, w, h, s='', c=BLUE, size=32, fill='white'):
    d.rounded_rectangle((x, y, x+w, y+h), radius=12, outline=c, width=3, fill=fill)
    if s:
        assert d.textlength(s, font=ImageFont.truetype(FONT, size)) <= w-22, s
        text(d, x+w/2, y+h/2, s, size, c, 'mm')


def arrow(d, x1, y1, x2, y2, c=MUTED):
    d.line((x1,y1,x2,y2), fill=c, width=4)
    a = math.atan2(y2-y1, x2-x1)
    d.polygon([(x2,y2),(x2-17*math.cos(a)+8*math.sin(a),y2-17*math.sin(a)-8*math.cos(a)),
               (x2-17*math.cos(a)-8*math.sin(a),y2-17*math.sin(a)+8*math.cos(a))], fill=c)


def canvas(title, subtitle):
    im = Image.new('RGB', (1600,900), 'white')
    d = ImageDraw.Draw(im)
    text(d,80,85,title,50)
    text(d,80,160,subtitle,28,MUTED)
    return im,d


def save(im, name):
    im.save(A/name, optimize=True)


def lifecycle():
    im,d = canvas('One model. Two phases.', 'Known input → first output → autoregressive continuation')
    box(d,80,300,270,98,'Queue + input',MUTED,30)
    box(d,390,300,440,98,'Prefill',BLUE,42)
    box(d,930,300,580,98,'Decode',GREEN,42)
    arrow(d,350,349,386,349)
    arrow(d,832,349,925,349)
    d.line((875,240,875,620),fill=ORANGE,width=3)
    text(d,875,215,'y1',36,ORANGE,'mm')
    for x,s in [(1050,'y2'),(1230,'y3'),(1410,'y4')]:
        text(d,x,470,s,32,GREEN,'mm')
        d.ellipse((x-9,416,x+9,434),fill=GREEN)
    text(d,610,453,'Build prompt KV',30,PURPLE,'mm')
    arrow(d,80,568,869,568,ORANGE)
    text(d,470,614,'TTFT',37,ORANGE,'mm')
    arrow(d,1060,568,1220,568,GREEN)
    text(d,1140,614,'ITL',37,GREEN,'mm')
    text(d,800,738,'Same parameters · different input shapes',36,FG,'mm')
    text(d,80,844,'Order only, not timing. Client latency also includes transport and other overhead.',25,MUTED)
    save(im,FILES[0])


def first_token():
    im,d = canvas('The first token is a boundary.', 'Known prompt positions can run together within each layer')
    text(d,305,241,'Causal attention',34,FG,'mm')
    for i in range(4):
        text(d,240+i*83,300,f'x{i+1}',26,BLUE,'mm')
        text(d,160,363+i*83,f'x{i+1}',26,BLUE,'mm')
        for j in range(4):
            x,y=208+j*83,330+i*83
            d.rounded_rectangle((x,y,x+65,y+65),radius=7,fill=BLUE if j<=i else '#F1F5F9')
    text(d,368,704,'Allowed: self + earlier positions',27,MUTED,'mm')
    d.line((705,238,705,785),fill=GRID,width=2)
    text(d,1120,247,'Cache at each layer',34,PURPLE,'mm')
    for y,n,label in [(337,4,'Select y1'),(628,5,'Predict y2')]:
        for i in range(n):
            box(d,820+i*112,y,95,70,f'x{i+1}' if i<4 else 'y1',PURPLE if i<4 else GREEN,29)
        text(d,1460,y+35,f'{n} entries',26,MUTED,'mm')
        text(d,1090,y+113,label,32,ORANGE if n==4 else GREEN,'mm')
    arrow(d,1090,484,1090,605,GREEN)
    box(d,1160,500,270,64,'Feed y1 forward',GREEN,27)
    text(d,80,844,'y1 has no KV until its forward pass. Every layer owns a separate cache.',25,MUTED)
    save(im,FILES[1])


def matrix(d,x,y,rows,cols,c,cell=42):
    for i in range(rows):
        for j in range(cols):
            d.rounded_rectangle((x+j*cell,y+i*cell,x+j*cell+cell-5,y+i*cell+cell-5),radius=4,fill=c)


def shapes():
    im,d = canvas('Same model. Different workloads.', 'How much must it read? How much must it write?')
    d.line((800,224,800,755),fill=GRID,width=2)
    for x,title,left_n,right_n,focus in [(80,'Long report → 3 bullets',9,3,'Prefill'),
                                        (900,'Short prompt → Long speech',2,9,'Decode')]:
        text(d,x,258,title,34)
        box(d,x+10,335,220,300,c=GRID)
        box(d,x+380,335,220,300,c=GRID)
        for j in range(left_n):
            y=360+j*29
            d.rounded_rectangle((x+32,y,x+207,y+10),radius=4,fill=BLUE)
        for j in range(right_n):
            y=360+j*29
            if right_n==3:
                d.ellipse((x+403,y,x+413,y+10),fill=GREEN)
                start=x+430
            else:
                start=x+402
            d.rounded_rectangle((start,y,x+577,y+10),radius=4,fill=GREEN)
        arrow(d,x+260,485,x+352,485,PURPLE)
        text(d,x+120,680,'Input',28,BLUE,'mm')
        text(d,x+490,680,'Answer',28,GREEN,'mm')
        text(d,x+305,751,f'Look at {focus.lower()}',32,BLUE if focus=='Prefill' else GREEN,'mm')
    text(d,80,844,'Illustrative input / output lengths, not measured token counts or latency.',25,MUTED)
    save(im,FILES[2])


def scheduling():
    im,d = canvas('Keep A writing while B enters.', 'A: an answer in progress · B: a newly submitted document')
    text(d,80,253,'One resource pool',35)
    for i in range(3):
        x=465+i*344
        text(d,x+147,240,f'Iteration {i+1}',29,MUTED,'mm')
        box(d,x,285,296,134,c=GRID)
        box(d,x+13,299,82,106,'A',GREEN,35)
        box(d,x+107,299,176,106,f'B{i+1}',BLUE,35)
    text(d,80,365,'A = decode',28,GREEN)
    text(d,80,408,'B1–B3 = input chunks',28,BLUE)
    text(d,966,465,'B retains its context across chunks',29,PURPLE,'mm')
    d.line((80,520,1520,520),fill=GRID,width=2)
    text(d,80,580,'Separate resource pools',35)
    box(d,100,666,390,103,'Prefill workers',BLUE,36)
    box(d,1110,666,390,103,'Decode workers',GREEN,36)
    arrow(d,530,714,1070,714,PURPLE)
    text(d,800,665,'KV + request state',33,PURPLE,'mm')
    text(d,800,766,'compatible model + cache format',25,MUTED,'mm')
    text(d,80,844,'A and B keep separate request state. Schematic budgets; no timing measurement.',25,MUTED)
    save(im,FILES[3])


def handoff():
    im,d = canvas('A handoff can overlap with compute.', 'Illustrative Spark → Mac placement from the EXO demonstration')
    box(d,80,226,550,95,'DGX Spark · prefill',BLUE,37)
    box(d,970,226,550,95,'M3 Ultra · decode',GREEN,37)
    arrow(d,667,273,935,273,PURPLE)
    text(d,800,237,'KV',30,PURPLE,'mm')
    text(d,80,386,'Prefill',32,BLUE)
    for i in range(3):
        box(d,350+i*220,354,205,65,f'L{i+1}',BLUE,30)
    text(d,80,510,'KV transfer',32,PURPLE)
    for i in range(3):
        x=565+i*245
        y=472+i*81
        box(d,x,y,245,58,f'KV · L{i+1}',PURPLE,28)
    d.line((1300,415,1300,750),fill=GRID,width=2)
    box(d,1330,634,190,80,'Decode',GREEN,31)
    arrow(d,1302,673,1326,673,GREEN)
    text(d,1170,756,'Transfer tail',28,ORANGE,'mm')
    arrow(d,350,801,1510,801,MUTED)
    text(d,80,853,'Order only. Overlap needs suitable software and bandwidth; communication still costs time.',25,MUTED)
    save(im,FILES[4])


def cover():
    im=Image.new('RGB',(2000,800),'white')
    d=ImageDraw.Draw(im)
    text(d,1000,244,'Prefill and Decode',152,FG,'mm')
    for i in range(5):
        box(d,250+i*91,438,71,125,c=BLUE)
    arrow(d,745,501,990,501,PURPLE)
    for i in range(4):
        box(d,1050+i*167,463,101,76,c=GREEN)
        if i<3:
            arrow(d,1168+i*167,501,1199+i*167,501,GREEN)
    save(im,'cover.png')


def decode_loop():
    im,d = canvas('One selected token becomes the next input.',
                  'Feed y1 → compute its KV → select y2 → repeat')
    box(d,130,293,200,94,'y1',GREEN,48)
    text(d,230,249,'Selected',28,GREEN,'mm')
    box(d,580,280,440,120,'Same model',BLUE,42)
    box(d,1270,293,200,94,'y2',GREEN,48)
    text(d,1370,249,'Next selected token',28,GREEN,'mm')
    arrow(d,358,340,548,340,GREEN)
    arrow(d,1050,340,1240,340,GREEN)
    text(d,1115,292,'Predict + select',25,MUTED,'mm')
    # A single persistent cache, with a distinct appended entry.
    for i in range(4):
        box(d,400+i*150,585,120,86,f'x{i+1}',PURPLE,34)
    box(d,1000,585,120,86,'y1',GREEN,34)
    text(d,685,720,'Retain prefix KV',30,PURPLE,'mm')
    text(d,1060,720,'New KV',30,GREEN,'mm')
    # History enters the model; new per-layer entries return to the bank.
    d.line((385,626,355,626,355,445,635,445),fill=PURPLE,width=4)
    arrow(d,635,445,635,415,PURPLE)
    text(d,470,492,'Read KV',28,PURPLE,'mm')
    d.line((960,415,960,453,1060,453),fill=GREEN,width=4)
    arrow(d,1060,453,1060,568,GREEN)
    text(d,1165,509,'Append',28,GREEN,'mm')
    # Feedback travels outside the cache; it does not duplicate cached entries.
    d.line((1370,415,1370,790,230,790,230,445),fill=GRID,width=4)
    arrow(d,230,445,230,407,GREEN)
    text(d,800,844,'Per-layer cache shown schematically. y2 receives its KV on the next forward pass.',25,MUTED,'mm')
    save(im,'06-decode-loop.png')


def article():
    source=(Z/'x-editor-draft-body-with-image-placeholders.md').read_text()
    title,body=source.split('\n',1)
    title=title.removeprefix('# ')
    raw=mistune.html(body)
    (Z/'x-editor-draft-body-with-image-placeholders.html').write_text(raw)
    order='# Inline image upload order\n\n'
    for i,static_name in enumerate(ORDER,1):
        stem=CLIPS[i-1]
        name=stem+'.gif' if ANIMATED else static_name
        m=re.search(r'\[IMAGE '+f'{i:02d}'+r'\] ([^\n]+)',source)
        assert m
        caption=m[1]
        order+=f'{i:02d}. `../assets/{name}` — {caption}\n'
        if ANIMATED:
            poster=stem+'-poster.png'
            figure=f'<figure class="motion"><img id="motion-{i}" src="../assets/{poster}" data-gif="../assets/{name}" data-still="../assets/{poster}" width="960" height="540" alt="{html.escape(caption,quote=True)}"><button type="button" aria-controls="motion-{i}" aria-pressed="false" hidden>播放动画</button><figcaption>{html.escape(caption)}</figcaption></figure>'
        else:
            figure=f'<figure><img src="../assets/{name}" width="1600" height="900" alt="{html.escape(caption,quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        raw,n=re.subn(r'<p>\[IMAGE '+f'{i:02d}'+r'\] .*?</p>',lambda _:figure,raw)
        assert n==1
    (Z/'image-upload-order.md').write_text(order)
    css='''*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#fff;color:#111827;font-family:Arial,"Songti SC",STSong,serif}main{max-width:940px;margin:45px auto 100px;padding:0 28px}header img{width:100%;height:auto}h1{font-size:44px;font-weight:400;margin:40px 0}h2{font-size:29px;line-height:1.5;margin:60px 0 26px}p{font-size:20px;line-height:1.95;margin:0 0 22px;overflow-wrap:anywhere}figure{margin:36px -14px}figure img{width:100%;height:auto;display:block;border:1px solid #e2e8f0;border-radius:8px}figcaption{font-size:15px;line-height:1.8;color:#64748b;padding:14px 10px}@media(max-width:600px){main{padding:0 20px;margin-top:18px}h1{font-size:30px}h2{font-size:24px;margin-top:40px}p{font-size:18px}figure{margin:28px -12px}}'''
    css+='button{margin:10px 10px 0;padding:7px 12px;border:1px solid #cbd5e1;border-radius:6px;background:white;color:#475569;cursor:pointer;font-size:15px}button:focus-visible{outline:2px solid #2563eb}'
    script='<script>'+ (P/'review-player.js').read_text()+'</script>' if ANIMATED else ''
    page=f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main><header><img src="../assets/{COVER}" width="2000" height="800" alt="Prefill and Decode"><h1>{title}</h1></header><article>{raw}</article></main>{script}</body></html>\n'
    (Z/'review.html').write_text(page)


def parallel_positions():
    im,d = canvas('Read the inputs. Produce separate outputs.',
                  'One causal attention head · two positions in the same layer')
    for x,s in [(180,'Input'),(530,'Projections'),(1060,'Attention'),(1430,'Output')]:
        text(d,x,247,s,30,MUTED,'mm')
    for i,y in enumerate([365,650],1):
        box(d,110,y-45,140,90,f'h{i}',BLUE,40)
        box(d,380,y-45,300,90,f'Q{i} / K{i} / V{i}',PURPLE,33)
        arrow(d,275,y,350,y,BLUE)
        box(d,940,y-54,240,108,f'Position {i}',BLUE,34)
        arrow(d,705,y,910,y,PURPLE)
        box(d,1370,y-45,120,90,f'O{i}',GREEN,40)
        arrow(d,1210,y,1340,y,GREEN)
    arrow(d,699,417,917,598,PURPLE)
    text(d,845,460,'K1 / V1',30,PURPLE,'mm')
    text(d,1060,770,'Both rows can run in parallel',33,FG,'mm')
    text(d,80,844,'O2 reads Q2, K1/V1 and K2/V2. It does not need O1. Other block operations omitted.',25,MUTED)
    save(im,'07-parallel-positions.png')


def main():
    A.mkdir(exist_ok=True)
    animation_files=[]
    if ANIMATED:
        for stem in CLIPS:
            for source,name in [(stem+'.gif',stem+'.gif'),('still.png',stem+'-poster.png')]:
                shutil.copy2(ANIMATION_ROOT/'rendered'/stem/source,A/name)
                animation_files.append('assets/'+name)
    for fn in [lifecycle,first_token,shapes,scheduling,handoff,decode_loop,parallel_positions,cover]:
        fn()
    article()
    sheet=Image.new('RGB',(1000,4*306),'#E2E8F0')
    with Image.open(A/COVER) as selected_cover:
        assert selected_cover.size == (2000,800)
    for i,name in enumerate(ORDER+[COVER]):
        im=Image.open(A/name)
        im.thumbnail((500,281),Image.Resampling.LANCZOS)
        x,y=i%2*500,i//2*306
        sheet.paste(im,(x,y))
        text(ImageDraw.Draw(sheet),x+12,y+292,name,17)
    sheet.save(P/'contact-sheet.png')
    metadata={'kind':'original animated schematics' if ANIMATED else 'original static schematics','animation_render':ANIMATED,'model_benchmark':False,
              'command':'uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-prefill-decode-001/build.py'+(' --animations' if ANIMATED else ''),
              'dependencies':{s:importlib.metadata.version(s) for s in ['Pillow','mistune']},
              'text_bounds_checked':len(BOUNDS),'sources':{},'outputs':{},
              'cover':{'path':'assets/'+COVER,'method':'built-in image generation',
                       'prompt':'cover-concepts/prefill-decode-v2.md','ratio':'5:2'}}
    for name in ['build.py','review-player.js','zh/x-editor-draft-body-with-image-placeholders.md','source-map.md','visual-plan.md','cover-concepts/prefill-decode-v2.md','cover-concepts/prefill-decode-v2-source.png']:
        metadata['sources'][name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
    for name in [*(f'assets/{s}' for s in ORDER+['cover.png',COVER]),'zh/review.html','contact-sheet.png',*animation_files]:
        metadata['outputs'][name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
    (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if ANIMATED:
        metadata['animation_manifest_sha256']=hashlib.sha256((ANIMATION_ROOT/'metadata.json').read_bytes()).hexdigest()
        (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Built seven '+('animated' if ANIMATED else 'static')+' figures, 5:2 cover and Chinese local review.')


if __name__=='__main__':
    main()
