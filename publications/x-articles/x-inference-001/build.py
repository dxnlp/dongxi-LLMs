"""Render original static figures and a local Chinese X Article review.

No model execution, browser control or external asset download. All diagrams
are schematic except the explicit weight-payload arithmetic in figure 05.
"""
from pathlib import Path
import hashlib
import html
import json
import math
import re
import importlib.metadata

import mistune
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
ASSETS = P / 'assets'
ZH = P / 'zh'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
FG, MUTED, GRID = '#111827', '#64748B', '#CBD5E1'
BLUE, PURPLE, GREEN, ORANGE = '#2563EB', '#7C3AED', '#047857', '#B45309'
PALE = '#F8FAFC'
FILES = ['01-latency.png', '02-batching.png', '03-phases.png',
         '04-speculation.png', '05-quantization.png', '06-parallelism.png',
         '07-map.png']
TEXT_BOUNDS = []


def text(d, x, y, s, size=32, color=FG, anchor='lm'):
    font = ImageFont.truetype(FONT, size)
    box = d.textbbox((x, y), s, font=font, anchor=anchor)
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= d._image.width and box[3] <= d._image.height, (s, box)
    TEXT_BOUNDS.append({'text': s, 'bounds': box})
    d.text((x, y), s, font=font, fill=color, anchor=anchor)


def box(d, x, y, w, h, label='', color=BLUE, fill='white', size=32):
    d.rounded_rectangle((x, y, x+w, y+h), radius=16, outline=color, width=3, fill=fill)
    if label:
        font = ImageFont.truetype(FONT, size)
        assert d.textlength(label, font=font) <= w-24, label
        text(d, x+w/2, y+h/2, label, size, color, 'mm')


def arrow(d, x1, y1, x2, y2, color=MUTED):
    d.line((x1, y1, x2, y2), fill=color, width=4)
    a = math.atan2(y2-y1, x2-x1)
    d.polygon([(x2,y2), (x2-17*math.cos(a)+8*math.sin(a), y2-17*math.sin(a)-8*math.cos(a)),
               (x2-17*math.cos(a)-8*math.sin(a),y2-17*math.sin(a)+8*math.cos(a))], fill=color)


def canvas(title, subtitle):
    im = Image.new('RGB', (1600,900), 'white')
    d = ImageDraw.Draw(im)
    text(d,80,90,title,52)
    text(d,80,164,subtitle,28,MUTED)
    return im,d


def save(im, name):
    im.save(ASSETS/name, optimize=True)


def latency():
    im,d = canvas('One request. Several clocks.', 'Request sent → first output → subsequent outputs')
    xs = [80,340,640]
    for x,w,s,c in [(xs[0],230,'Queue + input',MUTED),(xs[1],270,'Prefill',BLUE),(xs[2],880,'Decode',GREEN)]:
        box(d,x,285,w,86,s,c)
    text(d,650,245,'First token',30,ORANGE)
    d.line((650,278,650,618),fill=ORANGE,width=3)
    for x in [650,835,1020,1205,1390]:
        d.ellipse((x-11,430,x+11,452),fill=GREEN)
    arrow(d,80,440,1520,440)
    arrow(d,80,548,640,548,ORANGE)
    text(d,355,591,'TTFT',39,ORANGE,'mm')
    arrow(d,845,540,1010,540,GREEN)
    text(d,928,591,'ITL',39,GREEN,'mm')
    arrow(d,80,705,1490,705,PURPLE)
    text(d,790,755,'End-to-end latency',35,PURPLE,'mm')
    text(d,80,846,'Schematic timing. Client / server measurement boundaries matter.',25,MUTED)
    save(im,FILES[0])


def batching():
    im,d = canvas('Keep the work moving.', 'Continuous batching: request membership changes between iterations')
    for x,title in [(110,'Iteration n'),(900,'Next iteration')]:
        text(d,x,266,title,34)
    for y,label,color in [(335,'A · finishes',GREEN),(475,'B · continues',BLUE)]:
        box(d,110,y,430,90,label,color)
    arrow(d,585,520,850,520)
    text(d,720,574,'reschedule',29,MUTED,'mm')
    box(d,900,335,490,90,'C · joins',PURPLE)
    box(d,900,475,490,90,'B · continues',BLUE)
    box(d,110,695,430,75,'C · waiting',PURPLE)
    text(d,790,730,'Separate context and KV for each request',31,MUTED)
    text(d,80,846,'Admission needs input processing and available token / KV capacity.',25,MUTED)
    save(im,FILES[1])


def phases():
    im,d = canvas('Schedule phases. Or separate them.', 'Chunk size and worker placement are different decisions')
    text(d,80,265,'One worker pool',34)
    labels = [('P1',BLUE),('D',GREEN),('P2',BLUE),('D',GREEN),('P3',BLUE),('D',GREEN)]
    for i,(s,c) in enumerate(labels):
        box(d,440+i*174,225,150,84,s,c)
    text(d,445,355,'P = prefill chunk     D = decode work',29,MUTED)
    d.line((80,431,1520,431),fill=GRID,width=2)
    text(d,80,503,'Separate pools',34)
    box(d,120,585,400,120,'Prefill workers',BLUE)
    box(d,1060,585,420,120,'Decode workers',GREEN)
    arrow(d,555,645,1020,645,PURPLE)
    text(d,785,593,'KV transfer',36,PURPLE,'mm')
    text(d,785,720,'bandwidth + coordination',27,MUTED,'mm')
    text(d,80,846,'Same model context retained. Separation is not a throughput guarantee.',25,MUTED)
    save(im,FILES[2])


def speculation():
    im,d = canvas('Propose. Verify. Continue.', 'Illustrative rejection branch of speculative decoding')
    text(d,90,288,'Draft',34)
    text(d,90,505,'Verify',34)
    text(d,90,730,'Continue',34)
    for i,s in enumerate('abcd'):
        x=390+i*245
        box(d,x,244,165,86,s,BLUE,size=40)
        arrow(d,x+82,350,x+82,436,MUTED)
        c = GREEN if i<2 else ORANGE
        box(d,x,460,165,86,s,c,size=40)
        text(d,x+82,587,['accept','accept','reject','discard'][i],28,c,'mm')
        if i>=2:
            d.line((x+20,530,x+145,475),fill=ORANGE,width=4)
    for i,(s,c) in enumerate([('a',GREEN),('b',GREEN),('x',ORANGE)]):
        box(d,390+i*245,685,165,86,s,c,size=40)
    text(d,1257,729,'new branch',29,MUTED,'mm')
    text(d,80,846,'Probability-rule verification, not fact-checking. Acceptance count is illustrative.',25,MUTED)
    save(im,FILES[3])


def quantization():
    im,d = canvas('Same weights. Fewer storage bits.', '8 billion parameters · ideal weight payload only')
    for y,bits,color in [(315,16,BLUE),(555,4,PURPLE)]:
        payload=8_000_000_000*bits//8
        text(d,80,y+50,f'{bits} bit',40,color)
        width=payload/16_000_000_000*950
        d.rounded_rectangle((300,y,300+width,y+100),radius=10,fill=color)
        text(d,330+width,y+50,f'{payload//1_000_000_000} GB',40,color)
    text(d,80,760,'Weights       +       KV cache       +       runtime buffers',36,MUTED)
    text(d,80,846,'Decimal GB. Bars exclude metadata and KV; no speed or quality measurement.',25,MUTED)
    save(im,FILES[4])


def parallelism():
    im,d = canvas('Split the model. Or add replicas.', 'Model parallelism and request-level replication serve different needs')
    text(d,80,254,'One model replica · tensor-parallel sketch',34)
    box(d,100,338,200,88,'Request A',BLUE, size=30)
    box(d,480,315,375,135,'GPU 1 · shard',PURPLE)
    box(d,1085,315,375,135,'GPU 2 · shard',PURPLE)
    arrow(d,325,382,445,382,BLUE)
    arrow(d,885,362,1055,362,PURPLE)
    arrow(d,1055,405,885,405,PURPLE)
    text(d,970,488,'communicate',28,MUTED,'mm')
    d.line((80,550,1520,550),fill=GRID,width=2)
    text(d,80,608,'Two replicas · separate requests',34)
    for x,s in [(100,'A'),(900,'B')]:
        box(d,x,685,185,90,'Request '+s,BLUE,size=28)
        arrow(d,x+202,730,x+274,730,BLUE)
        box(d,x+294,668,325,125,'Full model',GREEN)
    text(d,80,846,'Replicas can themselves span GPUs. More devices do not imply linear speedup.',25,MUTED)
    save(im,FILES[5])


def overview():
    im,d = canvas('Put each optimization in its place.', 'Choose by bottleneck, then check the trade-off')
    rows=[('REUSE','Prefix caching','Skip matching prefill',PURPLE),
          ('SCHEDULE','Continuous batching / chunks','Organize concurrent work',BLUE),
          ('PLACE','Prefill / decode separation','Isolate phase workloads',BLUE),
          ('PROPOSE','Speculative decoding','Verify multiple candidates',GREEN),
          ('REPRESENT','Quantization','Store lower-bit tensors',PURPLE),
          ('DISTRIBUTE','Model shards / replicas','Spread model or requests',ORANGE)]
    for i,(name,method,role,c) in enumerate(rows):
        y=248+i*92
        d.line((80,y+53,1520,y+53),fill=GRID,width=1)
        text(d,80,y+9,name,25,c)
        text(d,340,y+9,method,31)
        text(d,1020,y+9,role,28,MUTED)
    text(d,80,846,'Latency · throughput · memory · quality · cost',29,MUTED)
    save(im,FILES[6])


def cover():
    im=Image.new('RGB',(2000,800),'white'); d=ImageDraw.Draw(im)
    text(d,140,232,'LLM Inference',155)
    for i,c in enumerate([BLUE,PURPLE,GREEN]):
        y=475+i*77
        d.line((150,y,1825,y),fill=GRID,width=3)
        for j in range(9):
            x=180+j*185
            w=100 if j<3 else 38
            box(d,x,y-15,w,30,color=c,fill=c)
    save(im,'cover.png')


def article():
    source=(ZH/'x-editor-draft-body-with-image-placeholders.md').read_text()
    title=source.splitlines()[0][2:]
    body=source.split('\n',1)[1].strip()
    markdown=mistune.create_markdown(escape=True)
    raw=markdown(body)
    (ZH/'x-editor-draft-body-with-image-placeholders.html').write_text(raw)
    captions=re.findall(r'^\[IMAGE (\d{2})\] (.+)$',source,re.M)
    assert [n for n,_ in captions] == [f'{i:02}' for i in range(1,8)]
    order='# Image Upload Order\n\n'
    for (n,caption),filename in zip(captions,FILES):
        order+=f'{n}. `../assets/{filename}` — {caption}\n'
        pattern=r'<p>\[IMAGE '+n+r'\] .*?</p>'
        figure=f'<figure><img src="../assets/{filename}" width="1600" height="900" loading="lazy" alt="{html.escape(caption,quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        raw,count=re.subn(pattern,lambda _:figure,raw)
        assert count == 1
    (ZH/'image-upload-order.md').write_text(order)
    style='''*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#fff;color:#111827;font-family:Arial,"Songti SC",STSong,serif}main{max-width:940px;margin:48px auto 100px;padding:0 28px}header img{width:100%;height:auto;display:block;border:1px solid #e2e8f0}h1{font-size:42px;line-height:1.4;font-weight:400;margin:40px 0 26px}h2{font-size:29px;line-height:1.5;margin:62px 0 24px;font-weight:600}p,li{font-size:20px;line-height:1.95;overflow-wrap:anywhere}p{margin:0 0 21px}strong{font-weight:700}ul{padding-left:25px}li{margin:12px 0}figure{margin:36px -14px}figure img{display:block;width:100%;height:auto;border:1px solid #e2e8f0;border-radius:10px}figcaption{font-size:15px;line-height:1.8;color:#64748b;padding:14px 10px}::selection{background:#ede9fe}@media(max-width:600px){main{margin:16px auto 60px;padding:0 20px}h1{font-size:29px;margin:28px 0}h2{font-size:24px;margin-top:42px}p,li{font-size:18px}figure{margin:28px -12px}figcaption{font-size:14px}}'''
    page=f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{style}</style></head><body><main><header><img src="../assets/cover.png" width="2000" height="800" alt="LLM Inference"><h1>{html.escape(title)}</h1></header><article>{raw}</article></main></body></html>\n'
    (ZH/'review.html').write_text(page)


def main():
    ASSETS.mkdir(exist_ok=True)
    for fn in [latency,batching,phases,speculation,quantization,parallelism,overview,cover]:
        fn()
    article()
    sheet=Image.new('RGB',(1000,4*306),'#E2E8F0')
    d=ImageDraw.Draw(sheet)
    for i,name in enumerate(FILES+['cover.png']):
        im=Image.open(ASSETS/name)
        im.thumbnail((500,281),Image.Resampling.LANCZOS)
        x,y=(i%2)*500,(i//2)*306
        sheet.paste(im,(x,y))
        text(d,x+12,y+294,name,16)
    sheet.save(P/'contact-sheet.png')
    metadata={'type':'original static schematics','model_run':False,'animation_render':False,
              'command':'uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-inference-001/build.py',
              'dependencies':{name:importlib.metadata.version(name) for name in ['mistune','Pillow']},
              'cover':[2000,800],'inline_figures':7,'text_bounds_checked':len(TEXT_BOUNDS),
              'calculated_weight_bytes':{'16bit':16_000_000_000,'4bit':4_000_000_000},
              'sources':{},'outputs':{}}
    for path in [P/'build.py', ZH/'x-editor-draft-body-with-image-placeholders.md']:
        metadata['sources'][str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [*(ASSETS/name for name in FILES+['cover.png']),ZH/'review.html']:
        metadata['outputs'][str(path.relative_to(P))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Built seven original PNG figures, 5:2 cover, Chinese HTML and image order.')


if __name__ == '__main__':
    main()
