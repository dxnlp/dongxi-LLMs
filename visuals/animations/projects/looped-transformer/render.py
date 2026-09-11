"""Render local media, extract review frames, and record reproducibility data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from PIL import Image, ImageDraw, ImageFont
import manimpango

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
OUT=P/'rendered'

def run(cmd,capture=False):
    result=subprocess.run(cmd,check=True,cwd=ROOT,text=True,stdout=subprocess.PIPE if capture else None)
    return result.stdout

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    assert {'Arial','Songti SC'}<=set(manimpango.list_fonts())
    assert all((P/'assets'/f'{name}.svg').exists() for name in ('block','loop')), 'Finish export.mjs before rendering'
    trace=json.loads((P/'trace.json').read_text())
    assert all(trace['checks'].values())
    assert trace['source_sha256']==sha(ROOT/'src/dongxi_llms/decoder_lab.py')
    OUT.mkdir(exist_ok=True)
    size,fps=('960,540','15') if args.preview else ('1920,1080','30')
    video=OUT/('preview.mp4' if args.preview else 'looped-transformer.mp4')
    with tempfile.TemporaryDirectory(prefix='loop-manim-') as tmp:
        run([sys.executable,'-m','manim','render','-r',size,'--fps',fps,'--renderer','cairo',
             '--progress_bar','none','--disable_caching','--verbosity','WARNING','--media_dir',tmp,
             '-o','looped-transformer',str(P/'scene.py'),'LoopedTransformer'])
        files=list(Path(tmp).glob('videos/**/looped-transformer.mp4'));assert len(files)==1
        shutil.copy2(files[0],video)
    probe=json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],True))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'])==tuple(map(int,size.split(',')))
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    timeline=json.loads((P/'timeline.json').read_text())
    with tempfile.TemporaryDirectory(prefix='loop-frames-') as tmp:
        frames=[]
        for i,checkpoint in enumerate(timeline['checkpoints']):
            path=Path(tmp)/f'{i}.png'
            run(['ffmpeg','-y','-loglevel','error','-ss',str(checkpoint['seconds']-.15),'-i',str(video),'-frames:v','1','-update','1',str(path)])
            frames.append(Image.open(path).convert('RGB'))
        frames[-1].save(OUT/'still.png')
        frames[-1].resize((960,540),Image.Resampling.LANCZOS).save(OUT/'still-half.png')
        sheet=Image.new('RGB',(1440,300*((len(frames)+2)//3)),'#e2e8f0');draw=ImageDraw.Draw(sheet)
        for i,(frame,checkpoint) in enumerate(zip(frames,timeline['checkpoints'])):
            x,y=i%3*480,i//3*300
            sheet.paste(frame.resize((480,270),Image.Resampling.LANCZOS),(x,y))
            draw.text((x+8,y+276),f"{checkpoint['seconds']:.1f}s · {checkpoint['name']}",font=ImageFont.load_default(size=14),fill='#111827')
        sheet.save(OUT/'contact-sheet.png')
    run(['ffmpeg','-y','-loglevel','error','-i',str(video),'-filter_complex',
         'fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer',
         '-loop','0',str(OUT/'looped-transformer.gif')])
    (P/'review.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>循环 Transformer</title>
<style>body{{margin:42px auto;max-width:1200px;padding:0 24px;font:18px/1.8 Arial,"Songti SC",serif;color:#111827;background:#fff}}h1{{font-weight:400;font-size:34px;margin:0 0 22px}}video{{width:100%;aspect-ratio:16/9;border:1px solid #e2e8f0;border-radius:12px}}summary{{cursor:pointer}}details{{margin-top:22px;color:#64748b}}p{{margin:12px 0}}</style>
<h1>循环 Transformer：同一组权重，多次计算</h1>
<video controls playsinline preload="metadata" poster="rendered/still.png"><source src="rendered/{video.name}" type="video/mp4"></video>
<details><summary>如何理解这幅图</summary>
<p>彩色矩阵是同一批输入在不同计算阶段的隐藏状态。展示其中一个样本的 6 个位置、每个位置 16 个特征。蓝色代表负值，紫色代表正值，颜色深浅使用统一尺度。颜色变化表示数值变化。</p>
<p>同一个因果 Transformer 模块执行三次，始终共享 2,160 个参数。每次完整批次计算为 53,760 次浮点矩阵运算，三次为 161,280 次；乘加按两次运算计数。倍数以执行一次为基准。</p>
<p>这里仅统计模块的矩阵乘法，省略嵌入、输出层、归一化、激活函数和 softmax 等开销。整个序列长度保持不变；额外的隐藏状态计算不会自动增加输出 token。</p>
<p>这是固定递归的未训练示例。更多计算是否改善预测，需要经过训练和受控评估。动画没有实现自适应路由，也没有测量延迟或内存。</p></details></html>''')
    sources=[p for p in P.iterdir() if p.suffix in ('.py','.json','.mjs','.yaml','.html') and p.name not in ('metadata.json','qa.json')]
    sources+=list((P/'assets').glob('*'))
    sources += [ROOT/'src/dongxi_llms/decoder_lab.py',P.parents[1]/'manim_style.py',P.parents[1]/'STYLE_GUIDE.md',P.parents[1]/'uv.lock']
    outputs=[video,OUT/'looped-transformer.gif',OUT/'still.png',OUT/'still-half.png',OUT/'contact-sheet.png']
    metadata=dict(task_id='ANIM-LOOP-001',status='preview' if args.preview else 'local-review',
        base_commit=run(['git','rev-parse','HEAD'],True).strip(),branch=run(['git','branch','--show-current'],True).strip(),
        rendered_at=datetime.now(timezone.utc).isoformat(),
        render_command='uv run --project visuals/animations python visuals/animations/projects/looped-transformer/render.py'+(' --preview' if args.preview else ''),
        environment=dict(python=platform.python_version(),platform=platform.platform(),
            packages={p:version(p) for p in ('manim','manimpango','pillow','numpy','pycairo')},excalidraw='0.18.0',trace_torch=trace['torch_version'],fonts=['Arial','Songti SC']),
        video=dict(width=stream['width'],height=stream['height'],fps=stream['r_frame_rate'],codec=stream['codec_name'],pixel_format=stream['pix_fmt'],duration_seconds=float(probe['format']['duration'])),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(sources)],
        outputs=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in outputs],
        timeline=timeline,numerical_checks=trace['checks'],limitations=trace['limitations'])
    (P/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Review ready: {P / "review.html"}')

if __name__=='__main__':main()
