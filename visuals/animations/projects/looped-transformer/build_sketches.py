"""Named, deterministic architecture assets in a 960×540 design canvas."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
def rect(name,x,y,w,h,seed,stroke='#64748b'):
    return dict(type='rectangle',id=name,x=x,y=y,width=w,height=h,seed=seed,
                strokeColor=stroke,backgroundColor='transparent',strokeWidth=1.4,
                roughness=1.1,roundness={'type':3})
def arrow(name,points,seed):
    x,y=points[0]
    return dict(type='arrow',id=name,x=x,y=y,points=[[a-x,b-y] for a,b in points],
                width=points[-1][0]-x,height=points[-1][1]-y,seed=seed,
                strokeColor='#64748b',strokeWidth=1.5,roughness=.9,endArrowhead='arrow')
block=[rect('block',380,178,200,175,901),
       rect('attention',405,217,150,47,902,'#cbd5e1'),
       rect('ffn',405,281,150,47,903,'#cbd5e1')]
loop=[arrow('return',[[851.25,264.6],[881.625,264.6],[881.625,111.375],
                      [78.375,111.375],[78.375,264.6],[108.75,264.6]],904)]
scene=block+loop
for i,(s,x,y) in enumerate([('Transformer block',403,186),('Attention',442,230),('FFN',466,294),('shared weights',434,151)]):
    scene=scene+[dict(type='text',id=f'label-{i}',text=s,x=x,y=y,fontSize=16,fontFamily=2,seed=920+i,strokeColor='#111827')]
assets={'block':block,'loop':loop,'architecture':scene}
(P/'sketches.json').write_text(json.dumps(assets,indent=2)+'\n')
