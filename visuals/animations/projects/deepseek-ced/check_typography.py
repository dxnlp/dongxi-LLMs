"""Render and test whole-string spacing at actual scene sizes; no film changes."""
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from manim import Scene, Text, tempconfig

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from manim_style import label

CASES=[('Causal Encoder–Decoder',34),('Cross-layer sharing',24),
       ('Global KV · 21–40',22),('Illustrative states',18),
       ('Weighted sum',24),('0.768',23)]


def positions(text):
    tree=ET.parse(text.file_name)
    return np.array([float(e.attrib['x']) for e in tree.iter()
                     if e.tag.endswith('use') and 'x' in e.attrib])


class Comparison(Scene):
    def construct(self):
        self.add(label('Previous',22,layout_scale=16).move_to([-3.4,3.4,0]),
                 label('Refined',22,layout_scale=16).move_to([3.4,3.4,0]))
        for i,(s,size) in enumerate(CASES):
            y=2.6-i*1.05
            self.add(label(s,size).move_to([-3.4,y,0]),
                     label(s,size,layout_scale=16).move_to([3.4,y,0]))


def main():
    out=P/'rendered/typography'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='ced-typography-') as tmp, tempconfig(dict(
            pixel_width=1920,pixel_height=1080,background_color='#FFFFFF',
            media_dir=tmp,disable_caching=True)):
        for s,size in CASES:
            old=label(s,size);new=label(s,size,layout_scale=16)
            reference=label(s,size,layout_scale=32)
            # Compare glyph advances to a still-higher-resolution layout.
            a=np.diff(positions(old));b=np.diff(positions(new))/16
            c=np.diff(positions(reference))/32
            assert len(a)==len(b)==len(c)
            old_error=float(np.mean(abs(a-c)));new_error=float(np.mean(abs(b-c)))
            assert new_error<old_error,(s,old_error,new_error)
            assert new.width<13.2 and new.height<.65,(s,new.width,new.height)
            rows.append(dict(text=s,size=size,old_advance_mae=old_error,
                             refined_advance_mae=new_error,width=float(new.width)))
        scene=Comparison();scene.render()
        scene.renderer.camera.get_image().save(out/'comparison.png')
    (P/'typography-qa.json').write_text(json.dumps(dict(layout_scale=16,
        reference_layout_scale=32,units='SVG coordinates at nominal size',
        checks=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
