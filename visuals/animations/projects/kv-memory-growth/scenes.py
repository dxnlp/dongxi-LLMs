"""Calculated separate-K/V payload: sequence length and KV-head comparisons."""
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import label, BACKGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED
label = partial(label, layout_scale=16)


def payload(positions, heads):
    return 2 * 1 * 24 * positions * heads * 64 * 2 // (1024 ** 2)


def cell(row, column):
    # Each column groups 1024 positions; each row is one KV head.
    x, y = -3.7 + column * .75, 1.48 - row * .36
    return VGroup(*[
        RoundedRectangle(width=.65, height=.125, corner_radius=.025,
            stroke_width=0, fill_color=color, fill_opacity=.78).move_to([x,y+dy,0])
        for color,dy in [(INTERMEDIATE,.075),(COMPOSED,-.075)]])


def bar(mib):
    return Rectangle(width=10.7*mib/384, height=.28, stroke_width=0,
                     fill_color=BASE, fill_opacity=1).move_to([-5.35,-2.5,0], aligned_edge=LEFT)


class MemoryGrowth(Scene):
    def construct(self):
        self.camera.background_color=BACKGROUND
        self.events=[]
        self.add(label('The memory bill grows with the cache',32).to_edge(UP,buff=.38).to_edge(LEFT,buff=.55),
                 label('24 layers · batch 1 · head width 64 · 2 bytes / element',20,MUTED).move_to([0,2.85,0]),
                 label('K',23,INTERMEDIATE).move_to([-5.95,1.7,0]),
                 label('V',23,COMPOSED).move_to([-5.95,1.32,0]),
                 label('1 column = 1,024 positions',19,MUTED).move_to([-1,-1.56,0]),
                 label('Calculated KV tensor memory',21,MUTED).move_to([0,-3.05,0]),
                 label('Model weights and runtime overhead excluded',18,MUTED).to_edge(DOWN,buff=.22))
        stage=label('More cached positions',26,BASE).move_to([0,2.22,0])
        heads=label('8 KV heads',24).move_to([-5.55,-.25,0])
        count=label('4,096 positions',26).move_to([4.5,.9,0])
        number=label('192 MiB',40,BASE).move_to([4.5,-.02,0])
        grid=[[cell(r,c) for c in range(8)] for r in range(8)]
        initial=VGroup(*[grid[r][c] for r in range(8) for c in range(4)])
        extra=VGroup(*[grid[r][c] for r in range(8) for c in range(4,8)])
        track=Line([-5.35,-2.5,0],[5.35,-2.5,0],color=GRID,stroke_width=12)
        fill=bar(192)
        self.play(FadeIn(stage),FadeIn(heads),FadeIn(count),FadeIn(number),FadeIn(initial),Create(track),FadeIn(fill),run_time=.6)
        baseline=initial.get_all_points().copy()
        self.mark('baseline',4096,8,fill)
        self.wait(.65)
        self.play(FadeOut(count),FadeOut(number),run_time=.15)
        self.play(LaggedStart(*[FadeIn(VGroup(*[grid[r][c] for r in range(8)]),shift=RIGHT*.15)
                               for c in range(4,8)],lag_ratio=.15),
                  Transform(fill,bar(384)),run_time=1.6)
        count=label('8,192 positions',26).move_to([4.5,.9,0])
        number=label('384 MiB',40,BASE).move_to([4.5,-.02,0])
        self.play(FadeIn(count),FadeIn(number),run_time=.2)
        np.testing.assert_allclose(initial.get_all_points(),baseline,atol=1e-10)
        self.mark('double_positions',8192,8,fill)
        self.wait(1.0)
        # Explicitly leave the growing-request view before comparing architectures.
        self.play(FadeOut(initial),FadeOut(extra),FadeOut(count),FadeOut(number),FadeOut(fill),FadeOut(stage),run_time=.4)
        stage=label('Compare KV-head configurations',26,BASE).move_to([0,2.22,0])
        self.play(FadeIn(stage),run_time=.2)
        self.wait(.25)
        grid=[[cell(r,c) for c in range(4)] for r in range(8)]
        keep=VGroup(*[grid[r][c] for r in range(4) for c in range(4)])
        lower=VGroup(*[grid[r][c] for r in range(4,8) for c in range(4)])
        count=label('4,096 positions',26).move_to([4.5,.9,0])
        number=label('192 MiB',40,BASE).move_to([4.5,-.02,0])
        fill=bar(192)
        self.play(FadeIn(keep),FadeIn(lower),FadeIn(count),FadeIn(number),FadeIn(fill),run_time=.5)
        self.mark('comparison_baseline',4096,8,fill)
        self.wait(.5)
        kept=keep.get_all_points().copy()
        self.play(FadeOut(heads),FadeOut(number),run_time=.15)
        self.play(FadeOut(lower,shift=DOWN*.12),
                  Transform(fill,bar(96)),run_time=1.3)
        heads=label('4 KV heads',24).move_to([-5.55,-.25,0])
        number=label('96 MiB',40,BASE).move_to([4.5,-.02,0])
        self.play(FadeIn(heads),FadeIn(number),run_time=.2)
        np.testing.assert_allclose(keep.get_all_points(),kept,atol=1e-10)
        self.mark('half_heads',4096,4,fill)
        self.wait(1.35)
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.4)
        self.wait(.2)
        (P/'memory-growth-timeline.json').write_text(json.dumps(dict(events=self.events,
            duration_seconds=float(self.time),checks=dict(arithmetic=True,bar_ratios=True,
                old_positions_unchanged=True,remaining_heads_unchanged=True,inside_canvas=True)),indent=2)+'\n')

    def mark(self,name,positions,heads,fill):
        mib=payload(positions,heads)
        assert np.isclose(fill.width/10.7,mib/384)
        for m in self.mobjects:
            assert m.get_left()[0]>-7.1 and m.get_right()[0]<7.1, name
            assert m.get_bottom()[1]>-3.95 and m.get_top()[1]<3.95, name
        self.events.append(dict(name=name,seconds=round(float(self.time),3),positions=positions,
                                heads=heads,mib=mib,bar_fraction=mib/384))
