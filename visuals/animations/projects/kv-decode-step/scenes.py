"""One decode step: project, append, match, normalize, mix, retain."""
from functools import partial
import json
from pathlib import Path
import sys
import numpy as np
from manim import *

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));sys.path.insert(0,str(P.parents[1]))
from example import fixture
from manim_style import label, BACKGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED
label=partial(label,layout_scale=16)


def box(text,point,width=1.3,color=BASE,size=28):
    b=RoundedRectangle(width=width,height=.63,corner_radius=.1,stroke_color=color,
        stroke_width=2,fill_color=color,fill_opacity=.05)
    return VGroup(b,label(text,size,color).move_to(b)).move_to(point)


def vector(point,color,values):
    b=RoundedRectangle(width=1.25,height=.56,corner_radius=.08,stroke_color=color,
        stroke_width=2,fill_color=color,fill_opacity=.05)
    bars=VGroup(*[Rectangle(width=.18,height=.08+.26*v,stroke_width=0,
        fill_color=color,fill_opacity=.8).move_to([x,-.18,0],aligned_edge=DOWN)
        for x,v in zip([-.23,.23],values)])
    return VGroup(b,bars).move_to(point)


def path(points,color):
    return VMobject(stroke_color=color,stroke_width=2.3).set_points_as_corners([np.array(p) for p in points])


class DecodeStep(Scene):
    def construct(self):
        self.camera.background_color=BACKGROUND;self.events=[]
        f=fixture();xs=[-1.1,.6,2.3]
        title=label('One decode step',34).to_edge(UP,buff=.38).to_edge(LEFT,buff=.55)
        scope=label('One head · illustrative vectors · positional operations omitted',18,MUTED).to_edge(DOWN,buff=.22)
        bank=RoundedRectangle(width=5.2,height=2.42,corner_radius=.14,stroke_color=GRID,
            stroke_width=2).move_to([.6,-.25,0])
        names=VGroup(label('K',29,INTERMEDIATE).move_to([-2.5,.5,0]),
                     label('V',29,COMPOSED).move_to([-2.5,-1,0]),
                     label('KV cache',24,MUTED).move_to([.6,1.28,0]))
        positions=VGroup(*[label(s,20,MUTED).move_to([x,-.25,0]) for s,x in zip(['past','past','current'],xs)])
        keys=VGroup(*[vector([x,.5,0],INTERMEDIATE,v) for x,v in zip(xs,f['K'])])
        values=VGroup(*[vector([x,-1,0],COMPOSED,v) for x,v in zip(xs,f['V'])])
        self.add(title,scope,bank,names,positions[:2],keys[:2],values[:2])
        old=[m.get_all_points().copy() for m in [*keys[:2],*values[:2]]]
        h=box('New h',[-5,2.25,0],1.75)
        project=box('Q / K / V projections',[-.5,2.25,0],4.2,size=26)
        entry=Arrow(h.get_right(),project.get_left(),buff=.1,color=BASE,stroke_width=2,tip_length=.12)
        q=box('Q',[-5,.5,0],color=BASE,size=34)
        self.play(FadeIn(h),Create(entry),FadeIn(project),run_time=.55)
        self.mark('new_hidden_state',cached=2)
        new_k=keys[2].copy().move_to([4.85,.5,0])
        new_v=values[2].copy().move_to([4.85,-1,0])
        qpath=path([project.get_left(),[-3.4,2.25,0],[-3.4,1.2,0],[-5,1.2,0],q.get_top()],BASE)
        kpath=path([project.get_right(),[4.85,2.25,0],new_k.get_top()],INTERMEDIATE)
        vpath=path([project.get_right(),[5.85,2.25,0],[5.85,-1,0],new_v.get_right()],COMPOSED)
        self.play(*[ShowPassingFlash(p,time_width=.6) for p in [qpath,kpath,vpath]],
                  FadeIn(q),FadeIn(new_k),FadeIn(new_v),run_time=.85)
        self.mark('new_qkv',cached=2)
        self.play(new_k.animate.move_to(keys[2]),new_v.animate.move_to(values[2]),FadeIn(positions[2]),run_time=.65)
        self.remove(new_k,new_v);self.add(keys[2],values[2])
        self.mark('append_before_read',cached=3,self_included=True)
        self.wait(.4)
        matching=VGroup(*[path([q.get_top(),[-5,1.05,0],[x,1.05,0],k.get_top()],BASE) for x,k in zip(xs,keys)])
        self.play(*[ShowPassingFlash(p,time_width=.7) for p in matching],run_time=.65)
        self.mark('query_matches_all_keys',cached=3,read_positions=[1,2,3])
        soft=label('Scaled scores → softmax',22,BASE).move_to([-4.4,-2.45,0])
        weights=VGroup(*[Rectangle(width=1.18*w/.5,height=.19,stroke_width=0,
            fill_color=BASE,fill_opacity=.82).move_to([x,-2.45,0]) for x,w in zip(xs,f['weights'])])
        self.play(FadeIn(soft),LaggedStart(*[GrowFromCenter(w) for w in weights],lag_ratio=.1),run_time=.65)
        self.mark('softmax_weights',cached=3,weights=f['weights'])
        self.wait(.35)
        apply_weights=VGroup(*[path([w.get_top(),v.get_bottom()],BASE) for w,v in zip(weights,values)])
        self.play(*[ShowPassingFlash(p,time_width=.7) for p in apply_weights],run_time=.45)
        output=box('Head output',[5,-2.45,0],2.7,COMPOSED,25)
        flows=VGroup(*[path([v.get_bottom(),[x,-1.58,0],[3.6,-1.58,0],output.get_top()],COMPOSED)
                      .set_stroke(width=2+8*w) for x,v,w in zip(xs,values,f['weights'])])
        mix=label('Weighted sum',22,COMPOSED).move_to([5,-1.6,0])
        self.play(*[ShowPassingFlash(p,time_width=.75) for p in flows],FadeIn(mix),FadeIn(output),run_time=.8)
        self.mark('weighted_value_sum',cached=3,output=f['output'])
        self.wait(.8)
        self.play(FadeOut(q),FadeOut(soft),FadeOut(weights),FadeOut(mix),FadeOut(output),
                  FadeOut(h),FadeOut(entry),FadeOut(project),run_time=.45)
        retain=label('Retain K / V for the next step',28,COMPOSED).move_to([.6,-2.25,0])
        self.play(FadeIn(retain),run_time=.3)
        for before,m in zip(old,[*keys[:2],*values[:2]]):np.testing.assert_allclose(before,m.get_all_points(),atol=1e-10)
        self.mark('cache_retained',cached=3,query_retained=False)
        self.wait(1.1)
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.4);self.wait(.15)
        (P/'decode-step-timeline.json').write_text(json.dumps(dict(events=self.events,fixture=f,
            checks=dict(old_cache_unchanged=True,inside_canvas=True,self_included=True),duration_seconds=float(self.time)),indent=2)+'\n')

    def mark(self,name,**state):
        for m in self.mobjects:
            assert m.get_left()[0]>-7.1 and m.get_right()[0]<7.1,name
            assert m.get_bottom()[1]>-3.95 and m.get_top()[1]<3.95,name
        self.events.append(dict(name=name,seconds=round(float(self.time),3),**state))
