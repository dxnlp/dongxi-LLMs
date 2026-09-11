"""Three focused English explanations of fixed recurrent depth."""
import json
from pathlib import Path
import sys
import numpy as np
from manim import *

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
sys.path.insert(0,str(P.parents[2]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED
from one_block import OneBlock

COLORS=[BASE,INTERMEDIATE,COMPOSED]

class OneBlockEnglish(OneBlock):
    pass


class Lesson(Scene):
    def setup(self):
        self.camera.background_color=BACKGROUND
        self.trace=json.loads((P/'trace.json').read_text())
        self.checkpoints=[]

    def text(self,s,size=26,pos=(0,0),color=FOREGROUND):
        return label(s,size,color).move_to([*pos,0])

    def mark(self,name):
        for item in self.mobjects:
            assert item.get_left()[0]>=-7.05 and item.get_right()[0]<=7.05,(name,'horizontal')
            assert item.get_bottom()[1]>=-3.95 and item.get_top()[1]<=3.95,(name,'vertical')
        self.checkpoints.append(dict(name=name,seconds=round(float(self.time),3)))

    def finish(self,name):
        (P/f'{name}-timeline.json').write_text(json.dumps(dict(checkpoints=self.checkpoints,duration=float(self.time),geometry_checks=True),indent=2)+'\n')

    def block(self,index,pos,width=1.90):
        color=COLORS[index]
        shape=SVGMobject(str(P.parent/'assets/block.svg')).stretch_to_fit_width(width).stretch_to_fit_height(width*.875).set_color(color)
        shape.move_to([*pos,0])
        factor=width/1.9
        heading=self.text(f'Block {"ABC"[index]}',int(23*factor),(pos[0],pos[1]+.64*factor),color)
        inner=VGroup(self.text('Attention',int(19*factor),(pos[0],pos[1]+.24*factor)),
                     self.text('FFN',int(19*factor),(pos[0],pos[1]-.38*factor)))
        return VGroup(shape,heading,inner)

    def heatmap(self,index,width=1.1):
        values=np.array(self.trace['states'][index])[0]
        rows=VGroup()
        for row in values:
            cells=VGroup()
            for v in row:
                color=interpolate_color(ManimColor(BACKGROUND),ManimColor(BASE if v<0 else INTERMEDIATE),abs(v)/self.trace['heatmap_abs_max'])
                cells.add(Square(side_length=.12,stroke_color=GRID,stroke_width=.25,fill_color=color,fill_opacity=1))
            rows.add(cells.arrange(RIGHT,buff=.014))
        return rows.arrange(DOWN,buff=.027).scale_to_fit_width(width).set_z_index(5)

    def caption(self,s):
        fresh=self.text(s,27,(0,-1.70))
        if hasattr(self,'current_caption'):
            self.play(FadeOut(self.current_caption),FadeIn(fresh),run_time=.4)
        else:self.play(FadeIn(fresh),run_time=.4)
        self.current_caption=fresh


class RepeatedStack(Lesson):
    def construct(self):
        title=self.text('Repeat a stack',34,(-4.55,3.38))
        scope=self.text('Three blocks · two passes',23,(3.85,3.38),MUTED)
        self.play(FadeIn(title),FadeIn(scope),run_time=.8)
        xs=[-2.8,0,2.8]
        blocks=VGroup(*[self.block(i,(x,.3)) for i,x in enumerate(xs)])
        params=VGroup(*[self.text(f'θ{"ABC"[i]}',26,(x,1.60),COLORS[i]) for i,x in enumerate(xs)])
        self.play(LaggedStart(*[FadeIn(b) for b in blocks],lag_ratio=.2),FadeIn(params),run_time=1.4)
        self.caption('A, B and C each have their own weights')
        labels=VGroup(self.text('Stored parameters',23,(-4.25,-2.62),MUTED),
                      self.text('Block applications',23,(0,-2.62),MUTED),
                      self.text('Stack passes',23,(4.25,-2.62),MUTED))
        count=self.text('0',39,(0,-3.2),BASE)
        passes=self.text('0',39,(4.25,-3.2),BASE)
        stored=self.text(f"{self.trace['shared_parameters']:,}",39,(-4.25,-3.2),INTERMEDIATE)
        self.play(FadeIn(labels),FadeIn(count),FadeIn(passes),FadeIn(stored),run_time=.7)
        self.wait(1);self.mark('three-distinct-blocks')
        links=VGroup(*[Arrow([a,.3,0],[b,.3,0],buff=0,tip_length=.12,stroke_width=2,color=MUTED)
                      for a,b in [(-4.8,-3.83),(-1.8,-1.0),(1.0,1.8),(3.83,4.8)]])
        self.play(Create(links),run_time=.7)
        packet=self.heatmap(0).move_to([-5.5,.3,0])
        state=self.text('h₀',25,(-5.5,-.24),BASE)
        self.play(FadeIn(packet),FadeIn(state),run_time=.7)
        n=0
        loop_path=VMobject().set_points_as_corners([
            [5.5,.3,0],[6.45,.3,0],[6.45,2.38,0],[-6.45,2.38,0],[-6.45,.3,0],[-5.5,.3,0]])
        loop_ink=VMobject(color=MUTED,stroke_width=2).set_points_as_corners([
            [6.07,.3,0],[6.45,.3,0],[6.45,2.38,0],[-6.45,2.38,0],[-6.45,.3,0]])
        loop_tip=Arrow([-6.45,.3,0],[-6.07,.3,0],buff=0,color=MUTED,stroke_width=2,tip_length=.1)
        loop=VGroup(loop_ink,loop_tip)
        for turn in range(2):
            self.caption('First pass: A → B → C' if turn==0 else 'Second pass: the same A → B → C')
            self.play(FadeOut(state),run_time=.25)
            for i,x in enumerate(xs):
                self.play(blocks[i][2].animate.set_opacity(.08),packet.animate.move_to([x,.3,0]),run_time=.95)
                n+=1
                updated=self.heatmap(n).move_to(packet)
                self.play(Transform(packet,updated),Indicate(params[i],color=COLORS[i],scale_factor=1.12),run_time=.7)
                new_count=self.text(str(n),39,(0,-3.2),BASE)
                self.play(FadeOut(count),FadeIn(new_count),run_time=.2);count=new_count
                self.play(packet.animate.move_to([x+1.43,.3,0]),blocks[i][2].animate.set_opacity(1),run_time=.6)
            self.play(packet.animate.move_to([5.5,.3,0]),run_time=.5)
            newpasses=self.text(str(turn+1),39,(4.25,-3.2),BASE)
            state=self.text('h₃' if turn==0 else 'h₆',25,(5.5,-.24),BASE)
            self.play(FadeOut(passes),FadeIn(newpasses),FadeIn(state),run_time=.4);passes=newpasses
            self.wait(1.1);self.mark(f'pass-{turn+1}')
            if turn==0:
                self.caption('Feed the updated state back to the start')
                self.play(Create(loop),run_time=.7)
                self.play(FadeOut(state),packet.animate.scale(.7),run_time=.3)
                self.play(MoveAlongPath(packet,loop_path),run_time=3.0,rate_func=linear)
                self.play(packet.animate.scale(1/.7),run_time=.3)
                state=self.text('h₃',25,(-5.5,-.24),BASE)
                self.play(FadeIn(state),run_time=.3)
                self.wait(.6);self.mark('same-stack-return')
        self.caption('Three parameter sets. Six block applications.')
        self.play(Indicate(stored,color=INTERMEDIATE,scale_factor=1.06),Indicate(count,color=BASE,scale_factor=1.06),run_time=.8)
        caveat=self.text('Block parameters only; no claim of better predictions.',18,(0,-3.82),MUTED)
        self.play(FadeIn(caveat),run_time=.5)
        self.wait(3);self.mark('final')
        self.finish('02-repeat-stack')


class UnrolledSharing(Lesson):
    def construct(self):
        title=self.text('Unfold the computation',34,(-3.75,3.38))
        self.play(FadeIn(title),run_time=.8)
        original=VGroup(*[self.block(i,(x,.40)) for i,x in enumerate((-2.8,0,2.8))])
        repeat=self.text('2 passes',27,(0,2.25),MUTED)
        self.play(FadeIn(original),FadeIn(repeat),run_time=1.2)
        self.caption('One stack, applied twice')
        self.wait(1.2);self.mark('folded-stack')

        xs=[-5.4,-3.24,-1.08,1.08,3.24,5.4]
        row1=[self.block(i,(xs[i],.4),width=1.62) for i in range(3)]
        row2=[self.block(i,(xs[i+3],.4),width=1.62) for i in range(3)]
        self.caption('Unfold the six applications')
        # The first three drawings move; the other three are application views.
        self.play(*[Transform(original[i],row1[i]) for i in range(3)],FadeOut(repeat),run_time=1.5)
        row2=[original[i].copy() for i in range(3)]
        self.add(*row2)
        self.play(*[row2[i].animate.move_to([xs[i],2.02,0]) for i in range(3)],run_time=.65)
        self.play(*[row2[i].animate.move_to([xs[i+3],2.02,0]) for i in range(3)],run_time=.95)
        self.play(*[row2[i].animate.move_to([xs[i+3],.4,0]) for i in range(3)],run_time=.65)
        apps=list(original)+row2
        labels=VGroup(*[self.text(str(i+1),23,(x,-.60),MUTED) for i,x in enumerate(xs)])
        arrows=VGroup(*[Arrow([xs[i]+.87,.4,0],[xs[i+1]-.87,.4,0],buff=0,tip_length=.1,color=MUTED,stroke_width=2) for i in range(5)])
        pass_labels=VGroup(self.text('Pass 1',25,(-3.24,-1.11),MUTED),self.text('Pass 2',25,(3.24,-1.11),MUTED))
        divider=DashedLine([0,-1.22,0],[0,1.40,0],color=GRID,stroke_width=1)
        self.play(FadeIn(labels),Create(arrows),FadeIn(pass_labels),Create(divider),run_time=.9)
        self.wait(1);self.mark('six-applications')

        owners=VGroup()
        for i in range(3):
            self.caption(f'Both applications of {"ABC"[i]} share θ{"ABC"[i]}')
            curve=CubicBezier([xs[i],1.18,0],[xs[i],2.55,0],[xs[i+3],2.55,0],[xs[i+3],1.18,0],color=COLORS[i],stroke_width=3)
            owner=self.text(f'θ{"ABC"[i]}',30,((xs[i]+xs[i+3])/2,2.46),COLORS[i])
            self.play(Create(curve),FadeIn(owner),Indicate(VGroup(apps[i],apps[i+3]),color=COLORS[i],scale_factor=1.04),run_time=1.2)
            self.wait(1.4);self.mark(f'paired-{"ABC"[i]}')
            small=self.text(f'θ{"ABC"[i]}',28,((-3.7,0,3.7)[i],-2.50),COLORS[i])
            self.play(FadeOut(curve),ReplacementTransform(owner,small),run_time=.6)
            owners.add(small)

        self.caption('Six applications still use only three parameter sets')
        totals=self.text('6,480 stored parameters',32,(0,-3.17),INTERMEDIATE)
        self.play(FadeIn(totals),run_time=.6)
        self.wait(1.2);self.mark('shared-storage')
        # One pulse executes the unfolded application order; no extra weights appear.
        dot=Dot([xs[0]-.8,.4,0],radius=.08,color=BASE).set_z_index(10)
        self.add(dot)
        for i in range(6):
            self.play(dot.animate.move_to([xs[i],.4,0]).set_color(COLORS[i%3]),run_time=.4)
        self.play(FadeOut(dot),run_time=.2)
        caveat=self.text('Unfolding displays computation; it does not allocate new weights.',18,(0,-3.82),MUTED)
        self.play(FadeIn(caveat),run_time=.4)
        self.wait(3);self.mark('final')
        self.finish('03-unfold-sharing')
