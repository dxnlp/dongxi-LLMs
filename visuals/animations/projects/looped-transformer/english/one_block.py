"""Continuous fixed recurrence with measured activations and one shared block."""
import json
from pathlib import Path
import sys
import numpy as np
from manim import *

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[2]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED


class OneBlock(Scene):
    def construct(self):
        self.camera.background_color=BACKGROUND
        trace=json.loads((P.parent/'trace.json').read_text())
        checkpoints=[]
        def mark(name):
            for item in self.mobjects:
                assert item.get_left()[0]>=-7.05 and item.get_right()[0]<=7.05,(name,'horizontal')
                assert item.get_bottom()[1]>=-3.95 and item.get_top()[1]<=3.95,(name,'vertical')
            checkpoints.append(dict(name=name,seconds=round(float(self.time),3)))
        def txt(s,size,pos,color=FOREGROUND,cjk=False):
            return label(s,size,color,cjk=cjk).move_to([*pos,0])
        def heatmap(index):
            values=np.array(trace['states'][index])[0]
            grid=VGroup()
            for row in values:
                cells=VGroup()
                for v in row:
                    color=interpolate_color(ManimColor(BACKGROUND),ManimColor(BASE if v<0 else INTERMEDIATE),min(abs(v)/trace['heatmap_abs_max'],1))
                    cells.add(RoundedRectangle(width=.122,height=.122,corner_radius=.025,
                        stroke_width=.35,stroke_color=GRID,fill_color=color,fill_opacity=1))
                grid.add(cells.arrange(RIGHT,buff=.013))
            return grid.arrange(DOWN,buff=.028)

        title=txt('Reuse one block',34,[-3.9,3.38],cjk=False)
        def subscript(base,sub):
            main=label(base,28,MUTED)
            small=label(sub,15,MUTED).next_to(main,RIGHT,buff=.025).shift(DOWN*.13)
            return VGroup(main,small)
        formula=VGroup(subscript('h','r+1'),label('=',28,MUTED),subscript('F','θ'),
                       label('(',28,MUTED),subscript('h','r'),label(')',28,MUTED)).arrange(RIGHT,buff=.08).move_to([4.15,3.38,0])
        self.play(FadeIn(title),FadeIn(formula),run_time=.9)

        block=SVGMobject(str(P.parent/'assets/block.svg')).stretch_to_fit_width(2.96).stretch_to_fit_height(2.59).move_to([0,.05,0])
        heading=txt('Transformer block',21,[0,1.10])
        attention=txt('Attention',25,[0,.46])
        ffn=txt('FFN',25,[0,-.49])
        innerlabels=VGroup(attention,ffn)
        # θ stays attached to the exact same block throughout every application.
        theta=txt('θ',35,[0,1.74],INTERMEDIATE)
        theta_line=Line([0,1.49,0],[0,1.36,0],stroke_width=2,color=INTERMEDIATE)
        self.play(Create(block),FadeIn(heading),FadeIn(innerlabels),FadeIn(theta),Create(theta_line),run_time=1.25)

        caption=txt('One set of weights, one block',27,[0,-1.89],cjk=False)
        def explain(s):
            nonlocal caption
            new=txt(s,27,[0,-1.89],cjk=False)
            self.play(FadeOut(caption,shift=DOWN*.04),FadeIn(new,shift=UP*.04),run_time=.4)
            caption=new
        self.play(FadeIn(caption),run_time=.4)
        counter_labels=VGroup(*[txt(s,24,[x,-2.66],MUTED,cjk=False) for x,s in zip((-4.35,0,4.35),('Stored parameters','Block applications','Matrix work'))])
        values=[txt('2,160',39,[-4.35,-3.24],INTERMEDIATE),txt('0',39,[0,-3.24],BASE),txt('0×',39,[4.35,-3.24],BASE)]
        baselines=VGroup(*[Line([x-1.37,-2.32,0],[x+1.37,-2.32,0],color=GRID,stroke_width=1) for x in (-4.35,0,4.35)])
        self.play(FadeIn(counter_labels),FadeIn(VGroup(*values)),Create(baselines),run_time=.8)
        self.wait(.8);mark('one-parameter-set')

        packet=heatmap(0).move_to([-4.30,.08,0])
        state_label=txt('h₀',32,[-4.30,-.72],BASE)
        shape=txt('6 × 16',17,[-4.30,.90],MUTED)
        # Scale key stays fixed; activation brightness is never renormalized per pass.
        neg=Square(side_length=.12,fill_color=BASE,fill_opacity=.8,stroke_width=0)
        pos=Square(side_length=.12,fill_color=INTERMEDIATE,fill_opacity=.8,stroke_width=0)
        key=VGroup(label('−',17,MUTED),neg,label('0',17,MUTED),pos,label('+',17,MUTED)).arrange(RIGHT,buff=.09).move_to([-4.30,-1.15,0])
        entry=Arrow([-3.1,.08,0],[-1.57,.08,0],buff=0,color=MUTED,stroke_width=2,tip_length=.13)
        exit_arrow=Arrow([1.55,.08,0],[3.03,.08,0],buff=0,color=MUTED,stroke_width=2,tip_length=.13)
        self.play(FadeIn(packet,shift=RIGHT*.1),FadeIn(state_label),FadeIn(shape),FadeIn(key),Create(entry),Create(exit_arrow),run_time=.9)
        self.wait(.9);mark('initial-hidden-state')

        # The sketch shows the persistent return route; motion follows the same loop.
        return_route=VMobject().set_points_as_corners([
            [4.30,.08,0],[5.95,.08,0],[5.95,2.35,0],
            [-5.95,2.35,0],[-5.95,.08,0],[-4.30,.08,0]])
        loop_ink=SVGMobject(str(P.parent/'assets/loop.svg')).stretch_to_fit_width(11.9).stretch_to_fit_height(2.27).move_to([0,1.215,0])

        for r in range(1,4):
            explain(('Hidden states enter the block','A new state, the same block','A third pass through the same weights')[r-1])
            self.play(FadeOut(state_label),FadeOut(shape),FadeOut(key) if r==1 else Wait(0),
                      innerlabels.animate.set_opacity(.12),run_time=.4)
            self.play(packet.animate.scale(.68).move_to([0,.08,0]),run_time=1.3,rate_func=smooth)
            target=heatmap(r).scale(.68).move_to(packet)
            self.play(Transform(packet,target),Indicate(theta,color=INTERMEDIATE,scale_factor=1.12),run_time=1.0)
            self.play(packet.animate.scale(1/.68).move_to([4.30,.08,0]),innerlabels.animate.set_opacity(1),run_time=1.3,rate_func=smooth)
            state_label=txt(('h₁','h₂','h₃')[r-1],32,[4.30,-.72],BASE)
            shape=txt('6 × 16',17,[4.30,.90],MUTED)
            next_values=[txt(str(r),39,[0,-3.24],BASE),txt(f'{r}×',39,[4.35,-3.24],BASE)]
            self.play(FadeIn(state_label),FadeIn(shape),
                      FadeOut(values[1]),FadeIn(next_values[0],shift=UP*.06),
                      FadeOut(values[2]),FadeIn(next_values[1],shift=UP*.06),run_time=.6)
            values[1:]=next_values
            self.wait(1.05);mark(f'application-{r}')
            if r<3:
                explain('States change; weights stay shared')
                if r==1:self.play(Create(loop_ink),run_time=1.15)
                self.play(FadeOut(state_label),FadeOut(shape),run_time=.3)
                self.play(packet.animate.scale(.58),run_time=.35)
                self.play(MoveAlongPath(packet,return_route),run_time=3.0,rate_func=linear)
                self.play(packet.animate.scale(1/.58),run_time=.35)
                state_label=txt(('h₁','h₂')[r-1],32,[-4.30,-.72],BASE)
                shape=txt('6 × 16',17,[-4.30,.90],MUTED)
                self.play(FadeIn(state_label),FadeIn(shape),run_time=.35)
                self.wait(.5);mark(f'return-{r}')

        explain('Fixed parameters, more computation')
        self.play(Indicate(values[0],color=INTERMEDIATE,scale_factor=1.06),run_time=.7)
        self.play(Indicate(VGroup(*values[1:]),color=BASE,scale_factor=1.05),run_time=.8)
        caveat=txt('More computation does not guarantee better predictions',20,[0,-3.81],MUTED,cjk=False)
        self.play(FadeIn(caveat),run_time=.6)
        self.wait(3);mark('final')
        (P/'01-one-block-timeline.json').write_text(json.dumps(dict(checkpoints=checkpoints,duration=float(self.time),geometry_checks=True),indent=2)+'\n')
