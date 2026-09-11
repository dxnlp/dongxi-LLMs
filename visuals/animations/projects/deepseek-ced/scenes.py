"""Motion-first global-KV dependencies and two-role shared representations."""
import json
from functools import partial
import sys
from pathlib import Path
import numpy as np
from manim import *

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED, ACCENT

# Whole-string, high-resolution shaping preserves Arial's natural advances.
label=partial(label,layout_scale=16)

class Film(Scene):
    def setup(self):
        self.camera.background_color=BACKGROUND
        self.data=json.loads((P/'trace.json').read_text())
        self.checkpoints=[]

    def txt(self,s,pos,size=25,color=FOREGROUND):
        return label(s,size,color).move_to([*pos,0])

    def start(self,title,scope):
        self.heading=label(title,34,FOREGROUND).to_edge(UP,buff=.35).to_edge(LEFT,buff=.5)
        self.scope=label(scope,18,MUTED).to_edge(DOWN,buff=.18)
        self.play(FadeIn(self.heading),FadeIn(self.scope),run_time=.7)

    def stage(self,s):
        fresh=label(s,24,BASE).to_edge(LEFT,buff=.5).shift(UP*2.6)
        if hasattr(self,'stage_label'):self.play(FadeOut(self.stage_label),FadeIn(fresh),run_time=.45)
        else:self.play(FadeIn(fresh),run_time=.45)
        self.stage_label=fresh

    def mark(self,name):
        for m in self.mobjects:
            assert m.get_left()[0]>-7.1 and m.get_right()[0]<7.1,(name,'x',m.get_center())
            assert m.get_bottom()[1]>-4 and m.get_top()[1]<4,(name,'y')
        self.checkpoints.append(dict(name=name,seconds=round(float(self.time),3)))

    def finish(self,id):
        (P/f'{id}-timeline.json').write_text(json.dumps(dict(checkpoints=self.checkpoints,duration=float(self.time),geometry_checks=True),indent=2)+'\n')

    def strip(self,values,pos,color=INTERMEDIATE,width=1.45,numbers=False):
        cells=VGroup()
        for v in values:
            cell=RoundedRectangle(width=width/len(values)-.035,height=.33 if not numbers else .58,corner_radius=.045,stroke_color=color,stroke_width=1,fill_color=color,fill_opacity=.12+min(abs(float(v))/2,1)*.5)
            if numbers:cell=VGroup(cell,label(f'{v:g}',23,FOREGROUND).move_to(cell))
            cells.add(cell)
        return cells.arrange(RIGHT,buff=.035).move_to([*pos,0]).set_z_index(4)

    def bank(self,values,pos,color=INTERMEDIATE,width=1.45):
        rows=VGroup(*[self.strip(row,(0,0),color,width) for row in values]).arrange(DOWN,buff=.075)
        return rows.move_to([*pos,0])

    def arrow(self,a,b,color=MUTED):
        return Arrow([*a,0],[*b,0],buff=0,tip_length=.12,stroke_width=2,color=color).set_z_index(-1)

    def route(self,points,color=INTERMEDIATE):
        return VMobject(color=color,stroke_width=2).set_points_as_corners([[*p,0] for p in points]).set_z_index(-1)

    def signal(self,path,color=BASE,duration=.75):
        dot=Dot(path.get_start(),radius=.065,color=color).set_z_index(8)
        self.add(dot);self.play(MoveAlongPath(dot,path),run_time=duration,rate_func=linear);self.remove(dot)

class GlobalKV(Film):
    def construct(self):
        d=self.data['ced'];xs=[-2.5,.8,4.1]
        self.start('Causal Encoder–Decoder','Illustrative states')
        self.stage('Layer-local KV')
        blocks=VGroup(*[RoundedRectangle(width=2.05,height=.9,corner_radius=.12,stroke_color=GRID,fill_color=BACKGROUND,fill_opacity=1).move_to([x,-.15,0]) for x in xs])
        ids=VGroup(*[self.txt(f'Layer {21+i}',(x,-.9),22,MUTED) for i,x in enumerate(xs)])
        states=VGroup(*[self.strip(f['input'][-1],(x,-.15),BASE) for x,f in zip(xs,d['layers'])])
        enc=self.bank(d['encoder'],(-5.6,1.55),BASE,1.2)
        enc_label=self.txt('Encoder output',(-5.6,2.25),21,BASE)
        links=VGroup(self.arrow((-4.5,-.15),(-3.55,-.15)),self.arrow((-1.45,-.15),(-.25,-.15)),self.arrow((1.85,-.15),(3.05,-.15)))
        tail=self.txt('… 40',(6.15,-.15),24,MUTED)
        self.play(FadeIn(blocks),FadeIn(ids),FadeIn(states),Create(links),FadeIn(tail),run_time=1.4)
        banks=VGroup(*[self.bank(f['reference_kv'],(x,1.55)) for x,f in zip(xs,d['layers'])])
        writes=VGroup(*[self.arrow((x,.35),(x,.9),INTERMEDIATE) for x in xs])
        for i in range(3):
            self.play(Create(writes[i]),run_time=.4);self.signal(writes[i],INTERMEDIATE,.6)
            self.play(FadeIn(banks[i]),run_time=.5)
        self.wait(2);self.mark('layer-local-banks')
        self.play(Indicate(states[2],color=BASE),Indicate(banks[2],color=INTERMEDIATE),run_time=1.1)
        self.signal(writes[2],INTERMEDIATE,1)
        self.wait(1.2)
        self.stage('Encoder-derived KV')
        self.play(FadeOut(banks),FadeOut(writes),run_time=.8)
        self.play(FadeIn(enc),FadeIn(enc_label),run_time=.8)
        source_link=self.route([(-5.6,.95),(-5.6,-.15),(-4.5,-.15)],BASE)
        self.play(Create(source_link),run_time=.6)
        fans=VGroup(*[self.route([(-5,1.55),(-4.5,1.55),(-4.5,2.3),(x,2.3),(x,2.13)]) for x in xs])
        banks=VGroup(*[self.bank(f['ced_projection'],(x,1.55)) for x,f in zip(xs,d['layers'])])
        for i in range(3):
            self.play(Create(fans[i]),run_time=.5);self.signal(fans[i],INTERMEDIATE,.95)
            self.play(FadeIn(banks[i]),run_time=.45)
        self.wait(2.2);self.mark('ced-source-before-sharing')
        self.stage('Cross-layer sharing')
        self.play(FadeOut(banks),FadeOut(fans),run_time=.9)
        global_bank=self.bank(d['bank'],(-2.5,1.55))
        producer=self.route([(-5,1.55),(-3.3,1.55)])
        self.play(Create(producer),run_time=.5);self.signal(producer,INTERMEDIATE,.8)
        self.play(FadeIn(global_bank),run_time=.6)
        shared_label=self.txt('Global KV · 21–40',(-2.5,2.32),22,INTERMEDIATE)
        self.play(FadeIn(shared_label),run_time=.5)
        reads=VGroup(*[self.route([(-2.5,1.0),(-2.5,.7),(x,.7),(x,.32)]) for x in xs])
        self.play(LaggedStart(*[Create(r) for r in reads],lag_ratio=.2),run_time=1.3)
        self.wait(2);self.mark('shared-bank')
        locals_=VGroup();outputs=VGroup();qlabels=VGroup()
        for i,(x,f) in enumerate(zip(xs,d['layers'])):
            q=self.strip(f['q'][-1],(x,-.15),BASE)
            qlabel=self.txt('Q',(x+.78,-.15),18,BASE)
            local=self.bank(f['local'][-2:],(x,-1.6),MUTED,1.25)
            localtitle=self.txt('Local KV',(x,-2.17),21,MUTED)
            localgroup=VGroup(local,localtitle);locals_.add(localgroup);qlabels.add(qlabel)
            input_copy=states[i].copy()
            self.play(Transform(states[i],q),TransformFromCopy(input_copy,local),FadeIn(qlabel),FadeIn(localtitle),run_time=1.1)
            localread=self.route([(x+.8,-1.6),(x+1.16,-1.6),(x+1.16,.2),(x+1.025,.2)],MUTED)
            self.play(Create(localread),run_time=.35)
            self.signal(reads[i],INTERMEDIATE,.85);self.signal(localread,MUTED,.7)
            result=self.strip(f['output'][-1],(x,-.15),COMPOSED)
            self.play(Transform(states[i],result),FadeOut(qlabel),run_time=.7)
            self.play(FadeOut(localread),run_time=.25)
        self.wait(1.5);self.mark('distinct-local-states-and-outputs')
        self.play(Indicate(global_bank,color=INTERMEDIATE,scale_factor=1.06),run_time=.7)
        self.wait(4);self.mark('final');self.finish('01-global-kv')

class SharedRoles(Film):
    def construct(self):
        d=self.data['roles'];c=np.array(d['c']);weights=np.array(d['weights']);outputs=np.array(d['outputs'])
        self.start('Shared KV','2D example')
        self.stage('Separate K / V')
        k=self.bank([[.8,.2],[.1,.7],[-.5,.3]],(-2,1.0),INTERMEDIATE,1.6)
        v=self.bank([[.2,1],[.8,.1],[-.2,.5]],(2,1.0),COMPOSED,1.6)
        kl=self.txt('K',(-2,2.0),30,INTERMEDIATE);vl=self.txt('V',(2,2.0),30,COMPOSED)
        q=self.strip([2,0],(-5,1),BASE,1.5,True);ql=self.txt('Q',(-5,2),27,BASE)
        line=self.arrow((-4.1,1),(-2.95,1),BASE)
        self.play(FadeIn(k),FadeIn(v),FadeIn(kl),FadeIn(vl),FadeIn(q),FadeIn(ql),Create(line),run_time=1.2)
        self.signal(line,BASE,.9)
        match=self.txt('Match',(0,1),25);mix=self.txt('Mix',(4.5,1),25)
        self.play(FadeIn(match),run_time=.4)
        step=self.arrow((.65,1),(1.15,1),MUTED);self.play(Create(step),run_time=.4);self.signal(step,MUTED,.7)
        self.play(FadeIn(mix),run_time=.5)
        self.wait(2);self.mark('separate-roles')
        self.play(*[FadeOut(m) for m in [k,v,kl,vl,q,ql,line,match,mix,step]],run_time=1)
        self.stage('Direct projection')
        source=self.strip([.6,-.2],(-4.8,.6),BASE,1.5)
        projection=RoundedRectangle(width=1.7,height=.8,corner_radius=.1,stroke_color=GRID).move_to([-1.8,.6,0])
        pl=self.txt('Project',(-1.8,.6),24)
        one=self.strip(c[0],(1.3,.6),INTERMEDIATE,1.7,True)
        a=self.arrow((-3.95,.6),(-2.75,.6));b=self.arrow((-.85,.6),(.35,.6),INTERMEDIATE)
        self.play(FadeIn(source),FadeIn(projection),FadeIn(pl),Create(a),Create(b),run_time=.9)
        self.signal(a,BASE,.8);self.signal(b,INTERMEDIATE,.8);self.play(FadeIn(one),run_time=.5)
        self.wait(1.8);self.mark('direct-shared-projection')
        self.play(FadeOut(source),FadeOut(projection),FadeOut(pl),FadeOut(a),FadeOut(b),run_time=.7)
        ys=[1.45,.35,-.75]
        rows=VGroup(*[self.strip(row,(-2.0,y),INTERMEDIATE,1.8,True) for row,y in zip(c,ys)])
        self.play(one.animate.move_to(rows[0]),run_time=.7)
        self.remove(one);self.add(rows[0]);self.play(FadeIn(rows[1]),FadeIn(rows[2]),run_time=.6)
        banklabel=self.txt('Shared KV',(-2,2.25),25,INTERMEDIATE)
        q=self.strip(d['q'][0],(-5.1,.35),BASE,1.55,True);ql=self.txt('Q',(-5.1,1.12),24,BASE)
        self.play(FadeIn(banklabel),FadeIn(q),FadeIn(ql),run_time=.65)
        self.stage('Score')
        scoretitle=self.txt('Scores',(.55,2.25),24,MUTED)
        scores=VGroup(*[self.txt(f'{v:+.2f}',(.55,y),29) for v,y in zip(d['scores'][0],ys)])
        self.play(FadeIn(scoretitle),run_time=.4)
        for j in range(3):
            read=self.arrow((-4.22,.35),(-3.02,ys[j]),BASE)
            self.play(Create(read),run_time=.25);self.signal(read,BASE,.55)
            self.play(Indicate(rows[j],color=INTERMEDIATE,scale_factor=1.05),FadeIn(scores[j]),run_time=.55)
            self.play(FadeOut(read),run_time=.2)
        self.wait(1.6);self.mark('scores')
        self.stage('Normalize')
        newtitle=self.txt('Weights',(.55,2.25),24,MUTED)
        self.play(Transform(scoretitle,newtitle),run_time=.4)
        bars=VGroup()
        for j,(w,y) in enumerate(zip(weights[0],ys)):
            bar=Rectangle(width=float(w)*1.8,height=.11,stroke_width=0,fill_color=BASE,fill_opacity=.8).move_to([-.35+float(w)*.9,y-.35,0])
            bars.add(bar)
            self.play(Transform(scores[j],self.txt(f'{w:.3f}',(.55,y),29)),GrowFromEdge(bar,LEFT),run_time=.55)
        self.wait(1.5);self.mark('normalized-weights')
        self.stage('Mix')
        contributions=VGroup();total=np.zeros(2)
        for j,(row,w,y) in enumerate(zip(c,weights[0],ys)):
            contribution=self.strip(np.round(w*row,3),(4,y),COMPOSED,2.5,True)
            contributions.add(contribution)
            moving=rows[j].copy()
            for cell in moving:cell[1].set_opacity(0)
            self.add(moving)
            self.play(moving.animate.scale(.5),run_time=.2)
            path=self.route([(-2,y),(-2,y-.62),(4,y-.62),(4,y)],COMPOSED)
            self.play(MoveAlongPath(moving,path),run_time=1.05)
            self.play(Transform(moving,contribution),run_time=.55)
            self.remove(moving);self.add(contribution)
            total+=w*row
        sumlabel=self.txt('Weighted sum',(4,2.25),24,COMPOSED)
        self.play(FadeIn(sumlabel),run_time=.5)
        final=self.strip([0,0],(4,-1.9),COMPOSED,2.5,True)
        self.play(FadeIn(final),run_time=.4)
        accumulated=np.zeros(2)
        for j,part in enumerate(contributions):
            copy=part.copy()
            for cell in copy:cell[1].set_opacity(0)
            self.add(copy)
            self.play(copy.animate.scale(.5),part.animate.set_opacity(.28),run_time=.25)
            route=self.route([(4,ys[j]),(6.0,ys[j]),(6.0,-1.9),(4,-1.9)],COMPOSED)
            self.play(MoveAlongPath(copy,route),run_time=.9)
            self.remove(copy);accumulated+=weights[0,j]*c[j]
            self.play(Transform(final,self.strip(np.round(accumulated,3),(4,-1.9),COMPOSED,2.5,True)),run_time=.35)
        self.wait(2);self.mark('same-rows-weighted-sum')
        self.play(FadeOut(contributions),FadeOut(sumlabel),FadeOut(final),FadeOut(scores),FadeOut(bars),FadeOut(scoretitle),FadeOut(q),FadeOut(ql),run_time=.9)
        self.stage('Query heads')
        headys=[1.45,.35,-.75];qs=VGroup();outs=VGroup();headlines=VGroup()
        outtitle=self.txt('Head outputs',(4,2.25),24,COMPOSED)
        querytitle=self.txt('Queries',(-5.1,2.25),24,BASE)
        self.play(FadeIn(outtitle),FadeIn(querytitle),run_time=.4)
        for i,y in enumerate(headys):
            qi=self.strip(d['q'][i],(-5.1,y),BASE,1.55,True);oi=self.strip(np.round(outputs[i],3),(4,y),COMPOSED,2.5,True)
            qs.add(qi);outs.add(oi)
            self.play(FadeIn(qi),run_time=.35)
            link=self.route([(-4.2,y),(-3.5,y),(-3.5,.35),(-3.03,.35)],BASE)
            self.signal(link,BASE,.6)
            wrow=VGroup(*[self.txt(f'{w:.2f}',(.15+j*.75,y),21,BASE) for j,w in enumerate(weights[i])])
            headlines.add(wrow)
            self.play(FadeIn(wrow),FadeIn(oi),run_time=.65)
        wt=self.txt('Weights',(.9,2.25),24,MUTED);self.play(FadeIn(wt),run_time=.4)
        self.wait(2);self.mark('three-toy-heads')
        self.wait(3.2);self.mark('final');self.finish('02-shared-roles')
