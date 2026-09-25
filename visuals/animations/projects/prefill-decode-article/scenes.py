"""Seven mechanism-first article loops. All timelines are explanatory, not timings."""
from functools import partial
import json
from pathlib import Path
import sys
import numpy as np
from manim import *

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from manim_style import label, BACKGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED, ACCENT
label=partial(label,layout_scale=16)


def tile(s,x,y,c=BASE,w=1.05,h=.65,size=28):
    b=RoundedRectangle(width=w,height=h,corner_radius=.10,stroke_color=c,
                       stroke_width=2,fill_color=c,fill_opacity=.035)
    t=label(s,size,c)
    assert t.width < w-.14,(s,t.width,w)
    return VGroup(b,t.move_to(b)).move_to([x,y,0])


def route(points,c=INTERMEDIATE):
    return VMobject(stroke_color=c,stroke_width=2.5).set_points_as_corners(
        [np.array(p,dtype=float) for p in points])


def edge(a,b,c=MUTED):
    return Arrow(a,b,buff=.10,color=c,stroke_width=2,tip_length=.13)


class Loop(Scene):
    def begin(self,title,note):
        self.camera.background_color=BACKGROUND
        self.events=[]
        self.title=label(title,34).to_edge(UP,buff=.35).to_edge(LEFT,buff=.55)
        self.add(self.title,label(note,18,MUTED).to_edge(DOWN,buff=.23))

    def mark(self,name,**state):
        for m in self.mobjects:
            assert m.get_left()[0]>-7.1 and m.get_right()[0]<7.1,(name,'x')
            assert m.get_bottom()[1]>-3.95 and m.get_top()[1]<3.95,(name,'y')
        self.events.append(dict(name=name,seconds=round(float(self.time),3),**state))
        self.wait(.35)

    def finish(self,stem):
        self.wait(1.2)
        self.mark('complete')
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.5)
        self.wait(.2)
        (P/(stem+'-timeline.json')).write_text(json.dumps(dict(events=self.events,
            checks={'inside_canvas':True},duration_seconds=float(self.time)),indent=2)+'\n')


class Lifecycle(Loop):
    def construct(self):
        self.begin('Two phases. Two kinds of waiting.','Phase overview · schematic order, not measured time')
        inputs=VGroup(*[tile(f'x{i+1}',-5.2+i*1.1,1.4,size=24,w=.85) for i in range(4)])
        self.add(label('Prefill',30,BASE).move_to([-3.6,2.4,0]),
                 label('Decode',30,COMPOSED).move_to([3.5,2.4,0]))
        model=tile('Same model',-3.6,0,w=3.7,h=.8,size=28)
        self.play(FadeIn(inputs),FadeIn(model),run_time=.6)
        self.mark('prompt_received',prompt=4)
        beams=[route([x.get_bottom(),model.get_top()],BASE) for x in inputs]
        self.play(*[ShowPassingFlash(p,time_width=.6) for p in beams],run_time=1.1)
        cache=VGroup(*[tile(f'x{i+1}',-5.2+i*1.1,-1.3,INTERMEDIATE,w=.85,size=24) for i in range(4)])
        self.play(*[TransformFromCopy(x,k) for x,k in zip(inputs,cache)],run_time=.8)
        self.add(label('Prompt KV · each layer',21,INTERMEDIATE).move_to([-3.6,-2,0]))
        self.mark('prompt_kv_ready',prompt_cache=4)
        outputs=VGroup(*[tile(f'y{i+1}',1.5+i*1.8,1.4,COMPOSED) for i in range(3)])
        e=edge(model.get_right(),outputs[0].get_left(),COMPOSED)
        self.play(GrowArrow(e),FadeIn(outputs[0]),run_time=.65)
        tt=edge([-6,-2.7,0],[1.5,-2.7,0],ACCENT)
        self.play(GrowArrow(tt),FadeIn(label('TTFT',26,ACCENT).move_to([-2.3,-3.15,0])),run_time=.5)
        self.mark('first_output',selected='y1',prompt_cache=4)
        for i in [1,2]:
            self.play(GrowArrow(edge(outputs[i-1].get_right(),outputs[i].get_left(),COMPOSED)),
                      FadeIn(outputs[i]),run_time=.9)
            self.mark(f'output_{i+1}',selected=f'y{i+1}')
        self.play(GrowArrow(edge([3.3,-.25,0],[5.1,-.25,0],COMPOSED)),
                  FadeIn(label('ITL',26,COMPOSED).move_to([4.2,-.75,0])),run_time=.5)
        self.finish('01-lifecycle')


class ParallelPositions(Loop):
    def construct(self):
        self.begin('Read input-derived K/V. Produce separate outputs.',
                   'One head in one layer · other block operations omitted')
        rows=[1.25,-1.25]
        hs=VGroup(*[tile(f'h{i+1}',-5.6,y) for i,y in enumerate(rows)])
        qs=VGroup(*[tile(f'Q{i+1} / K{i+1} / V{i+1}',-2.8,y,INTERMEDIATE,w=2.8,size=25) for i,y in enumerate(rows)])
        att=VGroup(*[tile(f'Position {i+1}',1.2,y,w=2.4,size=27) for i,y in enumerate(rows)])
        outs=VGroup(*[tile(f'O{i+1}',5.2,y,COMPOSED) for i,y in enumerate(rows)])
        self.add(hs)
        self.mark('inputs_ready')
        self.play(*[TransformFromCopy(h,q) for h,q in zip(hs,qs)],run_time=.8)
        self.add(*[edge(h.get_right(),q.get_left(),BASE) for h,q in zip(hs,qs)])
        self.mark('qkv_ready')
        paths=[edge(q.get_right(),a.get_left(),INTERMEDIATE) for q,a in zip(qs,att)]
        cross=edge(qs[0].get_right()+DOWN*.25,att[1].get_left()+UP*.25,INTERMEDIATE)
        self.play(*[GrowArrow(p) for p in paths],GrowArrow(cross),FadeIn(att),run_time=.8)
        self.add(label('K1 / V1',23,INTERMEDIATE).move_to([-.1,.55,0]))
        self.play(*[ShowPassingFlash(route([q.get_right(),a.get_left()]),time_width=.6)
                    for q,a in zip(qs,att)],
                  ShowPassingFlash(route([qs[0].get_right()+DOWN*.25,att[1].get_left()+UP*.25]),time_width=.6),run_time=.9)
        self.mark('causal_reads',reads=[[1],[1,2]],uses_O1=False)
        self.play(*[GrowArrow(edge(a.get_right(),o.get_left(),COMPOSED)) for a,o in zip(att,outs)],
                  *[FadeIn(o) for o in outs],run_time=.8)
        self.mark('outputs_together',same_layer_output_dependency=False)
        self.play(FadeIn(label('No O1 → O2 dependency',27,COMPOSED).move_to([1,-2.65,0])),run_time=.4)
        self.finish('02-parallel')


class FirstToken(Loop):
    def construct(self):
        self.begin('Selected does not yet mean cached.','Four prompt positions · cache rows stand for per-layer state')
        cells=VGroup()
        for i in range(4):
            for j in range(4):
                cells.add(Square(side_length=.55,stroke_width=0,fill_color=BASE if j<=i else GRID,
                    fill_opacity=0).move_to([-5+j*.7,1.2-i*.7,0]))
        self.add(cells,label('Causal prefill',26,BASE).move_to([-3.9,2.1,0]))
        self.play(*[c.animate.set_fill(opacity=.95 if i//4>=i%4 else .18) for i,c in enumerate(cells)],run_time=.8)
        self.mark('causal_grid',allowed=[[j<=i for j in range(4)] for i in range(4)])
        model=tile('Prefill',1.2,1.2,w=2.6,h=.8)
        cache=VGroup(*[tile(f'x{i+1}',-.6+i*1.1,-.85,INTERMEDIATE,w=.85,size=24) for i in range(4)])
        self.play(FadeIn(model),FadeIn(cache),run_time=.65)
        count=label('4 entries',27,INTERMEDIATE).move_to([1.5,-1.65,0])
        self.add(count)
        y=tile('y1',5,1.2,COMPOSED)
        self.play(GrowArrow(edge(model.get_right(),y.get_left(),COMPOSED)),FadeIn(y),run_time=.7)
        self.mark('y1_selected',cache=4,selected='y1')
        feed=route([y.get_bottom(),[5,0,0],[1.2,0,0],model.get_bottom()],COMPOSED)
        self.play(MoveAlongPath(y,feed),Transform(model,tile('Forward y1',1.2,1.2,w=2.6,h=.8,size=26)),run_time=1.2)
        self.play(FadeOut(y),run_time=.25)
        k=tile('y1',3.8,-.85,COMPOSED,w=.85,size=24)
        self.play(TransformFromCopy(model,k),Transform(count,label('5 entries',27,INTERMEDIATE).move_to(count)),run_time=.7)
        self.mark('y1_kv_appended',cache=5)
        y2=tile('y2',5,1.2,COMPOSED)
        self.play(FadeIn(y2),run_time=.6)
        self.mark('y2_selected',cache=5,selected='y2')
        self.finish('03-first-token')


class DecodeLoop(Loop):
    def construct(self):
        self.begin('Feed back. Append KV. Select the next token.',
                   'Two-layer schematic · vocabulary head and other operations compressed')
        layers=VGroup(tile('Layer 1',-1.3,1.5,w=2.2),tile('Layer 2',1.3,1.5,w=2.2))
        self.add(layers,edge(layers[0].get_right(),layers[1].get_left()),
                 label('Selected',24,COMPOSED).move_to([5.3,2.25,0]))
        xs=[-2.7,-1.4,-.1,1.2,2.5,3.8]
        ys=[-.6,-1.5]
        cache=[VGroup(*[tile(f'x{i+1}',xs[i],y,INTERMEDIATE,w=.95,size=24) for i in range(4)]) for y in ys]
        self.add(*cache,label('Layer 1 KV',22,INTERMEDIATE).move_to([-4.4,-.6,0]),
                 label('Layer 2 KV',22,INTERMEDIATE).move_to([-4.4,-1.5,0]))
        returning=route([[5.3,1.5,0],[5.8,1.5,0],[5.8,-2.65,0],[-5.8,-2.65,0],[-5.8,1.5,0],[-4.9,1.5,0]],GRID)
        self.add(returning)
        selected=tile('y1',5.3,1.5,COMPOSED)
        self.play(FadeIn(selected),run_time=.5)
        counts=[4,4]
        self.mark('y1_selected',cache=counts.copy(),selected='y1')
        for step in [1,2]:
            snapshots=[m.get_all_points().copy() for row in cache for m in row]
            self.play(MoveAlongPath(selected,returning),run_time=1.4)
            self.mark(f'y{step}_input',cache=counts.copy())
            dot=Dot(selected.get_center(),radius=.09,color=BASE)
            self.play(FadeOut(selected),FadeIn(dot),run_time=.2)
            for layer in range(2):
                self.play(dot.animate.move_to(layers[layer].get_center()+DOWN*.22),run_time=.45)
                kv=tile(f'y{step}',xs[3+step],ys[layer],COMPOSED,w=.95,size=24)
                self.play(TransformFromCopy(dot,kv),run_time=.45)
                cache[layer].add(kv);counts[layer]+=1
                read=route([cache[layer].get_left(),[-3.55,ys[layer],0],[-3.55,.6,0],layers[layer].get_bottom()],INTERMEDIATE)
                self.play(ShowPassingFlash(read,time_width=.55),Indicate(layers[layer],color=COMPOSED,scale_factor=1.04),run_time=.55)
                self.mark(f'y{step}_layer{layer+1}',cache=counts.copy(),read_includes_self=True)
            self.play(dot.animate.move_to([4.45,1.5,0]),run_time=.5)
            selected=tile(f'y{step+1}',5.3,1.5,COMPOSED)
            self.play(FadeOut(dot),FadeIn(selected),run_time=.4)
            old_now=[m for row in cache for m in list(row)[:-1]]
            for before,m in zip(snapshots,old_now):
                np.testing.assert_allclose(before,m.get_all_points(),atol=1e-10)
            self.mark(f'y{step+1}_selected',cache=counts.copy(),selected=f'y{step+1}',old_cache_unchanged=True)
        self.finish('04-decode-loop')


def line_stack(x,y,n,c,width=2.7,spacing=.26):
    return VGroup(*[Line([x-width/2,y-i*spacing,0],[x+width/2,y-i*spacing,0],
                        color=c,stroke_width=7) for i in range(n)])


class Workloads(Loop):
    def construct(self):
        self.begin('Read a long report. Or write a long speech.',
                   'Illustrative input/output volume · line counts and timing are schematic')
        self.add(Line([0,2.5,0],[0,-2.75,0],color=GRID,stroke_width=1),
                 label('Long report',27,BASE).move_to([-3.4,2.35,0]),
                 label('Short prompt',27,BASE).move_to([3.4,2.35,0]))
        left=line_stack(-3.4,1.85,7,BASE,spacing=.20)
        right=line_stack(3.4,1.85,2,BASE,spacing=.20)
        self.play(FadeIn(left),FadeIn(right),run_time=.7)
        self.mark('inputs',input_lines=[7,2])
        self.play(GrowArrow(edge([-3.4,.4,0],[-3.4,-.3,0],INTERMEDIATE)),
                  GrowArrow(edge([3.4,.4,0],[3.4,-.3,0],INTERMEDIATE)),run_time=.7)
        a=line_stack(-3.4,-.65,3,COMPOSED,spacing=.25)
        b=line_stack(3.4,-.65,7,COMPOSED,spacing=.25)
        for i in range(7):
            animations=[Create(b[i])]
            if i<3:animations.append(Create(a[i]))
            self.play(*animations,run_time=.42)
        self.mark('outputs',output_lines=[3,7],measured=False)
        self.play(FadeIn(label('3 conclusions',25,COMPOSED).move_to([-3.4,-2.55,0])),
                  FadeIn(label('Long speech',25,COMPOSED).move_to([3.4,-2.8,0])),run_time=.4)
        self.finish('05-workloads')


class Scheduling(Loop):
    def construct(self):
        self.begin('Keep A writing while B enters.', 'Separate A/B state · conceptual scheduling, not measured throughput')
        a=tile('A',-4.8,1.7,COMPOSED)
        bs=VGroup(*[tile(f'B{i+1}',-1.4+i*1.5,1.7) for i in range(3)])
        self.add(a,bs,label('Answer',22,COMPOSED).move_to([-4.8,2.45,0]),
                 label('Input chunks',22,BASE).move_to([.1,2.45,0]))
        iteration=RoundedRectangle(width=4,height=1.1,corner_radius=.14,stroke_color=GRID).move_to([0,0,0])
        self.add(iteration,label('A output',22,COMPOSED).move_to([-4.7,-.8,0]),
                 label('B cache',22,INTERMEDIATE).move_to([1.2,-.8,0]))
        bank=VGroup();answer=VGroup()
        self.mark('requests',separate_state=True)
        for i in range(3):
            acopy=a.copy();self.add(acopy)
            self.play(acopy.animate.move_to([-1,0,0]),bs[i].animate.move_to([1,0,0]),run_time=.65)
            if i:
                self.play(ShowPassingFlash(route([bank.get_top(),bs[i].get_bottom()]),time_width=.6),run_time=.4)
            self.mark(f'iteration_{i+1}',decode_request='A',prefill_chunk=f'B{i+1}')
            kv=tile(f'B{i+1}',-.3+i*1.5,-1.6,INTERMEDIATE)
            out=Line([-5.5,-1.35-i*.32,0],[-3.7,-1.35-i*.32,0],color=COMPOSED,stroke_width=7)
            self.play(TransformFromCopy(bs[i],kv),Create(out),FadeOut(acopy),FadeOut(bs[i]),run_time=.6)
            bank.add(kv);answer.add(out)
            self.mark(f'chunk_{i+1}_retained',B_chunks=i+1,A_outputs=i+1)
        # Preserve B's identity when introducing an alternative placement.
        keep=[self.title,bank]
        for item in bank:self.remove(item)
        self.add(bank)
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m not in keep],
                  bank.animate.scale(.8).move_to([-3,-1.4,0]),
                  Transform(self.title,label('Or separate prefill and decode resources.',34).move_to(self.title,aligned_edge=LEFT)),run_time=.65)
        pre=tile('Prefill',-3,.6,w=3.4,h=.9,size=30)
        dec=tile('Decode',3,.6,COMPOSED,w=3.4,h=.9,size=30)
        self.play(FadeIn(pre),FadeIn(dec),GrowArrow(edge(pre.get_right(),dec.get_left(),INTERMEDIATE)),
                  FadeIn(label('B: KV + request state',24,INTERMEDIATE).move_to([0,-.35,0])),run_time=.6)
        self.mark('separate_pools',request='B',B_chunks=3)
        self.play(bank.animate.move_to([3,-1.4,0]),run_time=1.4)
        self.mark('B_handoff',B_chunks=3,transferred_request='B')
        self.add(label('Transfer and coordination have a cost',23,MUTED).move_to([0,-2.55,0]))
        self.finish('06-scheduling')


class Handoff(Loop):
    def construct(self):
        self.begin('Send ready KV while later layers compute.',
                   'Third-party Spark → M3 Ultra example · schematic overlap, not a speed measurement')
        self.add(tile('DGX Spark',-3.8,2,w=3.6,h=.8,size=30),
                 tile('M3 Ultra',3.8,2,COMPOSED,w=3.6,h=.8,size=30))
        ys=[.65,-.35,-1.35]
        layers=VGroup(*[tile(f'L{i+1}',-4.7,y,w=1.1) for i,y in enumerate(ys)])
        self.add(layers,label('Prefill',24,BASE).move_to([-4.7,1.28,0]),
                 label('Received KV',24,INTERMEDIATE).move_to([4.4,1.28,0]))
        packets=[]
        self.play(Indicate(layers[0],color=COMPOSED,scale_factor=1.08),run_time=.8)
        layers[0].set_color(COMPOSED)
        self.mark('L1_ready',computed=[1],received=[])
        p1=tile('KV 1',-2.8,ys[0],INTERMEDIATE,w=1.3,size=25);packets.append(p1)
        self.add(p1)
        self.play(p1.animate.move_to([0,ys[0],0]),Indicate(layers[1],color=COMPOSED,scale_factor=1.08),run_time=1.1)
        layers[1].set_color(COMPOSED)
        self.mark('L2_ready',computed=[1,2],received=[],transfer_in_flight=[1])
        p2=tile('KV 2',-2.8,ys[1],INTERMEDIATE,w=1.3,size=25);packets.append(p2);self.add(p2)
        self.play(p1.animate.move_to([4.4,ys[0],0]),p2.animate.move_to([0,ys[1],0]),
                  Indicate(layers[2],color=COMPOSED,scale_factor=1.08),run_time=1.1)
        layers[2].set_color(COMPOSED)
        self.mark('L3_ready',computed=[1,2,3],received=[1],transfer_in_flight=[2])
        p3=tile('KV 3',-2.8,ys[2],INTERMEDIATE,w=1.3,size=25);packets.append(p3);self.add(p3)
        self.play(p2.animate.move_to([4.4,ys[1],0]),p3.animate.move_to([0,ys[2],0]),run_time=1.1)
        tail=label('Transfer tail',27,ACCENT).move_to([0,-2.3,0])
        self.play(FadeIn(tail),run_time=.3)
        self.mark('transfer_tail',computed=[1,2,3],received=[1,2],decode_started=False)
        self.play(p3.animate.move_to([4.4,ys[2],0]),run_time=1.2)
        self.mark('all_received',received=[1,2,3])
        self.play(FadeOut(tail),FadeIn(label('Continue decode',29,COMPOSED).move_to([3.6,-2.5,0])),run_time=.5)
        self.mark('decode_started',received=[1,2,3],decode_started=True)
        self.finish('07-handoff')
