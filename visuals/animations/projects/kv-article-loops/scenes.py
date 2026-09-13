"""Two short, source-grounded cache schematics for the bilingual article."""
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED, ACCENT

label = partial(label, layout_scale=16)


def card(text, point, width=1.35, height=.65, size=31, color=BASE):
    shape = RoundedRectangle(width=width, height=height, corner_radius=.11,
        stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=.045)
    return VGroup(shape, label(text, size, color).move_to(shape)).move_to(point)


def strip(point, color, width=1.35, height=.32):
    outer = RoundedRectangle(width=width, height=height, corner_radius=.055,
        stroke_color=color, stroke_width=1.6, fill_color=color, fill_opacity=.06)
    bars = VGroup(*[RoundedRectangle(width=.14, height=height*.53, corner_radius=.02,
        stroke_width=0, fill_color=color, fill_opacity=.7) for _ in range(3)]).arrange(RIGHT, buff=.17).move_to(outer)
    return VGroup(outer, bars).move_to(point)


def kv(point):
    return VGroup(strip([0, .21, 0], INTERMEDIATE), strip([0, -.21, 0], COMPOSED)).move_to(point)


def route(points, color=INTERMEDIATE, width=2.2):
    return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners([np.array(p) for p in points])


class Loop(Scene):
    def begin(self, title, scope):
        self.camera.background_color = BACKGROUND
        self.events = []
        self.add(label(title, 34).to_edge(UP, buff=.38).to_edge(LEFT, buff=.55),
                 label(scope, 18, MUTED).to_edge(DOWN, buff=.22))

    def mark(self, name, **state):
        for m in self.mobjects:
            assert m.get_left()[0] > -7.1 and m.get_right()[0] < 7.1, name
            assert m.get_bottom()[1] > -3.95 and m.get_top()[1] < 3.95, name
        self.events.append(dict(name=name, seconds=round(float(self.time),3), **state))

    def finish(self, stem, checks):
        self.wait(.9)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.4)
        self.wait(.15)
        assert all(checks.values())
        (P/(stem+'-timeline.json')).write_text(json.dumps(dict(
            events=self.events, duration_seconds=float(self.time),
            checks={**checks, 'inside_canvas':True}), indent=2)+'\n')


class AppendEdit(Loop):
    def construct(self):
        self.begin('Append or edit?', 'Fixed model and positions · dependency schematic')
        xs = [-2.9, -1.15, .6, 2.35, 4.1]
        top = VGroup(*[card(w,[xs[i],2.0,0]) for i,w in enumerate('ABCDE')])
        bottom = VGroup(*[card(w,[xs[i],-1.15,0]) for i,w in enumerate('ABCD')])
        upper = VGroup(*[kv([x,.88,0]) for x in xs])
        lower = VGroup(*[kv([x,-2.25,0]) for x in xs[:4]])
        self.add(label('Append',29,BASE).move_to([-5.3,2,0]),
                 label('Edit',29,ACCENT).move_to([-5.3,-1.15,0]),
                 label('K / V',24,MUTED).move_to([-5.3,.88,0]),
                 label('K / V',24,MUTED).move_to([-5.3,-2.25,0]),
                 Line([-6.2,-.25,0],[6.2,-.25,0],color=GRID,stroke_width=1))
        self.play(FadeIn(top[:4]),FadeIn(bottom),FadeIn(upper[:4]),FadeIn(lower),run_time=.6)
        self.mark('two_prefixes', original_positions=4)
        past_points = [m.get_all_points().copy() for m in upper[:4]]
        first_points = lower[0].get_all_points().copy()
        self.wait(.25)
        self.play(FadeIn(top[4],shift=LEFT*.25),run_time=.4)
        self.mark('append_input', reusable=[1,2,3,4])
        self.play(TransformFromCopy(top[4],upper[4]),run_time=.65)
        reuse = label('Reuse',24,COMPOSED).move_to([-.25,.15,0])
        brace = Line([-3.58,.35,0],[3.03,.35,0],color=COMPOSED,stroke_width=2)
        self.play(Create(brace),FadeIn(reuse),run_time=.35)
        for old,m in zip(past_points,upper[:4]):np.testing.assert_allclose(old,m.get_all_points(),atol=1e-10)
        self.mark('append_kv', reusable=[1,2,3,4], appended=5)
        self.wait(.55)
        edit = card('X',[xs[1],-1.15,0],color=ACCENT)
        self.play(Transform(bottom[1],edit),run_time=.45)
        self.mark('edit_input', changed_position=2)
        crosses=VGroup()
        for m in lower[1:]:
            crosses.add(VGroup(Line(m.get_corner(UL),m.get_corner(DR),color=ACCENT,stroke_width=3),
                               Line(m.get_corner(DL),m.get_corner(UR),color=ACCENT,stroke_width=3)))
        recompute = label('Recompute suffix',24,ACCENT).move_to([.6,-3.02,0])
        keep = label('Reuse',24,COMPOSED).move_to([xs[0],-3.02,0])
        pulses=[route([bottom[1].get_bottom(),[xs[1],-1.67,0],[xs[i],-1.67,0],lower[i].get_top()],ACCENT) for i in [1,2,3]]
        self.play(*[ShowPassingFlash(p,time_width=.8) for p in pulses],run_time=.6)
        self.play(FadeIn(crosses),FadeIn(recompute),FadeIn(keep),run_time=.35)
        self.mark('suffix_invalidated', reusable=[1], invalidated=[2,3,4])
        self.wait(.45)
        fresh=VGroup(*[kv([xs[i],-2.25,0]) for i in [1,2,3]])
        self.play(FadeOut(crosses),*[FadeOut(m) for m in lower[1:]],run_time=.25)
        self.play(*[TransformFromCopy(bottom[i],fresh[i-1]) for i in [1,2,3]],run_time=.7)
        np.testing.assert_allclose(first_points,lower[0].get_all_points(),atol=1e-10)
        self.mark('suffix_recomputed', reusable=[1], recomputed=[2,3,4])
        self.wait(.6)
        self.finish('append-edit', dict(append_preserves_old_geometry=True,edit_preserves_earlier_geometry=True,
                                      suffix_recomputed_from_edited_position=True))


class GlobalSharing(Loop):
    def construct(self):
        self.begin('Causal Encoder–Decoder', 'Decoder schematic · selected global entries + local KV')
        encoder=card('Encoder output',[-4.55,1.8,0],width=2.8,height=.8,size=26)
        bank_outline=RoundedRectangle(width=2.5,height=.9,corner_radius=.12,
            stroke_color=INTERMEDIATE,stroke_width=2,fill_color=INTERMEDIATE,fill_opacity=.04).move_to([0,1.8,0])
        entries=VGroup(*[strip([x,1.8,0],INTERMEDIATE,width=.6,height=.5) for x in [-.75,0,.75]])
        # The narrow entries use their own scaled glyphs to stay within the bank.
        for e in entries:e[1].scale(.5).move_to(e[0])
        bank=VGroup(bank_outline,entries)
        title=label('Global KV',27,INTERMEDIATE).move_to([0,2.5,0])
        projection=Arrow(encoder.get_right(),bank.get_left(),buff=.10,color=BASE,stroke_width=2,tip_length=.12)
        projection_name=label('L21 projection',20,MUTED).move_to([-2.2,2.33,0])
        self.play(FadeIn(encoder),run_time=.45)
        self.mark('encoder_output', global_banks=0)
        self.play(Create(projection),FadeIn(projection_name),TransformFromCopy(encoder,bank),FadeIn(title),run_time=.75)
        bank_points=bank.get_all_points().copy()
        self.mark('global_bank_created', global_banks=1,source='encoder_output',producer_layer=21)
        layer_xs=[-4.2,0,4.2]
        layers=VGroup();queries=[];locals_=[]
        for x,n in zip(layer_xs,[21,22,40]):
            outline=RoundedRectangle(width=2.8,height=1.4,corner_radius=.13,
                stroke_color=GRID,stroke_width=2,fill_color=BACKGROUND,fill_opacity=1).move_to([x,-.6,0])
            heading=label(f'Layer {n}',25).move_to([x,-.22,0])
            q=card('Q',[x-.82,-.89,0],width=.62,height=.51,size=26)
            local=card('Local KV',[x+.48,-.89,0],width=1.62,height=.51,size=21,color=COMPOSED)
            queries.append(q);locals_.append(local);layers.add(VGroup(outline,heading,q,local))
        self.play(FadeIn(layers),run_time=.55)
        routes=VGroup(*[route([bank.get_bottom(),[0,.7,0],[x,.7,0],layer.get_top()],INTERMEDIATE)
                        for x,layer in zip(layer_xs,layers)])
        self.play(*[Create(p.copy().set_color(GRID)) for p in routes],run_time=.35)
        self.mark('layer_local_state',global_banks=1,decoder_layers=[21,22,40],distinct_queries=3,local_caches=3)
        for n,p,q,local,layer in zip([21,22,40],routes,queries,locals_,layers):
            self.play(Indicate(q,color=BASE,scale_factor=1.08),
                      Indicate(local,color=COMPOSED,scale_factor=1.03),
                      ShowPassingFlash(p,time_width=.7),run_time=.6)
            self.mark(f'layer_{n}_read',global_banks=1,bank_source_layer=21,reader=n)
        np.testing.assert_allclose(bank_points,bank.get_all_points(),atol=1e-10)
        self.wait(.3)
        # This is a magnified view of one selected entry, not another allocated bank.
        selected=strip([-3.7,-2.7,0],INTERMEDIATE,width=1.7,height=.65)
        selected_title=label('One selected vector',24,INTERMEDIATE).move_to([-3.7,-1.97,0])
        matching=card('Q matching',[.7,-2.2,0],width=3,height=.62,size=27,color=BASE)
        mixture=card('Weighted sum',[.7,-3.08,0],width=3,height=.62,size=27,color=COMPOSED)
        self.play(Indicate(entries[0],color=INTERMEDIATE,scale_factor=1.08),
                  TransformFromCopy(entries[0],selected),FadeIn(selected_title),
                  FadeIn(matching),FadeIn(mixture),run_time=.7)
        self.mark('selected_entry_detail',global_banks=1,detail_of_entry=0)
        arrows=VGroup(Arrow(selected.get_right(),matching.get_left(),buff=.06,color=INTERMEDIATE,stroke_width=2,tip_length=.11),
                      Arrow(selected.get_right(),mixture.get_left(),buff=.06,color=COMPOSED,stroke_width=2,tip_length=.11))
        self.play(Create(arrows),run_time=.3)
        self.play(ShowPassingFlash(arrows[0].copy(),time_width=.8),Indicate(matching,color=BASE,scale_factor=1.03),run_time=.55)
        self.mark('same_entry_scores',global_banks=1,entry=0,role='query_matching')
        self.play(ShowPassingFlash(arrows[1].copy(),time_width=.8),Indicate(mixture,color=COMPOSED,scale_factor=1.03),run_time=.55)
        self.mark('same_entry_accumulates',global_banks=1,entry=0,role='weighted_accumulation')
        assert len(queries)==len(locals_)==3 and len({id(q) for q in queries})==3
        np.testing.assert_allclose(bank_points,bank.get_all_points(),atol=1e-10)
        self.finish('global-sharing',dict(one_unchanged_global_bank=True,distinct_local_queries_and_caches=True,
                                          same_selected_entry_in_both_roles=True))
