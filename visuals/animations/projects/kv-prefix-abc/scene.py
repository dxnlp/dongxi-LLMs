"""A/B/C document requests: computed states versus exact-prefix reuse."""
from functools import partial
from pathlib import Path
import json
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED, ACCENT
from mechanism import REQUESTS, shared_prefix, state_fingerprint

label = partial(label, layout_scale=16)


def cn(text, size=30, color=FOREGROUND):
    return label(text, size, color, cjk=True)


def cell(text, x, color=BASE, width=1.65, cjk=True):
    box = RoundedRectangle(width=width, height=.78, corner_radius=.10,
        stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=.035)
    word = (cn(text, 31, color) if cjk else label(text, 31, color)).move_to(box)
    assert word.width < width-.14
    return VGroup(box, word).move_to([x, 1.3, 0])


def entry(x, prefix, color=INTERMEDIATE):
    """Paired schematic K/V glyphs; deterministic identities, not activations."""
    rows = VGroup()
    values = state_fingerprint(prefix)
    for r in range(2):
        box = RoundedRectangle(width=1.65, height=.39, corner_radius=.055,
            stroke_color=color, stroke_width=1.8, fill_color=color, fill_opacity=.04)
        bars = VGroup(*[Rectangle(width=.14, height=.10+.13*values[r*4+j],
            stroke_width=0, fill_color=color, fill_opacity=.8) for j in range(4)])
        bars.arrange(RIGHT, buff=.17).move_to(box)
        rows.add(VGroup(box, bars).move_to([x, -.58-r*.56, 0]))
    return rows


class PrefixABC(Scene):
    def mark(self, name, **state):
        for m in self.mobjects:
            assert m.get_left()[0] > -7.08 and m.get_right()[0] < 7.08, name
            assert m.get_bottom()[1] > -3.95 and m.get_top()[1] < 3.95, name
        self.events.append(dict(name=name, seconds=round(float(self.time), 3), **state))

    def construct(self):
        self.camera.background_color = BACKGROUND
        self.events = []
        xs = [-4.85, -2.95, -1.05, .85, 4.40]
        names = ['项目', '100 万', '进度', '交付']
        title = cn('同一份方案，三次提问', 35).to_edge(UP, buff=.38).to_edge(LEFT, buff=.55)
        badge = VGroup(label('A', 32, BASE), cn('先看方案', 25, BASE)).arrange(RIGHT, buff=.22).to_edge(UP, buff=.42).to_edge(RIGHT, buff=.6)
        doc_box = RoundedRectangle(width=7.55, height=1.36, corner_radius=.17,
            stroke_color=GRID, stroke_width=1.6).move_to([-2, 1.3, 0])
        doc_name = cn('项目方案', 24, MUTED).move_to([-2, 2.25, 0])
        instruction_name = cn('你的指令', 24, MUTED).move_to([4.4, 2.25, 0])
        plus = label('+', 32, MUTED).move_to([2.55, 1.3, 0])
        input_cells = VGroup(*[cell(t, x) for t, x in zip(names, xs)])
        instruction = cell('看看风险', xs[4], width=2.7)
        kv_label = label('KV cache', 27, INTERMEDIATE).move_to([-2, -1.83, 0])
        k = label('K', 24, MUTED).move_to([-6.15, -.58, 0])
        v = label('V', 24, MUTED).move_to([-6.15, -1.14, 0])
        caveat = cn('相同执行设置 · 缓存仍可用 · 输入片段示意', 19, MUTED).to_edge(DOWN, buff=.24)
        phase = VGroup(label('Prefill', 29, COMPOSED), cn('计算输入', 25, COMPOSED)).arrange(RIGHT, buff=.3).move_to([0, -2.76, 0])
        self.add(title, badge, doc_box, doc_name, instruction_name, plus, k, v, caveat)
        self.play(FadeIn(input_cells), FadeIn(instruction), FadeIn(phase), run_time=.65)
        self.mark('A_input', document=REQUESTS['A'][:4], instruction=REQUESTS['A'][4])
        self.wait(.8)
        cache = VGroup(*[entry(x, REQUESTS['A'][:i+1], COMPOSED) for i, x in enumerate(xs)])
        routes = [Arrow([x,.8,0], [x,-.28,0], color=COMPOSED, buff=.12, stroke_width=2, tip_length=.12) for x in xs]
        # A visual sweep denotes computation, not serial scheduling inside prefill.
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(routes[i]), FadeIn(cache[i], shift=DOWN*.12)) for i in range(5)], lag_ratio=.20), run_time=1.65)
        self.play(*[FadeOut(a) for a in routes], *[m.animate.set_color(INTERMEDIATE) for m in cache], FadeIn(kv_label), run_time=.4)
        self.mark('A_cache', computed=list(range(5)), reused=[])
        doc_points = [m.get_all_points().copy() for m in input_cells]
        cache_points = [m.get_all_points().copy() for m in cache[:4]]
        self.wait(1.0)

        # B is a new request suffix. The document and its shared states do not change.
        b_badge = VGroup(label('B',32,BASE),cn('要求修改',25,BASE)).arrange(RIGHT,buff=.22).move_to(badge, aligned_edge=RIGHT)
        same = cn('文档未变',23,INTERMEDIATE).move_to(doc_name)
        b_phase = VGroup(label('Prefix caching',29,INTERMEDIATE),cn('复用前缀',25,INTERMEDIATE)).arrange(RIGHT,buff=.3).move_to(phase)
        self.play(Transform(badge,b_badge),Transform(doc_name,same),
            Transform(instruction,cell('预算减半',xs[4],width=2.7)),FadeOut(cache[4]),Transform(phase,b_phase),run_time=.65)
        self.mark('B_instruction_only', document=REQUESTS['B'][:4], reused=list(range(4)), changed=[4])
        self.play(Indicate(input_cells[1],color=INTERMEDIATE,scale_factor=1.05),run_time=.55)
        reuse_line = Line([-5.67,-2.15,0],[1.67,-2.15,0],stroke_width=3,color=INTERMEDIATE)
        self.play(Create(reuse_line),run_time=.45)
        self.wait(.65)
        fresh_b = entry(xs[4],REQUESTS['B'],COMPOSED)
        work = label('Prefill',23,COMPOSED).move_to([4.4,.33,0])
        access = VMobject(stroke_color=INTERMEDIATE,stroke_width=2.5).set_points_as_corners(
            [[-2,-1.46,0],[-2,-2.18,0],[4.4,-2.18,0],[4.4,-1.47,0]])
        self.play(FadeIn(work),GrowArrow(routes[4]),FadeIn(fresh_b,shift=DOWN*.12),run_time=.85)
        self.play(ShowPassingFlash(access,time_width=.7),run_time=.65)
        self.play(FadeOut(routes[4]),run_time=.2)
        for m, points in zip(input_cells,doc_points):np.testing.assert_allclose(m.get_all_points(),points,atol=1e-10)
        for m, points in zip(cache[:4],cache_points):np.testing.assert_allclose(m.get_all_points(),points,atol=1e-10)
        self.mark('B_new_instruction_computed', reused=list(range(4)), computed=[4], document_geometry_unchanged=True, cache_geometry_unchanged=True)
        self.wait(1.05)

        # C supplies the edited input. An unchanged textual tail can have new states.
        c_badge = VGroup(label('C',32,ACCENT),cn('读修改后的方案',25,ACCENT)).arrange(RIGHT,buff=.22).move_to(badge,aligned_edge=RIGHT)
        c_phase = VGroup(cn('保留前缀',25,INTERMEDIATE),label('·',25,MUTED),cn('重算后缀',25,COMPOSED)).arrange(RIGHT,buff=.3).move_to(phase)
        self.play(Transform(badge,c_badge),Transform(doc_name,cn('方案已修改',24,ACCENT).move_to(doc_name)),
            Transform(instruction,cell('预算够吗',xs[4],width=2.7)),
            FadeOut(fresh_b),FadeOut(work),FadeOut(reuse_line),Transform(phase,c_phase),run_time=.65)
        self.play(Transform(input_cells[1],cell('50 万',xs[1],color=ACCENT)),run_time=.55)
        self.mark('C_document_edited', shared_prefix=shared_prefix(REQUESTS['A'],REQUESTS['C']), changed_position=1)
        self.wait(.55)
        crosses = VGroup(*[Line(m.get_corner(DL),m.get_corner(UR),color=ACCENT,stroke_width=3) for m in cache[1:4]])
        keep = Line([-5.67,-2.15,0],[-4.03,-2.15,0],stroke_width=3,color=INTERMEDIATE)
        self.play(*[m.animate.set_opacity(.2) for m in cache[1:4]],FadeIn(crosses),Create(keep),run_time=.45)
        self.mark('C_suffix_invalidated', reused=[0], invalidated=[1,2,3])
        self.wait(.65)
        fresh_c = VGroup(*[entry(xs[i],REQUESTS['C'][:i+1],COMPOSED) for i in range(1,5)])
        self.play(FadeOut(crosses),*[FadeOut(m) for m in cache[1:4]],run_time=.2)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(routes[i]),FadeIn(fresh_c[i-1],shift=DOWN*.12)) for i in range(1,5)],lag_ratio=.20),run_time=1.5)
        self.play(*[FadeOut(routes[i]) for i in range(1,5)],run_time=.25)
        np.testing.assert_allclose(cache[0].get_all_points(),cache_points[0],atol=1e-10)
        for i in [0,2,3]:np.testing.assert_allclose(input_cells[i].get_all_points(),doc_points[i],atol=1e-10)
        self.mark('C_suffix_recomputed', reused=[0], computed=[1,2,3,4], unchanged_text_recomputed=[2,3])
        self.wait(1.2)

        # A compact closing key uses the same three terms, without production labels.
        objects = [m for m in self.mobjects if m is not title and m is not caveat]
        self.play(*[FadeOut(m) for m in objects],run_time=.5)
        summary = VGroup()
        for x,en,zh,col in [(-4.5,'Prefill','计算输入',COMPOSED),(0,'KV cache','保存状态',INTERMEDIATE),(4.5,'Prefix caching','复用前缀',BASE)]:
            summary.add(VGroup(label(en,32,col),cn(zh,28)).arrange(DOWN,buff=.4).move_to([x,.2,0]))
        self.play(FadeIn(summary,shift=UP*.12),run_time=.5)
        self.mark('summary', terms=['Prefill','KV cache','Prefix caching'])
        self.wait(1.65)
        self.play(*[FadeOut(m) for m in list(self.mobjects)],run_time=.45)
        self.wait(.2)
        (P/'timeline.json').write_text(json.dumps(dict(events=self.events,duration_seconds=float(self.time),
            checks=dict(inside_canvas=True,B_document_unchanged=True,B_cache_unchanged=True,C_earlier_cache_unchanged=True,
                        edit_precedes_invalidation=True,new_instruction_computed_in_all_requests=True)),indent=2)+'\n')
