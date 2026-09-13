"""A single-layer cache schematic with causally correct prediction timing."""
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE, INTERMEDIATE, COMPOSED

label = partial(label, layout_scale=16)


def box(text, point, width=1.22, color=BASE, size=32, height=.65):
    shape = RoundedRectangle(width=width, height=height, corner_radius=.12,
                             stroke_color=color, stroke_width=2,
                             fill_color=color, fill_opacity=.055)
    return VGroup(shape, label(text, size, color).move_to(shape)).move_to(point)


def vector(point, color):
    shape = RoundedRectangle(width=1.22, height=.62, corner_radius=.10,
                             stroke_color=color, stroke_width=2,
                             fill_color=color, fill_opacity=.065)
    # Equal bars are abstract vector glyphs, not measured activation magnitudes.
    bars = VGroup(*[RoundedRectangle(width=.16, height=.28, corner_radius=.025,
                                    stroke_width=0, fill_color=color, fill_opacity=.72)
                   for _ in range(3)]).arrange(RIGHT, buff=.13).move_to(shape)
    return VGroup(shape, bars).move_to(point)


class PrefillDecode(Scene):
    def mark(self, name, count, **extra):
        for m in self.mobjects:
            assert m.get_left()[0] > -7.1 and m.get_right()[0] < 7.1, name
            assert m.get_bottom()[1] > -3.95 and m.get_top()[1] < 3.95, name
        self.events.append(dict(name=name, seconds=round(float(self.time), 3),
                                cached_positions=count, **extra))

    def read_cache(self, q, keys, values, attention):
        # Q -> K matching, then V -> attention mixture. No numeric weights implied.
        paths = VGroup()
        for key in keys:
            x = key.get_center()[0]
            paths.add(VMobject(stroke_color=INTERMEDIATE, stroke_width=2).set_points_as_corners(
                [q.get_top(), [-5.2, 1.13, 0], [x, 1.13, 0], key.get_top()]))
        self.play(*[ShowPassingFlash(p, time_width=.65) for p in paths], run_time=.55)
        self.play(*[Indicate(k, color=INTERMEDIATE, scale_factor=1.06) for k in keys], run_time=.3)
        paths = VGroup()
        for value in values:
            x = value.get_center()[0]
            paths.add(VMobject(stroke_color=COMPOSED, stroke_width=2).set_points_as_corners(
                [value.get_bottom(), [x, -1.6, 0], [4.7, -1.6, 0], attention.get_bottom()]))
        self.play(*[ShowPassingFlash(p, time_width=.65) for p in paths],
                  Indicate(attention, color=COMPOSED, scale_factor=1.04), run_time=.6)

    def predict(self, attention, tail, word):
        p1 = Arrow(attention.get_top(), tail.get_bottom(), buff=.07, color=MUTED,
                   stroke_width=2, tip_length=.10)
        p2 = Arrow(tail.get_top(), word.get_bottom(), buff=.07, color=MUTED,
                   stroke_width=2, tip_length=.10)
        self.play(ShowPassingFlash(p1, time_width=.8), run_time=.2)
        self.play(Indicate(tail, color=BASE, scale_factor=1.03), run_time=.2)
        self.play(ShowPassingFlash(p2, time_width=.8), FadeIn(word), run_time=.3)

    def construct(self):
        self.camera.background_color = BACKGROUND
        self.events = []
        title = label('Prefill → decode', 34).to_edge(UP, buff=.38).to_edge(LEFT, buff=.55)
        scope = label('One layer · illustrative tokens', 19, MUTED).to_edge(DOWN, buff=.33)
        phase = label('Prefill', 27, BASE).move_to([5.4, 3.3, 0])
        labels = VGroup(label('Input', 25, MUTED).move_to([-5.2, 2.05, 0]),
                        label('K', 31, INTERMEDIATE).move_to([-3.87, .35, 0]),
                        label('V', 31, COMPOSED).move_to([-3.87, -.75, 0]),
                        label('Next token', 23, MUTED).move_to([4.7, 2.65, 0]),
                        label('KV cache', 23, MUTED).move_to([-.55, -2.2, 0]))
        attention = box('Attention', [4.7, -.75, 0], 2.3, COMPOSED, 26, .75)
        tail = box('Rest of model', [4.7, .65, 0], 2.3, MUTED, 22, .7)
        connections = VGroup(
            Arrow(attention.get_top(), tail.get_bottom(), buff=.06, color=GRID, stroke_width=2, tip_length=.10),
            Arrow(tail.get_top(), [4.7, 1.70, 0], buff=.06, color=GRID, stroke_width=2, tip_length=.10))
        scaffold = VGroup(title, scope, phase, labels, attention, tail, connections)
        self.add(scaffold)
        xs = [-2.8, -1.3, .2, 1.7]
        inputs = VGroup(*[box(w, [xs[i], 2.05, 0]) for i, w in enumerate(['The', 'sky', 'is'])])
        keys = VGroup(*[vector([x, .35, 0], INTERMEDIATE) for x in xs])
        values = VGroup(*[vector([x, -.75, 0], COMPOSED) for x in xs])
        q = VGroup(Circle(radius=.42, color=BASE, stroke_width=2, fill_opacity=.06, fill_color=BASE),
                   label('Q', 34, BASE)).move_to([-5.2, .35, 0])
        q_template = q.copy()
        blue = box('blue', [4.7, 2.05, 0], color=COMPOSED)
        period = box('.', [4.7, 2.05, 0], color=COMPOSED)
        self.wait(.2)
        self.play(FadeIn(inputs), run_time=.4)
        self.mark('prompt_input', 0)
        self.play(*[TransformFromCopy(inputs[i], VGroup(keys[i], values[i])) for i in range(3)],
                  TransformFromCopy(inputs[2], q), run_time=.65)
        self.mark('prompt_kv_ready', 3, query_position=3)
        snapshots = [m.get_all_points().copy() for m in [*keys[:3], *values[:3]]]
        self.read_cache(q, keys[:3], values[:3], attention)
        self.mark('prefill_read', 3, query_position=3, visible_positions=[1, 2, 3])
        self.predict(attention, tail, blue)
        self.mark('blue_selected', 3, selected_token='blue', selected_token_cached=False)
        self.wait(.55)
        self.play(FadeOut(q), run_time=.2)
        decode = label('Decode', 27, BASE).move_to(phase)
        self.play(Transform(phase, decode), blue.animate.move_to([xs[3], 2.05, 0]),
                  inputs.animate.set_color(MUTED), run_time=.55)
        self.mark('blue_fed_back', 3)
        # A selected token gets its K/V only after being processed as new input.
        q2 = q_template.copy()
        self.play(TransformFromCopy(blue, VGroup(keys[3], values[3])),
                  TransformFromCopy(blue, q2), run_time=.6)
        self.mark('new_kv_appended', 4, query_position=4)
        self.read_cache(q2, keys, values, attention)
        self.mark('decode_read', 4, query_position=4, visible_positions=[1, 2, 3, 4])
        for before, m in zip(snapshots, [*keys[:3], *values[:3]]):
            np.testing.assert_allclose(before, m.get_all_points(), atol=1e-10)
        self.predict(attention, tail, period)
        self.mark('period_selected', 4, selected_token='.', selected_token_cached=False)
        self.wait(.9)
        # Fade is an editorial loop reset, never an animation of reversing inference.
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.4)
        self.wait(.15)
        timeline = dict(events=self.events, duration_seconds=round(float(self.time), 3),
                        unchanged_past_geometry=True, all_objects_inside_canvas=True,
                        scope='One-layer schematic; no numeric attention weights or model predictions measured')
        (P/'timeline.json').write_text(json.dumps(timeline, indent=2)+'\n')
