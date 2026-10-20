"""ANIM-POST-001: five beginner loops and one static comparison for X-POST-001.

Every probability and stream volume here is an authored illustration, never a
measurement. Timelines record explanatory order, not timing.
"""
from functools import partial
import json
import os
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import (label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE,
                         INTERMEDIATE, COMPOSED, ACCENT)

label = partial(label, layout_scale=16)
WRONG = ACCENT


# ---------------------------------------------------------------- primitives
def card(text, color=FOREGROUND, size=26, w=None, h=.66, fill=.05, text_color=None):
    t = label(text, size, text_color or color)
    width = w or t.width + .6
    assert t.width < width - .2, (text, t.width, width)
    box = RoundedRectangle(width=width, height=h, corner_radius=.14, stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def tile(text, color, size=26, h=.7):
    t = label(text, size, color)
    box = RoundedRectangle(width=max(.9, t.width + .42), height=h, corner_radius=.12,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=.05)
    return VGroup(box, t.move_to(box))


def model_block(name, color, w=2.9, h=1.25, size=26):
    box = RoundedRectangle(width=w, height=h, corner_radius=.22, stroke_color=color,
                           stroke_width=2.6, fill_color=color, fill_opacity=.06)
    dots = VGroup(*[Dot(radius=.045, color=color, fill_opacity=.55)
                    for _ in range(14)]).arrange_in_grid(2, 7, buff=.2)
    t = label(name, size, color)
    inner = VGroup(t, dots).arrange(DOWN, buff=.2).move_to(box)
    return VGroup(box, inner)


def check_mark(color=COMPOSED, s=1.0):
    m = VMobject(stroke_color=color, stroke_width=7 * s)
    m.set_points_as_corners([np.array(p) * s for p in [(-.2, 0, 0), (-.06, -.15, 0), (.22, .2, 0)]])
    return m


def cross_mark(color=WRONG, s=1.0):
    a = Line([-.16 * s, -.16 * s, 0], [.16 * s, .16 * s, 0], stroke_width=7 * s, color=color)
    b = Line([-.16 * s, .16 * s, 0], [.16 * s, -.16 * s, 0], stroke_width=7 * s, color=color)
    return VGroup(a, b)


def hbar(length, color, h=.34):
    return Rectangle(width=max(length, .001), height=h, stroke_width=0,
                     fill_color=color, fill_opacity=.9)


class Loop(Scene):
    def begin(self, title, badge=None):
        self.camera.background_color = BACKGROUND
        self.events = []
        self.title = label(title, 40).to_edge(UP, buff=.42).to_edge(LEFT, buff=.55)
        self.add(self.title)
        if badge:
            b = label(badge, 23, ACCENT)
            box = RoundedRectangle(width=b.width + .46, height=.56, corner_radius=.22, stroke_color=ACCENT,
                                   stroke_width=1.6, fill_color=ACCENT, fill_opacity=.07)
            self.badge = VGroup(box, b.move_to(box)).to_edge(UP, buff=.44).to_edge(RIGHT, buff=.5)
            self.add(self.badge)

    def mark(self, name, wait=.35, **state):
        for m in self.mobjects:
            if m.get_width() == 0:
                continue
            assert m.get_left()[0] > -7.12 and m.get_right()[0] < 7.12, (name, 'x', m)
            assert m.get_bottom()[1] > -4.0 and m.get_top()[1] < 4.0, (name, 'y', m)
        self.events.append(dict(name=name, seconds=round(float(self.time), 3), **state))
        self.wait(wait)

    def finish(self, stem, hold=2.2, **state):
        self.mark('complete', wait=hold, **state)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.6)
        self.wait(.25)
        (P / (stem + '-timeline.json')).write_text(json.dumps(dict(
            events=self.events, checks={'inside_canvas': True},
            duration_seconds=round(float(self.time), 3)), indent=2) + '\n')


# ---------------------------------------------------------------- A1
BASE_BARS = [('Explain', .41, False), ('Write', .2, False), ('Describe', .13, False), ('People', .06, True)]
POST_BARS = [('People', .43, True), ('In', .24, True), ('The', .15, True), ('Explain', .04, False)]


class SamePrompt(Loop):
    def construct(self):
        self.begin('Same request, two models', 'illustrative probabilities')
        req = card('Explain the moon landing to a 6 year old.', FOREGROUND, 32, h=.8).move_to([0, 2.3, 0])
        self.play(FadeIn(req, shift=DOWN * .2), run_time=.7)
        self.mark('request')
        cols = []
        for x, name, color, bars, reply, reply_color in [
            (-3.6, 'base model', MUTED, BASE_BARS, 'Explain gravity to a 6 year old.', MUTED),
            (3.6, 'post-trained', COMPOSED, POST_BARS, 'People flew to the moon and took pictures.', COMPOSED)]:
            block = model_block(name, color, w=3.3, h=1.35, size=30).move_to([x, .85, 0])
            cols.append((x, block, bars, reply, reply_color))
        self.play(*[FadeIn(c[1]) for c in cols], run_time=.6)
        paths = [Line(req.get_bottom() + [0, -.05, 0], c[1].get_top(), color=FOREGROUND, stroke_width=2.4)
                 for c in cols]
        self.play(*[ShowPassingFlash(pth.copy(), time_width=.5) for pth in paths],
                  *[Indicate(c[1][0], color=c[1][0].get_stroke_color(), scale_factor=1.04) for c in cols],
                  run_time=.9)
        self.mark('both_models_read_request')
        head = label('likely next word', 24, MUTED).move_to([0, -.05, 0])
        self.play(FadeIn(head), run_time=.3)
        groups = []
        for x, block, bars, reply, rc in cols:
            rows = VGroup()
            for i, (word, p, answer_like) in enumerate(bars):
                y = -.65 - i * .6
                w = label(word, 30, FOREGROUND)
                w.move_to([x - 1.5, y, 0], aligned_edge=RIGHT)
                bar = hbar(p * 7.2, COMPOSED if answer_like else MUTED, .42)
                bar.move_to([x - 1.2, y, 0], aligned_edge=LEFT)
                rows.add(VGroup(w, bar))
            groups.append(rows)
        self.play(*[FadeIn(r[0]) for g in groups for r in g], run_time=.4)
        self.play(*[GrowFromEdge(r[1], LEFT) for g in groups for r in g], run_time=1.2)
        self.mark('distributions', base_top='Explain', post_top='People',
                  base_question_mass=round(sum(p for _, p, a in BASE_BARS if not a), 2),
                  post_answer_mass=round(sum(p for _, p, a in POST_BARS if a), 2))
        boxes = [SurroundingRectangle(g[0], color=ACCENT, buff=.15, corner_radius=.08, stroke_width=3) for g in groups]
        self.play(*[Create(b) for b in boxes], run_time=.5)
        replies = []
        for (x, block, bars, reply, rc) in cols:
            r = label(reply, 30, rc)
            r.scale_to_fit_width(min(r.width, 6.6)).move_to([x, -3.15, 0])
            replies.append(r)
        self.play(*[AddTextLetterByLetter(r, time_per_char=.035) for r in replies], run_time=1.4)
        tags = [label('another request', 24, MUTED).next_to(replies[0], DOWN, buff=.14),
                label('an answer', 24, COMPOSED).next_to(replies[1], DOWN, buff=.14)]
        self.play(*[FadeIn(t) for t in tags], run_time=.4)
        self.finish('01-same-prompt-two-models', hold=2.8)


# ---------------------------------------------------------------- A2
class TwoStages(Loop):
    def construct(self):
        self.begin('One model, two stages', 'volumes not to scale')
        my = -.6
        model = model_block('model', FOREGROUND, w=3.0, h=2.0, size=32).move_to([1.0, my, 0])
        box, (name, dots) = model[0], model[1]
        dots.set_opacity(.12)
        out_arrow = Arrow(model.get_right(), [3.65, my, 0], buff=.12, color=MUTED, stroke_width=3.5)
        out = card('continues text', MUTED, 26, h=.74).next_to(out_arrow, RIGHT, buff=.1)
        stage = label('pretraining', 36, FOREGROUND).move_to([-3.6, 2.25, 0])
        sub = label('predict the next word', 26, MUTED).next_to(stage, DOWN, buff=.16)
        self.add(model)
        self.play(FadeIn(stage), FadeIn(sub), run_time=.5)

        rng = np.random.default_rng(7)
        n = 110
        left, right = -6.75, model.get_left()[0] - .1
        x0 = rng.uniform(left, right, n); y0 = rng.uniform(-2.55, 1.2, n)
        widths = rng.uniform(.18, .7, n)
        lines = VGroup(*[Rectangle(width=w, height=.08, stroke_width=0, fill_color=MUTED,
                                   fill_opacity=.75) for w in widths])
        t = ValueTracker(0)

        def place(group):
            for i, m in enumerate(group):
                x = left + .35 + (x0[i] - left + 2.4 * t.get_value()) % (right - left - .7)
                prog = (x - left) / (right - left)
                y = my + (y0[i] - my) * (1 - .78 * prog ** 1.6)
                m.move_to([x, y, 0])
                m.set_fill(opacity=.8 * min(1, (right - .35 - x) / .8) * min(1, (x - left - .35) / .6))
        lines.add_updater(place); place(lines)
        self.add(lines)
        vol = label('trillions of words of raw text', 26, MUTED).move_to([-3.4, -3.2, 0])
        self.play(FadeIn(vol), run_time=.4)
        self.play(t.animate.set_value(3.6), dots.animate.set_opacity(.9), run_time=3.6, rate_func=linear)
        self.play(GrowArrow(out_arrow), FadeIn(out), t.animate.set_value(4.1), run_time=.5, rate_func=linear)
        self.mark('pretrained', output='continues text')
        lines.clear_updaters()
        self.play(FadeOut(lines), FadeOut(vol), run_time=.5)

        stage2 = label('post-training', 36, FOREGROUND).move_to(stage)
        sub2 = label('small, curated data', 26, MUTED).next_to(stage2, DOWN, buff=.16)
        self.play(FadeTransform(stage, stage2), FadeTransform(sub, sub2), run_time=.5)
        kinds = [('examples', BASE), ('comparisons', INTERMEDIATE), ('checked answers', COMPOSED)]
        target = model.get_left() + RIGHT * .05
        for k, (name_k, color) in enumerate(kinds):
            y = my + 1.05 - k * 1.05
            tag = label(name_k, 27, color).move_to([-6.7, y, 0], aligned_edge=LEFT)
            items = VGroup()
            for j in range(3):
                if color == BASE:
                    icon = VGroup(Rectangle(width=.5, height=.08, stroke_width=0, fill_color=color, fill_opacity=.9),
                                  Rectangle(width=.34, height=.08, stroke_width=0, fill_color=color, fill_opacity=.9)
                                  ).arrange(DOWN, buff=.08, aligned_edge=LEFT)
                elif color == INTERMEDIATE:
                    icon = VGroup(label('A', 20, color), label('>', 20, color), label('B', 20, color)).arrange(RIGHT, buff=.05)
                else:
                    icon = check_mark(color, .75)
                c = RoundedRectangle(width=.8, height=.54, corner_radius=.08, stroke_color=color,
                                     stroke_width=2.2, fill_color=color, fill_opacity=.08)
                items.add(VGroup(c, icon.move_to(c)).move_to([-3.35 + j * .95, y, 0]))
            path = Line(tag.get_right() + RIGHT * .15, target + UP * (y - my) * .45, color=color,
                        stroke_width=3, stroke_opacity=.35)
            self.play(FadeIn(tag), Create(path), FadeIn(items, shift=RIGHT * .2), run_time=.5)
            self.play(*[it.animate.move_to(path.get_end()).scale(.35).set_opacity(0) for it in items],
                      lag_ratio=.18, run_time=1.0)
            self.remove(items)
            ring = box.copy().set_stroke(color, 7).set_fill(opacity=0)
            self.play(ShowPassingFlash(ring, time_width=.8), run_time=.6)
            self.mark('post_' + name_k.split()[0], kind=name_k)
        new_out = card('answers helpfully', COMPOSED, 26, h=.74).move_to(out, aligned_edge=LEFT)
        self.play(FadeTransform(out, new_out), out_arrow.animate.set_color(COMPOSED), run_time=.6)
        self.finish('02-two-stages', hold=2.6, same_model=True, stages=['pretraining', 'post-training'])


# ---------------------------------------------------------------- comparison (static)
COMPARE = {
    'en': dict(title='Pretraining vs. post-training', cols=('Pretraining', 'Post-training'), cjk=False, rows=[
        ('Goal', 'Predict the next word', 'Behave the way people want'),
        ('Data', 'Trillions of words of raw text', 'Small, curated: at most millions'),
        ('Signal', 'Every next word in the text', 'Examples · comparisons · a checker'),
        ('Cost', 'Most of the compute', 'A small fraction'),
        ('Change', 'Builds knowledge and language', 'Nudges a model that already works')],
        foot='Pretraining gives the model what it knows; post-training shapes how it uses it.'),
    'zh': dict(title='预训练 vs 后训练', cols=('预训练', '后训练'), cjk=True, rows=[
        ('目标', '猜准下一个词', '按人们希望的方式行事'),
        ('数据', '万亿级的原始文本', '小而精，最多几百万条'),
        ('信号', '文本里的每一个下一个词', '示范 · 比较 · 检查器'),
        ('成本', '绝大部分算力', '其中一小部分'),
        ('改动', '从零建立知识和语言', '轻轻调整已经能干的模型')],
        foot='预训练决定模型知道什么，后训练决定它怎样运用这些知识。'),
}


class Comparison(Scene):
    def construct(self):
        lang = os.environ.get('COMPARE_LANG', 'en')
        d = COMPARE[lang]; cjk = d['cjk']
        L = partial(label, cjk=cjk)
        self.camera.background_color = BACKGROUND
        self.add(L(d['title'], 36).to_edge(UP, buff=.45).to_edge(LEFT, buff=.6))
        xs = [-5.2, -2.2, 3.75]
        for x, text, color in [(xs[1], d['cols'][0], MUTED), (xs[2], d['cols'][1], COMPOSED)]:
            t = L(text, 30, color)
            box = RoundedRectangle(width=5.4, height=.66, corner_radius=.16, stroke_color=color,
                                   stroke_width=2, fill_color=color, fill_opacity=.08)
            self.add(VGroup(box, t.move_to(box)).move_to([x, 2.35, 0]))
        for i, (k, a, b) in enumerate(d['rows']):
            y = 1.35 - i * .95
            self.add(Line([-6.6, y + .47, 0], [6.6, y + .47, 0], color=GRID, stroke_width=1.5))
            self.add(L(k, 26, FOREGROUND).move_to([-6.45, y, 0], aligned_edge=LEFT))
            for x, text, color in [(xs[1], a, FOREGROUND), (xs[2], b, COMPOSED)]:
                t = L(text, 24, color)
                assert t.width < 5.6, (text, t.width)
                self.add(t.move_to([x, y, 0]))
        self.add(Line([-6.6, 1.35 - 4 * .95 - .47, 0], [6.6, 1.35 - 4 * .95 - .47, 0], color=GRID, stroke_width=1.5))
        foot = L(d['foot'], 24, ACCENT).move_to([0, -3.45, 0])
        assert foot.width < 13
        self.add(foot)
        for m in self.mobjects:
            assert m.get_left()[0] > -7.12 and m.get_right()[0] < 7.12, m
        self.wait(.1)


# ---------------------------------------------------------------- A3
INSTRUCTION = ['Translate', 'to', 'English:', 'No', 'tenía', 'lavandería.']
RESPONSE = ['I', "didn't", 'have', 'a', 'laundry.']


def dim(t):
    return AnimationGroup(t[0].animate.set_stroke(MUTED).set_fill(MUTED, .03), t[1].animate.set_color(MUTED))


class InstructionTuning(Loop):
    def construct(self):
        self.begin('Instruction tuning: grade only the response')
        inst = VGroup(*[tile(w, BASE, 32, .86) for w in INSTRUCTION]).arrange(RIGHT, buff=.2).move_to([.95, 1.55, 0])
        resp = VGroup(*[tile(w, COMPOSED, 32, .86) for w in RESPONSE]).arrange(RIGHT, buff=.2)
        resp.move_to([0, -.45, 0]).align_to(inst, LEFT)
        masks = VGroup()
        for r in resp:
            q = label('?', 32, COMPOSED)
            b = r[0].copy()
            masks.add(VGroup(b, q.move_to(b)))
        tag_i = label('instruction', 27, BASE).next_to(inst, LEFT, buff=.35)
        tag_r = label('response', 27, COMPOSED).next_to(resp, LEFT, buff=.35).align_to(tag_i, RIGHT)
        self.play(FadeIn(inst, lag_ratio=.1), FadeIn(tag_i), run_time=.8)
        self.play(FadeIn(masks, lag_ratio=.1), FadeIn(tag_r), run_time=.6)
        self.mark('pair_shown', instruction_tokens=len(INSTRUCTION), response_tokens=len(RESPONSE))
        ctx = label('context only · not graded', 26, MUTED).next_to(inst, DOWN, buff=.28)
        self.play(*[dim(t) for t in inst], tag_i.animate.set_color(MUTED), FadeIn(ctx), run_time=.7)
        self.mark('instruction_is_context', graded=[])
        graded = []
        for i, (m, r) in enumerate(zip(masks, resp)):
            self.play(m[0].animate.set_stroke(ACCENT, 4), run_time=.3)
            tick = check_mark(COMPOSED, .9).next_to(r, DOWN, buff=.24)
            self.play(FadeOut(m, scale=.9), FadeIn(r, scale=1.08), FadeIn(tick, scale=1.4), run_time=.42)
            graded.append(RESPONSE[i])
            self.mark(f'graded_{i + 1}', wait=.1, graded=list(graded))
        claim = VGroup(label('Same next-word rule.', 32, FOREGROUND),
                       label('Only the response words count.', 32, COMPOSED)).arrange(DOWN, buff=.2)
        claim.move_to([0, -2.85, 0])
        self.play(FadeIn(claim, shift=UP * .15), run_time=.6)
        self.finish('03-instruction-tuning', hold=2.6, graded=list(graded), instruction_graded=[])


# ---------------------------------------------------------------- A4
class PreferenceAlignment(Loop):
    def construct(self):
        self.begin('Preference alignment: learn from comparisons')
        q = card('Does garlic cure a cold?', FOREGROUND, 30, h=.78).move_to([0, 2.3, 0])
        xs = [-3.6, 3.6]
        a = card("It may help a bit. It won't cure it.", FOREGROUND, 27, w=6.0, h=.8).move_to([xs[0], 1.05, 0])
        b = card("Yes! It's the best antibiotic.", FOREGROUND, 27, w=6.0, h=.8).move_to([xs[1], 1.05, 0])
        self.play(FadeIn(q, shift=DOWN * .2), run_time=.5)
        self.play(FadeIn(a, shift=UP * .15), FadeIn(b, shift=UP * .15), run_time=.6)
        self.mark('pair_shown')
        stamp = VGroup(check_mark(INTERMEDIATE, .9), label('preferred', 26, INTERMEDIATE)).arrange(RIGHT, buff=.14)
        stamp.next_to(a, DOWN, buff=.18)
        self.play(a[0].animate.set_stroke(INTERMEDIATE, 4), FadeIn(stamp, scale=1.3),
                  b[0].animate.set_stroke(GRID).set_fill(MUTED, .02), b[1].animate.set_color(MUTED).set_opacity(.6),
                  run_time=.6)
        self.mark('preference_marked', chosen='A', rejected='B')

        base_y, h0, bw = -3.25, 1.3, 1.6
        ghosts = VGroup(*[DashedVMobject(Rectangle(width=bw, height=h0, stroke_color=ACCENT, stroke_width=2.6),
                                         num_dashes=32).move_to([x, base_y + h0 / 2, 0]) for x in xs])
        bars = VGroup(*[Rectangle(width=bw, height=h0, stroke_width=0, fill_color=c, fill_opacity=.85)
                        .move_to([x, base_y + h0 / 2, 0]) for x, c in zip(xs, [INTERMEDIATE, MUTED])])
        axis = Line([-6.6, base_y, 0], [6.6, base_y, 0], color=GRID, stroke_width=2.2)
        cap = label("model's chance of writing each answer", 24, MUTED).move_to([0, base_y - .38, 0])
        self.play(Create(axis), FadeIn(bars), FadeIn(cap), run_time=.6)
        self.add(ghosts)
        orig = VGroup(DashedVMobject(Rectangle(width=.5, height=.32, stroke_color=ACCENT, stroke_width=2.4), num_dashes=10),
                      label('original model', 24, ACCENT)).arrange(RIGHT, buff=.15).move_to([0, base_y + .5, 0])
        self.play(FadeIn(orig), run_time=.3)
        self.mark('start_equals_original', heights=[h0, h0])

        hA, hB = ValueTracker(h0), ValueTracker(h0)
        bars[0].add_updater(lambda m: m.stretch_to_fit_height(hA.get_value()).move_to([xs[0], base_y + hA.get_value() / 2, 0]))
        bars[1].add_updater(lambda m: m.stretch_to_fit_height(hB.get_value()).move_to([xs[1], base_y + hB.get_value() / 2, 0]))

        def leash(i, tracker):
            lx = xs[i] + (1 if i else -1) * (bw / 2 + .22)
            return always_redraw(lambda: Line([lx, base_y + h0, 0],
                                              [lx, base_y + tracker.get_value(), 0],
                                              color=ACCENT, stroke_width=3 + 3 * abs(tracker.get_value() - h0)))
        leashes = VGroup(leash(0, hA), leash(1, hB))
        self.add(leashes)
        # Pulled toward the preferred answer, held back by the leash: a long ease-out.
        self.play(hA.animate.set_value(2.35), hB.animate.set_value(.55), run_time=2.4,
                  rate_func=lambda s: 1 - (1 - s) ** 3)
        leash_tag = label('leash', 26, ACCENT).next_to(leashes[0], LEFT, buff=.15)
        self.play(FadeIn(leash_tag), run_time=.3)
        self.mark('nudged', heights=[2.35, .55], within_leash=True)
        for m in bars: m.clear_updaters()
        routes = VGroup(card('with a judge: RLHF, e.g. PPO', INTERMEDIATE, 24, h=.62),
                        card('straight from pairs: DPO', INTERMEDIATE, 24, h=.62)).arrange(DOWN, buff=.22)
        routes.move_to([0, -.8, 0])
        assert routes.width < 2 * (xs[1] - bw / 2 - .1)
        self.play(FadeIn(routes, shift=UP * .15), run_time=.5)
        self.finish('04-preference-alignment', hold=2.8)


# ---------------------------------------------------------------- A5
ROUNDS = [[51, 41, 51, 57], [51, 51, 41, 51]]
CHANCES = [[('51', .45), ('41', .3), ('57', .25)],
           [('51', .7), ('41', .17), ('57', .13)],
           [('51', .85), ('41', .09), ('57', .06)]]


class RLVR(Loop):
    def construct(self):
        assert 17 * 3 == 51
        self.begin('RLVR: learn from a checker', 'checks the final answer only')
        prob = card('17 × 3 = ?', FOREGROUND, 34, h=.82).move_to([-4.7, 2.15, 0])
        model = model_block('model', BASE, w=2.9, h=1.35, size=30).move_to([-4.7, .55, 0])
        checker = model_block('checker', COMPOSED, w=2.6, h=1.35, size=30).move_to([5.1, .55, 0])
        self.play(FadeIn(prob), FadeIn(model), FadeIn(checker), run_time=.6)
        cap = label("model's chances", 24, MUTED).move_to([-4.7, -.55, 0])
        rows = []
        for i, (ans, p) in enumerate(CHANCES[0]):
            y = -1.2 - i * .72
            t = label(ans, 30, FOREGROUND).move_to([-6.25, y, 0])
            bar = hbar(p * 4.2, BASE, .42).move_to([-5.75, y, 0], aligned_edge=LEFT)
            rows.append(VGroup(t, bar))
        self.play(FadeIn(cap), *[FadeIn(r) for r in rows], run_time=.5)
        self.mark('setup', chances=dict(CHANCES[0]))

        for rnd, answers in enumerate(ROUNDS):
            cards = VGroup(*[card(f'17 × 3 = {v}', FOREGROUND, 30, w=3.5, h=.78).move_to([.85, 2.0 - k * 1.12, 0])
                             for k, v in enumerate(answers)])
            arrows = VGroup(*[Arrow(model.get_right(), c.get_left(), buff=.12, color=BASE, stroke_width=2.6,
                                    tip_length=.16) for c in cards])
            self.play(LaggedStart(*[AnimationGroup(GrowArrow(ar), FadeIn(c, shift=RIGHT * .2))
                                    for ar, c in zip(arrows, cards)], lag_ratio=.2), run_time=1.0)
            self.mark(f'round_{rnd + 1}_sampled', answers=answers)
            marks = VGroup()
            for c, v in zip(cards, answers):
                ok = v == 17 * 3
                scan = Line(checker.get_left(), c.get_right(), color=COMPOSED, stroke_width=3)
                m = (check_mark(COMPOSED, 1.15) if ok else cross_mark(WRONG, 1.15)).next_to(c, RIGHT, buff=.32)
                self.play(ShowPassingFlash(scan.copy().reverse_direction(), time_width=.7),
                          FadeIn(m, scale=1.5), c[0].animate.set_stroke(COMPOSED if ok else WRONG, 3.4),
                          run_time=.38)
                marks.add(m)
            self.mark(f'round_{rnd + 1}_checked', correct=[v == 51 for v in answers])
            new = CHANCES[rnd + 1]
            self.play(*[r[1].animate.stretch_to_fit_width(p * 4.2).move_to([-5.75, r[1].get_center()[1], 0],
                                                                         aligned_edge=LEFT)
                        for r, (_, p) in zip(rows, new)],
                      *[c.animate.set_opacity(.25) for c, v in zip(cards, answers) if v != 51],
                      run_time=1.0)
            self.mark(f'round_{rnd + 1}_update', chances=dict(new))
            if rnd == 0:
                self.wait(.4)
                self.play(FadeOut(cards), FadeOut(arrows), FadeOut(marks), run_time=.4)
        self.finish('05-rlvr', hold=2.6)
