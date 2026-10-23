"""ANIM-POST-002: six article GIFs for X-POST-002 (instruction tuning / SFT).

Redesigned 2026-10-22 after learner review ("too crowded, too much text"): one idea per
GIF, few large labels, motion carries the meaning. GIF 01 uses the Stanford Alpaca team's
published numbers; GIF 05 uses loss values measured in Day 12 · 03. Everything else is an
authored illustration. Timelines record explanatory order, not timing.
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


# ---------------------------------------------------------------- primitives
def card(text, color=FOREGROUND, size=26, w=None, h=.66, fill=.05, text_color=None):
    t = label(text, size, text_color or color)
    width = w or t.width + .6
    assert t.width < width - .2, (text, t.width, width)
    box = RoundedRectangle(width=width, height=h, corner_radius=.14, stroke_color=color,
                           stroke_width=2, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def chip(text, color=FOREGROUND, size=22, h=.54, fill=.05):
    t = label(text, size, color)
    box = RoundedRectangle(width=max(.46, t.width + .26), height=h, corner_radius=.1,
                           stroke_color=color, stroke_width=1.8, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def row(words, color=FOREGROUND, size=22, buff=.1):
    return VGroup(*[chip(w, color, size) for w in words]).arrange(RIGHT, buff=buff)


def recolor(c, color, fill=.06):
    return AnimationGroup(c[0].animate.set_stroke(color).set_fill(color, fill), c[1].animate.set_color(color))


def check_mark(color=COMPOSED, s=1.0):
    m = VMobject(stroke_color=color, stroke_width=7 * s)
    m.set_points_as_corners([np.array(p) * s for p in [(-.2, 0, 0), (-.06, -.15, 0), (.22, .2, 0)]])
    return m


def cross_mark(color=ACCENT, s=1.0):
    a = Line([-.16 * s, -.16 * s, 0], [.16 * s, .16 * s, 0], stroke_width=7 * s, color=color)
    b = Line([-.16 * s, .16 * s, 0], [.16 * s, -.16 * s, 0], stroke_width=7 * s, color=color)
    return VGroup(a, b)


def fit(group, width=13.4):
    if group.width > width:
        group.scale_to_fit_width(width)
    return group


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
            assert self.badge.get_left()[0] > self.title.get_right()[0] + .3, 'badge overlaps title'
            self.add(self.badge)

    def mark(self, name, wait=.35, **state):
        for m in self.mobjects:
            if m.get_width() == 0:
                continue
            assert m.get_left()[0] > -7.12 and m.get_right()[0] < 7.12, (name, 'x', m)
            assert m.get_bottom()[1] > -4.0 and m.get_top()[1] < 4.0, (name, 'y', m)
        self.events.append(dict(name=name, seconds=round(float(self.time), 3), **state))
        self.wait(wait)

    def finish(self, stem, hold=2.4, **state):
        self.mark('complete', wait=hold, **state)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.6)
        self.wait(.25)
        (P / (stem + '-timeline.json')).write_text(json.dumps(dict(
            events=self.events, checks={'inside_canvas': True},
            duration_seconds=round(float(self.time), 3)), indent=2) + '\n')


# ---------------------------------------------------------------- 01
ALPACA = dict(examples=52000, data_cost='< $500', train='3 hours on 8 GPUs', blind='90 vs 89')


def example_card(w=.9, h=.62):
    box = RoundedRectangle(width=w, height=h, corner_radius=.08, stroke_color=INTERMEDIATE, stroke_width=1.6,
                           fill_color=WHITE, fill_opacity=1)
    q = Rectangle(width=w * .45, height=.07, stroke_width=0, fill_color=BASE, fill_opacity=.9)
    a1 = Rectangle(width=w * .7, height=.07, stroke_width=0, fill_color=COMPOSED, fill_opacity=.9)
    a2 = Rectangle(width=w * .55, height=.07, stroke_width=0, fill_color=COMPOSED, fill_opacity=.9)
    lines = VGroup(q, a1, a2).arrange(DOWN, buff=.08, aligned_edge=LEFT).move_to(box)
    return VGroup(box, lines)


class AlpacaRecipe(Loop):
    def construct(self):
        self.begin('52,000 examples make an assistant', 'Stanford Alpaca, 2023')
        teacher = card('AI teacher', INTERMEDIATE, 32, w=3.3, h=1.1, fill=.08).move_to([-4.6, .7, 0])
        model = card('LLaMA 7B', MUTED, 36, w=3.6, h=1.3, fill=.06).move_to([3.9, .7, 0])
        sub_t = label('writes the examples', 24, MUTED).next_to(teacher, DOWN, buff=.2)
        sub_m = label('predicts text', 24, MUTED).next_to(model, DOWN, buff=.2)
        self.play(FadeIn(teacher), FadeIn(model), FadeIn(sub_t), FadeIn(sub_m), run_time=.7)
        self.mark('start')
        count = ValueTracker(0)
        counter = always_redraw(lambda: label(f'{int(count.get_value()):,} examples', 34, FOREGROUND).move_to([-.35, -1.5, 0]))
        self.add(counter)
        cards = [example_card() for _ in range(18)]
        anims = []
        for k, c in enumerate(cards):
            c.move_to(teacher.get_right() + RIGHT * .5 + UP * ((k % 3) - 1) * .35)
            anims.append(Succession(FadeIn(c, run_time=.15),
                                    c.animate(run_time=.75, rate_func=smooth).move_to(model.get_left() + LEFT * .3).scale(.4).set_opacity(0)))
        self.play(LaggedStart(*anims, lag_ratio=.12), count.animate.set_value(ALPACA['examples']), run_time=3.2, rate_func=linear)
        self.remove(*cards)
        self.mark('examples', n=ALPACA['examples'])
        alpaca = card('Alpaca 7B', COMPOSED, 36, w=3.6, h=1.3, fill=.12).move_to(model)
        sub_a = label('follows instructions', 24, COMPOSED).next_to(alpaca, DOWN, buff=.2)
        self.play(FadeTransform(model, alpaca), FadeTransform(sub_m, sub_a), run_time=.7)
        self.play(Indicate(alpaca, color=COMPOSED, scale_factor=1.06), run_time=.6)
        self.mark('assistant')
        stats = VGroup(card('data < $500', FOREGROUND, 28, h=.72), card('3 hours of training', FOREGROUND, 28, h=.72),
                       card('blind test: 90 vs 89', FOREGROUND, 28, h=.72)).arrange(RIGHT, buff=.35).move_to([0, -2.75, 0])
        fit(stats)
        flow = Arrow(teacher.get_right(), alpaca.get_left(), buff=.25, color=INTERMEDIATE, stroke_width=4, tip_length=.22)
        flow_lab = label('52,000 examples', 34, FOREGROUND).next_to(flow, UP, buff=.2)
        self.play(FadeOut(counter), GrowArrow(flow), FadeIn(flow_lab), run_time=.6)
        self.play(LaggedStart(*[FadeIn(st, shift=UP * .15) for st in stats], lag_ratio=.35), run_time=1.3)
        self.mark('stats', **ALPACA)
        self.finish('01-alpaca-recipe', hold=3.0, **ALPACA)


# ---------------------------------------------------------------- 02
SEQ = [('<user>', 'm'), ('Happy word?', 'u'), ('<end>', 'm'), ('<assistant>', 'm'), ('Elated.', 'a'), ('<end>', 'e')]


def seq_row(size=30):
    chips = []
    for text, kind in SEQ:
        col = INTERMEDIATE if kind in 'm' else (BASE if kind == 'u' else BASE)
        if kind == 'e':
            col = INTERMEDIATE
        chips.append(chip(text, col, size, h=.74))
    return VGroup(*chips).arrange(RIGHT, buff=.14)


class ThreeForms(Loop):
    def construct(self):
        self.begin('One example, three forms')
        stage = label('1 · a chat', 30, MUTED).move_to([0, 2.3, 0])
        u = card('Happy word?', BASE, 32, h=.8, fill=.1).move_to([-2.2, .9, 0])
        a = card('Elated.', COMPOSED, 32, h=.8, fill=.1).move_to([2.2, -.1, 0])
        self.play(FadeIn(stage), FadeIn(u, shift=RIGHT * .3), run_time=.6)
        self.play(FadeIn(a, shift=LEFT * .3), run_time=.5)
        self.mark('chat')
        r = fit(seq_row()).move_to([0, .4, 0])
        stage2 = label('2 · one token sequence', 30, MUTED).move_to(stage)
        self.play(FadeTransform(stage, stage2), ReplacementTransform(u, r[1]), ReplacementTransform(a, r[4]), run_time=.9)
        self.play(LaggedStart(*[FadeIn(r[i], scale=1.2) for i in (0, 2, 3, 5)], lag_ratio=.2), run_time=.9)
        tag = label('template markers', 26, INTERMEDIATE).next_to(r, DOWN, buff=.45)
        self.play(FadeIn(tag), run_time=.3)
        self.mark('sequence', tokens=[t for t, _ in SEQ])
        stage3 = label('3 · which tokens are graded', 30, MUTED).move_to(stage)
        self.play(FadeTransform(stage2, stage3), FadeOut(tag), *[recolor(r[i], GRID, .02) for i in (0, 1, 2, 3)], run_time=.7)
        self.play(*[recolor(r[i], COMPOSED, .18) for i in (4, 5)], run_time=.5)
        ticks = VGroup(*[check_mark(COMPOSED, 1.0).next_to(r[i], DOWN, buff=.18) for i in (4, 5)])
        self.play(FadeIn(ticks, scale=1.4), run_time=.4)
        claim = label('Read everything. Grade only the answer.', 34, FOREGROUND).move_to([0, -2.1, 0])
        self.play(FadeIn(claim, shift=UP * .15), run_time=.6)
        self.mark('mask', graded=['Elated.', '<end>'])
        self.finish('05-three-forms', hold=3.0, graded=['Elated.', '<end>'])


# ---------------------------------------------------------------- 04
class MaskedNotUntouched(Loop):
    def construct(self):
        self.begin('Masked doesn\'t mean untouched')
        prompt = VGroup(chip('Happy', MUTED, 34, h=.82, fill=.03), chip('word?', MUTED, 34, h=.82, fill=.03)).arrange(RIGHT, buff=.25)
        ans = chip('Elated.', COMPOSED, 34, h=.82, fill=.18)
        row_ = VGroup(prompt, ans).arrange(RIGHT, buff=1.4).move_to([0, -.9, 0])
        lp = label('not graded', 28, MUTED).next_to(prompt, DOWN, buff=.35)
        la = label('graded', 28, COMPOSED).next_to(ans, DOWN, buff=.35)
        self.play(FadeIn(row_), FadeIn(lp), FadeIn(la), run_time=.7)
        self.mark('tokens')
        loss = card('loss', ACCENT, 32, h=.74, fill=.12).next_to(ans, UP, buff=.9)
        link = Line(ans.get_top(), loss.get_bottom(), color=ACCENT, stroke_width=3)
        self.play(FadeIn(loss, shift=DOWN * .15), Create(link), run_time=.6)
        arcs = VGroup(*[ArcBetweenPoints(ans.get_top() + UP * .05, p.get_top() + UP * .05, angle=PI / 2.4,
                                         color=BASE, stroke_width=3, stroke_opacity=.6) for p in prompt])
        read = label('attention reads the prompt', 26, BASE).move_to([-1.1, 1.75, 0])
        self.play(Create(arcs), FadeIn(read), run_time=.9)
        self.mark('attention')
        for _ in range(2):
            pulses = [ShowPassingFlash(a.copy().set_stroke(ACCENT, 7, 1), time_width=.5) for a in arcs]
            self.play(ShowPassingFlash(link.copy().reverse_direction().set_stroke(ACCENT, 7), time_width=.6), run_time=.35)
            self.play(*pulses, run_time=.8)
        self.play(*[p[0].animate.set_stroke(ACCENT, 5).set_fill(ACCENT, .15) for p in prompt],
                  *[p[1].animate.set_color(ACCENT) for p in prompt], run_time=.5)
        lp2 = label('not graded, still trained', 28, ACCENT).move_to(lp)
        self.play(FadeTransform(lp, lp2), FadeOut(read), run_time=.5)
        self.mark('gradient_reaches_prompt', prompt_gradient_nonzero=True)
        claim = label('The mask decides what is graded, not what is trained.', 30, FOREGROUND).move_to([0, -3.0, 0])
        self.play(FadeIn(claim, shift=UP * .15), run_time=.6)
        self.finish('08-masked-not-untouched', hold=3.0)


# ---------------------------------------------------------------- 05
WORDS5 = ['The', 'cat', 'sat', 'down']


class OnePositionShift(Loop):
    def construct(self):
        self.begin('The one-position shift')
        toks = VGroup(*[chip(w, BASE, 38, h=.9, fill=.06) for w in WORDS5]).arrange(RIGHT, buff=1.0).move_to([0, -.6, 0])
        self.play(FadeIn(toks, lag_ratio=.15), run_time=.7)
        self.mark('tokens')
        good = VGroup(*[CurvedArrow(toks[i].get_top() + UP * .08, toks[i + 1].get_top() + UP * .08, angle=-PI / 2.2,
                                    color=COMPOSED, stroke_width=4, tip_length=.2) for i in range(3)])
        cap1 = label('each position guesses the next token', 30, COMPOSED).move_to([0, 2.0, 0])
        self.play(LaggedStart(*[Create(g) for g in good], lag_ratio=.3), FadeIn(cap1), run_time=1.2)
        ok = card('✓ predicts', COMPOSED, 30, h=.74, fill=.1).move_to([0, -2.3, 0])
        self.play(FadeIn(ok), run_time=.4)
        self.mark('shifted', alignment='next token')
        self.wait(.8)
        bad = VGroup(*[CurvedArrow(toks[i].get_top() + UP * .08 + LEFT * .2, toks[i].get_top() + UP * .08 + RIGHT * .2,
                                   angle=-PI * 1.6, color=ACCENT, stroke_width=4, tip_length=.18) for i in range(4)])
        cap2 = label('forgot the shift: it "guesses" the token it already sees', 30, ACCENT).move_to(cap1)
        self.play(FadeOut(good), FadeTransform(cap1, cap2), run_time=.5)
        self.play(LaggedStart(*[Create(b) for b in bad], lag_ratio=.2), run_time=1.0)
        oops = card('✗ copies · runs without error', ACCENT, 30, h=.74, fill=.1).move_to(ok)
        self.play(FadeTransform(ok, oops), run_time=.5)
        self.mark('unshifted', alignment='same token', error_raised=False)
        self.finish('09-one-position-shift', hold=3.0)


# ---------------------------------------------------------------- 06
SHORT_N, LONG_N = 2, 500


class TokenShare(Loop):
    def construct(self):
        self.begin('Long answers weigh more', 'illustrative counts')
        cell = .14
        a_sq = VGroup(*[Square(cell, stroke_width=0, fill_color=BASE, fill_opacity=.95) for _ in range(SHORT_N)]).arrange(RIGHT, buff=.04).scale(2.0)
        b_sq = VGroup(*[Square(cell, stroke_width=0, fill_color=INTERMEDIATE, fill_opacity=.85) for _ in range(LONG_N)]).arrange_in_grid(20, 25, buff=.035)
        a_lab = label(f'{SHORT_N} tokens', 34, BASE); b_lab = label(f'{LONG_N} tokens', 34, INTERMEDIATE)
        VGroup(a_sq, a_lab).arrange(DOWN, buff=.35).move_to([-4.0, .5, 0])
        VGroup(b_sq, b_lab).arrange(DOWN, buff=.3).move_to([2.6, .5, 0])
        self.play(FadeIn(a_sq), FadeIn(a_lab), run_time=.5)
        self.play(LaggedStart(*[FadeIn(s) for s in b_sq], lag_ratio=.002), FadeIn(b_lab), run_time=1.3)
        self.mark('two_examples', tokens=[SHORT_N, LONG_N])
        W_ = 11.0
        frame = Rectangle(width=W_, height=.6, stroke_color=GRID, stroke_width=1.6).move_to([0, -2.2, 0])
        wa = W_ * SHORT_N / (SHORT_N + LONG_N)
        seg_a = Rectangle(width=max(wa, .06), height=.6, stroke_width=0, fill_color=BASE, fill_opacity=1).move_to(frame.get_left(), aligned_edge=LEFT)
        seg_b = Rectangle(width=W_ - wa, height=.6, stroke_width=0, fill_color=INTERMEDIATE, fill_opacity=.85).move_to(frame.get_right(), aligned_edge=RIGHT)
        cap = label('share of the loss', 26, MUTED).next_to(frame, UP, buff=.18)
        self.play(Create(frame), FadeIn(cap), run_time=.4)
        self.play(ReplacementTransform(a_sq, seg_a), ReplacementTransform(b_sq, seg_b), FadeOut(a_lab), FadeOut(b_lab), run_time=1.3)
        big = label(f'{LONG_N // SHORT_N}×', 96, INTERMEDIATE).move_to([0, .7, 0])
        sub = label('one long answer outweighs a short one', 30, FOREGROUND).next_to(big, DOWN, buff=.2)
        self.play(FadeIn(big, scale=1.4), FadeIn(sub), run_time=.7)
        self.mark('shares', share_short=round(SHORT_N / (SHORT_N + LONG_N), 4), ratio=LONG_N // SHORT_N)
        claim = label('Balance a mix by tokens, not just by examples', 28, COMPOSED).next_to(frame, DOWN, buff=.3)
        self.play(FadeIn(claim, shift=UP * .1), run_time=.5)
        self.finish('10-token-share', hold=3.0)


# ---------------------------------------------------------------- LIMA
class Lima(Loop):
    def construct(self):
        self.begin('LIMA: 1,000 examples', 'Zhou et al., 2023')
        base = Rectangle(width=4.9, height=2.2, stroke_color=MUTED, stroke_width=2, fill_color=MUTED, fill_opacity=.12)
        base.move_to([-3.2, .1, 0])
        base_l = label('knowledge from pretraining', 26, MUTED).move_to(base)
        top_slot = Rectangle(width=4.9, height=.5, stroke_color=GRID, stroke_width=1.6).next_to(base, UP, buff=0)
        name = label('LLaMA 65B', 30, FOREGROUND).next_to(top_slot, UP, buff=.2)
        self.play(FadeIn(base), FadeIn(base_l), Create(top_slot), FadeIn(name), run_time=.8)
        self.mark('base_model')
        grid = VGroup(*[Square(.075, stroke_width=0, fill_color=BASE, fill_opacity=.9) for _ in range(1000)]).arrange_in_grid(25, 40, buff=.03)
        grid.move_to([3.6, .55, 0])
        g_l = label('1,000 examples · no RL', 28, BASE).next_to(grid, DOWN, buff=.3)
        self.play(LaggedStart(*[FadeIn(q) for q in grid], lag_ratio=.0006), FadeIn(g_l), run_time=1.2)
        self.mark('examples', n=1000)
        layer = Rectangle(width=4.9, height=.5, stroke_width=0, fill_color=BASE, fill_opacity=.85).move_to(top_slot)
        layer_l = label('how to answer', 24, WHITE).move_to(layer)
        note = label('+ 1,000 examples, no RL', 28, BASE).next_to(layer, RIGHT, buff=.5)
        self.play(ReplacementTransform(grid, layer), FadeTransform(g_l, note), run_time=1.3)
        self.play(FadeIn(layer_l), run_time=.3)
        self.mark('thin_layer')
        frame = Rectangle(width=8.0, height=.5, stroke_color=GRID, stroke_width=1.6).move_to([0, -2.55, 0])
        cap = label('answers rated equal to or better than GPT-4', 26, FOREGROUND).next_to(frame, UP, buff=.2)
        v = ValueTracker(0)
        fill = always_redraw(lambda: Rectangle(width=max(.001, 8.0 * v.get_value() / 100), height=.5, stroke_width=0,
                                               fill_color=COMPOSED, fill_opacity=.9).move_to(frame.get_left(), aligned_edge=LEFT))
        pct = always_redraw(lambda: label(f'{int(round(v.get_value()))}%', 30, COMPOSED).next_to(frame, RIGHT, buff=.25))
        self.play(Create(frame), FadeIn(cap), run_time=.4)
        self.add(fill, pct)
        self.play(v.animate.set_value(43), run_time=1.2)
        self.mark('result', pct=43)
        self.finish('02-lima', hold=3.0, examples=1000, pct=43)


# ---------------------------------------------------------------- FLAN
FLAN_WINS = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]


class Flan(Loop):
    def construct(self):
        assert len(FLAN_WINS) == 25 and sum(FLAN_WINS) == 20
        self.begin('FLAN: 60+ tasks as instructions', 'Wei et al., 2021')
        cols = [BASE, INTERMEDIATE, COMPOSED]
        train = VGroup(*[RoundedRectangle(width=.36, height=.36, corner_radius=.05, stroke_width=0,
                                          fill_color=cols[k % 3], fill_opacity=.75) for k in range(60)]).arrange_in_grid(10, 6, buff=.08)
        train.move_to([-5.0, .2, 0])
        t_l = label('60+ training tasks', 26, FOREGROUND).next_to(train, DOWN, buff=.3)
        model = card('137B model', FOREGROUND, 32, w=3.0, h=1.3, fill=.06).move_to([-.9, .2, 0])
        self.play(FadeIn(train, lag_ratio=.02), FadeIn(t_l), FadeIn(model), run_time=1.0)
        self.mark('training_tasks', n=60)
        self.play(LaggedStart(*[q.animate.move_to(model.get_left()).scale(.2).set_opacity(0) for q in train], lag_ratio=.015),
                  FadeOut(t_l), run_time=1.4)
        self.remove(train)
        self.play(Indicate(model, color=FOREGROUND, scale_factor=1.05), run_time=.4)
        test = VGroup(*[RoundedRectangle(width=.5, height=.5, corner_radius=.06, stroke_color=GRID, stroke_width=2,
                                         fill_opacity=0) for _ in range(25)]).arrange_in_grid(5, 5, buff=.12).move_to([4.2, .45, 0])
        u_l = label('25 unseen tasks', 26, FOREGROUND).next_to(test, DOWN, buff=.3)
        arrow = Arrow(model.get_right(), test.get_left(), buff=.25, color=MUTED, stroke_width=3, tip_length=.18)
        self.play(GrowArrow(arrow), FadeIn(test, lag_ratio=.02), FadeIn(u_l), run_time=.8)
        self.mark('unseen_tasks', n=25)
        flips = [q.animate.set_fill(COMPOSED if w else GRID, .85 if w else .5).set_stroke(COMPOSED if w else GRID)
                 for q, w in zip(test, FLAN_WINS)]
        self.play(LaggedStart(*flips, lag_ratio=.08), run_time=2.0)
        res = label('20 / 25 beat GPT-3 175B (zero-shot)', 32, COMPOSED).move_to([0, -2.75, 0])
        self.play(FadeIn(res, shift=UP * .15), run_time=.5)
        self.mark('result', wins=sum(FLAN_WINS), total=25)
        self.finish('03-flan', hold=3.0, wins=sum(FLAN_WINS), total=25)


# ---------------------------------------------------------------- recipe
STEPS = ['Collect', 'Format', 'Mask', 'Train', 'Evaluate']


class Recipe(Loop):
    def construct(self):
        self.begin('The training recipe')
        xs = [-5.4 + 2.7 * k for k in range(5)]
        chips = VGroup(*[card(f'{k + 1} · {s}', MUTED, 26, w=2.35, h=.72, fill=.03).move_to([x, 1.55, 0]) for k, (s, x) in enumerate(zip(STEPS, xs))])
        arrows = VGroup(*[Arrow(chips[k].get_right(), chips[k + 1].get_left(), buff=.06, color=GRID, stroke_width=2.4, tip_length=.14) for k in range(4)])
        self.play(FadeIn(chips, lag_ratio=.1), FadeIn(arrows), run_time=.8)
        self.mark('steps')
        y = -.3
        def tok(t, c, size=20):
            return chip(t, c, size, h=.5, fill=.12)
        icons = [
            VGroup(example_card(1.5, .9), example_card(1.5, .9)).arrange(DOWN, buff=.15),
            VGroup(VGroup(tok('<user>', INTERMEDIATE, 24), tok('Q', BASE, 24)).arrange(RIGHT, buff=.08),
                   VGroup(tok('<asst>', INTERMEDIATE, 24), tok('A', BASE, 24)).arrange(RIGHT, buff=.08)).arrange(DOWN, buff=.15, aligned_edge=LEFT),
            VGroup(VGroup(tok('<user>', GRID, 24), tok('Q', GRID, 24)).arrange(RIGHT, buff=.08),
                   VGroup(tok('<asst>', GRID, 24), tok('A', COMPOSED, 24)).arrange(RIGHT, buff=.08)).arrange(DOWN, buff=.15, aligned_edge=LEFT),
            VGroup(card('model', FOREGROUND, 24, w=1.8, h=.8, fill=.05), label('loss ↓', 24, COMPOSED)).arrange(DOWN, buff=.2),
            VGroup(card('new task', FOREGROUND, 24, w=1.8, h=.8, fill=.05), check_mark(COMPOSED, 1.1)).arrange(DOWN, buff=.25),
        ]
        for ic, x in zip(icons, xs):
            if ic.width > 2.5:
                ic.scale_to_fit_width(2.5)
            ic.move_to([x, y, 0])
        for k in range(5):
            light = [chips[k][0].animate.set_stroke(BASE if k < 4 else COMPOSED, 3).set_fill(BASE if k < 4 else COMPOSED, .1),
                     chips[k][1].animate.set_color(FOREGROUND)]
            if k:
                light.append(arrows[k - 1].animate.set_color(BASE))
            self.play(*light, FadeIn(icons[k], shift=DOWN * .15), run_time=.6)
            self.wait(.35)
            self.mark(f'step_{k + 1}', step=STEPS[k], wait=.1)
        claim = label('Same next-token loss, on instruction → response pairs', 30, FOREGROUND).move_to([0, -2.6, 0])
        self.play(FadeIn(claim, shift=UP * .15), run_time=.6)
        self.finish('04-recipe', hold=3.0, steps=STEPS)


# ---------------------------------------------------------------- which tokens get a loss
CONV = [('<system>', 'You are helpful.'), ('<user>', 'Happy word?'), ('<assistant>', 'Elated.'),
        ('<user>', 'Another one?'), ('<assistant>', 'Joyful.')]


class WhatGetsLoss(Loop):
    def construct(self):
        self.begin('Which tokens get a loss')
        rows = VGroup()
        for role, text in CONV:
            rows.add(VGroup(chip(role, INTERMEDIATE, 28, h=.66), chip(text, FOREGROUND, 28, h=.66), chip('<end>', INTERMEDIATE, 28, h=.66)).arrange(RIGHT, buff=.12))
        rows.arrange(DOWN, buff=.24, aligned_edge=LEFT).move_to([-.6, .05, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * .2) for r in rows], lag_ratio=.15), run_time=1.2)
        self.mark('conversation', turns=len(CONV))
        dim = []
        for (role, _), r in zip(CONV, rows):
            dim += [recolor(r[0], GRID, .02)]
            if role != '<assistant>':
                dim += [recolor(r[1], GRID, .02), recolor(r[2], GRID, .02)]
        self.play(*dim, run_time=.8)
        graded, ticks = [], VGroup()
        for (role, text), r in zip(CONV, rows):
            if role == '<assistant>':
                graded += [text, '<end>']
                ticks.add(check_mark(COMPOSED, 1.0).next_to(r, RIGHT, buff=.3))
        self.play(*[recolor(r[k], COMPOSED, .2) for (role, _), r in zip(CONV, rows) if role == '<assistant>' for k in (1, 2)], run_time=.7)
        self.play(FadeIn(ticks, scale=1.4), run_time=.4)
        legend = VGroup(chip('loss', COMPOSED, 24, fill=.2), chip('no loss', GRID, 24, fill=.02)).arrange(DOWN, buff=.2, aligned_edge=LEFT)
        legend.move_to([5.6, .05, 0])
        self.play(FadeIn(legend), run_time=.4)
        self.mark('mask', graded=graded)
        claim = label('Loss only on what the model must write', 32, FOREGROUND).move_to([0, -2.9, 0])
        self.play(FadeIn(claim, shift=UP * .15), run_time=.6)
        self.finish('07-what-gets-loss', hold=3.0, graded=graded)


# ---------------------------------------------------------------- static: three forms side by side
SUMMARY = {
    'en': dict(cjk=False, heads=['Stored', 'What the model reads', 'What gets a loss'],
               legend=['loss', 'no loss'],
               note='The model reads every token; only the response and its end marker get a loss.'),
    'zh': dict(cjk=True, heads=['存储形态', '模型读取的序列', '计算损失的部分'],
               legend=['计算损失', '不计算'],
               note='模型读取全部 token，只有回答和结尾标记计算损失。'),
}
TOKS = ['<|im_start|>user', 'Happy word?', '<|im_end|>', '<|im_start|>assistant', 'Elated.', '<|im_end|>']


class ThreeFormsSummary(Scene):
    def construct(self):
        d = SUMMARY[os.environ.get('SUMMARY_LANG', 'en')]
        L = partial(label, cjk=d['cjk'])
        self.camera.background_color = BACKGROUND
        xs = [-4.75, 0, 4.75]
        for x, h in zip(xs, d['heads']):
            self.add(L(h, 30, FOREGROUND).move_to([x, 2.85, 0]))
        msgs = VGroup()
        for role, text, col in [('user', 'Happy word?', BASE), ('assistant', 'Elated.', COMPOSED)]:
            box = RoundedRectangle(width=3.6, height=1.25, corner_radius=.14, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=.06)
            inner = VGroup(label(f'role: {role}', 22, MUTED), label(text, 28, col)).arrange(DOWN, buff=.15, aligned_edge=LEFT)
            msgs.add(VGroup(box, inner.move_to(box)))
        msgs.arrange(DOWN, buff=.4).move_to([xs[0], .35, 0])
        self.add(msgs)
        def column(colored):
            col = VGroup()
            for i, t in enumerate(TOKS):
                marker = t.startswith('<|')
                if not colored:
                    c = INTERMEDIATE if marker else BASE
                    fill = .06
                else:
                    c = COMPOSED if i >= 4 else GRID
                    fill = .2 if i >= 4 else .02
                col.add(chip(t, c, 24, h=.56, fill=fill))
            return col.arrange(DOWN, buff=.14)
        c2 = column(False).move_to([xs[1], .35, 0]); c3 = column(True).move_to([xs[2], .35, 0])
        self.add(c2, c3)
        for i in (4, 5):
            self.add(check_mark(COMPOSED, .8).next_to(c3[i], RIGHT, buff=.2))
        for a, b in ((msgs, c2), (c2, c3)):
            self.add(Arrow(a.get_right(), b.get_left(), buff=.25, color=MUTED, stroke_width=3, tip_length=.18))
        leg = VGroup(VGroup(chip(' ', COMPOSED, 20, fill=.2), L(d['legend'][0], 22, COMPOSED)).arrange(RIGHT, buff=.12),
                     VGroup(chip(' ', GRID, 20, fill=.02), L(d['legend'][1], 22, MUTED)).arrange(RIGHT, buff=.12)).arrange(RIGHT, buff=.5)
        leg.next_to(c3, DOWN, buff=.35)
        self.add(leg)
        note = L(d['note'], 26, FOREGROUND).move_to([0, -3.3, 0])
        assert note.width < 13.4
        self.add(note)
        for m in self.mobjects:
            assert m.get_left()[0] > -7.12 and m.get_right()[0] < 7.12, m
            assert m.get_bottom()[1] > -4 and m.get_top()[1] < 4, m
        self.wait(.1)
