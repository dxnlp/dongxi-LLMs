"""DPO article GIF set (CAND-ANIM-079 as two clips, 067, 057, 052, 069 + 053, 066, 070, 071, 073, 074, 077, 078).

One running example, the egg question, shared with the article and the KL set. Shares, rewards and gold
values are illustrative; every ratio, log, score and optimum on them is exact (they match the article's
check_numbers.py). Clip 09 replays one simulated run of check_numbers.py. Clip 11 uses invented curves. No
numbers from our own runs appear.

Cast, shared by every clip:

- slate: the reference
- blue: the trained model, the reward
- green: raised answers, the chosen answer, gold
- amber: lowered answers, the rejected answer
- violet: KL, the loss

The bottom-right corner is kept free for the DONGXI LABS mark (brand.py), checked at every mark().
Timelines record explanatory order, not measured time.
"""
from functools import partial
import json
import math
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import (label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE,
                         INTERMEDIATE, COMPOSED, ACCENT)

label = partial(label, layout_scale=16)
REFC, NEWC, UP_C, DOWN_C, KLC = '#475569', BASE, COMPOSED, ACCENT, INTERMEDIATE
MARK_X, MARK_Y = 4.9, -3.35          # keep x > MARK_X and y < MARK_Y empty for the brand mark

NAMES = ['vague', '10 min', 'long', 'best', 'wrong']
NAMES6 = ['vague', '10 min', 'long', 'best', 'wrong', '20 min']
REF = np.array([.40, .25, .15, .12, .08])
SMALL = np.array([.30, .25, .15, .22, .08])
R = np.array([0, 1, .5, 2, -1.])
REF6 = np.array([.40, .25, .15, .12, .075, .005])
R6 = np.array([0, 1, .5, 2, -1, 3.])
GOLD6 = np.array([0, 1, .5, 2, -1, -2.])
BEST, TEN, VAGUE, WRONG = 3, 1, 0, 4
# One simulated run from check_numbers.py: seed 0, 1,000 pairs, Adam, full batch. Shares at steps
# 0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 100, 200 and 1,500.
DPO_PATH = [[.4, .25, .15, .12, .08], [.325, .302, .163, .145, .065], [.267, .356, .152, .173, .052],
            [.221, .39, .148, .199, .041], [.195, .399, .148, .226, .032], [.185, .391, .143, .255, .026],
            [.187, .359, .132, .303, .019], [.193, .319, .129, .344, .015], [.193, .289, .124, .383, .012],
            [.177, .294, .118, .401, .01], [.16, .318, .118, .394, .009], [.178, .318, .123, .369, .012],
            [.194, .329, .131, .331, .016], [.19, .324, .129, .343, .015], [.189, .325, .128, .343, .014],
            [.189, .325, .128, .343, .014]]
DPO_RUN = np.array(DPO_PATH[-1])
FIRST_PAIRS = [(2, 1), (4, 3), (0, 4), (0, 1), (0, 4), (3, 4), (2, 3), (0, 4)]
AVG_DISTANCE = [('10', .38), ('100', .15), ('1,000', .04), ('10,000', .01)]


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def optimum(ref, r, beta):
    w = ref * np.exp(r / beta)
    return w / w.sum()


def score(p, ref=REF, r=R, beta=1.):
    return float(p @ r) - beta * kl(p, ref)


def sig(x):
    return 1 / (1 + math.exp(-x))


def bt(m):
    return -math.log(sig(m))


def f3(v, sign=False, d=3):
    s = f'{v:+.{d}f}' if sign else f'{v:.{d}f}'
    return s.replace('-', '−')


STAR = optimum(REF, R, 1)


# ---------------------------------------------------------------- primitives
def pill(text, col, size=26, fill=WHITE, text_color=None):
    t = label(text, size, text_color or col)
    box = RoundedRectangle(width=t.width + .34, height=t.height + .26, corner_radius=.16, stroke_color=col,
                           stroke_width=2.5, fill_color=fill, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def harrow(x0, x1, y, color=MUTED, sw=4, tip=.15):
    return Arrow([x0, y, 0], [x1, y, 0], buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=.5)


def varrow(x, y0, y1, color=MUTED, sw=4, tip=.15):
    return Arrow([x, y0, 0], [x, y1, 0], buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=.5)


def bar(x, v, base, unit, col, w=.9, opacity=.9):
    """A vertical bar from `base`; negative values hang below it."""
    h = max(abs(v) * unit, .012)
    b = Rectangle(width=w, height=h, stroke_width=0, fill_color=col, fill_opacity=opacity)
    return b.move_to([x, base, 0], aligned_edge=DOWN if v >= 0 else UP)


def hbar(x, v, y, unit, col, h=.42, opacity=.9):
    """A horizontal bar from `x`; negative values extend to the left."""
    w = max(abs(v) * unit, .012)
    b = Rectangle(width=w, height=h, stroke_width=0, fill_color=col, fill_opacity=opacity)
    return b.move_to([x, y, 0], aligned_edge=LEFT if v >= 0 else RIGHT)


def pi_ref(size, color):
    p = label('π', size, color)
    s = label('ref', int(size * .62), color).next_to(p, RIGHT, buff=.02).shift(DOWN * .1 * size / 30)
    return VGroup(p, s)


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
            for f in m.family_members_with_points():
                if not isinstance(f, VMobject) or (f.get_fill_opacity() < .01 and f.get_stroke_opacity() < .01):
                    continue
                assert f.get_left()[0] > -7.12 and f.get_right()[0] < 7.12, (name, 'x', f)
                assert f.get_bottom()[1] > -4.0 and f.get_top()[1] < 4.0, (name, 'y', f)
                assert not (f.get_right()[0] > MARK_X and f.get_bottom()[1] < MARK_Y), (name, 'brand corner', f)
        for m in self.mobjects:
            for f in m.get_family():
                if isinstance(f, CurvedArrow):
                    raise AssertionError((name, 'curved arrow', f))
                if isinstance(f, Arrow):
                    d = f.get_end() - f.get_start()
                    assert abs(d[0]) < 1e-3 or abs(d[1]) < 1e-3, (name, 'slanted arrow', d)
        self.events.append(dict(name=name, seconds=round(float(self.time), 3), **state))
        self.wait(wait)

    def finish(self, stem, hold=2.4, **state):
        self.mark('complete', wait=hold, **state)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=.6)
        self.wait(.25)
        (P / (stem + '-timeline.json')).write_text(json.dumps(dict(
            events=self.events, checks={'inside_canvas': True, 'brand_corner_free': True},
            duration_seconds=round(float(self.time), 3)), indent=2) + '\n')

    def clear_stage(self, keep=(), run_time=.5):
        """Fade everything except the title, the badge and `keep`."""
        fixed = {self.title, getattr(self, 'badge', None), *keep}
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m not in fixed], run_time=run_time)


def node(text, w, h, col, size=28, opacity=.08):
    box = RoundedRectangle(width=w, height=h, corner_radius=.14, stroke_color=col, stroke_width=2.5,
                           fill_color=col, fill_opacity=opacity)
    return VGroup(box, label(text, size, col).move_to(box))


def ghost(x, v, base, unit, col, w):
    """Dashed outline of a target bar."""
    h = max(v * unit, .02)
    return DashedVMobject(Rectangle(width=w, height=h, stroke_color=col, stroke_width=3).move_to(
        [x, base, 0], aligned_edge=DOWN), num_dashes=max(14, int(h * 14)))


class LossAxes:
    """The DPO loss as a curve over the gap, shared by clips 06 and 07."""

    def __init__(self, x0=-3.35, y0=-2.3, sx=1.0, sy=1.42):
        self.x0, self.y0, self.sx, self.sy = x0, y0, sx, sy
        self.xa = Line(self.pt(-3.05, 0), self.pt(3.05, 0), color=GRID, stroke_width=3)
        self.ya = Line(self.pt(0, 0), self.pt(0, 3.2), color=GRID, stroke_width=3)
        self.ticks = VGroup(*[VGroup(Line(self.pt(m, -.03), self.pt(m, .03), color=MUTED, stroke_width=2),
                                     label(f3(m, d=0), 24, MUTED).next_to(self.pt(m, 0), DOWN, buff=.14))
                              for m in (-3, 0, 3)])
        self.xl = label('gap', 26, MUTED).next_to(self.pt(1.5, 0), DOWN, buff=.14)
        self.yl = label('loss', 26, KLC).next_to(self.pt(0, 3.2), RIGHT, buff=.14)
        self.fixed = VGroup(self.xa, self.ya, self.ticks, self.xl, self.yl)

    def pt(self, m, v=None):
        return np.array([self.x0 + m * self.sx, self.y0 + (bt(m) if v is None else v) * self.sy, 0])

    def curve(self, a=-3, b=3, col=KLC):
        return ParametricFunction(self.pt, t_range=[a, b, .05], color=col, stroke_width=5)

    def tangent(self, m, half=.75):
        d = sig(-m)
        return Line(self.pt(m - half, bt(m) + half * d), self.pt(m + half, bt(m) - half * d), color=NEWC, stroke_width=4)


# ================================================================ 00a the long way and the shortcut
class TwoRoutes(Loop):
    """RLHF goes from human pairs through a reward model and RL to the chat model. DPO goes straight there."""

    def lane(self, y, items, col):
        nodes = [node(t, w, .92, c, 28).move_to([x, y, 0]) for t, w, x, c in items]
        arrows = [harrow(a.get_right()[0] + .1, b.get_left()[0] - .1, y, col, 4) for a, b in zip(nodes, nodes[1:])]
        return nodes, arrows

    def construct(self):
        self.begin('The long way and the shortcut')
        y1, y2 = 1.25, -1.25
        head1 = label('RLHF', 32, REFC).move_to([-6.5, y1 + .95, 0], aligned_edge=LEFT)
        n1, a1 = self.lane(y1, [('human pairs', 2.6, -5.25, UP_C), ('reward model', 2.75, -1.85, REFC),
                                ('RL on samples', 2.9, 1.7, REFC), ('chat model', 2.5, 5.3, NEWC)], REFC)
        self.play(FadeIn(head1), FadeIn(n1[0]), run_time=.5)
        s1 = label('trained first', 26, MUTED).next_to(n1[1], DOWN, buff=.16)
        s2 = label('writes answers, scores them', 26, MUTED).next_to(n1[2], DOWN, buff=.16)
        for a, n, t in zip(a1, n1[1:], [s1, s2, None]):
            self.play(GrowArrow(a), FadeIn(n, shift=RIGHT * .1), run_time=.6)
            if t is not None:
                self.play(FadeIn(t), run_time=.35)
        self.mark('long_way', stages=4, wait=1.2)
        head2 = label('DPO', 32, UP_C).move_to([-6.5, y2 + .95, 0], aligned_edge=LEFT)
        n2, _ = self.lane(y2, [('human pairs', 2.6, -5.25, UP_C), ('chat model', 2.5, 5.3, NEWC)], UP_C)
        short = harrow(n2[0].get_right()[0] + .1, n2[1].get_left()[0] - .1, y2, UP_C, 8, .24)
        self.play(FadeIn(head2), FadeIn(n2[0]), run_time=.5)
        self.play(GrowArrow(short), FadeIn(n2[1], shift=RIGHT * .1), run_time=.8)
        t3 = label('one loss on stored pairs', 28, UP_C).next_to(short, UP, buff=.16)
        self.play(FadeIn(t3), run_time=.35)
        self.mark('shortcut', stages=2, wait=1.2)
        q = label('no reward model: where does the score come from?', 30, FOREGROUND).move_to([-6.5, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(q, shift=UP * .08), run_time=.45)
        self.finish('00a-two-routes', hold=3.0, stages=[4, 2])


# ================================================================ 00b the change is the score
class ChangeIsScore(Loop):
    """The model's change from the frozen reference is itself a score: raised answers above 0, lowered answers
    below 0, unchanged answers 0."""

    LX, LB, LU, LW = [-6.3, -5.25, -4.2, -3.15, -2.1], -2.35, 5.8, .84
    RX, RY, RU, RW = [.75, 1.95, 3.15, 4.35, 5.55], .25, .95, .84

    def shares(self, p):
        return VGroup(*[bar(x, v, self.LB, self.LU, NEWC, self.LW) for x, v in zip(self.LX, p)])

    def scores(self, p):
        lr = np.log(np.asarray(p) / REF)
        b = VGroup(*[bar(x, v, self.RY, self.RU, UP_C if v > .0005 else DOWN_C, self.RW) for x, v in zip(self.RX, lr)])
        t = VGroup(*[label('0' if abs(v) < .0005 else f3(v, True), 26, UP_C if v > .0005 else DOWN_C if v < -.0005 else MUTED)
                     .next_to([x, self.RY + v * self.RU, 0], UP if v >= -.0005 else DOWN, buff=.1) for x, v in zip(self.RX, lr)])
        return b, t

    def construct(self):
        self.begin('The change is the score')
        hl = label('shares; dashed: reference (frozen)', 28, FOREGROUND).move_to([-6.75, 2.5, 0], aligned_edge=LEFT)
        hr = label('score = β × log-ratio, β = 1', 28, FOREGROUND).move_to([.3, 2.5, 0], aligned_edge=LEFT)
        base = Line([-6.85, self.LB, 0], [-1.55, self.LB, 0], color=GRID, stroke_width=2)
        zero = Line([.2, self.RY, 0], [6.1, self.RY, 0], color=GRID, stroke_width=2)
        ln = VGroup(*[label(n, 24, FOREGROUND).move_to([x, self.LB - .32, 0]) for n, x in zip(NAMES, self.LX)])
        rn = VGroup(*[label(n, 24, FOREGROUND).move_to([x, -2.15, 0]) for n, x in zip(NAMES, self.RX)])
        ref = VGroup(*[ghost(x, v, self.LB, self.LU, REFC, self.LW + .12) for x, v in zip(self.LX, REF)])
        sh = self.shares(REF)
        sb, st = self.scores(REF)
        self.play(FadeIn(hl), FadeIn(hr), FadeIn(base), FadeIn(zero), FadeIn(ln), FadeIn(rn), run_time=.5)
        self.play(FadeIn(sh), Create(ref), run_time=.6)
        self.play(FadeIn(st), run_time=.3)
        self.add(sb)
        same = label('start: copy of the reference', 28, REFC).move_to([.3, 1.95, 0], aligned_edge=LEFT)
        self.play(FadeIn(same), run_time=.3)
        self.mark('start', scores=[0, 0, 0, 0, 0], wait=1.1)

        def go(to, frm, text, col, n=12):
            nonlocal sb, st, same
            self.play(FadeOut(same), run_time=.2)
            for t in np.linspace(0, 1, n + 1)[1:]:
                p = (1 - t) * frm + t * to                # a straight line in shares: unmoved answers stay at 0
                b2, t2 = self.scores(p)
                self.play(Transform(sh, self.shares(p)), run_time=.11, rate_func=linear)
                self.remove(sb, st); sb, st = b2, t2; self.add(sb, st)
            same = label(text, 28, col).move_to([.3, 1.95, 0], aligned_edge=LEFT)
            self.play(FadeIn(same), run_time=.3)

        go(SMALL, REF, 'raised: above 0, lowered: below 0', UP_C)
        self.mark('small_move', scores=[round(float(v), 3) + 0 for v in np.log(SMALL / REF)], wait=1.7)
        go(STAR, SMALL, 'the best compromise', NEWC)
        end = pill('no score head, no second network', UP_C, 26).move_to([.3, -2.85, 0], aligned_edge=LEFT)
        self.play(FadeIn(end, scale=1.08), run_time=.4)
        self.finish('00b-change-is-the-score', hold=3.0, scores=[round(float(v), 3) for v in np.log(STAR / REF)])


# ================================================================ 01 secretly a reward model
class Secretly(Loop):
    """A separate reward model, then the same score from two probabilities: the model and its frozen
    reference read an answer, and ratio, log and β give its score."""

    def construct(self):
        self.begin('Your model is secretly a reward model')
        y = .55
        a = pill('an answer', FOREGROUND, 28).move_to([-5.5, y, 0])
        body = node('LLM body', 3.0, 1.7, REFC).move_to([-1.7, y, 0])
        head = node('score head', 2.0, 1.7, NEWC, 26, .16).next_to(body, RIGHT, buff=.08)
        s = pill('one score', NEWC, 30).move_to([4.6, y, 0])
        a1 = harrow(a.get_right()[0] + .12, body.get_left()[0] - .12, y)
        a2 = harrow(head.get_right()[0] + .12, s.get_left()[0] - .12, y)
        self.play(FadeIn(a, shift=RIGHT * .1), run_time=.4)
        self.play(GrowArrow(a1), FadeIn(body), run_time=.5)
        self.play(FadeIn(head, scale=1.1), run_time=.4)
        self.play(GrowArrow(a2), FadeIn(s, scale=1.15), run_time=.5)
        c = label('a reward model is a separate network', 30, FOREGROUND).move_to([0, -1.45, 0])
        self.play(FadeIn(c), run_time=.4)
        self.mark('separate', wait=1.3)
        self.clear_stage(run_time=.4)

        # the same job, done by two probabilities
        rows = []
        for yy, name, col, share in [(2.05, 'trained model', NEWC, .22), (.75, 'reference (frozen)', REFC, .12)]:
            ans = pill('best answer', UP_C, 26).move_to([-5.45, yy, 0])
            m = node(name, 3.7, .92, col).move_to([-1.55, yy, 0])
            v = label(f'{share * 100:.0f}%', 34, col).move_to([1.45, yy, 0])
            ar1 = harrow(ans.get_right()[0] + .1, m.get_left()[0] - .1, yy)
            ar2 = harrow(m.get_right()[0] + .1, v.get_left()[0] - .1, yy)
            rows.append((ans, ar1, m, ar2, v))
        for r in rows:
            self.play(FadeIn(r[0]), GrowArrow(r[1]), FadeIn(r[2]), run_time=.5)
            self.play(GrowArrow(r[3]), FadeIn(r[4], scale=1.2), run_time=.4)
        self.mark('two_shares', shares=[.22, .12], wait=.6)
        ratio, log_r = .22 / .12, math.log(.22 / .12)
        y2 = -1.1
        steps = [pill(f'ratio {ratio:.2f}', FOREGROUND, 28), pill(f'log {log_r:.3f}', KLC, 28),
                 pill('× β = 1', REFC, 28), pill(f'score {log_r:.3f}', NEWC, 30, NEWC, WHITE)]
        for st, x in zip(steps, [-5.3, -1.85, 1.3, 4.75]):
            st.move_to([x, y2, 0])
        chain = VGroup(steps[0])
        self.play(FadeIn(steps[0], shift=DOWN * .1), run_time=.45)
        for i in (1, 2, 3):
            ar = harrow(steps[i - 1].get_right()[0] + .1, steps[i].get_left()[0] - .1, y2)
            chain.add(ar, steps[i])
            self.play(GrowArrow(ar), FadeIn(steps[i], scale=1.1), run_time=.45)
        note = label('this score is the implicit reward', 30, FOREGROUND).move_to([0, -2.35, 0])
        self.play(FadeIn(note), run_time=.4)
        self.mark('score', ratio=round(ratio, 2), log=round(log_r, 3), wait=1.5)
        self.clear_stage(run_time=.4)

        # every answer gets one
        XS, Y0, U = [-5.2, -3.1, -1.0, 1.1, 3.2], .45, 1.75
        lr = np.log(SMALL / REF)
        head2 = label('every answer gets a score', 32, FOREGROUND).move_to([-6.55, 2.5, 0], aligned_edge=LEFT)
        axis = Line([-6.2, Y0, 0], [4.2, Y0, 0], color=GRID, stroke_width=2.5)
        names = VGroup(*[label(n, 28, FOREGROUND).move_to([x, Y0 - 1.4, 0]) for n, x in zip(NAMES, XS)])
        bs = VGroup(*[bar(x, v, Y0, U, NEWC, 1.1) for x, v in zip(XS, lr)])
        vals = VGroup(*[label(f3(v, True) if abs(v) > 1e-9 else '0', 28, NEWC).next_to(b, UP if v >= 0 else DOWN, buff=.12)
                        for v, b in zip(lr, bs)])
        self.play(FadeIn(head2), Create(axis), FadeIn(names), run_time=.5)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v > 0 else UP) for b, v in zip(bs, lr)], lag_ratio=.15),
                  FadeIn(vals), run_time=.9)
        q1 = label('Your Language Model is Secretly a Reward Model', 30, NEWC).move_to([-6.55, -2.2, 0], aligned_edge=LEFT)
        q2 = label('Rafailov et al., 2023', 26, MUTED).next_to(q1, DOWN, buff=.16).align_to(q1, LEFT)
        self.play(FadeIn(q1), FadeIn(q2), run_time=.5)
        self.finish('01-secretly-a-reward-model', hold=3.0, scores=[round(float(v), 3) + 0 for v in lr])


# ================================================================ 02 three signals
class ThreeSignals(Loop):
    """Reward and KL enter the update; gold only judges checkpoints. DPO replaces the reward model with
    stored pairs and puts β and the reference straight into the loss."""

    def construct(self):
        self.begin('Three signals, two jobs')
        mx, wx = -3.6, 2.0
        model = node('model', 2.6, 1.0, NEWC, 32).move_to([mx, -.15, 0])
        rm = node('reward model (frozen)', 4.4, .9, NEWC, 28, .04).move_to([mx, 2.2, 0])
        rf = node('reference (frozen)', 4.4, .9, REFC, 28).move_to([mx, -2.5, 0])
        a1 = varrow(mx, rm.get_bottom()[1] - .1, model.get_top()[1] + .1, NEWC, 5)
        a2 = varrow(mx, rf.get_top()[1] + .1, model.get_bottom()[1] - .1, KLC, 5)
        t1 = label('reward: which way', 28, NEWC).next_to(a1, RIGHT, buff=.2)
        t2 = label('KL: how far', 28, KLC).next_to(a2, RIGHT, buff=.2)
        self.play(FadeIn(model, scale=1.1), run_time=.4)
        self.play(FadeIn(rm), GrowArrow(a1), FadeIn(t1), run_time=.6)
        self.mark('reward', wait=.7)
        self.play(FadeIn(rf), GrowArrow(a2), FadeIn(t2), run_time=.6)
        self.mark('kl', wait=.7)
        wall = DashedLine([wx, 2.75, 0], [wx, -3.05, 0], color=MUTED, stroke_width=3)
        wl = label('never enters training', 26, MUTED).move_to([wx + .2, 2.75, 0], aligned_edge=LEFT)
        gold = node('gold', 3.2, 1.0, UP_C, 32).move_to([4.75, -.15, 0])
        items = VGroup(*[label(t, 26, UP_C) for t in ('held-out comparisons', 'benchmarks', 'reading samples')]
                       ).arrange(DOWN, buff=.14).next_to(gold, DOWN, buff=.3)
        self.play(Create(wall), FadeIn(wl), run_time=.5)
        self.play(FadeIn(gold, scale=1.1), FadeIn(items), run_time=.5)
        cp = pill('checkpoint', NEWC, 26).move_to(model)
        self.add(cp)
        self.play(cp.animate.move_to([.6, -.15, 0]), run_time=.5)
        self.play(cp.animate.move_to([gold.get_left()[0] - 1.0, 1.25, 0]), run_time=.8)
        self.play(cp.animate.next_to(gold, UP, buff=.14), run_time=.4)
        self.mark('gold', wait=1.0)
        self.play(FadeOut(cp), run_time=.3)

        # what DPO changes
        tag = pill('in DPO', KLC, 30, WHITE).move_to([-6.15, -.15, 0])
        rm2 = node('stored human pairs', 4.4, .9, UP_C, 28).move_to(rm)
        t1b = label('chosen, rejected', 28, UP_C).next_to(a1, RIGHT, buff=.2)
        self.play(FadeIn(tag, scale=1.15), run_time=.35)
        self.play(FadeOut(t1), run_time=.2)
        self.play(FadeTransform(rm, rm2), FadeIn(t1b), a1.animate.set_color(UP_C), run_time=.6)
        self.mark('pairs', wait=.9)
        t2b = label('β and reference in the loss', 28, KLC).next_to(a2, RIGHT, buff=.2)
        t2b.align_to(t2, LEFT)
        self.play(FadeOut(t2), run_time=.2)
        self.play(FadeIn(t2b), run_time=.4)
        hold = pill('gold stays outside', UP_C, 26).next_to(gold, UP, buff=.16)
        self.play(FadeIn(hold, scale=1.1), run_time=.4)
        self.finish('02-three-signals', hold=3.0, in_update=['reward', 'KL'], outside=['gold'])


# ================================================================ 03 why pay for distance
class WhyPay(Loop):
    """With reward alone, all share flows to the answer the reward model overrates. With a price on KL, the
    same run stops near the reference."""

    B, U = -1.75, 3.65

    def lane(self, x0):
        return [x0 + i * 1.1 for i in range(6)]

    def bars(self, xs, p):
        cols = [REFC, REFC, REFC, UP_C, REFC, DOWN_C]
        return VGroup(*[bar(x, v, self.B, self.U, c, .72) for x, v, c in zip(xs, p, cols)])

    def read(self, cx, p):
        g = VGroup(label(f'reward {f3(p @ R6, d=2)}', 28, NEWC), label(f'gold {f3(p @ GOLD6, d=2)}', 28, UP_C)).arrange(RIGHT, buff=.5)
        return g.move_to([cx, -2.85, 0])

    def construct(self):
        self.begin('Why pay for distance')
        L, Rr = self.lane(-6.45), self.lane(.95)
        cl, cr = (L[0] + L[-1]) / 2, (Rr[0] + Rr[-1]) / 2
        hl = label('reward alone', 30, FOREGROUND).move_to([cl, 2.62, 0])
        hr = label('reward − β × KL, β = 1', 30, FOREGROUND).move_to([cr, 2.62, 0])
        bases = VGroup(*[Line([xs[0] - .5, self.B, 0], [xs[-1] + .5, self.B, 0], color=GRID, stroke_width=2) for xs in (L, Rr)])
        names = VGroup(*[label(n, 24, DOWN_C if n == '20 min' else FOREGROUND).move_to([x, self.B - .3, 0])
                         for xs in (L, Rr) for n, x in zip(NAMES6, xs)])
        bl, br = self.bars(L, REF6), self.bars(Rr, REF6)
        self.play(FadeIn(hl), FadeIn(hr), FadeIn(bases), FadeIn(names), run_time=.5)
        self.play(FadeIn(bl), FadeIn(br), run_time=.5)
        rl, rr = self.read(cl, REF6), self.read(cr, REF6)
        self.play(FadeIn(rl), FadeIn(rr), run_time=.4)
        self.mark('reference', reward=round(float(REF6 @ R6), 2), gold=round(float(REF6 @ GOLD6), 2), wait=.7)
        t1 = label('20 min: reward model says 3', 28, DOWN_C).move_to([6.75, 1.75, 0], aligned_edge=RIGHT)
        t2 = label('true value (gold) is −2', 28, DOWN_C).next_to(t1, DOWN, buff=.14).align_to(t1, RIGHT)
        self.play(FadeIn(t1), FadeIn(t2), run_time=.5)
        self.mark('overrated', proxy=3, gold=-2, wait=1.2)

        # reward alone: the share flows to the highest score
        seq = [optimum(REF6, R6, b) for b in np.geomspace(6, .06, 22)] + [np.array([0, 0, 0, 0, 0, 1.])]
        for p in seq:
            self.play(Transform(bl, self.bars(L, p)), run_time=.13, rate_func=linear)
            self.remove(rl); rl = self.read(cl, p); self.add(rl)
        top = label('100%', 28, DOWN_C).next_to(bl[5], UP, buff=.08)
        self.play(FadeIn(top), run_time=.3)
        self.mark('reward_alone', reward=3.0, gold=-2.0, share=1.0, wait=1.0)

        # with a price the same run stops early
        self.play(FadeOut(t1), FadeOut(t2), run_time=.3)
        final = optimum(REF6, R6, 1)
        for b in np.geomspace(6, 1, 14):
            p = optimum(REF6, R6, b)
            self.play(Transform(br, self.bars(Rr, p)), run_time=.14, rate_func=linear)
            self.remove(rr); rr = self.read(cr, p); self.add(rr)
        a = label(f'{final[3] * 100:.1f}%', 28, UP_C).next_to(br[3], UP, buff=.08)
        b_ = label(f'{final[5] * 100:.1f}%', 28, DOWN_C).next_to(br[5], UP, buff=.08)
        self.play(FadeIn(a), FadeIn(b_), run_time=.4)
        self.finish('03-why-pay-for-distance', hold=3.2, reward=round(float(final @ R6), 2), gold=round(float(final @ GOLD6), 2),
                    trick=round(float(final[5]), 3), best=round(float(final[3]), 3))


# ================================================================ 04 no move helps
class NoMoveHelps(Loop):
    """Net values level out at 0.808, where no small move raises the score. That condition gives the
    recipe: multiply, add, divide."""

    XS, Y0, U, BW = [-5.7, -4.2, -2.7, -1.2, .3], -.65, 1.3, 1.0
    PX = 1.75

    def nets(self, p):
        return R - np.log(p / REF)

    def bars(self, p):
        n = self.nets(p)
        return VGroup(*[bar(x, v, self.Y0, self.U, NEWC, self.BW) for x, v in zip(self.XS, n)])

    def vals(self, p):
        n = self.nets(p)
        return VGroup(*[label(f3(v), 26, NEWC).next_to([x, self.Y0 + v * self.U, 0], UP if v >= 0 else DOWN, buff=.1)
                        for x, v in zip(self.XS, n)])

    def sc(self, p):
        return label(f'score {score(p):.3f}', 32, FOREGROUND).move_to([self.PX, .1, 0], aligned_edge=LEFT)

    def chip(self, a, b, y, text):
        c = pill(text, UP_C, 24).move_to([self.XS[a], y, 0])
        self.play(FadeIn(c, scale=1.1), run_time=.3)
        self.play(c.animate.move_to([self.XS[b], y, 0]), run_time=.8)
        self.play(FadeOut(c), run_time=.25)

    def construct(self):
        self.begin('The best model: no move helps')
        axis = Line([-6.4, self.Y0, 0], [1.0, self.Y0, 0], color=GRID, stroke_width=2.5)
        names = VGroup(*[label(n, 26, FOREGROUND).move_to([x, -2.9, 0]) for n, x in zip(NAMES, self.XS)])
        h1 = label('net value', 32, NEWC).move_to([self.PX, 2.35, 0], aligned_edge=LEFT)
        h2 = label('reward − β × log-ratio', 28, NEWC).next_to(h1, DOWN, buff=.14).align_to(h1, LEFT)
        bs, vs = self.bars(SMALL), self.vals(SMALL)
        self.play(Create(axis), FadeIn(names), FadeIn(h1), FadeIn(h2), run_time=.5)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v > 0 else UP) for b, v in zip(bs, self.nets(SMALL))], lag_ratio=.12),
                  FadeIn(vs), run_time=.9)
        sc = self.sc(SMALL)
        self.play(FadeIn(sc), run_time=.3)
        self.mark('unequal', net=[round(float(v), 3) for v in self.nets(SMALL)], score=round(score(SMALL), 3), wait=1.0)

        # a chip of 0.01 moves up the net-value ladder
        self.chip(VAGUE, BEST, 2.05, '0.01')
        q = SMALL.copy(); q[VAGUE] -= .01; q[BEST] += .01
        gain = label(f'+{score(q) - score(SMALL):.3f}: not the top yet', 30, UP_C).move_to([self.PX, -.5, 0], aligned_edge=LEFT)
        s2 = label(f'score {score(q):.3f}', 32, FOREGROUND).move_to(sc, aligned_edge=LEFT)
        self.play(FadeTransform(sc, s2), FadeIn(gain), run_time=.4)
        sc = s2
        self.mark('chip', after=round(score(q), 3), gain=round(score(q) - score(SMALL), 3), wait=1.2)
        self.play(FadeOut(gain), FadeOut(vs), run_time=.3)

        # share keeps moving until the bars level
        for t in np.linspace(0, 1, 15)[1:]:
            p = SMALL ** (1 - t) * STAR ** t; p /= p.sum()
            self.play(Transform(bs, self.bars(p)), run_time=.15, rate_func=linear)
            self.remove(sc); sc = self.sc(p); self.add(sc)
        c = float(self.nets(STAR)[0])
        lvl = DashedLine([-6.4, self.Y0 + c * self.U, 0], [1.0, self.Y0 + c * self.U, 0], color=UP_C, stroke_width=3)
        v_all = label(f'all five: {c:.3f}', 30, UP_C).move_to([-6.3, 1.25, 0], aligned_edge=LEFT)
        self.play(Create(lvl), FadeIn(v_all), run_time=.5)
        self.mark('level', constant=round(c, 3), score=round(score(STAR), 3), net=[round(float(v), 3) for v in self.nets(STAR)], wait=.8)
        self.chip(VAGUE, BEST, 2.05, '0.01')
        q = STAR.copy(); q[VAGUE] -= .01; q[BEST] += .01
        q2 = STAR.copy(); q2[VAGUE] += .01; q2[BEST] -= .01
        lose = label('−0.0004 either way', 30, DOWN_C).move_to([self.PX, -.5, 0], aligned_edge=LEFT)
        done = pill('no move helps', UP_C, 30).move_to([self.PX + 1.55, -1.5, 0])
        self.play(FadeIn(lose), run_time=.35)
        self.play(FadeIn(done, scale=1.15), run_time=.4)
        self.mark('no_move', forward=round(score(q) - score(STAR), 4), back=round(score(q2) - score(STAR), 4), wait=1.4)
        self.clear_stage(keep=(), run_time=.4)

        # the recipe
        B, U = -2.25, 3.95
        names = VGroup(*[label(n, 26, FOREGROUND).move_to([x, B - .35, 0]) for n, x in zip(NAMES, self.XS)])
        base = Line([-6.4, B, 0], [1.0, B, 0], color=GRID, stroke_width=2.5)
        bars_ = lambda v, col=REFC: VGroup(*[bar(x, a, B, U, col, self.BW) for x, a in zip(self.XS, v)])
        tops = lambda v, col: VGroup(*[label(f3(a), 26, col).next_to([x, B + a * U, 0], UP, buff=.1) for x, a in zip(self.XS, v)])
        bs = bars_(REF); ts = tops(REF, REFC)
        self.play(Create(base), FadeIn(names), FadeIn(bs), FadeIn(ts), run_time=.5)
        step = lambda t, y, col: label(t, 30, col).move_to([self.PX, y, 0], aligned_edge=LEFT)
        s1 = step('1  multiply by e^reward', 2.3, NEWC)
        self.play(FadeIn(s1), run_time=.35)
        fac = VGroup(*[label(f'× {v:.3f}'.replace('.000', ''), 24, NEWC).next_to([x, B + a * U, 0], UP, buff=.5) for x, v, a in zip(self.XS, np.exp(R), REF)])
        self.play(FadeIn(fac), run_time=.4)
        self.wait(.7)
        w = REF * np.exp(R)
        tw = tops(w, NEWC)
        self.play(Transform(bs, bars_(w, NEWC)), FadeOut(fac), FadeOut(ts), run_time=.9)
        self.play(FadeIn(tw), run_time=.3)
        self.mark('multiply', weights=[round(float(v), 3) for v in w], wait=.8)
        s2 = step('2  add them up', 1.5, KLC)
        self.play(FadeIn(s2), run_time=.3)
        k, x = 1.95, self.PX
        seg = VGroup()
        cols = [REFC, NEWC, REFC, UP_C, DOWN_C]
        for a, col in zip(w, cols):
            r = Rectangle(width=a * k, height=.46, stroke_color=WHITE, stroke_width=2, fill_color=col, fill_opacity=.9)
            r.move_to([x, .75, 0], aligned_edge=LEFT); x += a * k; seg.add(r)
        self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in seg], lag_ratio=.4), run_time=.8)
        z = float(w.sum())
        zl = label(f'Z = {z:.3f}', 32, KLC).move_to([self.PX, .12, 0], aligned_edge=LEFT)
        self.play(FadeIn(zl), run_time=.3)
        self.mark('add', z=round(z, 3), wait=.9)
        s3 = step('3  divide by Z', -.55, UP_C)
        self.play(FadeIn(s3), run_time=.3)
        ts2 = VGroup(*[label(f'{a * 100:.1f}%', 28, UP_C).next_to([x_, B + a * U, 0], UP, buff=.1) for x_, a in zip(self.XS, STAR)])
        self.play(Transform(bs, bars_(STAR, UP_C)), FadeOut(tw), run_time=.9)
        self.play(FadeIn(ts2), run_time=.3)
        out = VGroup(label('best: 12% to 39.5%', 30, UP_C), label('wrong: 8% to 1.3%', 30, DOWN_C)).arrange(DOWN, buff=.14, aligned_edge=LEFT)
        out.move_to([self.PX, -1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(out), run_time=.4)
        self.finish('04-no-move-helps', hold=3.0, shares=[round(float(v), 3) for v in STAR], z=round(z, 3))


# ================================================================ 05 the implicit reward
class ImplicitReward(Loop):
    """At the best compromise every log-ratio equals the reward minus 0.808. A comparison cancels the
    constant, and DPO scores pairs with exactly this log-ratio."""

    XS, Y0, UNIT = [-6.0, -4.65, -3.3, -1.95, -.6], -.1, 1.05

    def construct(self):
        self.begin('The log-ratio is also a reward')
        p = optimum(REF, R, 1)
        lr = np.log(p / REF)
        c = float(np.log((REF * np.exp(R)).sum()))
        axis = Line([self.XS[0] - .65, self.Y0, 0], [self.XS[-1] + .65, self.Y0, 0], color=GRID, stroke_width=2.5)
        names = VGroup(*[label(n, 24, FOREGROUND).move_to([x, -3.2, 0]) for n, x in zip(NAMES, self.XS)])
        self.play(Create(axis), FadeIn(names), run_time=.5)
        rb = VGroup(*[bar(x, v, self.Y0, self.UNIT, UP_C, 1.0, .25) for x, v in zip(self.XS, R)])
        lb = VGroup(*[bar(x, v, self.Y0, self.UNIT, NEWC, .62) for x, v in zip(self.XS, lr)])
        ll = VGroup(*[label(f3(v, True), 24, NEWC).next_to(b, UP if v > 0 else DOWN, buff=.1) for v, b in zip(lr, lb)])
        head1 = label('log-ratio at the best compromise', 28, NEWC).move_to([-6.6, 2.3, 0], aligned_edge=LEFT)
        head2 = label('reward', 28, UP_C).move_to([-6.6, 1.8, 0], aligned_edge=LEFT)
        self.play(FadeIn(head1), LaggedStart(*[GrowFromEdge(b, DOWN if v > 0 else UP) for b, v in zip(lb, lr)], lag_ratio=.12), run_time=.9)
        self.play(FadeIn(ll), run_time=.4)
        self.mark('log_ratio', values=[round(float(x), 3) for x in lr], wait=.8)
        self.play(FadeIn(head2), FadeIn(rb), run_time=.5)
        self.mark('reward', values=R.tolist(), wait=.8)
        lift = pill(f'+ {f3(c)}', NEWC, 30).move_to([2.0, 1.75, 0])
        self.play(FadeIn(lift, scale=1.15), FadeOut(ll), run_time=.45)
        lifted = VGroup(*[bar(x, v + c, self.Y0, self.UNIT, NEWC, .62) for x, v in zip(self.XS, lr)])
        self.play(Transform(lb, lifted), run_time=1.0)
        same = label('every log-ratio = reward − 0.808', 30, FOREGROUND).move_to([.55, .85, 0], aligned_edge=LEFT)
        self.play(FadeIn(same), run_time=.45)
        self.mark('match', constant=round(c, 3), diffs=[round(float(a - b), 3) for a, b in zip(lr, R)], wait=1.3)

        # a comparison cancels the constant
        self.play(FadeOut(lift), run_time=.25)
        sig1 = 1 / (1 + math.exp(-1))
        rows = VGroup(label('best vs 10 min', 30, FOREGROUND),
                      label('reward:  2 − 1 = 1', 30, UP_C),
                      label(f'log-ratio:  {f3(lr[3])} − {f3(lr[1])} = 1', 30, NEWC),
                      label(f'P(best wins) = σ(1) = {sig1:.3f}', 30, KLC)).arrange(DOWN, buff=.2, aligned_edge=LEFT)
        rows.move_to([.55, -.45, 0], aligned_edge=LEFT).shift(DOWN * .35)
        self.play(same.animate.shift(UP * 1.0), run_time=.3)
        for r in rows:
            self.play(FadeIn(r, shift=DOWN * .08), run_time=.45)
        self.mark('cancel', gap=round(float(lr[3] - lr[1]), 3), p=round(sig1, 3), wait=1.3)
        self.play(FadeOut(rows[1:]), FadeOut(same), run_time=.3)
        l0, l1 = math.log(2), -math.log(sig1)
        title = label('DPO loss on this pair', 30, KLC).move_to(rows[0], aligned_edge=LEFT)
        self.play(FadeOut(rows[0]), FadeIn(title), run_time=.3)
        y = -1.45
        bar_ = hbar(.55, l0, y, 5, KLC, .5)
        t = label(f'start {l0:.3f}', 30, KLC).next_to(bar_, UP, buff=.14).align_to(bar_, LEFT)
        self.play(GrowFromEdge(bar_, LEFT), FadeIn(t), run_time=.6)
        self.wait(.6)
        b2 = hbar(.55, l1, y, 5, UP_C, .5)
        t2 = label(f'best compromise {l1:.3f}', 30, UP_C).move_to(t, aligned_edge=LEFT)
        self.play(Transform(bar_, b2), FadeTransform(t, t2), run_time=.9)
        self.finish('05-implicit-reward', hold=3.0, constant=round(c, 3), p=round(sig1, 3), loss=[round(l0, 3), round(l1, 3)])


# ================================================================ 06 the loss on one pair
class LossOnPair(Loop):
    """The gap between two scores becomes a chance and a loss: large when wrong, small when right, never 0."""

    PX = .95

    def read(self, m):
        g = VGroup(label(f'gap {f3(m, d=2)}', 30, FOREGROUND), label(f'chance {sig(m):.3f}', 30, NEWC), label(f'loss {bt(m):.3f}', 30, KLC))
        return g.arrange(DOWN, buff=.18, aligned_edge=LEFT).move_to([self.PX, .1, 0], aligned_edge=LEFT)

    def construct(self):
        self.begin('The DPO loss on one pair')
        ax = LossAxes()
        c1 = label('chosen: best answer', 30, UP_C).move_to([self.PX, 2.35, 0], aligned_edge=LEFT)
        c2 = label('rejected: About 10 minutes', 30, DOWN_C).next_to(c1, DOWN, buff=.14).align_to(c1, LEFT)
        self.play(FadeIn(c1), FadeIn(c2), run_time=.5)
        self.play(FadeIn(ax.fixed), run_time=.5)
        self.mark('pair', wait=.6)
        t = ValueTracker(-3)
        curve = always_redraw(lambda: ax.curve(-3, max(t.get_value(), -2.95)))
        dot = always_redraw(lambda: Dot(ax.pt(t.get_value()), radius=.11, color=KLC))
        r = self.read(-3)
        self.add(curve, dot); self.play(FadeIn(r), run_time=.3)
        self.mark('wrong_order', gap=-3, loss=round(bt(-3), 3), wait=.7)
        for a in np.linspace(-3, 3, 25)[1:]:
            self.play(t.animate.set_value(a), run_time=.13, rate_func=linear)
            self.remove(r); r = self.read(float(a)); self.add(r)
        self.mark('right_order', gap=3, loss=round(bt(3), 3), wait=.7)
        self.remove(dot, r); self.remove(curve)
        self.add(ax.curve())
        pts = [('start', 0., REFC), ('small move', math.log(.22 / .12), NEWC), ('best model', 1., UP_C)]
        rows = VGroup()
        dots = VGroup(*[Dot(ax.pt(m), radius=.11, color=c) for _, m, c in pts])
        for i, (n, m, c) in enumerate(pts):
            row = label(f'{n}: loss {bt(m):.3f}', 30, c).move_to([self.PX, .55 - i * .62, 0], aligned_edge=LEFT)
            rows.add(row)
            self.play(FadeIn(dots[i], scale=1.5), FadeIn(row), run_time=.55)
            self.wait(.35)
        self.mark('three_points', loss=[round(bt(m), 3) for _, m, _ in pts], gap=[round(m, 3) for _, m, _ in pts], wait=.7)
        n = pill('the loss never reaches 0', KLC, 28).move_to([self.PX + 2.45, -1.75, 0])
        self.play(FadeIn(n, scale=1.1), run_time=.4)
        self.finish('06-loss-on-one-pair', hold=3.0, loss=[.693, .435, .313], chance=[.5, round(sig(math.log(.22 / .12)), 3), .731],
                    ends=[round(bt(-3), 3), round(bt(3), 3)])


# ================================================================ 07 the push
class ThePush(Loop):
    """A nudge test finds the slope; its size is the push: hard on wrong pairs, faint on won pairs. Trained
    step by step, one pair slows down and never stops."""

    PX = 1.0

    def construct(self):
        self.begin('The push')
        ax = LossAxes()
        self.play(FadeIn(ax.fixed), Create(ax.curve()), run_time=.8)
        dot = Dot(ax.pt(0), radius=.11, color=NEWC)
        n1 = label('nudge the gap by 0.01', 30, FOREGROUND).move_to([self.PX, 2.35, 0], aligned_edge=LEFT)
        n2 = label(f'loss {bt(0):.4f} to {bt(.01):.4f}', 30, KLC).next_to(n1, DOWN, buff=.16).align_to(n1, LEFT)
        n3 = label('slope about −0.5', 30, NEWC).next_to(n2, DOWN, buff=.16).align_to(n1, LEFT)
        self.play(FadeIn(dot, scale=1.5), FadeIn(n1), run_time=.4)
        self.play(FadeIn(n2), run_time=.4)
        tan = ax.tangent(0)
        self.play(Create(tan), FadeIn(n3), run_time=.5)
        self.mark('nudge', before=round(bt(0), 4), after=round(bt(.01), 4), wait=1.1)
        self.play(FadeOut(n1), FadeOut(n2), FadeOut(n3), run_time=.3)

        # the slope's size is the push
        rows = [('gap −3, wrong order', -3., UP_C), ('gap 0', 0., UP_C), ('gap 1', 1., UP_C), ('gap 3, already won', 3., UP_C)]
        head = label('the push', 32, UP_C).move_to([self.PX, 2.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), run_time=.3)
        shown = VGroup()
        for i, (t, m, c) in enumerate(rows):
            y = 1.7 - i * 1.12
            tx = label(t, 28, FOREGROUND).move_to([self.PX, y, 0], aligned_edge=LEFT)
            ar = harrow(self.PX, self.PX + max(sig(-m) * 3.9, .42), y - .46, c, 6, .17)
            v = label(f'{sig(-m):.3f}', 28, c).next_to(ar, RIGHT, buff=.14)
            self.play(dot.animate.move_to(ax.pt(m)), Transform(tan, ax.tangent(m)), run_time=.5)
            self.play(FadeIn(tx), GrowArrow(ar), FadeIn(v), run_time=.45)
            shown.add(tx, ar, v)
            self.wait(.35)
        self.mark('sizes', push=[round(sig(-m), 3) for _, m, _ in rows], wait=1.4)
        self.play(FadeOut(shown), FadeOut(head), FadeOut(tan), run_time=.35)

        # one pair, trained step by step
        h2 = label('one pair, learning rate 1', 30, FOREGROUND).move_to([self.PX, 2.35, 0], aligned_edge=LEFT)
        self.play(dot.animate.move_to(ax.pt(0)), FadeIn(h2), run_time=.4)
        m, gaps, losses, pushes = 0., [], [], []
        for i in range(5):
            y = 1.7 - i * .82
            row = label(f'gap {m:.3f}   loss {bt(m):.3f}', 28, KLC).move_to([self.PX, y, 0], aligned_edge=LEFT)
            ar = harrow(self.PX, self.PX + sig(-m) * 3.9, y - .34, UP_C, 5, .15)
            self.play(FadeIn(row), GrowArrow(ar), run_time=.42)
            gaps.append(round(m, 3)); losses.append(round(bt(m), 3)); pushes.append(round(sig(-m), 3))
            if i < 4:
                m += sig(-m)
                self.play(dot.animate.move_to(ax.pt(m)), run_time=.42)
        end = pill('steps shrink and never stop', UP_C, 26).move_to([self.PX + 2.3, -2.75, 0])
        self.play(FadeIn(end, scale=1.1), run_time=.4)
        self.finish('07-the-push', hold=3.0, gaps=gaps, losses=losses, pushes=pushes)


# ================================================================ 08 where the probability comes from
class WhereFrom(Loop):
    """One step moves only the pair's two logits, and every share changes. The wrong answer rises, because
    no pair mentions it, and the result is not the target."""

    XS, B, U, W = [-5.4, -3.2, -1.0, 1.2, 3.4], -2.35, 7.2, 1.15

    def bars(self, p):
        return VGroup(*[bar(x, v, self.B, self.U, c, self.W) for x, v, c in
                        zip(self.XS, p, [REFC, DOWN_C, REFC, UP_C, REFC])])

    def pcts(self, p, cols=None):
        cols = cols or [REFC, DOWN_C, REFC, UP_C, REFC]
        return VGroup(*[label(f'{v * 100:.1f}%', 28, c).next_to([x, self.B + v * self.U, 0], UP, buff=.1)
                        for x, v, c in zip(self.XS, p, cols)])

    def construct(self):
        self.begin('Where the probability comes from')
        base = Line([-6.3, self.B, 0], [4.3, self.B, 0], color=GRID, stroke_width=2)
        names = VGroup(*[label(n, 28, FOREGROUND).move_to([x, self.B - .36, 0]) for n, x in zip(NAMES, self.XS)])
        bs, ps = self.bars(REF), self.pcts(REF)
        self.play(FadeIn(base), FadeIn(names), FadeIn(bs), FadeIn(ps), run_time=.6)
        c1 = label('chosen: best', 30, UP_C).move_to([.4, 2.42, 0], aligned_edge=LEFT)
        c2 = label('rejected: 10 min', 30, DOWN_C).next_to(c1, RIGHT, buff=.5)
        self.play(FadeIn(c1), FadeIn(c2), run_time=.4)
        self.mark('reference', shares=REF.tolist(), wait=.7)
        y1 = self.B + REF[BEST] * self.U + .6
        y2 = self.B + REF[TEN] * self.U + .6
        up = varrow(self.XS[BEST], y1, y1 + .95, UP_C, 7, .22)
        dn = varrow(self.XS[TEN], y2 + .95, y2, DOWN_C, 7, .22)
        t1 = label('+0.5', 28, UP_C).next_to(up, RIGHT, buff=.14)
        t2 = label('−0.5', 28, DOWN_C).next_to(dn, RIGHT, buff=.14)
        self.play(GrowArrow(up), GrowArrow(dn), FadeIn(t1), FadeIn(t2), run_time=.6)
        none = label('the other three logits get no push', 28, MUTED).move_to([-6.5, 2.42, 0], aligned_edge=LEFT)
        self.play(FadeIn(none), run_time=.4)
        self.mark('push', logits=[0, -.5, 0, .5, 0], wait=1.2)
        self.play(FadeOut(VGroup(up, dn, t1, t2, none)), FadeOut(ps), run_time=.3)
        th = np.log(REF).copy(); th[BEST] += .5; th[TEN] -= .5
        one = np.exp(th) / np.exp(th).sum()
        self.play(Transform(bs, self.bars(one)), run_time=1.0)
        cols = [UP_C, DOWN_C, UP_C, UP_C, ACCENT]
        ps = self.pcts(one, cols)
        self.play(FadeIn(ps), run_time=.4)
        self.mark('stepped', shares=[round(float(v), 3) for v in one], wait=.8)
        w = label('the wrong answer rose: 8.0% to 8.2%', 30, DOWN_C).move_to([-6.5, 2.42, 0], aligned_edge=LEFT)
        self.play(FadeOut(c1), FadeOut(c2), FadeIn(w), run_time=.4)
        box = SurroundingRectangle(VGroup(ps[WRONG], names[WRONG]), color=DOWN_C, buff=.12, corner_radius=.1)
        self.play(Create(box), run_time=.5)
        self.mark('wrong_rose', before=.08, after=round(float(one[WRONG]), 3), wait=1.5)
        self.play(FadeOut(box), FadeOut(w), run_time=.3)
        gs = VGroup(*[ghost(x, v, self.B, self.U, NEWC, self.W + .14) for x, v in zip(self.XS, STAR)])
        leg = label('dashed: the best model', 30, NEWC).move_to([-6.5, 2.42, 0], aligned_edge=LEFT)
        self.play(Create(gs), FadeIn(leg), run_time=.8)
        gap = math.log(one[BEST] / REF[BEST]) - math.log(one[TEN] / REF[TEN])
        same = label('same gap: 1', 30, FOREGROUND).move_to([3.0, 1.75, 0], aligned_edge=LEFT)
        diff = label('different shares', 30, FOREGROUND).next_to(same, DOWN, buff=.14).align_to(same, LEFT)
        self.play(FadeIn(same), FadeIn(diff), run_time=.4)
        self.finish('08-where-probability-comes-from', hold=3.2, gap=round(gap, 3), shares=[round(float(v), 3) for v in one])


# ================================================================ 09 the ideal test
class IdealTest(Loop):
    """Who-won records stream in with the rewards hidden. The bars move from the reference to the target."""

    XS, B, U, W = [-6.1, -4.95, -3.8, -2.65, -1.5], -2.3, 6.9, .86
    PX = .35

    def bars(self, p):
        return VGroup(*[bar(x, v, self.B, self.U, NEWC, self.W) for x, v in zip(self.XS, p)])

    def construct(self):
        self.begin('The ideal test')
        base = Line([-6.65, self.B, 0], [-.95, self.B, 0], color=GRID, stroke_width=2)
        names = VGroup(*[label(n, 24, FOREGROUND).move_to([x, self.B - .32, 0]) for n, x in zip(NAMES, self.XS)])
        gs = VGroup(*[ghost(x, v, self.B, self.U, UP_C, self.W + .12) for x, v in zip(self.XS, STAR)])
        bs = self.bars(REF)
        leg = label('dashed: target', 28, UP_C).move_to([-6.6, 2.45, 0], aligned_edge=LEFT)
        hide = pill('rewards hidden from training', REFC, 26).move_to([self.PX, 2.45, 0], aligned_edge=LEFT)
        self.play(FadeIn(base), FadeIn(names), FadeIn(bs), run_time=.5)
        self.play(Create(gs), FadeIn(leg), run_time=.6)
        self.play(FadeIn(hide, scale=1.08), run_time=.35)
        tv = lambda p: .5 * float(np.abs(p - STAR).sum())
        h = label('who won', 28, MUTED).move_to([self.PX, 1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(h), run_time=.25)
        self.mark('start', distance=round(tv(REF), 2), wait=.4)
        rows = []
        for k, (w, l) in enumerate(FIRST_PAIRS):
            row = VGroup(label(NAMES[w], 28, UP_C), label('beat', 28, MUTED), label(NAMES[l], 28, DOWN_C)).arrange(RIGHT, buff=.16)
            row.move_to([self.PX, 1.2, 0], aligned_edge=LEFT)
            shift = [r.animate.shift(DOWN * .52) for r in rows]
            drop = [FadeOut(rows.pop(0))] if len(rows) == 4 else []
            self.play(FadeIn(row, shift=DOWN * .1), *shift, *drop, run_time=.3)
            rows.append(row)
        h2 = label('1,000 pairs, nothing else', 28, FOREGROUND).move_to([self.PX, 1.75, 0], aligned_edge=LEFT)
        self.play(FadeOut(h), FadeIn(h2), run_time=.3)
        self.mark('records', pairs=1000, wait=1.0)

        # training on those records: the run's own shares, step by step
        self.play(FadeOut(VGroup(*rows)), FadeOut(h2), run_time=.3)
        m_y = -2.05
        mt = lambda p: label(f'distance to target {tv(p):.2f}', 30, KLC).move_to([self.PX, m_y + .55, 0], aligned_edge=LEFT)
        mb = lambda p: hbar(self.PX, tv(p), m_y, 13.5, KLC, .44)
        t, mbar = mt(REF), mb(REF)
        self.play(FadeIn(t), GrowFromEdge(mbar, LEFT), run_time=.4)
        self.mark('distance', distance=round(tv(REF), 2), wait=.5)
        for sh in DPO_PATH[1:]:
            p = np.array(sh)
            self.play(Transform(bs, self.bars(p)), Transform(mbar, mb(p)), run_time=.3, rate_func=linear)
            self.remove(t); t = mt(p); self.add(t)
        n = label('after training on 1,000 pairs', 30, NEWC).move_to([self.PX, 1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(n), run_time=.3)
        got = VGroup(*[label(f'{v * 100:.1f}', 26, NEWC).next_to([x, self.B + max(v, s) * self.U, 0], UP, buff=.1)
                       for x, v, s in zip(self.XS, DPO_RUN, STAR)])
        self.play(FadeIn(got), run_time=.3)
        self.mark('after_run', shares=DPO_RUN.tolist(), distance=round(tv(DPO_RUN), 2), wait=1.8)
        self.play(FadeOut(n), FadeOut(t), FadeOut(mbar), FadeOut(hide), run_time=.3)

        # more pairs, closer
        h3b = label('more pairs, closer to the target', 30, FOREGROUND).move_to([self.PX, 2.45, 0], aligned_edge=LEFT)
        h3 = label('average of 8 simulated datasets', 26, MUTED).next_to(h3b, DOWN, buff=.14).align_to(h3b, LEFT)
        self.play(FadeIn(h3b), FadeIn(h3), run_time=.4)
        for i, (nm, d) in enumerate(AVG_DISTANCE):
            y = 1.05 - i * .95
            lab = label(f'{nm} pairs', 28, FOREGROUND).move_to([self.PX, y, 0], aligned_edge=LEFT)
            b = hbar(self.PX + 2.4, d, y, 8.5, KLC, .44)
            v = label(f'{d:.2f}', 28, KLC).next_to(b, RIGHT, buff=.14)
            self.play(FadeIn(lab), GrowFromEdge(b, LEFT), FadeIn(v), run_time=.5)
        self.finish('09-ideal-test', hold=3.0, distance=[round(tv(REF), 2), round(tv(DPO_RUN), 2)], average=[d for _, d in AVG_DISTANCE],
                    pairs_shown=len(FIRST_PAIRS))


# ================================================================ 10 answers to tokens
class AnswersToTokens(Loop):
    """A reply's log-probability is a sum over its answer tokens and the end-of-turn token; the prompt is
    not counted. Raw totals mostly measure length, so each answer is compared with its own reference value."""

    def tok(self, text, col, opacity=.1):
        t = label(text, 28, col)
        box = RoundedRectangle(width=max(t.width + .3, .62), height=.66, corner_radius=.1, stroke_color=col,
                               stroke_width=2, fill_color=col, fill_opacity=opacity)
        return VGroup(box, t.move_to(box))

    def construct(self):
        self.begin('From answers to tokens')
        prompt = VGroup(*[self.tok(t, MUTED, .05) for t in ['How', 'long', 'should', 'I', 'boil', 'an', 'egg', '?']]).arrange(RIGHT, buff=.1)
        prompt.move_to([-6.5, 1.95, 0], aligned_edge=LEFT)
        pl = label('prompt: context, not counted', 28, MUTED).next_to(prompt, RIGHT, buff=.3)
        self.play(FadeIn(prompt), run_time=.5)
        self.play(FadeIn(pl), run_time=.3)
        self.mark('prompt', wait=.7)
        ans = ['About', '10', 'minutes', '.', 'end of turn']
        lps = [-1.2, -.4, -.6, -.3, -.5]
        toks = VGroup(*[self.tok(t, UP_C if i < 4 else NEWC, .12) for i, t in enumerate(ans)]).arrange(RIGHT, buff=.32)
        toks.move_to([-6.5, .45, 0], aligned_edge=LEFT)
        vs = VGroup(*[label(f3(v, d=1), 28, KLC).next_to(t, DOWN, buff=.18) for v, t in zip(lps, toks)])
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * .08) for t in toks], lag_ratio=.15), run_time=.7)
        stop = label('stopping is part of the answer', 28, NEWC).next_to(toks[-1], RIGHT, buff=.28)
        self.play(FadeIn(stop), run_time=.3)
        self.mark('answer', wait=.5)
        self.play(LaggedStart(*[FadeIn(v) for v in vs], lag_ratio=.15), run_time=.7)
        total = sum(lps)
        s = label(f'sum of log-probabilities: {f3(total, d=1)}', 32, KLC).move_to([-6.5, -1.5, 0], aligned_edge=LEFT)
        self.play(FadeIn(s), run_time=.4)
        self.mark('sum', total=round(total, 1), wait=1.2)
        self.clear_stage(run_time=.4)

        # raw totals follow length
        X0, U = -3.6, .09
        head = label('raw totals mostly measure length', 32, FOREGROUND).move_to([-6.5, 2.45, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), run_time=.35)
        rows = [('40-token reply', -40, -38, 1.45), ('60-token reply', -60, -59, -.05)]
        drawn = []
        for n, r0, r1, y in rows:
            t = label(n, 28, FOREGROUND).move_to([-6.5, y, 0], aligned_edge=LEFT)
            b = hbar(X0, -r0, y, U, REFC, .46)
            v = label(str(r0).replace('-', '−'), 28, REFC).next_to(b, RIGHT, buff=.14)
            self.play(FadeIn(t), GrowFromEdge(b, LEFT), FadeIn(v), run_time=.5)
            drawn.append((b, v))
        self.mark('raw', totals=[-40, -60], wait=1.3)
        h2 = label('compare each answer with its own reference value', 32, FOREGROUND).move_to(head, aligned_edge=LEFT)
        self.play(FadeOut(head), FadeIn(h2), run_time=.35)
        changes = []
        for (n, r0, r1, y), (b, v) in zip(rows, drawn):
            b2 = hbar(X0, -r1, y - .62, U, NEWC, .46)
            v2 = label(f'trained {r1}'.replace('-', '−'), 28, NEWC).next_to(b2, RIGHT, buff=.14)
            v0 = label(f'reference {r0}'.replace('-', '−'), 28, REFC).move_to(v, aligned_edge=LEFT)
            self.play(FadeTransform(v, v0), GrowFromEdge(b2, LEFT), FadeIn(v2), run_time=.55)
            c = pill(f'change +{r1 - r0}', UP_C, 28).move_to([5.7, y - .3, 0])
            self.play(FadeIn(c, scale=1.12), run_time=.35)
            changes.append(r1 - r0)
        c = label('the length is mostly subtracted away', 30, UP_C).move_to([-6.5, -2.55, 0], aligned_edge=LEFT)
        self.play(FadeIn(c), run_time=.35)
        self.finish('10-answers-to-tokens', hold=3.0, changes=changes, total=round(total, 1))


# ================================================================ 11 what to watch
class WhatToWatch(Loop):
    """Four panels of invented curves: pair accuracy from 50%, chosen and rejected, a run where both fall, and
    KL measured from samples. No measured values."""

    W, H = 5.0, 1.72

    def frame(self, x, y, title, mid=True):
        tt = label(title, 28, FOREGROUND).move_to([x, y + self.H + .28, 0], aligned_edge=LEFT)
        ya = Line([x, y, 0], [x, y + self.H, 0], color=GRID, stroke_width=2.5)
        xa = Line([x, y + (self.H / 2 if mid else 0), 0], [x + self.W, y + (self.H / 2 if mid else 0), 0], color=GRID, stroke_width=2.5)
        return VGroup(tt, ya, xa)

    def line(self, x, y, f, col):
        return ParametricFunction(lambda t: np.array([x + t * self.W, y + f(t) * self.H, 0]), t_range=[0, 1, .02],
                                  color=col, stroke_width=5)

    def construct(self):
        self.begin('What to watch on a real model')
        x1, x2, y1, y2 = -6.45, .75, .4, -2.6

        # 1 accuracy: both start at 50%
        f1 = self.frame(x1, y1, 'pair accuracy: starts at 50%', mid=False)
        tr = self.line(x1, y1, lambda t: .95 * (1 - math.exp(-4 * t)), NEWC)
        he = self.line(x1, y1, lambda t: .42 * (1 - math.exp(-3.5 * t)), UP_C)
        self.play(FadeIn(f1), run_time=.4)
        self.play(Create(tr), Create(he), run_time=1.3, rate_func=linear)
        l1 = label('training', 26, NEWC).next_to([x1 + self.W, y1 + .9 * self.H, 0], DOWN, buff=.12).shift(LEFT * .75)
        l2 = label('held-out', 26, UP_C).next_to([x1 + self.W, y1 + .42 * self.H, 0], DOWN, buff=.1).shift(LEFT * .7)
        self.play(FadeIn(l1), FadeIn(l2), run_time=.3)
        self.mark('accuracy', wait=1.0)

        # 2 healthy: rise, then split
        f2 = self.frame(x2, y1, 'chosen and rejected, change')
        mid = .5
        ch = self.line(x2, y1, lambda t: mid + .42 * (1 - math.exp(-4 * t)), UP_C)
        rj = self.line(x2, y1, lambda t: mid + 1.5 * t * (1 - 1.25 * t), DOWN_C)
        self.play(FadeIn(f2), run_time=.3)
        self.play(Create(ch), Create(rj), run_time=1.3, rate_func=linear)
        l3 = label('chosen', 26, UP_C).next_to([x2 + self.W, y1 + .92 * self.H, 0], DOWN, buff=.12).shift(LEFT * .6)
        l4 = label('rejected', 26, DOWN_C).move_to([x2 + 1.9, y1 + .2 * self.H, 0], aligned_edge=RIGHT)
        self.play(FadeIn(l3), FadeIn(l4), run_time=.3)
        self.mark('both_rise_then_split', wait=1.0)

        # 3 displacement: both fall
        f3_ = self.frame(x1, y2, 'a run where both fall')
        c3 = self.line(x1, y2, lambda t: mid - .22 * t, UP_C)
        r3 = self.line(x1, y2, lambda t: mid - .48 * t, DOWN_C)
        self.play(FadeIn(f3_), run_time=.3)
        self.play(Create(c3), Create(r3), run_time=1.2, rate_func=linear)
        l5 = label('gap wider, chosen lower', 26, DOWN_C).move_to([x1 + .3, y2 + .86 * self.H, 0], aligned_edge=LEFT)
        self.play(FadeIn(l5), run_time=.3)
        self.mark('both_fall', wait=1.0)

        # 4 KL
        f4 = self.frame(x2, y2, 'KL, measured from samples', mid=False)
        k1 = self.line(x2, y2, lambda t: .92 * (1 - math.exp(-2.2 * t)) / (1 - math.exp(-2.2)), KLC)
        k2 = self.line(x2, y2, lambda t: .46 * (1 - math.exp(-2.2 * t)) / (1 - math.exp(-2.2)), REFC)
        self.play(FadeIn(f4), run_time=.3)
        self.play(Create(k1), Create(k2), run_time=1.2, rate_func=linear)
        l6 = label('small β', 26, KLC).move_to([x2 + 4.1, y2 + .66 * self.H, 0], aligned_edge=RIGHT)
        l7 = label('large β', 26, REFC).move_to([x2 + 3.95, y2 + .23 * self.H, 0], aligned_edge=LEFT)
        self.play(FadeIn(l6), FadeIn(l7), run_time=.3)
        self.finish('11-what-to-watch', hold=3.4, panels=4, measured_values=0)
