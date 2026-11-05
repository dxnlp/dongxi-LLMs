"""KL article GIF set (CAND-ANIM-058 as two clips, 059, 065, 060, 061, 054, 055, 063, 037, 064, 066).

One running example, the egg question, shared with the article. Shares, rewards and gold values are
illustrative; every ratio, log, sum and optimum on them is exact (they match the article's
check_numbers.py). The sampled clips use one seeded draw. No numbers from our own runs appear.

Cast, shared by every clip:

- slate: the reference
- blue: the new model
- green: raised answers, positive terms, gold
- amber: lowered answers, negative terms, and the illustration badge
- violet: KL

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
REF = np.array([.40, .25, .15, .12, .08])
SMALL = np.array([.30, .25, .15, .22, .08])
AWAY = np.array([.50, .25, .15, .02, .08])
R = np.array([0, 1, .5, 2, -1.])
REF6 = np.array([.40, .25, .15, .12, .075, .005])
R6 = np.array([0, 1, .5, 2, -1, 3.])
GOLD6 = np.array([0, 1, .5, 2, -1, -2.])


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def optimum(ref, r, beta):
    w = ref * np.exp(r / beta)
    return w / w.sum()


def f3(v, sign=False, d=3):
    s = f'{v:+.{d}f}' if sign else f'{v:.{d}f}'
    return s.replace('-', '−')


def samples():
    """The same seeded draw as the article's check_numbers.py."""
    rng = np.random.default_rng(0)
    return np.log(SMALL / REF)[rng.choice(5, size=10_000, p=SMALL)]


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


# ================================================================ 01 the two splits
class TwoSplits(Loop):
    """The reference's five shares; 0.10 moves from the vague answer to the best answer; how much
    did the model change?"""

    XS, BASE_Y, UNIT = [-5.2, -2.9, -.6, 1.7, 4.0], -2.45, 8.4

    def pct(self, x, v, col):
        return label(f'{v * 100:.0f}%', 28, col).next_to([x, self.BASE_Y + v * self.UNIT, 0], UP, buff=.12)

    def construct(self):
        self.begin('One question, five answers', badge='illustrative shares')
        q = label('How long should I boil an egg?', 30, FOREGROUND)
        box = RoundedRectangle(width=q.width + .5, height=q.height + .32, corner_radius=.16, stroke_color=REFC,
                               stroke_width=2.2, fill_color=REFC, fill_opacity=.08)
        bubble = VGroup(box, q.move_to(box)).move_to([0, 2.42, 0]).to_edge(LEFT, buff=.55)
        self.play(FadeIn(bubble, shift=UP * .15), run_time=.5)
        base = Line([-6.1, self.BASE_Y, 0], [4.9, self.BASE_Y, 0], color=GRID, stroke_width=2)
        names = VGroup(*[label(n, 28, FOREGROUND).move_to([x, self.BASE_Y - .36, 0]) for n, x in zip(NAMES, self.XS)])
        self.play(FadeIn(base), FadeIn(names), run_time=.4)
        bars = VGroup(*[bar(x, v, self.BASE_Y, self.UNIT, REFC) for x, v in zip(self.XS, REF)])
        pcts = VGroup(*[self.pct(x, v, REFC) for x, v in zip(self.XS, REF)])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.15), run_time=1)
        self.play(FadeIn(pcts), run_time=.4)
        who = pill('reference', REFC, 28).move_to([3.6, 2.42, 0])
        self.play(FadeIn(who, scale=1.15), run_time=.35)
        self.mark('reference', shares=REF.tolist(), total=round(float(REF.sum()), 6), wait=.8)

        # 0.10 lifts off the vague answer and lands on the best answer
        ghosts = VGroup(*[DashedLine([self.XS[i] - .6, self.BASE_Y + REF[i] * self.UNIT, 0],
                                     [self.XS[i] + .6, self.BASE_Y + REF[i] * self.UNIT, 0],
                                     color=REFC, stroke_width=2.5) for i in (0, 3)])
        top = self.BASE_Y + REF[0] * self.UNIT
        chunk = Rectangle(width=.9, height=.10 * self.UNIT, stroke_width=0, fill_color=NEWC, fill_opacity=.9)
        chunk.move_to([self.XS[0], top - .05 * self.UNIT, 0])
        low = bar(self.XS[0], SMALL[0], self.BASE_Y, self.UNIT, REFC)
        self.add(chunk)
        self.play(FadeIn(ghosts), Transform(bars[0], low), FadeOut(pcts[0]), run_time=.5)
        tag = label('0.10', 28, NEWC).next_to(chunk, UP, buff=.1)
        self.play(FadeIn(tag), run_time=.3)
        lift = top + .32
        self.play(chunk.animate.move_to([self.XS[0], lift, 0]), tag.animate.move_to([self.XS[0], lift + .68, 0]), run_time=.4)
        self.play(chunk.animate.move_to([self.XS[3], lift, 0]), tag.animate.move_to([self.XS[3], lift + .68, 0]), run_time=1)
        land = self.BASE_Y + (REF[3] + .05) * self.UNIT
        self.play(chunk.animate.move_to([self.XS[3], land, 0]), FadeOut(tag), FadeOut(pcts[3]), run_time=.5)
        new_pcts = [self.pct(self.XS[0], SMALL[0], DOWN_C), self.pct(self.XS[3], SMALL[3], UP_C)]
        new_pcts[1].shift(UP * .04)
        who2 = pill('new model', NEWC, 28).move_to(who)
        self.play(FadeIn(new_pcts[0]), FadeIn(new_pcts[1]), FadeTransform(who, who2), run_time=.5)
        self.mark('moved', shares=SMALL.tolist(), wait=.9)

        q2 = label('How much did it change?', 32, KLC)
        qbox = pill('one number', KLC, 30)
        ask = VGroup(q2, qbox).arrange(RIGHT, buff=.35).move_to([1.2, 1.25, 0])
        self.play(FadeIn(q2, shift=UP * .1), run_time=.5)
        self.play(FadeIn(qbox, scale=1.15), run_time=.4)
        self.finish('01-two-splits', hold=2.6, shares=SMALL.tolist(), moved=.10)


# ================================================================ 02 why a log
class WhyLog(Loop):
    """On the ratio scale doubling and halving have unequal sizes. The log gives them the same size,
    turns multiplication into addition, and falls without limit near 0."""

    X0, Y0, SX, SY = -5.95, .55, 1.62, 1.16

    def pt(self, r, y=None):
        return np.array([self.X0 + r * self.SX, self.Y0 + (math.log(r) if y is None else y) * self.SY, 0])

    def construct(self):
        self.begin('Why a log')
        xa = Line(self.pt(.02, 0), self.pt(4.45, 0), color=GRID, stroke_width=3)
        ya = Line([self.X0, self.Y0 - 2.5 * self.SY, 0], [self.X0, self.Y0 + 1.72 * self.SY, 0], color=GRID, stroke_width=3)
        ticks = VGroup(*[VGroup(Line(self.pt(r, -.05), self.pt(r, .05), color=MUTED, stroke_width=2),
                                label(t, 24, MUTED).next_to(self.pt(r, 0), UP if r <= 1 else DOWN, buff=.14))
                         for r, t in [(.5, '0.5'), (1, '1'), (2, '2'), (4, '4')]])
        self.play(Create(xa), FadeIn(ticks), run_time=.6)
        rl = label('ratio', 24, MUTED).next_to(self.pt(4.45, 0), UP, buff=.12).shift(LEFT * .3)
        self.play(FadeIn(rl), run_time=.25)

        # on the ratio scale the two changes have unequal sizes
        y_u = self.Y0 - .95
        up = harrow(self.pt(1, 0)[0], self.pt(2, 0)[0], y_u, UP_C, 5)
        dn = harrow(self.pt(1, 0)[0], self.pt(.5, 0)[0], y_u, DOWN_C, 5)
        ul = label('double: +1', 26, UP_C).next_to(up, DOWN, buff=.14).align_to(up, LEFT).shift(RIGHT * .1)
        dl = label('halve: −0.5', 26, DOWN_C).next_to(dn, DOWN, buff=.14).align_to(dn, RIGHT).shift(LEFT * .1)
        self.play(GrowArrow(up), FadeIn(ul), run_time=.6)
        self.play(GrowArrow(dn), FadeIn(dl), run_time=.5)
        note = label('opposite changes, unequal sizes', 30, FOREGROUND).move_to([-.3, y_u, 0], aligned_edge=LEFT)
        self.play(FadeIn(note), run_time=.4)
        self.mark('unequal', double=1, halve=-.5, wait=1.0)
        self.play(FadeOut(VGroup(up, dn, ul, dl, note)), run_time=.4)

        # the log curve
        curve = ParametricFunction(lambda r: self.pt(r), t_range=[.1, 4.3], color=KLC, stroke_width=5)
        ll = label('log', 28, KLC).next_to(self.pt(4.3), UP, buff=.12).shift(LEFT * .2)
        self.play(Create(ya), Create(curve), FadeIn(ll), run_time=1)
        panel_x = 1.55
        rows = []

        def row(y, left, right, col):
            a, b = label(left, 28, FOREGROUND), label(right, 30, col)
            a.move_to([panel_x, y, 0], aligned_edge=LEFT)
            b.move_to([panel_x + 3.25, y, 0], aligned_edge=LEFT)
            return VGroup(a, b)

        def drop(r, col):
            d = Dot(self.pt(r), radius=.11, color=col)
            v = DashedLine(self.pt(r, 0), self.pt(r), color=col, stroke_width=4)
            return VGroup(v, d)

        d1 = Dot(self.pt(1), radius=.11, color=KLC)
        r1 = row(1.7, 'no change, 1', '0', KLC)
        self.play(FadeIn(d1, scale=.5), FadeIn(r1), run_time=.5)
        d2, r2 = drop(2, UP_C), row(1.02, 'double, 2', f3(math.log(2), True), UP_C)
        self.play(Create(d2), FadeIn(r2), run_time=.7)
        d3, r3 = drop(.5, DOWN_C), row(.34, 'halve, 0.5', f3(math.log(.5), True), DOWN_C)
        self.play(Create(d3), FadeIn(r3), run_time=.7)
        self.play(Indicate(r2[1], color=UP_C), Indicate(r3[1], color=DOWN_C), run_time=.7)
        self.mark('equal', double=round(math.log(2), 3), halve=round(math.log(.5), 3), wait=.9)

        # multiplication becomes addition
        x4 = self.pt(4, 0)[0]
        seg1 = Line(self.pt(4, 0), self.pt(4, math.log(2)), color=UP_C, stroke_width=9)
        seg2 = Line(self.pt(4, math.log(2)), self.pt(4, 2 * math.log(2)), color=NEWC, stroke_width=9)
        r4 = row(-.34, '×2, then ×2', '', UP_C)
        sum_l = label('0.693 + 0.693', 30, UP_C).move_to(r4[1], aligned_edge=LEFT)
        sum_l.move_to([panel_x + 2.6, -.34, 0], aligned_edge=LEFT)
        self.play(Create(seg1), FadeIn(r4[0]), run_time=.5)
        self.play(Create(seg2), FadeIn(sum_l), run_time=.5)
        r5 = row(-1.02, 'so ×4 is', f3(math.log(4), True), NEWC)
        self.play(FadeIn(r5), run_time=.4)
        self.mark('adds', four=round(math.log(4), 3), wait=.9)

        # near 0 it keeps falling
        d5 = drop(.1, DOWN_C)
        down = varrow(self.pt(.1)[0], self.pt(.1)[1] - .12, self.pt(.1)[1] - .6, DOWN_C, 5)
        r6 = row(-1.7, 'near 0, 0.1', f3(math.log(.1), True), DOWN_C)
        self.play(Create(d5), FadeIn(r6), run_time=.7)
        self.play(GrowArrow(down), run_time=.4)
        self.finish('02-why-log', hold=2.8, double=round(math.log(2), 3), halve=round(math.log(.5), 3),
                    four=round(math.log(4), 3), tenth=round(math.log(.1), 3))


# ================================================================ 03 term by term
class TermByTerm(Loop):
    """Ratio, log, weight by the new model, add: −0.086 + 0.133 = 0.047. The formula assembles above
    the columns, each piece lighting up the step it names."""

    XS, HX = [-2.05, -.1, 1.85, 3.8, 5.75], -6.75
    Y_NAME, Y_RATIO, Y_LOG, Y_W, Y_T = 1.32, .6, -.12, -.84, -1.56

    def formula(self):
        s = 34
        kl_ = label('KL  =', s, KLC)
        sig_ = label('Σ', s + 4, FOREGROUND)
        w = label('π(y)', s, FOREGROUND)
        dot = label('·', s, FOREGROUND)
        log_ = label('log', s, FOREGROUND)
        lp, rp = label('(', s, FOREGROUND), label(')', s, FOREGROUND)
        num = label('π(y)', s, FOREGROUND)
        sl = label('/', s, FOREGROUND)
        den = VGroup(pi_ref(s, FOREGROUND), label('(y)', s, FOREGROUND)).arrange(RIGHT, buff=.03)
        ratio = VGroup(num, sl, den).arrange(RIGHT, buff=.12)
        g = VGroup(kl_, sig_, w, dot, log_, lp, ratio, rp).arrange(RIGHT, buff=.14)
        return g.move_to([0, 2.42, 0]), dict(sigma=sig_, weight=w, log=log_, ratio=ratio)

    def cells(self, y, values, colors, size=30):
        return VGroup(*[label(v, size, c).move_to([x, y, 0]) for v, c, x in zip(values, colors, self.XS)])

    def construct(self):
        self.begin('KL, term by term', badge='illustrative shares')
        form, parts = self.formula()
        for k in parts:
            parts[k].set_opacity(.25)
        self.play(FadeIn(form), run_time=.5)
        names = VGroup(*[label(n, 28, FOREGROUND).move_to([x, self.Y_NAME, 0]) for n, x in zip(NAMES, self.XS)])
        rule = Line([self.HX, self.Y_NAME - .32, 0], [6.8, self.Y_NAME - .32, 0], color=GRID, stroke_width=2)
        self.play(FadeIn(names), Create(rule), run_time=.5)

        ratio = SMALL / REF
        logs = np.log(ratio)
        terms = SMALL * logs
        col = [DOWN_C if t < 0 else UP_C if t > 0 else MUTED for t in terms]

        def header(text, y):
            return label(text, 28, MUTED).move_to([self.HX, y, 0], aligned_edge=LEFT)

        h1 = header('new ÷ reference', self.Y_RATIO)
        r1 = self.cells(self.Y_RATIO, [f'{x:.2f}'.rstrip('0').rstrip('.') for x in ratio], col)
        self.play(parts['ratio'].animate.set_opacity(1), FadeIn(h1), run_time=.5)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * .1) for c in r1], lag_ratio=.12), run_time=.8)
        self.mark('ratio', ratio=[round(float(x), 3) for x in ratio], wait=.6)
        h2 = header('log', self.Y_LOG)
        r2 = self.cells(self.Y_LOG, [f3(x, True) if abs(x) > 1e-9 else '0' for x in logs], col)
        self.play(parts['log'].animate.set_opacity(1), FadeIn(h2), run_time=.5)
        self.play(*[TransformFromCopy(a, b) for a, b in zip(r1, r2)], run_time=.8)
        self.mark('log', log=[round(float(x), 3) for x in logs], wait=.6)
        h3 = header('× new share', self.Y_W)
        r3 = self.cells(self.Y_W, [f'{x:.2f}' for x in SMALL], [NEWC] * 5)
        self.play(parts['weight'].animate.set_opacity(1), FadeIn(h3), run_time=.5)
        self.play(LaggedStart(*[FadeIn(c, shift=DOWN * .1) for c in r3], lag_ratio=.12), run_time=.8)
        self.mark('weight', weight=SMALL.tolist(), wait=.4)
        h4 = header('term', self.Y_T)
        r4 = self.cells(self.Y_T, [f3(x, True) if abs(x) > 1e-9 else '0' for x in terms], col)
        self.play(FadeIn(h4), FadeIn(r4), run_time=.6)
        line = Line([self.HX, self.Y_T - .36, 0], [6.8, self.Y_T - .36, 0], color=GRID, stroke_width=2)
        self.play(Create(line), run_time=.3)
        self.mark('terms', terms=[round(float(x), 3) for x in terms], wait=.6)

        # the terms slide into a sum
        a, b = r4[0].copy(), r4[3].copy()
        plus, eq = label('+', 34, FOREGROUND), label('=', 34, FOREGROUND)
        total = pill(f3(kl(SMALL, REF)), KLC, 36)
        sigma_l = label('add', 28, MUTED).move_to([self.HX, -2.72, 0], aligned_edge=LEFT)
        row = VGroup(a.copy().scale(1.15), plus, b.copy().scale(1.15), eq, total).arrange(RIGHT, buff=.28).move_to([.9, -2.72, 0])
        self.play(parts['sigma'].animate.set_opacity(1), FadeIn(sigma_l), run_time=.4)
        self.play(TransformFromCopy(r4[0], row[0]), TransformFromCopy(r4[3], row[2]), FadeIn(plus), run_time=.8)
        self.play(FadeIn(eq), FadeIn(total, scale=1.2), run_time=.5)
        self.finish('03-kl-term-by-term', hold=3.0, kl=round(kl(SMALL, REF), 3), terms=[round(float(terms[0]), 3), round(float(terms[3]), 3)])


# ================================================================ 04 evidence
class Evidence(Loop):
    """Each reply adds its log-ratio to the evidence that the new model wrote it. KL is the average
    step. At 0.047 the evidence reaches 3.0, or 20-to-1 odds, after about 64 replies on average."""

    X0, Y0, SX, SY, N, LOW = -5.9, -1.3, .1, .95, 80, -1.0

    def pt(self, n, v):
        return np.array([self.X0 + n * self.SX, self.Y0 + v * self.SY, 0])

    def construct(self):
        self.begin('What the number means', badge='one simulated run')
        lr = samples()[:self.N]
        path = np.concatenate([[0], np.cumsum(lr)])
        xa = Line(self.pt(0, self.LOW), self.pt(self.N, self.LOW), color=GRID, stroke_width=3)
        ya = Line(self.pt(0, self.LOW), self.pt(0, 3.75), color=GRID, stroke_width=3)
        zero = DashedLine(self.pt(0, 0), self.pt(self.N, 0), color=GRID, stroke_width=2)
        zl = label('0', 24, MUTED).next_to(self.pt(0, 0), LEFT, buff=.12)
        yl = label('evidence for the new model', 26, MUTED).next_to(self.pt(0, 3.75), RIGHT, buff=.15)
        xl = label('replies read', 26, MUTED).next_to(self.pt(self.N, self.LOW), DOWN, buff=.14).shift(LEFT * .7)
        xt = VGroup(*[label(str(n), 24, MUTED).next_to(self.pt(n, self.LOW), DOWN, buff=.12) for n in (0, 20, 40, 60)])
        self.play(Create(xa), Create(ya), FadeIn(zero), FadeIn(zl), FadeIn(yl), FadeIn(xl), FadeIn(xt), run_time=.6)
        goal = DashedLine(self.pt(0, math.log(20)), self.pt(self.N, math.log(20)), color=UP_C, stroke_width=3)
        goal_l = label('20-to-1 odds (log 20 = 3.0)', 26, UP_C).next_to(self.pt(0, math.log(20)), UP, buff=.08).shift(RIGHT * 2.1)
        self.play(Create(goal), FadeIn(goal_l), run_time=.5)

        # the first replies, one at a time
        chips = VGroup()
        name_of = {round(float(v), 6): n for v, n in [(np.log(SMALL / REF)[0], 'vague'), (np.log(SMALL / REF)[3], 'best')]}
        n_t = ValueTracker(0)

        def trace():
            n = n_t.get_value()
            k = int(n)
            pts = [self.pt(i, path[i]) for i in range(k + 1)]
            if k < self.N:
                pts.append(self.pt(n, path[k] + (n - k) * lr[k]))
            line = VMobject(stroke_color=NEWC, stroke_width=4)
            if len(pts) > 1:
                line.set_points_as_corners(pts)
            else:
                line.set_points_as_corners([pts[0], pts[0] + RIGHT * .001])
            return line

        tr = always_redraw(trace)
        self.add(tr)
        cx = 3.0
        for i in range(5):
            v = float(lr[i])
            name = name_of.get(round(v, 6), 'unchanged')
            c = pill(f'{name}  {f3(v, True) if abs(v) > 1e-9 else "0"}', UP_C if v > 0 else DOWN_C if v < 0 else MUTED, 26)
            c.move_to([cx + 1.65, 1.0 - .66 * i, 0])
            self.play(FadeIn(c, shift=LEFT * .2), n_t.animate.set_value(i + 1), run_time=.45, rate_func=linear)
            chips.add(c)
            self.wait(.08)
        self.mark('first_replies', values=[round(float(x), 3) for x in lr[:5]], wait=.5)
        self.play(FadeOut(chips), run_time=.3)
        self.play(n_t.animate.set_value(self.N), run_time=3.2, rate_func=linear)
        self.mark('run', end=round(float(path[-1]), 3))

        # the average slope
        k = kl(SMALL, REF)
        n_cross = math.log(20) / k
        avg = Line(self.pt(0, 0), self.pt(self.N, k * self.N), color=KLC, stroke_width=4.5)
        self.play(tr.animate.set_stroke(opacity=.28), run_time=.3)
        self.remove(tr)
        faded = trace().set_stroke(opacity=.28)
        self.add(faded)
        self.play(Create(avg), run_time=.8)
        drop = DashedLine(self.pt(n_cross, math.log(20)), self.pt(n_cross, self.LOW), color=KLC, stroke_width=3)
        dot = Dot(self.pt(n_cross, math.log(20)), radius=.11, color=KLC)
        callout = VGroup(label('KL 0.047 is the average step', 28, KLC),
                         label('about 64 replies on average', 28, KLC)).arrange(DOWN, buff=.14, aligned_edge=LEFT)
        callout.move_to([2.0, -.25, 0], aligned_edge=LEFT)
        self.play(FadeIn(dot, scale=.5), Create(drop), run_time=.5)
        self.play(FadeIn(callout), run_time=.5)
        self.mark('average', slope=round(k, 3), replies=round(n_cross, 1), wait=1.4)

        # a model that always gives the best answer
        self.play(FadeOut(VGroup(callout, avg, faded, drop, dot)), run_time=.4)
        one = VMobject(stroke_color=UP_C, stroke_width=6).set_points_as_corners(
            [self.pt(0, 0), self.pt(0, math.log(1 / .12)), self.pt(1.2, math.log(1 / .12))])
        d2 = Dot(self.pt(1.2, math.log(1 / .12)), radius=.11, color=UP_C)
        c2 = VGroup(label('A model that always gives the best answer:', 28, FOREGROUND),
                    label('KL 2.12, so one reply gives 8.33 to 1', 28, UP_C)).arrange(DOWN, buff=.14, aligned_edge=LEFT)
        c2.move_to([-4.9, .35, 0], aligned_edge=LEFT)
        self.play(Create(one), FadeIn(d2), run_time=.6)
        self.play(FadeIn(c2), run_time=.5)
        self.finish('04-evidence', hold=2.8, kl=round(k, 3), replies=round(n_cross, 1), always=round(math.log(1 / .12), 2),
                    odds=round(1 / .12, 2))


# ================================================================ 05 never negative
class NeverNegative(Loop):
    """Gains add to 0, and every term is at least its gain, so the terms add to at least 0. A drop to
    0.01 has a large log-ratio and a tiny weight."""

    AX, UNIT = -.2, 21

    def construct(self):
        self.begin('Why KL is never negative', badge='illustrative shares')
        axis = Line([self.AX, 2.25, 0], [self.AX, -2.4, 0], color=GRID, stroke_width=3)
        zero = label('0', 24, MUTED).next_to(axis, UP, buff=.08)
        self.play(Create(axis), FadeIn(zero), run_time=.4)

        def rows(y, name, gain, term):
            n = label(name, 30, FOREGROUND).move_to([-6.6, y, 0], aligned_edge=LEFT)
            gb = hbar(self.AX, gain, y + .28, self.UNIT, MUTED, .42)
            tb = hbar(self.AX, term, y - .28, self.UNIT, UP_C if term > 0 else DOWN_C, .42)
            gl = label(f'gain {f3(gain, True, 2)}', 26, MUTED)
            tl = label(f'term {f3(term, True)}', 26, UP_C if term > 0 else DOWN_C)
            for lb, b, v in ((gl, gb, gain), (tl, tb, term)):
                lb.next_to(b, RIGHT if v > 0 else LEFT, buff=.14)
            return VGroup(n, gb, tb, gl, tl)

        t0, t3 = SMALL[0] * math.log(SMALL[0] / REF[0]), SMALL[3] * math.log(SMALL[3] / REF[3])
        a = rows(1.25, 'vague', -.10, t0)
        b = rows(-.35, 'best', .10, t3)
        for g in (a, b):
            self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT if g is b else RIGHT), FadeIn(g[3]), run_time=.6)
        self.mark('gains', gains=[-.10, .10], wait=.5)
        gs = label('gains add to 0', 30, MUTED).move_to([-5.5, -1.7, 0], aligned_edge=LEFT)
        self.play(FadeIn(gs), run_time=.4)
        self.wait(.6)
        for g in (a, b):
            self.play(GrowFromEdge(g[2], LEFT if g is b else RIGHT), FadeIn(g[4]), run_time=.7)
            self.wait(.35)
        ts = label(f'terms add to {f3(t0 + t3, True)}', 30, KLC).move_to([-5.5, -2.35, 0], aligned_edge=LEFT)
        self.play(FadeIn(ts), run_time=.4)
        rule = pill('each term ≥ its gain', KLC, 28).move_to([3.55, -2.0, 0])
        self.play(FadeIn(rule, scale=1.15), run_time=.4)
        self.mark('terms', terms=[round(t0, 3), round(t3, 3)], total=round(t0 + t3, 3), wait=1.4)

        # a huge drop: a large log-ratio, a tiny weight
        self.clear_stage()
        head = label('A huge drop: vague 0.40 → 0.01', 32, FOREGROUND).move_to([-6.6, 2.2, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), run_time=.4)
        lg = math.log(.01 / .40)
        ax2 = Line([5.3, 1.45, 0], [5.3, -.55, 0], color=GRID, stroke_width=3)
        self.play(Create(ax2), run_time=.3)
        unit = 2.1
        lb = hbar(5.3, lg, .9, unit, DOWN_C, .56)
        l1 = label('log-ratio', 28, MUTED).move_to([-6.6, .9, 0], aligned_edge=LEFT)
        v1 = label(f3(lg), 34, DOWN_C).next_to(lb, LEFT, buff=.14)
        assert v1.get_left()[0] > l1.get_right()[0] + .1
        self.play(FadeIn(l1), GrowFromEdge(lb, RIGHT), FadeIn(v1), run_time=.8)
        self.mark('drop', log_ratio=round(lg, 3), wait=.8)
        w = label('× new share 0.01', 28, NEWC).move_to([-6.6, 0, 0], aligned_edge=LEFT)
        self.play(FadeIn(w), run_time=.4)
        term = .01 * lg
        tb = hbar(5.3, term, 0, unit, DOWN_C, .56)
        v2 = label(f3(term), 34, DOWN_C).next_to(tb, LEFT, buff=.14)
        self.play(TransformFromCopy(lb, tb), FadeIn(v2), run_time=.9)
        self.mark('weighted', term=round(term, 3), wait=.8)
        up = .51 * math.log(.51 / .12)
        e = np.array([.01, .25, .15, .51, .08])
        res = VGroup(label(f'best 0.12 → 0.51: term {f3(up, True)}', 30, UP_C),
                     label(f'KL = {f3(term)} + {f3(up)} = {f3(kl(e, REF))}', 30, KLC)).arrange(DOWN, buff=.2, aligned_edge=LEFT)
        res.move_to([-6.6, -1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(res[0]), run_time=.5)
        self.play(FadeIn(res[1]), run_time=.5)
        self.finish('05-never-negative', hold=2.8, log_ratio=round(lg, 3), term=round(term, 3), up=round(up, 3), kl=round(kl(e, REF), 3))


# ================================================================ 06 no direction
class NoDirection(Loop):
    """Moves toward and away from the best answer both cost KL; the reward separates them; β sets how
    far the best answer's share rises."""

    def mini(self, shares, cx, base):
        xs = [cx + (i - 2) * .56 for i in range(5)]
        g = VGroup(Line([xs[0] - .35, base, 0], [xs[-1] + .35, base, 0], color=GRID, stroke_width=2))
        for i, (x, v, r) in enumerate(zip(xs, shares, REF)):
            g.add(bar(x, v, base, 2.0, UP_C if (i == 3 and v > r) else DOWN_C if (i == 3) else REFC, .42))
        g.add(label('best', 22, MUTED).move_to([xs[3], base - .24, 0]))
        return g

    def construct(self):
        self.begin('KL has no direction', badge='illustrative shares')
        cols = dict(R=-2.0, K=.9, S=3.9)
        heads = VGroup(label('average reward', 28, NEWC).move_to([cols['R'], 2.4, 0]),
                       label('− β × KL', 28, KLC).move_to([cols['K'], 2.4, 0]),
                       label('= score', 28, FOREGROUND).move_to([cols['S'], 2.4, 0]))
        beta = label('β = 1', 26, MUTED).move_to([-5.5, 2.4, 0])
        rows = [('toward best', SMALL, 1.0), ('away from best', AWAY, -1.05)]
        minis, kls, rews, scores = [], [], [], []
        self.play(FadeIn(heads[1]), run_time=.4)
        for name, s, y in rows:
            m = self.mini(s, -5.35, y - .55)
            n = label(name, 26, FOREGROUND).next_to(m, UP, buff=.72).align_to(m, LEFT)
            n.move_to([-6.6, y + .68, 0], aligned_edge=LEFT)
            k = pill(f3(kl(s, REF)), KLC, 34).move_to([cols['K'], y, 0])
            self.play(FadeIn(m), FadeIn(n), run_time=.5)
            self.play(FadeIn(k, scale=1.15), run_time=.4)
            minis.append(VGroup(m, n)); kls.append(k)
        self.mark('both_cost', kl=[round(kl(SMALL, REF), 3), round(kl(AWAY, REF), 3)], wait=1.3)
        self.play(FadeIn(heads[0]), FadeIn(beta), run_time=.4)
        for (name, s, y) in rows:
            r = pill(f3(float(s @ R)), NEWC, 34).move_to([cols['R'], y, 0])
            self.play(FadeIn(r, scale=1.15), run_time=.4)
            rews.append(r)
        self.play(FadeIn(heads[2]), run_time=.3)
        vals = []
        for (name, s, y), col in zip(rows, (UP_C, DOWN_C)):
            v = float(s @ R) - kl(s, REF)
            sc = pill(f3(v), col, 38).move_to([cols['S'], y, 0])
            self.play(FadeIn(sc, scale=1.15), run_time=.45)
            scores.append(sc); vals.append(round(v, 3))
        ref_l = label('keep the reference: 0.485', 26, MUTED).move_to([-.1, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(ref_l), run_time=.4)
        self.mark('reward_separates', score=vals, reference=round(float(REF @ R), 3), wait=2.0)

        # the β dial
        self.clear_stage()
        xs = [-5.9, -4.65, -3.4, -2.15, -.9]
        base, unit = -2.45, 4.3
        ax = Line([xs[0] - .6, base, 0], [xs[-1] + .6, base, 0], color=GRID, stroke_width=2)
        nm = VGroup(*[label(n, 24, FOREGROUND).move_to([x, base - .3, 0]) for n, x in zip(NAMES, xs)])
        steps = [(.02, '0'), (.5, '0.5'), (1, '1'), (2, '2'), (5, '5'), (1e6, 'very high')]
        tx = [1.3, 2.05, 2.8, 3.55, 4.3, 5.75]
        slider = Line([tx[0], -1.0, 0], [tx[-1], -1.0, 0], color=GRID, stroke_width=4)
        ticks = VGroup(*[label(t, 24, MUTED).move_to([x, -1.42, 0]) for (_, t), x in zip(steps, tx)])
        sl = label('β, the price of KL', 26, KLC).move_to([tx[0] - .1, -.5, 0], aligned_edge=LEFT)

        def shares(b):
            return optimum(REF, R, b)

        def chart(b):
            p = shares(b)
            return VGroup(*[bar(x, v, base, unit, UP_C if i == 3 else REFC, .9) for i, (x, v) in enumerate(zip(xs, p))])

        ghost = DashedLine([xs[3] - .6, base + .12 * unit, 0], [xs[3] + .6, base + .12 * unit, 0], color=REFC, stroke_width=3)
        cur = chart(steps[0][0])
        knob = Dot([tx[0], -1.0, 0], radius=.16, color=KLC)
        read = label(f'best {shares(steps[0][0])[3]:.2f}', 38, UP_C).move_to([1.3, .95, 0], aligned_edge=LEFT)
        self.play(FadeIn(ax), FadeIn(nm), FadeIn(cur), FadeIn(ghost), Create(slider), FadeIn(ticks), FadeIn(sl), FadeIn(knob),
                  FadeIn(read), run_time=.7)
        self.wait(.6)
        seen = [round(float(shares(steps[0][0])[3]), 2)]
        for (b, _), x in zip(steps[1:], tx[1:]):
            p = shares(b)
            r2 = label(f'best {p[3]:.2f}', 38, UP_C).move_to(read, aligned_edge=LEFT)
            self.play(Transform(cur, chart(b)), knob.animate.move_to([x, -1.0, 0]), FadeTransform(read, r2), run_time=.65)
            read = r2
            seen.append(round(float(p[3]), 2))
            self.wait(.3)
        self.mark('beta', best=seen, wait=.5)
        c = VGroup(label('The reward chooses the direction.', 28, FOREGROUND),
                   label('β × KL limits the distance.', 28, KLC)).arrange(DOWN, buff=.14, aligned_edge=LEFT)
        c.move_to([.75, 2.15, 0], aligned_edge=LEFT)
        self.play(FadeIn(c), run_time=.6)
        self.finish('06-no-direction', hold=2.8, best=seen, score=vals)


# ================================================================ 07 rare moves
class RareMoves(Loop):
    """The same 0.10 moved to a common answer and to a rare one: the share moved is equal, KL is not."""

    BASE_Y, UNIT = -2.4, 5.6

    def panel(self, cx, target, ref_t, name, r_col):
        x0, x1 = cx - 1.2, cx + 1.2
        base = Line([cx - 2.5, self.BASE_Y, 0], [cx + 2.5, self.BASE_Y, 0], color=GRID, stroke_width=2)
        vague = bar(x0, .40, self.BASE_Y, self.UNIT, REFC, 1.3)
        tb = bar(x1, ref_t, self.BASE_Y, self.UNIT, REFC, 1.3)
        n0 = label('vague', 28, FOREGROUND).move_to([x0, self.BASE_Y - .36, 0])
        n1 = label(name, 28, r_col).move_to([x1, self.BASE_Y - .36, 0])
        return dict(x0=x0, x1=x1, base=base, vague=vague, target=tb, n0=n0, n1=n1)

    def move(self, d, ref_t):
        h = .10 * self.UNIT
        top = self.BASE_Y + .40 * self.UNIT
        chunk = Rectangle(width=1.3, height=h, stroke_width=0, fill_color=NEWC, fill_opacity=.9)
        chunk.move_to([d['x0'], top - h / 2, 0])
        self.add(chunk)
        low = bar(d['x0'], .30, self.BASE_Y, self.UNIT, REFC, 1.3)
        self.play(Transform(d['vague'], low), run_time=.3)
        y = self.BASE_Y + (ref_t + .05) * self.UNIT
        self.play(chunk.animate.move_to([d['x1'], top + .3, 0]), run_time=.7)
        self.play(chunk.animate.move_to([d['x1'], y, 0]), run_time=.45)
        return chunk

    def construct(self):
        self.begin('Same move, different price', badge='illustrative shares')
        ref = REF6
        def after(i):
            p = ref.copy(); p[0] -= .1; p[i] += .1
            return p
        a, b = after(1), after(5)
        ka, kb = kl(a, ref), kl(b, ref)
        pa = self.panel(-3.6, 0, ref[1], '10 min', FOREGROUND)
        pb = self.panel(3.35, 5, ref[5], 'rare, wrong', DOWN_C)
        for d in (pa, pb):
            self.play(FadeIn(d['base']), FadeIn(d['vague']), FadeIn(d['target']), FadeIn(d['n0']), FadeIn(d['n1']), run_time=.5)
        ra = label('25% → 35%', 28, NEWC).move_to([pa['x1'], .3, 0])
        rb = label('0.5% → 10.5%', 28, NEWC).move_to([pb['x1'], .3, 0])
        self.move(pa, ref[1])
        self.play(FadeIn(ra), run_time=.3)
        self.move(pb, ref[5])
        self.play(FadeIn(rb), run_time=.3)
        moved = pill('share moved: 0.10 in both', MUTED, 28).move_to([0, 1.1, 0])
        self.play(FadeIn(moved, scale=1.1), run_time=.4)
        self.mark('moved', total_variation=[round(.5 * float(abs(a - ref).sum()), 3), round(.5 * float(abs(b - ref).sum()), 3)], wait=1.1)

        xa, xb = -5.4, 1.55
        qa = label(f'× {a[1] / ref[1]:.1f}', 32, UP_C).move_to([pa['x1'] - .05, 1.1, 0])
        qb = label(f'× {b[5] / ref[5]:.0f}', 32, DOWN_C).move_to([pb['x1'] + .1, 1.1, 0])
        self.play(FadeOut(moved), run_time=.25)
        self.play(FadeIn(qa, scale=1.2), FadeIn(qb, scale=1.2), run_time=.5)
        self.mark('ratio', ratios=[round(a[1] / ref[1], 2), round(b[5] / ref[5], 2)], wait=.9)
        y = 2.05
        ma = hbar(xa, ka, y, 11, KLC, .46)
        mb = hbar(xb, kb, y, 11, KLC, .46)
        la = label(f'KL {f3(ka)}', 32, KLC).next_to(ma, RIGHT, buff=.15)
        lb = label(f'KL {f3(kb)}', 32, KLC).next_to(mb, RIGHT, buff=.15)
        self.play(GrowFromEdge(ma, LEFT), FadeIn(la), run_time=.6)
        self.play(GrowFromEdge(mb, LEFT), FadeIn(lb), run_time=.8)
        times = pill(f'{kb / ka:.1f} times the price', KLC, 30).move_to([0, 1.1, 0])
        self.play(FadeOut(VGroup(ra, rb)), FadeOut(qa), FadeOut(qb), FadeIn(times, scale=1.15), run_time=.5)
        self.finish('07-rare-moves', hold=3.0, kl=[round(ka, 3), round(kb, 3)], times=round(kb / ka, 1),
                    moved=[.10, .10])


# ================================================================ 08 which way
class WhichWay(Loop):
    """KL is not symmetric. The order decides what is expensive: dropping an answer is finite one way
    and infinite the other; inventing an answer is the reverse."""

    XS, BASE_Y, UNIT = [-6.1, -5.15, -4.2, -3.25, -2.3], -2.3, 6.2
    MX, UNIT_M = .35, 26

    def chart(self, shares, xs, colors):
        return VGroup(*[bar(x, v, self.BASE_Y, self.UNIT, c, .72) for x, v, c in zip(xs, shares, colors)])

    def meter(self, y, v):
        if math.isinf(v):
            b = Rectangle(width=5.6, height=.5, stroke_width=0, fill_color=DOWN_C, fill_opacity=.9)
            b.move_to([self.MX, y, 0], aligned_edge=LEFT)
            t = label('∞', 44, WHITE).move_to(b).align_to(b, RIGHT).shift(LEFT * .2)
            return VGroup(b, t)
        b = hbar(self.MX, v, y, self.UNIT_M, KLC, .5)
        return VGroup(b, label(f3(v), 34, KLC).next_to(b, RIGHT, buff=.15))

    def construct(self):
        self.begin('Which model goes on top', badge='illustrative shares')
        def head(text_a, text_b, y, col):
            a = label(text_a, 30, col).move_to([self.MX, y, 0], aligned_edge=LEFT)
            b = label(text_b, 24, MUTED).next_to(a, RIGHT, buff=.3)
            return VGroup(a, b)
        h1 = head('KL(π ‖ π_ref)', 'weights: new model, RLHF', 1.72, KLC)
        h2 = head('KL(π_ref ‖ π)', 'weights: reference', -.28, REFC)
        y1, y2 = 1.1, -.9
        step = label('same small move', 30, FOREGROUND).move_to([-6.6, 2.3, 0], aligned_edge=LEFT)
        base = Line([self.XS[0] - .55, self.BASE_Y, 0], [-1.0, self.BASE_Y, 0], color=GRID, stroke_width=2)
        names = VGroup(*[label(n, 21, FOREGROUND).move_to([x, self.BASE_Y - .28, 0]) for n, x in zip(NAMES, self.XS)])
        ref_g = self.chart(REF, self.XS, [REFC] * 5)
        self.play(FadeIn(base), FadeIn(names), FadeIn(ref_g), FadeIn(step), FadeIn(h1), FadeIn(h2), run_time=.6)
        cur = self.chart(SMALL, self.XS, [DOWN_C, NEWC, NEWC, UP_C, NEWC])
        self.play(Transform(ref_g, cur), run_time=.7)
        m1, m2 = self.meter(y1, kl(SMALL, REF)), self.meter(y2, kl(REF, SMALL))
        self.play(GrowFromEdge(m1[0], LEFT), FadeIn(m1[1]), run_time=.5)
        self.play(GrowFromEdge(m2[0], LEFT), FadeIn(m2[1]), run_time=.5)
        self.mark('asymmetric', forward=round(kl(SMALL, REF), 3), reverse=round(kl(REF, SMALL), 3), wait=1.4)

        # dropping an answer
        drop = np.array([.40, .25, .15, .20, 0])
        step2 = label('drop the wrong answer', 30, FOREGROUND).move_to(step, aligned_edge=LEFT)
        self.play(FadeTransform(step, step2), FadeOut(m1), FadeOut(m2), run_time=.4)
        self.play(Transform(ref_g, self.chart(drop, self.XS, [NEWC, NEWC, NEWC, UP_C, DOWN_C])), run_time=.7)
        m1 = self.meter(y1, kl(drop, REF)); m2 = self.meter(y2, math.inf)
        self.play(GrowFromEdge(m1[0], LEFT), FadeIn(m1[1]), run_time=.5)
        self.play(GrowFromEdge(m2[0], LEFT), FadeIn(m2[1]), run_time=.7)
        self.mark('drop', forward=round(kl(drop, REF), 3), reverse='inf', wait=1.6)

        # inventing an answer
        xs6 = self.XS + [-1.35]
        new6 = np.array([.35, .25, .15, .12, .08, .05])
        ref6 = np.array([.40, .25, .15, .12, .08, 0])
        back = kl(ref6[:5], new6[:5])
        step3 = label('invent an answer', 30, FOREGROUND).move_to(step2, aligned_edge=LEFT)
        nm = label('new', 21, DOWN_C).move_to([xs6[5], self.BASE_Y - .28, 0])
        self.play(FadeTransform(step2, step3), FadeOut(m1), FadeOut(m2), run_time=.4)
        base2 = Line([self.XS[0] - .55, self.BASE_Y, 0], [xs6[5] + .5, self.BASE_Y, 0], color=GRID, stroke_width=2)
        ch6 = self.chart(new6, xs6, [NEWC, NEWC, NEWC, NEWC, NEWC, DOWN_C])
        self.play(Transform(ref_g, VGroup(*ch6[:5])), Transform(base, base2), FadeIn(ch6[5]), FadeIn(nm), run_time=.7)
        m1 = self.meter(y1, math.inf); m2 = self.meter(y2, back)
        self.play(GrowFromEdge(m1[0], LEFT), FadeIn(m1[1]), run_time=.7)
        self.play(GrowFromEdge(m2[0], LEFT), FadeIn(m2[1]), run_time=.5)
        self.mark('invent', forward='inf', reverse=round(back, 3), wait=1.2)
        self.finish('08-which-way', hold=2.6, forward=round(kl(SMALL, REF), 3), reverse=round(kl(REF, SMALL), 3),
                    drop=round(kl(drop, REF), 3), invent_reverse=round(back, 3))


# ================================================================ 09 from samples
class FromSamples(Loop):
    """The new model writes replies; each has a log-ratio; the running average settles near the exact
    KL. One reply can be negative; the average settles on the positive value."""

    X0, Y0, SX, SY, LO, HI = -5.7, -.05, 2.55, 8.0, -.26, .2

    def pt(self, n, v):
        return np.array([self.X0 + math.log10(n) * self.SX, self.Y0 + v * self.SY, 0])

    def construct(self):
        self.begin('KL from samples', badge='one simulated run')
        lr = samples()
        mean = np.cumsum(lr) / np.arange(1, len(lr) + 1)
        xa = Line(self.pt(1, self.LO), self.pt(10_000, self.LO), color=GRID, stroke_width=3)
        ya = Line(self.pt(1, self.LO), self.pt(1, self.HI), color=GRID, stroke_width=3)
        zero = DashedLine(self.pt(1, 0), self.pt(10_000, 0), color=GRID, stroke_width=2)
        zl = label('0', 24, MUTED).next_to(self.pt(1, 0), LEFT, buff=.12)
        ticks = VGroup(*[label(f'{n:,}', 24, MUTED).next_to(self.pt(n, self.LO), DOWN, buff=.12) for n in (1, 10, 100, 1000, 10_000)])
        xl = label('replies written', 26, MUTED).next_to(ticks, DOWN, buff=.12)
        yl = label('average log-ratio so far', 26, NEWC).next_to(self.pt(1, self.HI), RIGHT, buff=.15).shift(UP * .3)
        self.play(Create(xa), Create(ya), FadeIn(zero), FadeIn(zl), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=.6)
        exact = kl(SMALL, REF)
        el = DashedLine(self.pt(1, exact), self.pt(10_000, exact), color=KLC, stroke_width=3.5)
        ell = label(f'exact {f3(exact)}', 28, KLC).next_to(self.pt(10_000, exact), RIGHT, buff=.25)
        self.play(Create(el), FadeIn(ell), run_time=.5)
        n_t = ValueTracker(0)  # log10 of the replies so far

        def trace():
            m = 10 ** n_t.get_value()
            k = max(int(m), 1)
            ns = sorted(set(int(round(x)) for x in np.logspace(0, math.log10(k), 240)))
            line = VMobject(stroke_color=NEWC, stroke_width=4)
            pts = [self.pt(n, float(mean[n - 1])) for n in ns]
            line.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + RIGHT * .001])
            return line

        def head():
            m = max(int(10 ** n_t.get_value()), 1)
            return Dot(self.pt(m, float(mean[m - 1])), radius=.1, color=NEWC)

        tr, dot = always_redraw(trace), always_redraw(head)
        self.add(tr, dot)

        # the first replies, with one negative value picked out
        first_neg = int(np.argmax(lr < -1e-9))
        for n in range(1, first_neg + 2):
            self.play(n_t.animate.set_value(math.log10(n)), run_time=.28, rate_func=linear)
        neg = pill(f'one reply: {f3(float(lr[first_neg]), True)}', DOWN_C, 28).move_to([2.4, 2.35, 0])
        self.play(FadeIn(neg, scale=1.15), run_time=.4)
        self.mark('negative_reply', reply=first_neg + 1, value=round(float(lr[first_neg]), 3), wait=.9)
        self.play(FadeOut(neg), run_time=.3)
        seen = []
        for n in (10, 100, 10_000):
            self.play(n_t.animate.set_value(math.log10(n)), run_time=1.3 if n < 10_000 else 2.0, rate_func=linear)
            v = float(mean[n - 1])
            seen.append(round(v, 3))
            tag = label(f'{f3(v)}', 28, NEWC).next_to(self.pt(n, max(v, .13) if n == 10 else v), UP if n != 10_000 else DOWN, buff=.2 if n != 10_000 else .62)
            self.add(tag)
            self.wait(.35)
        self.mark('average', values=seen, wait=.4)
        c = label('The average settles on the positive value.', 28, FOREGROUND).move_to([2.4, 2.35, 0])
        self.play(FadeIn(c), run_time=.5)
        self.finish('09-from-samples', hold=2.8, values=seen, exact=round(exact, 3))


# ================================================================ 10 tokens
class Tokens(Loop):
    """A reply's probability is a product over tokens; the log turns it into a sum, so KL can be
    charged token by token. PPO puts it in the reward; GRPO puts it in the loss."""

    XS, RX = [-2.5, -.2, 2.1], 5.25
    Y_TOK, Y_REF, Y_NEW, Y_LOG = 1.95, .95, -.05, -1.45

    def construct(self):
        self.begin('KL, token by token', badge='illustrative probabilities')
        ref, new = [.40, .50, .80], [.50, .50, .96]
        toks = VGroup(*[pill(f'token {i + 1}', FOREGROUND, 28).move_to([x, self.Y_TOK, 0]) for i, x in enumerate(self.XS)])
        rl = label('reply', 28, FOREGROUND).move_to([self.RX, self.Y_TOK, 0])
        self.play(FadeIn(toks), FadeIn(rl), run_time=.5)

        def row(y, name, vals, col, total, times=True):
            h = label(name, 30, col).move_to([-6.6, y, 0], aligned_edge=LEFT)
            cells = VGroup(*[label(v, 32, col).move_to([x, y, 0]) for v, x in zip(vals, self.XS)])
            ops = VGroup(*[label('×' if times else '+', 30, MUTED).move_to([(a + b) / 2, y, 0]) for a, b in zip(self.XS, self.XS[1:])])
            eq = label('=', 30, MUTED).move_to([3.75, y, 0])
            res = label(total, 34, col).move_to([self.RX, y, 0])
            return VGroup(h, cells, ops, eq, res)

        a = row(self.Y_REF, 'reference', [f'{x:.2f}' for x in ref], REFC, f'{np.prod(ref):.2f}')
        b = row(self.Y_NEW, 'new model', [f'{x:.2f}' for x in new], NEWC, f'{np.prod(new):.2f}')
        for r in (a, b):
            self.play(FadeIn(r[0]), FadeIn(r[1]), run_time=.5)
            self.play(FadeIn(r[2]), FadeIn(r[3]), FadeIn(r[4], scale=1.15), run_time=.5)
        self.mark('product', ref=round(float(np.prod(ref)), 3), new=round(float(np.prod(new)), 3), wait=.9)
        rule = Line([-6.6, -.72, 0], [6.6, -.72, 0], color=GRID, stroke_width=2)
        ls = [math.log(n / r) for n, r in zip(new, ref)]
        total = math.log(np.prod(new) / np.prod(ref))
        c = row(self.Y_LOG, 'log-ratio', [f3(x, True) if abs(x) > 1e-9 else '0' for x in ls], KLC, f3(total, True), times=False)
        self.play(Create(rule), FadeIn(c[0]), run_time=.4)
        self.play(LaggedStart(*[FadeIn(x, shift=DOWN * .1) for x in c[1]], lag_ratio=.2), run_time=.8)
        self.play(FadeIn(c[2]), FadeIn(c[3]), FadeIn(c[4], scale=1.2), run_time=.6)
        note = label('the log turns × into +', 30, KLC).move_to([0, -2.45, 0])
        self.play(FadeIn(note), run_time=.4)
        self.mark('sum', logs=[round(x, 3) for x in ls], total=round(total, 3), wait=1.6)

        # where two published methods put it
        self.clear_stage(run_time=.45)
        self.play(FadeOut(self.badge), run_time=.2)
        self.remove(self.badge)
        def lane(y, name, src, expr, tail):
            n = label(name, 34, FOREGROUND).move_to([-6.6, y + .32, 0], aligned_edge=LEFT)
            s = label(src, 26, MUTED).move_to([-6.6, y - .24, 0], aligned_edge=LEFT)
            e = pill(expr, KLC, 30).move_to([-2.6, y, 0], aligned_edge=LEFT)
            t = label(tail, 28, MUTED).move_to([-2.45, y - .78, 0], aligned_edge=LEFT)
            return VGroup(n, s, e, t)
        l1 = lane(.95, 'PPO', 'InstructGPT', 'reward at each token − β × log-ratio', 'β = 0.02')
        l2 = lane(-1.3, 'GRPO', 'DeepSeekMath', 'loss + β × KL estimate', 'β = 0.04, per token x − log x − 1, x = π_ref / π')
        for l in (l1, l2):
            self.play(FadeIn(l[0]), FadeIn(l[1]), run_time=.4)
            self.play(FadeIn(l[2], scale=1.08), run_time=.45)
            self.play(FadeIn(l[3]), run_time=.35)
            self.wait(.5)
        self.finish('10-tokens', hold=3.0, logs=[round(x, 3) for x in ls], total=round(total, 3), betas=[.02, .04])


# ================================================================ 11 reading a run
class ReadingRun(Loop):
    """As KL grows the reward keeps rising, while gold rises, peaks and falls. Each point is the best
    compromise at a smaller β, standing in for training longer."""

    X0, Y0, SX, SY = -5.75, -1.3, 2.3, .98

    def pt(self, k, v):
        return np.array([self.X0 + k * self.SX, self.Y0 + v * self.SY, 0])

    def construct(self):
        self.begin('Reading KL in a run', badge='toy example')
        bs = np.exp(np.linspace(math.log(.25), math.log(1e4), 1500))
        rows = []
        for b in bs:
            p = optimum(REF6, R6, b)
            rows.append((kl(p, REF6), float(p @ R6), float(p @ GOLD6), float(p[5])))
        rows = sorted(rows)
        raw = np.array(rows)
        grid = np.linspace(0, raw[-1, 0], 700)
        arr = np.column_stack([grid] + [np.interp(grid, raw[:, 0], raw[:, c]) for c in (1, 2, 3)])   # even steps in KL
        peak = int(arr[:, 2].argmax())
        xa = Arrow(self.pt(0, -1), self.pt(3.95, -1), buff=0, color=GRID, stroke_width=3, tip_length=.15)
        ya = Line(self.pt(0, -1), self.pt(0, 3.1), color=GRID, stroke_width=3)
        zero = DashedLine(self.pt(0, 0), self.pt(3.9, 0), color=GRID, stroke_width=2)
        xt = VGroup(*[label(str(k), 24, MUTED).next_to(self.pt(k, -1), DOWN, buff=.12) for k in (0, 1, 2, 3)])
        xl = label('KL from the reference', 26, KLC).next_to(self.pt(1.9, -1), DOWN, buff=.52).shift(RIGHT * 0)
        self.play(GrowArrow(xa), Create(ya), FadeIn(zero), FadeIn(xt), FadeIn(xl), run_time=.6)

        i_t = ValueTracker(0)

        def poly(c, col):
            def f():
                k = max(int(i_t.get_value()), 1)
                idx = np.unique(np.linspace(0, k, min(k + 1, 160)).astype(int))
                m = VMobject(stroke_color=col, stroke_width=5)
                m.set_points_as_corners([self.pt(arr[j, 0], arr[j, c]) for j in idx])
                return m
            return f

        rc, gc = always_redraw(poly(1, NEWC)), always_redraw(poly(2, UP_C))
        rd = always_redraw(lambda: Dot(self.pt(arr[int(i_t.get_value()), 0], arr[int(i_t.get_value()), 1]), radius=.11, color=NEWC))
        gd = always_redraw(lambda: Dot(self.pt(arr[int(i_t.get_value()), 0], arr[int(i_t.get_value()), 2]), radius=.11, color=UP_C))
        px = 3.3

        def read(i):
            a = arr[i]
            g = VGroup(label(f'KL {a[0]:.2f}', 30, KLC), label(f'reward {f3(a[1], d=2)}', 30, NEWC),
                       label(f'gold {f3(a[2], d=2)}', 30, UP_C),
                       label(f'wrong answer {a[3] * 100:.1f}%' if a[3] < .01 else f'wrong answer {a[3] * 100:.0f}%', 30, DOWN_C))
            return g.arrange(DOWN, buff=.2, aligned_edge=LEFT).move_to([px, .9, 0], aligned_edge=LEFT)

        r = read(0)
        self.add(rc, gc, rd, gd)
        self.play(FadeIn(r), run_time=.4)
        self.mark('start', kl=0, reward=round(float(arr[0, 1]), 2), gold=round(float(arr[0, 2]), 2), wait=.7)

        def advance(i, t):
            nonlocal r
            n = 8
            for s in range(1, n + 1):
                j = int(i_t.get_value() + (i - i_t.get_value()) * 1 / (n + 1 - s))
                self.play(i_t.animate.set_value(j), run_time=t / n, rate_func=linear)
                r2 = read(j)
                self.remove(r); self.add(r2); r = r2

        advance(peak, 2.2)
        i_t.set_value(peak)
        v = DashedLine(self.pt(arr[peak, 0], -1), self.pt(arr[peak, 0], 3.05), color=UP_C, stroke_width=3)
        keep = pill('gold peak: keep this checkpoint', UP_C, 26).move_to(self.pt(arr[peak, 0], 2.55)).align_to(self.pt(arr[peak, 0] + .08, 0), LEFT)
        self.play(Create(v), FadeIn(keep, scale=1.1), run_time=.5)
        self.mark('peak', kl=round(float(arr[peak, 0]), 2), reward=round(float(arr[peak, 1]), 2), gold=round(float(arr[peak, 2]), 2), wait=1.1)
        self.play(FadeOut(keep), run_time=.25)
        advance(len(arr) - 1, 2.8)
        i_t.set_value(len(arr) - 1)
        self.remove(r)
        r = read(len(arr) - 1); self.add(r)
        rl = label('reward', 28, NEWC).next_to(self.pt(arr[-1, 0], arr[-1, 1]), UP, buff=.18).shift(LEFT * .35)
        gl = label('gold', 28, UP_C).next_to(self.pt(arr[-1, 0], arr[-1, 2]), DOWN, buff=.1).shift(LEFT * .75)
        self.play(FadeIn(rl), FadeIn(gl), run_time=.4)
        self.finish('11-reading-a-run', hold=3.0, start=[round(float(x), 2) for x in arr[0, 1:3]],
                    peak_kl=round(float(arr[peak, 0]), 2), peak_gold=round(float(arr[peak, 2]), 2),
                    peak_reward=round(float(arr[peak, 1]), 2), end=[round(float(x), 2) for x in arr[-1]],
                    last_reward_rise=bool((np.diff(arr[:, 1]) > -1e-9).all()))


# ================================================================ 12 the implicit reward
class ImplicitReward(Loop):
    """At the best compromise every log-ratio equals the reward minus 0.808. A comparison cancels the
    constant, and DPO scores pairs with exactly this log-ratio."""

    XS, Y0, UNIT = [-6.0, -4.65, -3.3, -1.95, -.6], -.1, 1.05

    def construct(self):
        self.begin('The log-ratio is also a reward', badge='illustrative numbers')
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
        self.finish('12-implicit-reward', hold=3.0, constant=round(c, 3), p=round(sig1, 3), loss=[round(l0, 3), round(l1, 3)])
