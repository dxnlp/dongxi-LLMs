"""X-POST-005 GIF set: reward models (CAND-ANIM-044, 041, 035, 042 as two clips, 048, 043 as two clips, 046, 050, 051).

One running example, shared with X-POST-004 and X-TOKEN-001. The prompt is "Is OpenAI open?".
Answer A, "Partly. Some weights are open.", is chosen; answer B, "Yes. OpenAI is open. It's in
the name.", is rejected. Token strings and IDs are real (Qwen3 tokenizer); the answers are
chosen, not generated. Scores, logits, hidden vectors and curves are illustrative; every
sigmoid, softmax, loss and gradient step on them is exact. No numbers from our own runs appear.

Cast, shared by every clip:

- green: the chosen answer
- slate: the rejected answer
- violet: network weights (body, heads)
- blue: scores, gaps, probabilities
- amber: the loss, label errors, and the illustration badge

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
CHOSEN, REJECTED, NET, SCORE, LOSS = COMPOSED, '#475569', INTERMEDIATE, BASE, ACCENT

PROMPT = 'Is OpenAI open?'
A_TEXT = 'Partly. Some weights are open.'
B_TEXT = "Yes. OpenAI is open. It's in the name."
A_TOKENS = [(5800, 'Part'), (398, 'ly'), (13, '.'), (4329, ' Some'), (14324, ' weights'), (525, ' are'),
            (1787, ' open'), (13, '.'), (151645, '<|im_end|>')]
A2_TOKENS = A_TOKENS[:3] + [(2009, ' All')] + A_TOKENS[4:]
assert ''.join(s for _, s in A_TOKENS[:-1]) == A_TEXT
VOCAB_ROWS, LAYERS = 151_936, 28
MARK_X, MARK_Y = 4.9, -3.35          # keep x > MARK_X and y < MARK_Y empty for the brand mark


def sig(x):
    return 1 / (1 + math.exp(-x))


def softmax(z):
    e = np.exp(np.array(z, float) - max(z))
    return e / e.sum()


# ---------------------------------------------------------------- primitives
def chip(text, color=FOREGROUND, size=26, h=.52, fill=.08, text_color=None, w=None):
    t = label(text, size, text_color or color)
    box = RoundedRectangle(width=w or max(.42, t.width + .3), height=h, corner_radius=.11,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def shown(s):
    return '␣' + s[1:] if s.startswith(' ') else s


def pill(text, col, size=26, fill=WHITE, text_color=None):
    t = label(text, size, text_color or col)
    box = RoundedRectangle(width=t.width + .34, height=t.height + .26, corner_radius=.2, stroke_color=col,
                           stroke_width=2.5, fill_color=fill, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def letter(ch, color, r=.24, size=26):
    c = Circle(r, stroke_width=0, fill_color=color, fill_opacity=1)
    return VGroup(c, label(ch, size, WHITE).move_to(c))


def person_icon(color=REJECTED, s=1.0):
    head = Circle(.1 * s, stroke_width=0, fill_color=color, fill_opacity=1)
    body = Arc(radius=.19 * s, start_angle=0, angle=PI, stroke_width=0, fill_color=color, fill_opacity=1)
    body.add_line_to(body.get_start()).next_to(head, DOWN, buff=.04 * s)
    return VGroup(head, body)


def bot_icon(color=BASE, s=1.0):
    face = RoundedRectangle(width=.4 * s, height=.32 * s, corner_radius=.08 * s, stroke_width=0,
                            fill_color=color, fill_opacity=1)
    eyes = VGroup(*[Dot(face.get_center() + RIGHT * dx * s, radius=.04 * s, color=WHITE) for dx in (-.09, .09)])
    antenna = Line(face.get_top(), face.get_top() + UP * .08 * s, color=color, stroke_width=3)
    return VGroup(face, eyes, antenna, Dot(antenna.get_end(), radius=.035 * s, color=color))


def lock_icon(color=NET, s=1.0):
    body = RoundedRectangle(width=.36 * s, height=.28 * s, corner_radius=.05 * s, stroke_width=0,
                            fill_color=color, fill_opacity=1)
    shackle = Arc(radius=.12 * s, start_angle=0, angle=PI, color=color, stroke_width=4 * s)
    shackle.next_to(body, UP, buff=-.02 * s)
    return VGroup(shackle, body)


def check_icon(color=CHOSEN, s=1.0):
    return VMobject(stroke_color=color, stroke_width=6 * s).set_points_as_corners(
        [[-.16 * s, 0, 0], [-.04 * s, -.12 * s, 0], [.18 * s, .14 * s, 0]])


def cursor_icon(s=1.0):
    pts = [[0, 0, 0], [0, -.5, 0], [.13, -.38, 0], [.24, -.6, 0], [.32, -.56, 0], [.21, -.34, 0], [.38, -.34, 0]]
    return Polygon(*[np.array(p) * s for p in pts], stroke_color=FOREGROUND, stroke_width=2,
                   fill_color=WHITE, fill_opacity=1)


def net_block(w, h, label_text=None, size=24, cols=0, rows=0):
    box = RoundedRectangle(width=w, height=h, corner_radius=.1, stroke_color=NET, stroke_width=2.2,
                           fill_color=NET, fill_opacity=.1)
    g = VGroup(box)
    for i in range(1, rows):
        y = -h / 2 + h * i / rows
        g.add(Line([-w / 2 + .06, y, 0], [w / 2 - .06, y, 0], stroke_color=NET, stroke_width=1, stroke_opacity=.3))
    for i in range(1, cols):
        x = -w / 2 + w * i / cols
        g.add(Line([x, -h / 2 + .06, 0], [x, h / 2 - .06, 0], stroke_color=NET, stroke_width=1, stroke_opacity=.3))
    if label_text:
        g.add(label(label_text, size, NET).move_to(box))
    return g


def harrow(x0, x1, y, color=MUTED, sw=4, tip=.15):
    """A strictly horizontal connector."""
    return Arrow([x0, y, 0], [x1, y, 0], buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=.5)


def varrow(x, y0, y1, color=MUTED, sw=4, tip=.15):
    """A strictly vertical connector."""
    return Arrow([x, y0, 0], [x, y1, 0], buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=.5)


def elbow(pts, color=MUTED, sw=4, tip=.17, opacity=1):
    """A right-angled connector through `pts` (each step horizontal or vertical), tip on the last step."""
    pts = [np.array([p[0], p[1], 0.]) for p in pts]
    for a, b in zip(pts, pts[1:]):
        assert abs(a[0] - b[0]) < 1e-6 or abs(a[1] - b[1]) < 1e-6, (a, b)
    path = VMobject(stroke_color=color, stroke_width=sw, stroke_opacity=opacity).set_points_as_corners(pts[:-1])
    head = Arrow(pts[-2], pts[-1], buff=0, color=color, stroke_width=sw, tip_length=tip,
                 max_tip_length_to_length_ratio=.5).set_stroke(opacity=opacity)
    return VGroup(path, head)


def answer_card(text, tag, width, size=24, color=GRID):
    t = label(text, size, FOREGROUND)
    tg = letter(tag, MUTED)
    box = RoundedRectangle(width=width, height=1.0, corner_radius=.16, stroke_color=color, stroke_width=2.4,
                           fill_color=WHITE, fill_opacity=1)
    tg.move_to(box).align_to(box, LEFT).shift(RIGHT * .22)
    t.next_to(tg, RIGHT, buff=.22)
    assert t.get_right()[0] < box.get_right()[0] - .12, (text, t.width, width)
    return VGroup(box, tg, t)


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
                if isinstance(f, Arrow) and not getattr(f, 'free', False):
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


def prompt_bubble(x_left, y, size=28):
    t = label(PROMPT, size, FOREGROUND)
    box = RoundedRectangle(width=t.width + .5, height=t.height + .3, corner_radius=.16, stroke_color=REJECTED,
                           stroke_width=2.2, fill_color=REJECTED, fill_opacity=.08)
    q = VGroup(box, t.move_to(box))
    q.move_to([x_left + q.width / 2, y, 0])
    return q, person_icon().next_to(q, LEFT, buff=.16)


# ================================================================ 01 one click, one row
class OneClick(Loop):
    """A person reads two answers and picks A. The choice is stored as one row (prompt, chosen,
    rejected); thousands of rows are the reward model's training set."""

    def construct(self):
        self.begin('One choice, one row')
        q, pic = prompt_bubble(-5.6, 1.75, 30)
        self.play(FadeIn(pic), FadeIn(q, shift=RIGHT * .2), run_time=.5)
        a = answer_card(A_TEXT, 'A', 5.8).move_to([-3.55, -.1, 0])
        b = answer_card(B_TEXT, 'B', 6.7).move_to([3.3, -.1, 0])
        self.play(FadeIn(a, shift=UP * .15), FadeIn(b, shift=UP * .15), run_time=.6)
        self.mark('answers')

        cur = cursor_icon(1.1).move_to([5.6, -2.4, 0])
        self.play(FadeIn(cur), run_time=.3)
        self.play(cur.animate.move_to(a[0].get_center() + RIGHT * 1.6 + DOWN * .2), run_time=.9)
        self.play(cur.animate.scale(.8), run_time=.12)
        self.play(cur.animate.scale(1.25), a[0].animate.set_stroke(CHOSEN, width=3.5).set_fill(CHOSEN, opacity=.08),
                  a[1][0].animate.set_fill(CHOSEN), b[1][0].animate.set_fill(REJECTED),
                  b.animate.set_opacity(.5), run_time=.35)
        tick = check_icon(CHOSEN, 1.3).next_to(a[0], UP, buff=.12).align_to(a[0], RIGHT).shift(LEFT * .3)
        self.play(Create(tick), run_time=.35)
        self.mark('click', chosen='A')

        # the choice folds into one row of a table
        widths, x0 = [2.6, 4.5, 5.4], -6.55
        xs = [x0 + sum(widths[:i]) + widths[i] / 2 for i in range(3)]
        y_head, y_row, rh = 1.25, .55, .62
        heads = VGroup(*[label(t, 24, MUTED).move_to([x, y_head, 0]) for t, x in zip(['prompt', 'chosen', 'rejected'], xs)])
        frame = VGroup(*[Rectangle(width=w, height=rh, stroke_color=GRID, stroke_width=2).move_to([x, y_row, 0])
                         for w, x in zip(widths, xs)])
        cells = VGroup(label(PROMPT, 22, FOREGROUND), label(A_TEXT, 22, CHOSEN), label(B_TEXT, 22, REJECTED))
        for c, x, w in zip(cells, xs, widths):
            c.move_to([x, y_row, 0])
            assert c.width < w - .15, (c, w)
        self.play(FadeOut(cur), FadeOut(tick), FadeOut(pic), run_time=.3)
        self.play(ReplacementTransform(q, VGroup(frame[0], cells[0])),
                  ReplacementTransform(a, VGroup(frame[1], cells[1])),
                  ReplacementTransform(b, VGroup(frame[2], cells[2])), FadeIn(heads), run_time=1.1)
        self.mark('row', columns=['prompt', 'chosen', 'rejected'])

        # more rows: the training set
        rows = VGroup()
        rng = np.random.default_rng(44)
        for k in range(1, 6):
            y = y_row - k * rh
            r = VGroup()
            for x, w, col in zip(xs, widths, [MUTED, CHOSEN, REJECTED]):
                r.add(Rectangle(width=w, height=rh, stroke_color=GRID, stroke_width=2).move_to([x, y, 0]))
                r.add(RoundedRectangle(width=float(rng.uniform(.45, .8)) * (w - .3), height=.14, corner_radius=.07,
                                       stroke_width=0, fill_color=col, fill_opacity=.35).move_to([x, y, 0]))
            rows.add(r)
        self.play(LaggedStart(*[FadeIn(r, shift=DOWN * .2) for r in rows], lag_ratio=.25), run_time=1.6)
        dots = label('⋮', 34, MUTED).move_to([xs[1], y_row - 6 * rh - .05, 0])
        self.play(FadeIn(dots), run_time=.3)
        self.mark('dataset', rows_drawn=6)

        table = VGroup(heads, frame, cells, rows, dots)
        rm = net_block(2.5, 1.5, 'reward model', 26).move_to([4.6, -.65, 0])
        self.play(table.animate.scale(.62).move_to([-3.2, -.65, 0]), run_time=.8)
        arr = harrow(table.get_right()[0] + .1, rm.get_left()[0] - .05, rm.get_y(), sw=5, tip=.24)
        self.play(GrowArrow(arr), FadeIn(rm), run_time=.6)
        out = pill('score ?', SCORE, 28).next_to(rm, UP, buff=.5)
        up = varrow(rm.get_x(), rm.get_top()[1] + .05, out.get_bottom()[1] - .05, SCORE, sw=5, tip=.2)
        self.play(GrowArrow(up), FadeIn(out, scale=1.2), run_time=.6)
        self.finish('01-one-click', hold=3.0, chosen='A', rejected='B', columns=3)


# ================================================================ 02 Bradley–Terry
class BradleyTerry(Loop):
    """Two scores; only their gap matters. The sigmoid turns the gap into the probability that the
    chosen answer wins; the loss is −log of it; each step raises A and lowers B by 1 − σ(gap)."""

    UNIT, BASE_Y = .78, -2.45
    XA, XB = -4.55, -2.75
    CX0, CY0, CSX, CSY = 1.35, -2.4, .66, 4.1       # sigmoid axes: gap −4…4 → x, probability 0…1 → y

    def bar(self, x, s, col):
        return Rectangle(width=.9, height=max(s * self.UNIT, .01), stroke_width=0, fill_color=col,
                         fill_opacity=.9).move_to([x, self.BASE_Y, 0], aligned_edge=DOWN)

    def cpt(self, g, p):
        return np.array([self.CX0 + (g + 4) * self.CSX, self.CY0 + p * self.CSY, 0])

    def dynamic(self, sa, sb, offset=0):
        g = sa - sb
        p = sig(g)
        ba, bb = self.bar(self.XA, sa, CHOSEN), self.bar(self.XB, sb, REJECTED)
        la = label(f'{sa + offset:.1f}', 26, CHOSEN).next_to(ba, UP, buff=.12)
        lb = label(f'{sb + offset:.1f}', 26, REJECTED).next_to(bb, UP, buff=.12)
        ya, yb = self.BASE_Y + sa * self.UNIT, self.BASE_Y + sb * self.UNIT
        xg = -1.75
        gap = VGroup(DashedLine([self.XA + .45, ya, 0], [xg, ya, 0], color=SCORE, stroke_width=2),
                     DashedLine([self.XB + .45, yb, 0], [xg, yb, 0], color=SCORE, stroke_width=2),
                     DoubleArrow([xg, yb, 0], [xg, ya, 0], buff=0, color=SCORE, stroke_width=4,
                                 tip_length=.16, max_tip_length_to_length_ratio=.4))
        gl = label(f'gap {g:+.2f}'.replace('-', '−'), 24, SCORE).next_to(gap[2], RIGHT, buff=.12)
        dot = Dot(self.cpt(g, p), radius=.11, color=SCORE)
        hline = DashedLine(self.cpt(-4, p), self.cpt(g, p), color=SCORE, stroke_width=2)
        pl = label(f'{p:.2f}', 26, SCORE).next_to(self.cpt(-4, p), LEFT, buff=.12)
        loss = -math.log(p)
        lbar = Rectangle(width=max(loss * 1.6, .01), height=.3, stroke_width=0, fill_color=LOSS, fill_opacity=.85)
        lbar.move_to([1.75, -3.1, 0], aligned_edge=LEFT)
        ll = label(f'{loss:.2f}', 24, LOSS).next_to(lbar, RIGHT, buff=.12)
        return dict(shapes=VGroup(ba, bb, gap, dot, hline, lbar), labels=VGroup(la, lb, gl, pl, ll))

    def construct(self):
        self.begin('Bradley–Terry', badge='illustrative scores')
        # the score axis and the two answers
        axis = Line([-5.45, self.BASE_Y, 0], [-5.45, self.BASE_Y + 3.4 * self.UNIT + .3, 0], color=GRID, stroke_width=2)
        base = Line([-5.45, self.BASE_Y, 0], [-2.0, self.BASE_Y, 0], color=GRID, stroke_width=2)
        ticks = VGroup(*[label(str(v), 22, MUTED).move_to([-5.75, self.BASE_Y + v * self.UNIT, 0]) for v in range(4)])
        tags = VGroup(letter('A', CHOSEN).move_to([self.XA, self.BASE_Y - .42, 0]),
                      letter('B', REJECTED).move_to([self.XB, self.BASE_Y - .42, 0]))
        tick = check_icon(CHOSEN, 1.1).next_to(tags[0], RIGHT, buff=.12)
        score_l = label('score', 24, MUTED).next_to(axis, UP, buff=.15)
        self.play(FadeIn(axis), FadeIn(base), FadeIn(ticks), FadeIn(tags), FadeIn(score_l), run_time=.5)
        self.play(Create(tick), run_time=.3)

        sa, sb = 1.0, 2.0
        d = self.dynamic(sa, sb)
        ba, bb, gap, dot, hline, lbar = d['shapes']
        la, lb, gl, pl, ll = d['labels']
        self.play(GrowFromEdge(ba, DOWN), GrowFromEdge(bb, DOWN), FadeIn(la), FadeIn(lb), run_time=.7)
        self.mark('scores', a=sa, b=sb)
        self.play(FadeIn(gap), FadeIn(gl), run_time=.5)
        self.mark('gap', gap=sa - sb)

        # shift both scores by 10: nothing after the gap changes
        t10 = VGroup(*[label(str(v + 10), 22, MUTED).move_to(t.get_center()) for v, t in zip(range(4), ticks)])
        la10 = label(f'{sa + 10:.1f}', 26, CHOSEN).move_to(la); lb10 = label(f'{sb + 10:.1f}', 26, REJECTED).move_to(lb)
        plus = pill('+10', MUTED, 24).move_to([-3.65, 1.25, 0])
        self.play(FadeIn(plus, scale=1.2), run_time=.3)
        self.play(FadeTransform(ticks, t10), FadeTransform(la, la10), FadeTransform(lb, lb10), Indicate(gl, color=SCORE),
                  run_time=.8)
        self.wait(.4)
        ticks0 = VGroup(*[label(str(v), 22, MUTED).move_to(t.get_center()) for v, t in zip(range(4), t10)])
        la, lb = d['labels'][0], d['labels'][1]
        self.play(FadeTransform(t10, ticks0), FadeTransform(la10, la), FadeTransform(lb10, lb), FadeOut(plus), run_time=.6)
        ticks = ticks0
        self.mark('shift', gap_unchanged=True)

        # the gap goes onto the S-curve
        xa = Line(self.cpt(-4, 0), self.cpt(4, 0), color=GRID, stroke_width=2)
        ya = Line(self.cpt(-4, 0), self.cpt(-4, 1), color=GRID, stroke_width=2)
        mid = DashedLine(self.cpt(0, 0), self.cpt(0, 1), color=GRID, stroke_width=1.5)
        curve = ParametricFunction(lambda g: self.cpt(g, sig(g)), t_range=[-4, 4], color=SCORE, stroke_width=4)
        xt = VGroup(*[label((f'{v:+d}' if v else '0').replace('-', '−'), 20, MUTED).next_to(self.cpt(v, 0), DOWN, buff=.1) for v in (-3, 0, 3)])
        xl = label('gap', 22, MUTED).next_to(self.cpt(4, 0), DOWN, buff=.1)
        yl = label('P(A wins)', 22, SCORE).next_to(self.cpt(-4, 1), UP, buff=.12).shift(RIGHT * .4)
        self.play(Create(xa), Create(ya), FadeIn(mid), FadeIn(xt), FadeIn(xl), FadeIn(yl), run_time=.5)
        self.play(Create(curve), run_time=.8)
        g0 = label(f'{sa - sb:+.2f}'.replace('-', '−'), 24, SCORE).move_to(self.cpt(sa - sb, 0) + DOWN * .55)
        self.play(TransformFromCopy(gl, g0), run_time=.6)
        self.play(FadeIn(dot, scale=.5), Create(hline), FadeIn(pl), FadeOut(g0), run_time=.7)
        self.mark('probability', p=round(sig(sa - sb), 4))
        loss_l = label('loss', 24, LOSS).next_to(lbar, LEFT, buff=.15)
        self.play(GrowFromEdge(lbar, LEFT), FadeIn(ll), FadeIn(loss_l), run_time=.6)
        self.mark('loss', loss=round(-math.log(sig(sa - sb)), 4))

        # training steps: A up and B down by the push 1 − σ(gap)
        steps, cur = [], d
        cur['labels'] = VGroup(la, lb, gl, pl, ll)
        for k in range(4):
            push = 1 - sig(sa - sb)
            pa = Arrow([self.XA + .7, ba.get_top()[1] - .05, 0], [self.XA + .7, ba.get_top()[1] + push * self.UNIT * 1.6, 0],
                       buff=0, color=CHOSEN, stroke_width=6, tip_length=.18, max_tip_length_to_length_ratio=.5)
            pb = Arrow([self.XB + .7, bb.get_top()[1] + .05, 0], [self.XB + .7, bb.get_top()[1] - push * self.UNIT * 1.6, 0],
                       buff=0, color=REJECTED, stroke_width=6, tip_length=.18, max_tip_length_to_length_ratio=.5)
            self.play(GrowArrow(pa), GrowArrow(pb), run_time=.35 if k else .6)
            sa, sb = sa + push, sb - push
            new = self.dynamic(sa, sb)
            self.play(Transform(cur['shapes'], new['shapes']), FadeTransform(cur['labels'], new['labels']),
                      FadeOut(pa), FadeOut(pb), run_time=.8)
            ba, bb = cur['shapes'][0], cur['shapes'][1]
            cur['labels'] = new['labels']
            steps.append(dict(push=round(push, 4), gap=round(sa - sb, 4), p=round(sig(sa - sb), 4),
                              loss=round(-math.log(sig(sa - sb)), 4)))
            if k == 0:
                self.mark('step_1', **steps[0])
        self.finish('02-bradley-terry', hold=3.0, steps=steps, start_gap=-1.0, start_p=round(sig(-1), 4),
                    start_loss=round(-math.log(sig(-1)), 4))


# ================================================================ 03 the gradient push
class GradientPush(Loop):
    """Many pairs on the gap axis. Each dot's push is 1 − σ(gap): large on pairs ranked wrong, small on
    pairs already right. Gradient steps move the wrong pairs most; the total push and the mean loss fall.
    The highlighted pair is the article's two-feature example (gap 0 → 1.0 in the first step)."""

    GAPS = [-3.2, -2.5, -1.8, -1.2, -.6, 0.0, .5, 1.1, 1.8, 2.6, 3.4, 4.2]
    OURS = 5
    X0, SX, AY, PH = -.2, .78, -2.3, 3.3         # gap 0 at x = X0; push 1 → PH units above the axis

    def gx(self, g):
        return self.X0 + g * self.SX

    def draw(self, gaps):
        g = VGroup()
        for i, v in enumerate(gaps):
            p = 1 - sig(v)
            col = CHOSEN if i == self.OURS else SCORE
            stem = Line([self.gx(v), self.AY, 0], [self.gx(v), self.AY + max(p * self.PH, .02), 0], color=col,
                        stroke_width=5, stroke_opacity=.75)
            dot = Dot([self.gx(v), self.AY, 0], radius=.07 + .16 * p, color=col)
            g.add(VGroup(stem, dot))
        return g

    def readout(self, v):
        p = sig(v)
        rows = [('gap', f'{v:.2f}', SCORE), ('loss', f'{-math.log(p):.3f}', LOSS), ('push', f'{1 - p:.3f}', SCORE)]
        g = VGroup()
        for k, (n, val, col) in enumerate(rows):
            y = 1.55 - k * .5
            g.add(label(n, 24, MUTED).move_to([-5.55, y, 0], aligned_edge=LEFT))
            g.add(label(val, 26, col).move_to([-3.75, y, 0], aligned_edge=RIGHT))
        return g

    def meters(self, gaps):
        loss = float(np.mean([-math.log(sig(v)) for v in gaps]))
        push = float(np.mean([1 - sig(v) for v in gaps]))
        g = VGroup()
        for k, (n, val, col, sc) in enumerate([('mean loss', loss, LOSS, 2.4), ('mean push', push, SCORE, 3.6)]):
            y = 1.55 - k * .6
            g.add(label(n, 24, MUTED).move_to([1.2, y, 0], aligned_edge=LEFT))
            g.add(Rectangle(width=max(val * sc, .02), height=.28, stroke_width=0, fill_color=col,
                            fill_opacity=.85).move_to([3.1, y, 0], aligned_edge=LEFT))
        return g, loss, push

    def construct(self):
        self.begin('Where the push goes', badge='illustrative pairs')
        axis = Arrow([self.gx(-4.2), self.AY, 0], [self.gx(6.2), self.AY, 0], buff=0, color=GRID, stroke_width=3,
                     tip_length=.18)
        zero = DashedLine([self.gx(0), self.AY - .25, 0], [self.gx(0), self.AY + self.PH + .2, 0], color=GRID,
                          stroke_width=1.5)
        gap_l = label('gap', 24, MUTED).next_to(axis, DOWN, buff=.12).align_to(axis, RIGHT)
        wrong = label('wrong order', 22, MUTED).move_to([self.gx(-2.4), self.AY - .5, 0])
        right = label('right order', 22, MUTED).move_to([self.gx(2.4), self.AY - .5, 0])
        curve = ParametricFunction(lambda g: [self.gx(g), self.AY + (1 - sig(g)) * self.PH, 0], t_range=[-4.2, 6.1],
                                   color=SCORE, stroke_width=2.5, stroke_opacity=.35)
        push_l = label('push = 1 − σ(gap)', 24, SCORE).move_to([self.gx(1.9) + .6, self.AY + 1.1, 0])
        self.play(GrowArrow(axis), FadeIn(zero), FadeIn(gap_l), FadeIn(wrong), FadeIn(right), run_time=.6)
        self.play(Create(curve), FadeIn(push_l), run_time=.7)

        gaps = list(self.GAPS)
        pairs = self.draw(gaps)
        self.play(LaggedStart(*[FadeIn(p, shift=UP * .2) for p in pairs], lag_ratio=.08), run_time=1.0)
        ours_l = VGroup(letter('A', CHOSEN, .17, 20), label('vs', 20, MUTED), letter('B', REJECTED, .17, 20)).arrange(RIGHT, buff=.08)
        ours_l.next_to(pairs[self.OURS][1], DOWN, buff=.75)
        rd = self.readout(gaps[self.OURS])
        rd_head = ours_l.copy().move_to([-4.7, 2.2, 0])
        mt, loss0, push0 = self.meters(gaps)
        self.play(FadeIn(ours_l), FadeIn(rd_head), FadeIn(rd), FadeIn(mt), run_time=.6)
        self.mark('pairs', n=len(gaps), mean_loss=round(loss0, 4), mean_push=round(push0, 4))

        history = []
        for k in range(4):
            new = [v + 2 * (1 - sig(v)) for v in gaps]          # lr = 1; the gap moves by 2 · push (two features)
            arrows = VGroup(*[Arrow([self.gx(a), self.AY + .45, 0], [self.gx(b), self.AY + .45, 0], buff=0,
                                    color=CHOSEN if i == self.OURS else SCORE, stroke_width=4, tip_length=.14,
                                    max_tip_length_to_length_ratio=.45)
                              for i, (a, b) in enumerate(zip(gaps, new)) if b - a > .08])
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=.05), run_time=.5)
            gaps = new
            m2, l2, p2 = self.meters(gaps)
            ours_target = ours_l.copy().next_to(self.draw(gaps)[self.OURS][1], DOWN, buff=.75)
            rd_new = self.readout(gaps[self.OURS])
            self.play(Transform(pairs, self.draw(gaps)), FadeOut(arrows), Transform(ours_l, ours_target),
                      FadeTransform(rd, rd_new), Transform(mt, m2), run_time=.9)
            rd = rd_new
            history.append(dict(ours_gap=round(gaps[self.OURS], 4), mean_loss=round(l2, 4), mean_push=round(p2, 4)))
            if k == 0:
                self.mark('step_1', **history[0])
        self.finish('03-gradient-push', hold=3.0, steps=history, ours_start=dict(gap=0.0, loss=round(math.log(2), 4), push=.5))


# ================================================================ 04 one head row
H_A, H_B = [.5, -1, 2, 0], [1, 0, 0, 0]
LM_ROWS = [('Part', [2, 0, .5, 1]), ('Yes', [0, -.5, .25, 3]), ('No', [0, 1, 0, 2])]
W_ROW = [1, -.5, .5, 2]


def fmt(v):
    return f'{v:g}'.replace('-', '−')


def num(v, d=1):
    return f'{v:.{d}f}'.replace('-', '−')


class HeadBase(Loop):
    """Shared by 04 and 05: the prompt goes through the body, and the final hidden vector h sits above the head,
    column over column, so each logit is h · one row."""

    COLS = [-.15, 1.0, 2.15, 3.3]      # x of the four numbers, shared by h and every head row
    HY = 1.45                          # y of h
    CW, CH, CS = .98, .68, 30          # chip width, height, font size

    def vec(self, vals, y, color, fill=.1):
        return VGroup(*[chip(fmt(v), color, self.CS, h=self.CH, fill=fill, text_color=FOREGROUND, w=self.CW).move_to([x, y, 0])
                        for v, x in zip(vals, self.COLS)])

    def dot(self, h, row):
        return LaggedStart(*[AnimationGroup(Indicate(a, color=SCORE, scale_factor=1.12), Indicate(b, color=NET, scale_factor=1.12))
                             for a, b in zip(h, row)], lag_ratio=.25)

    def front(self, hvals, tag=None):
        """Input → body → h, all on one horizontal line."""
        inp = chip(PROMPT, REJECTED, 26, h=.62).move_to([-5.45, self.HY, 0])
        body = net_block(1.85, 1.0, f'body × {LAYERS}', 28).move_to([-2.35, self.HY, 0])
        a_in = harrow(inp.get_right()[0] + .08, body.get_left()[0] - .08, self.HY)
        h = self.vec(hvals, self.HY, SCORE)
        h_l = label('h', 34, SCORE).next_to(h, RIGHT, buff=.3)
        a_h = harrow(body.get_right()[0] + .08, h.get_left()[0] - .1, self.HY)
        anims = [FadeIn(inp), GrowArrow(a_in), FadeIn(body)]
        if tag is not None:
            tag.next_to(inp, DOWN, buff=.2)
            anims.append(FadeIn(tag, scale=1.3))
        self.play(*anims, run_time=.6)
        self.play(GrowArrow(a_h), LaggedStart(*[FadeIn(c, shift=RIGHT * .15) for c in h], lag_ratio=.12), FadeIn(h_l), run_time=.8)
        return inp, h


class ChatHead(HeadBase):
    """A chat model's LM head has one row per vocabulary entry, 151,936 rows. Each logit is h · one row. Softmax,
    pick a token, and the loop runs again for the next one."""

    def construct(self):
        self.begin('Chat model: the LM head', badge='4 numbers stand for 1,024')
        inp, h = self.front(H_A)
        self.mark('hidden', h=H_A)
        ys = [.15, -.75, -1.65]
        rows = VGroup(*[self.vec(v, y, NET, .08) for (_, v), y in zip(LM_ROWS, ys)])
        names = VGroup(*[label(n, 30, NET).move_to([-.85, y, 0], aligned_edge=RIGHT) for (n, _), y in zip(LM_ROWS, ys)])
        more = label(f'⋮   {VOCAB_ROWS:,} rows', 28, NET).move_to([1.6, -2.45, 0])
        fx0, fx1 = -1.75, self.COLS[-1] + .66
        frame = RoundedRectangle(width=fx1 - fx0, height=3.45, corner_radius=.16, stroke_color=NET, stroke_width=2,
                                 fill_color=NET, fill_opacity=.03).move_to([(fx0 + fx1) / 2, -1.12, 0])
        head_l = label('LM head', 28, NET).next_to(frame, LEFT, buff=.2).align_to(frame, UP)
        assert head_l.get_left()[0] > -6.9 and names.get_left()[0] > frame.get_left()[0] + .05
        self.play(FadeIn(frame), FadeIn(head_l), LaggedStart(*[FadeIn(VGroup(n, r)) for n, r in zip(names, rows)], lag_ratio=.2),
                  FadeIn(more), run_time=.9)
        self.mark('lm_head', rows=VOCAB_ROWS)
        logits = VGroup()
        for (n, v), r, y in zip(LM_ROWS, rows, ys):
            val = float(np.dot(H_A, v))
            self.play(self.dot(h, r), run_time=.9 if n == 'Part' else .6)
            out = label(num(val), 34, SCORE).move_to([4.75, y, 0])
            self.play(FadeIn(out, shift=RIGHT * .15), run_time=.3)
            logits.add(out)
        logit_l = label('logits', 28, SCORE).move_to([4.75, .8, 0])
        self.play(FadeIn(logit_l), run_time=.3)
        self.mark('logits', logits={n: float(np.dot(H_A, v)) for n, v in LM_ROWS})

        sm = pill('softmax', SCORE, 28).move_to([6.0, ys[1], 0])
        pick = chip('Part', BASE, 32, h=.66, fill=1, text_color=WHITE).move_to([6.0, -2.15, 0])
        a1 = harrow(5.2, sm.get_left()[0] - .06, ys[1])
        a2 = varrow(sm.get_x(), sm.get_bottom()[1] - .06, pick.get_top()[1] + .06)
        loop_y = -3.05
        loop = elbow([(pick.get_x(), pick.get_bottom()[1] - .06), (pick.get_x(), loop_y), (inp.get_x(), loop_y),
                      (inp.get_x(), inp.get_bottom()[1] - .08)], BASE, sw=3.5, opacity=.75)
        nxt = label('next token', 28, BASE).move_to([-3.4, loop_y + .3, 0])
        assert pick.get_right()[0] < 6.95
        self.play(GrowArrow(a1), FadeIn(sm), run_time=.4)
        self.play(GrowArrow(a2), FadeIn(pick, scale=.7), run_time=.4)
        self.play(Create(loop), FadeIn(nxt), run_time=.9)
        self.finish('04-chat-head', hold=3.0, rows=VOCAB_ROWS, picked='Part',
                    logits={n: float(np.dot(H_A, v)) for n, v in LM_ROWS})


class ScoreHead(HeadBase):
    """A reward model keeps the body and replaces the LM head with a score head of one row. h · that row is the
    score. Answer B goes through its own pass; the two scores are compared."""

    def construct(self):
        self.begin('Reward model: one head row', badge='4 numbers stand for 1,024')
        tag_a = letter('A', CHOSEN, .26, 28)
        inp, h = self.front(H_A, tag_a)
        self.mark('hidden', h=H_A)
        wy = -.55
        w = self.vec(W_ROW, wy, NET, .08)
        wx0, wx1 = self.COLS[0] - .66, self.COLS[-1] + .66
        w_frame = RoundedRectangle(width=wx1 - wx0, height=1.1, corner_radius=.16, stroke_color=NET, stroke_width=2,
                                   fill_color=NET, fill_opacity=.03).move_to([(wx0 + wx1) / 2, wy, 0])
        w_l = label('score head', 28, NET).next_to(w_frame, DOWN, buff=.15).align_to(w_frame, LEFT)
        one = label('1 row', 28, NET).next_to(w_frame, DOWN, buff=.15).align_to(w_frame, RIGHT)
        self.play(FadeIn(VGroup(w_frame, w, w_l, one), shift=UP * .3), run_time=.7)
        self.mark('score_head', rows=1)
        self.play(self.dot(h, w), run_time=1.0)
        sA = float(np.dot(H_A, W_ROW))
        pa = pill(f'score  {num(sA)}', CHOSEN, 32).move_to([5.55, wy, 0])
        a3 = harrow(w_frame.get_right()[0] + .08, pa.get_left()[0] - .08, wy)
        self.play(GrowArrow(a3), FadeIn(pa, scale=1.2), run_time=.5)
        self.mark('score_a', score=sA)

        # answer B: its own pass, its own h
        tag_b = letter('B', REJECTED, .26, 28).move_to(tag_a)
        hb = self.vec(H_B, self.HY, SCORE)
        sB = float(np.dot(H_B, W_ROW))
        pb = pill(f'score  {num(sB)}', REJECTED, 32).move_to([5.55, -2.05, 0])
        self.play(FadeTransform(tag_a, tag_b), FadeTransform(h, hb), run_time=.7)
        self.play(self.dot(hb, w), run_time=1.0)
        bx = w_frame.get_right()[0] + .25
        a4 = elbow([(bx, wy), (bx, pb.get_y()), (pb.get_left()[0] - .08, pb.get_y())], MUTED, sw=4, tip=.15)
        self.play(Create(a4), FadeIn(pb, scale=1.2), run_time=.6)
        gt = VGroup(letter('A', CHOSEN, .24, 26), label('>', 34, SCORE), letter('B', REJECTED, .24, 26)).arrange(RIGHT, buff=.15)
        gt.next_to(pb, DOWN, buff=.35)
        self.play(FadeIn(gt), run_time=.4)
        self.finish('05-score-head', hold=3.0, rows=1, score_a=sA, score_b=sB)


# ================================================================ 05 which position gives the score
class LastToken(Loop):
    """The score head gives a number at every position. Two answers that differ in one token give
    identical numbers before it and different numbers from it on. Only the last position has read the
    whole answer; its number is the score."""

    BAR, LIFT = .85, .62

    def row(self, toks, y, changed=None):
        chips = [chip('prompt', REJECTED, 26, h=.6)]
        for k, (_, s) in enumerate(toks):
            special = s.startswith('<|')
            col = ACCENT if (changed is not None and k == changed) else (MUTED if special else FOREGROUND)
            chips.append(chip(shown(s), col, 26, h=.6, fill=.18 if col == ACCENT else .05))
        return VGroup(*chips).arrange(RIGHT, buff=.1).move_to([0, y, 0]).align_to([-6.75, 0, 0], LEFT)

    def bars(self, vals, chips, col_fn):
        g = VGroup()
        for i, (v, c) in enumerate(zip(vals, chips)):
            b = Rectangle(width=.26, height=max(abs(v) * self.BAR, .02), stroke_width=0, fill_color=col_fn(i), fill_opacity=.9)
            b.move_to([c.get_x(), c.get_top()[1] + self.LIFT, 0], aligned_edge=DOWN if v >= 0 else UP)
            g.add(b)
        return g

    def construct(self):
        self.begin('Which position gives the score', badge='illustrative values')
        n = len(A_TOKENS) + 1
        change = 4                                   # position of ␣Some / ␣All (after the prompt chip)
        rng = np.random.default_rng(55)
        v1 = [float(x) for x in rng.uniform(-.55, .95, n)]
        v1[-1] = .82                                 # the accurate answer ends higher than the wrong variant
        v2 = v1[:change] + [x - .45 if x > .2 else x + .45 for x in v1[change:]]
        assert v1[:change] == v2[:change] and all(abs(a - b) > .05 for a, b in zip(v1[change:], v2[change:]))

        r1 = self.row(A_TOKENS, .75)
        assert r1.get_right()[0] < 6.6, r1.get_right()
        self.play(LaggedStart(*[FadeIn(c, shift=UP * .1) for c in r1], lag_ratio=.06), run_time=.8)
        b1 = self.bars(v1, r1, lambda i: SCORE)
        base1 = Line([r1.get_left()[0], r1.get_top()[1] + self.LIFT, 0], [r1.get_right()[0], r1.get_top()[1] + self.LIFT, 0],
                     color=GRID, stroke_width=1.5)
        self.play(FadeIn(base1), LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(b1, v1)],
                                             lag_ratio=.12), run_time=1.3)
        self.mark('every_position', positions=n)

        r2 = self.row(A2_TOKENS, -2.1, changed=change - 1)
        self.play(TransformFromCopy(r1, r2), run_time=.8)
        b2 = self.bars(v2, r2, lambda i: SCORE if i < change else ACCENT)
        base2 = base1.copy().set_y(r2.get_top()[1] + self.LIFT)
        self.play(FadeIn(base2), LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(b2, v2)],
                                             lag_ratio=.12), run_time=1.3)
        x0, x1 = r1[change].get_left()[0] - .08, r1[-1].get_right()[0] + .08
        band = Rectangle(width=x1 - x0, height=5.15, stroke_width=0, fill_color=ACCENT, fill_opacity=.08)
        band.move_to([(x0 + x1) / 2, .05, 0])
        same = label('same', 28, MUTED).move_to([(r1[0].get_x() + r1[change - 1].get_x()) / 2, .03, 0])
        diff = label('different', 28, ACCENT).move_to([band.get_x(), .03, 0])
        self.add(band); self.bring_to_back(band)
        self.play(FadeIn(band), FadeIn(same), FadeIn(diff), run_time=.6)
        self.mark('compare', first_different=change, equal_before=True, different_after=True)

        # causal attention: the last position reads every earlier position
        last = r1[-1]
        arcs = VGroup(*[ArcBetweenPoints(last.get_top() + UP * .05, c.get_top() + UP * .05, angle=1.0, color=CHOSEN,
                                         stroke_width=2.5, stroke_opacity=.6) for c in r1[:-1]])
        self.play(FadeOut(b1), FadeOut(base1), FadeOut(same), FadeOut(diff), band.animate.set_opacity(0), run_time=.4)
        self.play(LaggedStart(*[Create(a) for a in arcs[::-1]], lag_ratio=.1), run_time=1.2)
        self.mark('attention', reads=n - 1)
        self.play(FadeOut(arcs), FadeIn(b1), FadeIn(base1), run_time=.5)

        # the last number is the score; the rest are discarded
        for row, bs, vals, dy in [(r1, b1, v1, -.78), (r2, b2, v2, -.78)]:
            box = SurroundingRectangle(VGroup(bs[-1], row[-1]), color=SCORE, buff=.1, corner_radius=.08, stroke_width=3)
            self.play(Create(box), *[b.animate.set_opacity(.15) for b in bs[:-1]], run_time=.5)
            p = pill(f'score  {vals[-1]:.2f}'.replace('-', '−'), SCORE, 28)
            p.move_to([row[-1].get_x(), row.get_y() + dy, 0])
            self.play(TransformFromCopy(bs[-1], p), run_time=.6)
        self.finish('06-last-token', hold=3.0, positions=n, first_different=change, used='last',
                    tokens=[s for _, s in A_TOKENS], variant=[s for _, s in A2_TOKENS])


# ================================================================ 06 two softmaxes
class SoftmaxBase(Loop):
    def barrow(self, vals, names, x0, base, colors, step=1.0, unit=.8, w=.7, size=30, digits=1):
        g, labs, vl = VGroup(), VGroup(), VGroup()
        for k, (v, nm, c) in enumerate(zip(vals, names, colors)):
            b = Rectangle(width=w, height=max(abs(v) * unit, .02), stroke_width=0, fill_color=c, fill_opacity=.9)
            b.move_to([x0 + k * step, base, 0], aligned_edge=DOWN if v >= 0 else UP)
            g.add(b)
            labs.add(label(nm, size, c).move_to([x0 + k * step, base - .32 if v >= 0 else base + .32, 0]))
            vl.add(label(num(v, digits), 28, c).next_to(b, UP if v >= 0 else DOWN, buff=.1))
        return g, labs, vl


class ChatSoftmax(SoftmaxBase):
    """A chat model: one pass gives a logit for every vocabulary entry; softmax over all of them gives real
    probabilities of the next token, column by column; one token is sampled. Training uses −log p(next token)."""

    XS = [-2.3, -1.05, .2]             # the Part / Yes / No columns

    def column_bars(self, vals, base, unit, digits, w=.75):
        g, names, vl = VGroup(), VGroup(), VGroup()
        for x, v, n in zip(self.XS, vals, ['Part', 'Yes', 'No']):
            b = Rectangle(width=w, height=max(abs(v) * unit, .02), stroke_width=0, fill_color=SCORE, fill_opacity=.9)
            b.move_to([x, base, 0], aligned_edge=DOWN if v >= 0 else UP)
            g.add(b)
            names.add(label(n, 30, SCORE).move_to([x, base - .33 if v >= 0 else base + .33, 0]))
            vl.add(label(num(v, digits), 28, SCORE).next_to(b, UP if v >= 0 else DOWN, buff=.1))
        return g, names, vl

    def tail(self, base, hs, rng):
        return VGroup(*[Rectangle(width=.1, height=float(hh), stroke_width=0, fill_color=SCORE, fill_opacity=.25)
                        .move_to([1.15 + k * .17, base, 0], aligned_edge=DOWN) for k, hh in enumerate(rng.uniform(*hs, 8))])

    def construct(self):
        self.begin('Chat model: softmax over all tokens', badge='illustrative logits')
        top, bot = .7, -2.6
        z = [2.0, 1.0, -1.0]
        lb, ln, lv = self.column_bars(z, top, .75, 1)
        t1 = self.tail(top, (.06, .35), np.random.default_rng(6))
        t1_l = label(f'+{VOCAB_ROWS - 3:,} more', 26, MUTED).next_to(t1, UP, buff=.2)
        row1 = label('logits', 30, SCORE).move_to([-4.95, top + .55, 0])
        one = label('one pass', 26, MUTED).next_to(row1, DOWN, buff=.12)
        self.play(FadeIn(row1), FadeIn(one), run_time=.4)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(lb, z)], lag_ratio=.15),
                  FadeIn(ln), FadeIn(lv), FadeIn(t1), FadeIn(t1_l), run_time=1.0)
        self.mark('logits', logits=dict(zip(['Part', 'Yes', 'No'], z)), vocabulary=VOCAB_ROWS)

        sm = pill('softmax', SCORE, 28).move_to([-4.95, (top + bot) / 2 + .1, 0])
        d1 = varrow(sm.get_x(), one.get_bottom()[1] - .12, sm.get_top()[1] + .08)
        row2 = label('probabilities', 30, SCORE).move_to([-4.95, bot + .55, 0])
        d2 = varrow(sm.get_x(), sm.get_bottom()[1] - .08, row2.get_top()[1] + .12)
        self.play(GrowArrow(d1), FadeIn(sm), run_time=.4)
        pz = softmax(z)
        pb, pn, pv = self.column_bars([float(x) for x in pz], bot, 2.2, 3)
        t2 = self.tail(bot, (.01, .04), np.random.default_rng(7))
        self.play(GrowArrow(d2), FadeIn(row2), TransformFromCopy(lb, pb), FadeIn(pn), FadeIn(pv), FadeIn(t2), run_time=.9)
        sumto = label('sum = 1', 26, MUTED).next_to(t2, RIGHT, buff=.3)
        self.play(FadeIn(sumto), run_time=.3)
        self.mark('probabilities', p=[round(float(x), 3) for x in pz])

        samp = chip('Part', BASE, 32, h=.66, fill=1, text_color=WHITE).move_to([5.6, -1.55, 0])
        a4 = harrow(3.55, samp.get_left()[0] - .1, samp.get_y(), BASE)
        sample_l = label('sample', 28, BASE).next_to(a4, UP, buff=.12)
        self.play(GrowArrow(a4), FadeIn(samp, scale=.7), FadeIn(sample_l), run_time=.6)
        loss = VGroup(label('training:', 28, LOSS), label('loss =', 28, LOSS), label('−log p(next token)', 28, LOSS)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=.12)
        loss.move_to([5.35, .95, 0])
        assert loss.get_left()[0] > t1_l.get_right()[0] + .3
        self.play(FadeIn(loss), Indicate(pb[0], color=LOSS), run_time=.6)
        self.finish('07-chat-softmax', hold=3.0, chat=[round(float(x), 3) for x in pz], sampled='Part')


class PairSoftmax(SoftmaxBase):
    """A reward model: one pass per answer, one score each. Softmax over the two compared answers equals σ(gap)
    and is used only inside the training loss, −log p(chosen). In use, the raw score is the reward."""

    def construct(self):
        self.begin('Reward model: a two-way softmax', badge='illustrative scores')
        ya, yb = 1.75, .6
        ins, passes, pills, arrows = VGroup(), VGroup(), VGroup(), VGroup()
        for y, tag, col, s in [(ya, 'A', CHOSEN, 2.0), (yb, 'B', REJECTED, 1.0)]:
            i_ = VGroup(chip(PROMPT, REJECTED, 28, h=.62), letter(tag, col, .25, 28)).arrange(RIGHT, buff=.12).move_to([-4.95, y, 0])
            m_ = net_block(1.1, .75, 'pass', 28).move_to([-2.3, y, 0])
            p_ = pill(num(s), col, 34).move_to([-.55, y, 0])
            ins.add(i_); passes.add(m_); pills.add(p_)
            arrows.add(harrow(i_.get_right()[0] + .08, m_.get_left()[0] - .08, y),
                       harrow(m_.get_right()[0] + .08, p_.get_left()[0] - .08, y))
        for k in range(2):
            self.play(FadeIn(ins[k]), GrowArrow(arrows[2 * k]), FadeIn(passes[k]), GrowArrow(arrows[2 * k + 1]),
                      FadeIn(pills[k], scale=1.2), run_time=.7)
        self.mark('scores', a=2.0, b=1.0, passes=2)

        # training: the two scores compete in a softmax over two
        base = -2.65
        sb, sl, sv = self.barrow([2.0, 1.0], ['A', 'B'], -4.6, base, [CHOSEN, REJECTED], step=1.05, unit=.75)
        self.play(TransformFromCopy(pills[0], VGroup(sb[0], sv[0])), TransformFromCopy(pills[1], VGroup(sb[1], sv[1])),
                  FadeIn(sl), run_time=.8)
        sm = pill('softmax', SCORE, 28).move_to([-1.75, base + .55, 0])
        a5 = harrow(sb.get_right()[0] + .25, sm.get_left()[0] - .08, sm.get_y())
        q = softmax([2.0, 1.0])
        qb, ql, qv = self.barrow([float(x) for x in q], ['A', 'B'], .1, base, [CHOSEN, REJECTED], step=1.05, unit=2.2, w=.7, digits=3)
        a6 = harrow(sm.get_right()[0] + .08, qb.get_left()[0] - .25, sm.get_y())
        self.play(GrowArrow(a5), FadeIn(sm), GrowArrow(a6), run_time=.5)
        self.play(TransformFromCopy(sb, qb), FadeIn(ql), FadeIn(qv), run_time=.8)
        eq = label('= σ(2.0 − 1.0)', 30, SCORE).move_to([3.95, base + 1.25, 0])
        self.play(FadeIn(eq, shift=LEFT * .2), run_time=.4)
        self.mark('pair_softmax', p=[round(float(x), 3) for x in q], sigma=round(sig(1.0), 3))
        loss = label('loss = −log p(A)', 30, LOSS).move_to([3.95, base + .5, 0])
        assert loss.get_left()[0] > qb.get_right()[0] + .5 and eq.get_left()[0] > qb.get_right()[0] + .5
        self.play(FadeIn(loss), Indicate(qb[0], color=LOSS), run_time=.6)
        self.mark('loss', loss=round(-math.log(q[0]), 3))

        # in use: no softmax; the raw score is the reward
        train = VGroup(sb, sl, sv, a5, sm, a6, qb, ql, qv, eq, loss)
        box = DashedVMobject(RoundedRectangle(width=train.width + .5, height=train.height + .4, corner_radius=.18,
                                              stroke_color=MUTED, stroke_width=2).move_to(train), num_dashes=70)
        only = label('training only', 26, MUTED).next_to(box, UP, buff=.1).align_to(box, RIGHT)
        self.play(Create(box), FadeIn(only), train.animate.set_opacity(.25), run_time=.7)
        reward = pill('reward', SCORE, 34).move_to([4.6, ya, 0])
        go = harrow(pills[0].get_right()[0] + .08, reward.get_left()[0] - .08, ya, CHOSEN, sw=5, tip=.2)
        self.play(GrowArrow(go), FadeIn(reward, scale=1.2), run_time=.7)
        self.finish('08-pair-softmax', hold=3.0, pair=[round(float(x), 3) for x in q], reward=2.0)


# ================================================================ 07 random noise vs systematic bias
class NoiseVsBias(Loop):
    """Each tile is one labelled pair. The learned reward is shown as two weights: on being accurate and on
    being long. Random flips favour no feature: both weights shrink and keep their ratio. Flips that always
    favour the longer answer add up: the weight on length grows relative to accuracy. More data of the same
    kind changes neither."""

    TRUE = (2.0, .5)                   # illustrative weights: accurate, long
    UNIT = .85
    BX = (1.6, 3.1)                    # x of the two bars

    def tiles(self, cy, x0=-6.5, n=12, cols=6):
        g = VGroup()
        for k in range(n):
            t = RoundedRectangle(width=.55, height=.55, corner_radius=.08, stroke_width=0, fill_color=CHOSEN, fill_opacity=.8)
            t.move_to([x0 + .3 + (k % cols) * .7, cy + .35 - (k // cols) * .7, 0])
            g.add(t)
        return g

    def weights(self, base, vals, dashed=False):
        g = VGroup()
        for x, v in zip(self.BX, vals):
            r = Rectangle(width=.85, height=max(v * self.UNIT, .02))
            r.move_to([x, base, 0], aligned_edge=DOWN)
            if dashed:
                g.add(DashedVMobject(r.set_stroke(MUTED, 2.5).set_fill(opacity=0), num_dashes=28))
            else:
                g.add(r.set_stroke(width=0).set_fill(SCORE, .85))
        return g

    def flip(self, tile, up):
        sym = Triangle(stroke_width=0, fill_color=WHITE, fill_opacity=1).scale(.13).move_to(tile)
        if not up:
            sym.rotate(PI)
        return AnimationGroup(tile.animate.set_fill(LOSS, opacity=.9), FadeIn(sym, scale=.4)), sym

    def construct(self):
        self.begin('Random noise vs systematic bias', badge='illustration')
        lanes = [('random noise', 1.15), ('systematic bias', -1.95)]
        heads = VGroup(*[label(t, 26, FOREGROUND).move_to([-4.75, cy + 1.05, 0]) for t, cy in lanes])
        sep = DashedLine([-6.8, -.25, 0], [6.8, -.25, 0], color=GRID, stroke_width=1.5)
        tiles = [self.tiles(cy) for _, cy in lanes]
        bases = [cy - 1.0 for _, cy in lanes]
        axes = VGroup()
        for b in bases:
            axes.add(Line([self.BX[0] - .75, b, 0], [self.BX[1] + .75, b, 0], color=GRID, stroke_width=2))
            for x, n in zip(self.BX, ['accurate', 'long']):
                axes.add(label(n, 24, MUTED).move_to([x, b - .28, 0]))
        true = [self.weights(b, self.TRUE, dashed=True) for b in bases]
        learned = [self.weights(b, self.TRUE) for b in bases]
        legend = VGroup(VGroup(DashedVMobject(Rectangle(width=.36, height=.28).set_stroke(MUTED, 2.5), num_dashes=10),
                               label('true', 24, MUTED)).arrange(RIGHT, buff=.12),
                        VGroup(Rectangle(width=.36, height=.28, stroke_width=0, fill_color=SCORE, fill_opacity=.85),
                               label('learned', 24, SCORE)).arrange(RIGHT, buff=.12)).arrange(DOWN, aligned_edge=LEFT, buff=.14)
        legend.move_to([5.75, 2.25, 0])
        self.play(FadeIn(heads), FadeIn(sep), FadeIn(axes), FadeIn(legend), run_time=.5)
        self.play(*[LaggedStart(*[FadeIn(t, scale=.6) for t in g], lag_ratio=.04) for g in tiles], run_time=.8)
        self.play(*[FadeIn(w) for w in true], *[GrowFromEdge(b, DOWN) for w in learned for b in w], run_time=.7)
        self.mark('clean', weights=self.TRUE, ratio=self.TRUE[1] / self.TRUE[0])

        # random flips: either direction
        rnd = [(1, True), (4, False), (8, False), (10, True)]
        self.play(LaggedStart(*[self.flip(tiles[0][k], up)[0] for k, up in rnd], lag_ratio=.2), run_time=1.0)
        shrink = .55
        noisy = (self.TRUE[0] * shrink, self.TRUE[1] * shrink)
        self.play(Transform(learned[0], self.weights(bases[0], noisy)), run_time=.9)
        lab0 = label('smaller, same ratio', 26, SCORE).move_to([5.35, bases[0] + .7, 0])
        assert lab0.get_left()[0] > self.BX[1] + .55
        self.play(FadeIn(lab0), run_time=.3)
        self.mark('random', flips=len(rnd), weights=noisy, ratio=round(noisy[1] / noisy[0], 4))

        # biased flips: always toward the longer answer
        bias = [3, 7, 11]
        self.play(LaggedStart(*[self.flip(tiles[1][k], True)[0] for k in bias], lag_ratio=.2), run_time=.9)
        tilted = (1.7, 1.15)
        self.play(Transform(learned[1], self.weights(bases[1], tilted)), run_time=.9)
        lab1 = label('length rewarded', 26, SCORE).move_to([5.35, bases[1] + .7, 0])
        self.play(FadeIn(lab1), run_time=.3)
        self.mark('bias', flips=len(bias), weights=tilted, ratio=round(tilted[1] / tilted[0], 4))

        # more data of the same kind
        more = [self.tiles(cy, x0=-2.3, n=6, cols=2) for _, cy in lanes]
        extra = VGroup()
        for g, ks, ups in [(more[0], [2], [False]), (more[1], [1, 4], [True, True])]:
            for k, up in zip(ks, ups):
                g[k].set_fill(LOSS, opacity=.9)
                s = Triangle(stroke_width=0, fill_color=WHITE, fill_opacity=1).scale(.13).move_to(g[k])
                if not up:
                    s.rotate(PI)
                extra.add(s)
        plus = label('more data', 26, MUTED).move_to([-1.65, 2.2, 0])
        self.play(FadeIn(plus), *[LaggedStart(*[FadeIn(t, scale=.6) for t in g], lag_ratio=.05) for g in more],
                  FadeIn(extra), run_time=.9)
        self.play(Indicate(learned[0], color=SCORE, scale_factor=1.04), Indicate(learned[1], color=SCORE, scale_factor=1.04),
                  run_time=.8)
        self.finish('09-noise-vs-bias', hold=3.0, true_ratio=self.TRUE[1] / self.TRUE[0],
                    random=dict(ratio=round(noisy[1] / noisy[0], 4), scale=shrink),
                    bias=dict(ratio=round(tilted[1] / tilted[0], 4)), more_data='no change')


# ================================================================ 08 a proxy, optimized
class Proxy(Loop):
    """A reward model can score a longer answer higher without new content. Optimized against, its
    score keeps rising while true quality rises, peaks and falls (shape after Gao et al., 2022)."""

    GX0, GY0, SX, SY = .5, -2.45, 1.45, 1.05

    def gp(self, t, v):
        return np.array([self.GX0 + t * self.SX, self.GY0 + v * self.SY, 0])

    @staticmethod
    def proxy(t):
        return 2.4 * (1 - math.exp(-t / 1.1)) + .3 * t

    @staticmethod
    def true(t):
        return 2.25 * (t / 1.5) * math.exp(1 - t / 1.5)

    def construct(self):
        self.begin('A proxy, optimized', badge='illustrative shape')
        short = VGroup(letter('A', CHOSEN, .2, 26), label(A_TEXT, 26, FOREGROUND)).arrange(RIGHT, buff=.15)
        longer = VGroup(letter('A', CHOSEN, .2, 26), VGroup(label(A_TEXT, 26, FOREGROUND),
                        label('To be clear, it is important to note that…', 26, LOSS)).arrange(DOWN, aligned_edge=LEFT, buff=.1)
                        ).arrange(RIGHT, buff=.15, aligned_edge=UP)
        short.move_to([-6.6, 1.7, 0], aligned_edge=LEFT)
        longer.move_to([-6.6, .25, 0], aligned_edge=LEFT)
        s1 = pill('2.0', SCORE, 26).next_to(short, DOWN, buff=.2).align_to(short, LEFT).shift(RIGHT * .55)
        s2 = pill('2.4', SCORE, 26).next_to(longer, DOWN, buff=.2).align_to(longer, LEFT).shift(RIGHT * .55)
        self.play(FadeIn(short), run_time=.4)
        self.play(FadeIn(s1, scale=1.2), run_time=.35)
        self.play(FadeIn(longer), run_time=.5)
        self.play(FadeIn(s2, scale=1.2), Indicate(longer[1][1], color=LOSS, scale_factor=1.05), run_time=.6)
        self.mark('longer_scores_higher', short=2.0, longer=2.4)

        # the over-optimization curves
        xa = Arrow(self.gp(0, 0), self.gp(4.3, 0), buff=0, color=GRID, stroke_width=3, tip_length=.15)
        ya = Arrow(self.gp(0, 0), self.gp(0, 4.6), buff=0, color=GRID, stroke_width=3, tip_length=.15)
        xl = label('optimization against the reward model', 24, MUTED).next_to(xa, DOWN, buff=.12).align_to(xa, LEFT)
        pc = ParametricFunction(lambda t: self.gp(t, self.proxy(t)), t_range=[0, 4.1], color=SCORE, stroke_width=4)
        tc = ParametricFunction(lambda t: self.gp(t, self.true(t)), t_range=[0, 4.1], color=CHOSEN, stroke_width=4)
        pl = label('reward model score', 26, SCORE).next_to(self.gp(4.1, self.proxy(4.1)), UP, buff=.15).shift(LEFT * 1.55)
        assert pl.get_right()[0] < 6.85
        tl = label('true quality', 26, CHOSEN).next_to(self.gp(4.1, self.true(4.1)), DOWN, buff=.15).shift(LEFT * .4)
        self.play(GrowArrow(xa), GrowArrow(ya), FadeIn(xl), run_time=.5)
        t = ValueTracker(0)
        dp = always_redraw(lambda: Dot(self.gp(t.get_value(), self.proxy(t.get_value())), radius=.1, color=SCORE))
        dt = always_redraw(lambda: Dot(self.gp(t.get_value(), self.true(t.get_value())), radius=.1, color=CHOSEN))
        pcr = always_redraw(lambda: ParametricFunction(lambda u: self.gp(u, self.proxy(u)), t_range=[0, max(t.get_value(), .01)],
                                                       color=SCORE, stroke_width=4))
        tcr = always_redraw(lambda: ParametricFunction(lambda u: self.gp(u, self.true(u)), t_range=[0, max(t.get_value(), .01)],
                                                       color=CHOSEN, stroke_width=4))
        self.add(pcr, tcr, dp, dt)
        self.play(t.animate.set_value(1.5), run_time=1.6, rate_func=linear)
        self.mark('together', t=1.5, proxy=round(self.proxy(1.5), 3), true=round(self.true(1.5), 3))
        self.play(t.animate.set_value(4.1), run_time=2.4, rate_func=linear)
        self.remove(pcr, tcr, dp, dt)
        self.add(pc, tc, Dot(self.gp(4.1, self.proxy(4.1)), radius=.1, color=SCORE),
                 Dot(self.gp(4.1, self.true(4.1)), radius=.1, color=CHOSEN))
        self.play(FadeIn(pl), FadeIn(tl), run_time=.4)
        ts = np.linspace(1.5, 4.1, 40)
        area = Polygon(*[self.gp(u, self.proxy(u)) for u in ts], *[self.gp(u, self.true(u)) for u in ts[::-1]],
                       stroke_width=0, fill_color=LOSS, fill_opacity=.15)
        hack = label('reward hacking', 24, LOSS).move_to(self.gp(3.05, (self.proxy(3.05) + self.true(3.05)) / 2))
        self.add(area); self.bring_to_back(area)
        self.play(FadeIn(area), FadeIn(hack), run_time=.6)
        peak = 1.5
        self.finish('10-proxy', hold=3.0, peak_t=peak, proxy_end=round(self.proxy(4.1), 3), true_end=round(self.true(4.1), 3),
                    true_peak=round(self.true(peak), 3))


# ================================================================ 09 three uses
class ThreeUses(Loop):
    """A frozen reward model only scores. Best-of-n keeps the highest-scoring answer; labeling turns two
    scored answers into a new preference row; in RLHF the score is the reward that updates the policy."""

    def construct(self):
        self.begin('One reward model, three uses', badge='illustrative scores')
        rm = net_block(3.0, .9, 'reward model', 26).move_to([0, 2.0, 0])
        lk = lock_icon(NET, 1.1).next_to(rm, RIGHT, buff=.2)
        self.play(FadeIn(rm), FadeIn(lk), run_time=.5)
        cols = [-4.6, 0, 4.25]
        heads = VGroup(*[label(t, 26, FOREGROUND).move_to([x, .7, 0]) for t, x in zip(['best-of-n', 'labeling', 'RLHF'], cols)])
        bus_y = 1.3
        beams = VGroup(Line(rm.get_bottom(), [0, bus_y, 0]), Line([cols[0], bus_y, 0], [cols[2], bus_y, 0]),
                       *[Line([x, bus_y, 0], [x, 1.02, 0]) for x in cols]).set_stroke(NET, 2, .45)
        self.play(FadeIn(heads), Create(beams), run_time=.5)
        self.mark('frozen')

        # best-of-n
        sc = [.4, 2.0, 1.1, -.3]
        ans = VGroup(*[chip(f'answer {k + 1}', SCORE, 26, h=.44, w=1.7) for k in range(4)]).arrange(DOWN, buff=.14)
        ans.move_to([cols[0] - .55, -1.45, 0])
        bot = bot_icon(BASE, 1.1).next_to(ans, LEFT, buff=.3)
        self.play(FadeIn(bot), LaggedStart(*[FadeIn(a, shift=RIGHT * .2) for a in ans], lag_ratio=.12), run_time=.7)
        sl = VGroup(*[label(num(s), 26, SCORE).next_to(a, RIGHT, buff=.25) for s, a in zip(sc, ans)])
        self.play(LaggedStart(*[FadeIn(s) for s in sl], lag_ratio=.12), run_time=.6)
        best = int(np.argmax(sc))
        keep = SurroundingRectangle(VGroup(ans[best], sl[best]), color=CHOSEN, buff=.08, corner_radius=.08, stroke_width=3)
        self.play(Create(keep), *[VGroup(a, s).animate.fade(.7) for i, (a, s) in enumerate(zip(ans, sl)) if i != best],
                  run_time=.6)
        self.mark('best_of_n', scores=sc, kept=best + 1)

        # labeling: two new answers become a new row
        xa = chip('answer X', SCORE, 26, h=.44, w=1.6).move_to([cols[1] - .5, -.45, 0])
        xb = chip('answer Y', SCORE, 26, h=.44, w=1.6).move_to([cols[1] - .5, -1.1, 0])
        sx = label('1.6', 26, SCORE).next_to(xa, RIGHT, buff=.25)
        sy = label('0.2', 26, SCORE).next_to(xb, RIGHT, buff=.25)
        self.play(FadeIn(xa), FadeIn(xb), run_time=.4)
        self.play(FadeIn(sx), FadeIn(sy), run_time=.4)
        hd = VGroup(label('chosen', 24, MUTED), label('rejected', 24, MUTED)).arrange(RIGHT, buff=.5).move_to([cols[1], -2.0, 0])
        cells = VGroup(chip('X', CHOSEN, 26, h=.44, w=1.2), chip('Y', REJECTED, 26, h=.44, w=1.2)).arrange(RIGHT, buff=.2)
        cells.move_to([cols[1], -2.6, 0])
        hd[0].set_x(cells[0].get_x()); hd[1].set_x(cells[1].get_x())
        self.play(FadeIn(hd), TransformFromCopy(xa, cells[0]), TransformFromCopy(xb, cells[1]), run_time=.8)
        self.mark('labeling', chosen='X', rejected='Y')

        # RLHF: the score is the reward that updates the policy
        pol = bot_icon(BASE, 1.3).move_to([cols[2] - 1.35, -.6, 0])
        pol_l = label('policy', 26, BASE).next_to(pol, LEFT, buff=.15)
        an = chip('answer', SCORE, 26, h=.44, w=1.4).move_to([cols[2] + .55, -.6, 0])
        a1 = harrow(pol.get_right()[0] + .08, an.get_left()[0] - .08, an.get_y())
        sc3 = pill('2.0', SCORE, 28).move_to([cols[2] + .55, -1.7, 0])
        rw = label('reward', 26, SCORE).next_to(sc3, RIGHT, buff=.15)
        upd = elbow([(sc3.get_left()[0] - .08, sc3.get_y()), (pol.get_x(), sc3.get_y()), (pol.get_x(), pol.get_bottom()[1] - .1)],
                    CHOSEN, sw=4, tip=.18)
        upd_l = label('update', 26, CHOSEN).next_to(upd[0], DOWN, buff=.15)
        self.play(FadeIn(pol), FadeIn(pol_l), GrowArrow(a1), FadeIn(an), run_time=.6)
        self.play(FadeIn(sc3, scale=1.2), FadeIn(rw), run_time=.4)
        self.play(Create(upd), FadeIn(upd_l), run_time=.6)
        self.play(Indicate(pol, color=CHOSEN), run_time=.6)
        self.finish('11-three-uses', hold=3.0, uses=['best-of-n', 'labeling', 'RLHF'], best_of_n=sc, rlhf_reward=2.0)
