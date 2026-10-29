"""CAND-ANIM-038: figures for X-POST-004 (RL terms for LLMs).

One running example throughout. Prompt "Is OpenAI open?", state "OpenAI is", next token
open / closed / partly with probabilities 0.5 / 0.3 / 0.2 and judge scores 2 / 5 / 9.
Every figure uses the same cast:

- token chips (blue): the answer
- the judge's violet score stamp: the reward
- the amber value line: V
- green / red arrows: signed advantage

All numbers are illustrative; the arithmetic on them is exact. Timelines record explanatory
order, not timing.
"""
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np
from manim import *

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from manim_style import (label, BACKGROUND, FOREGROUND, MUTED, GRID, BASE,
                         INTERMEDIATE, COMPOSED, ACCENT)

label = partial(label, layout_scale=16)
NEGATIVE = '#B91C1C'  # signed quantities only (negative advantage)

# ---------------------------------------------------------------- the running example
TOKENS = ['open', 'closed', 'partly']
PROB = dict(open=.5, closed=.3, partly=.2)
SCORE = dict(open=2, closed=5, partly=9)
V = sum(PROB[t] * SCORE[t] for t in TOKENS)                       # 4.3
ADV = {t: round(SCORE[t] - V, 2) for t in TOKENS}                  # −2.3, +0.7, +4.7
SAMPLES = ['open', 'partly', 'open', 'closed', 'open', 'open', 'closed', 'partly', 'open', 'closed']
assert abs(V - 4.3) < 1e-9 and abs(sum(PROB[t] * ADV[t] for t in TOKENS)) < 1e-9
assert {t: SAMPLES.count(t) for t in TOKENS} == {t: round(10 * PROB[t]) for t in TOKENS}
assert abs(np.mean([SCORE[t] for t in SAMPLES]) - V) < 1e-9       # the 10 samples average to V exactly
# 05: answers that start the same way still differ, so each sample has its own judge score;
# each branch's samples average to its Q, and all ten average to V
SAMPLE_SCORES = dict(open=[3, 1, 2, 1, 3], closed=[5, 4, 6], partly=[10, 8])
assert all(len(SAMPLE_SCORES[t]) == SAMPLES.count(t) and np.mean(SAMPLE_SCORES[t]) == SCORE[t] for t in TOKENS)
assert abs(np.mean(sum(SAMPLE_SCORES.values(), [])) - V) < 1e-9


# ---------------------------------------------------------------- primitives
def chip(text, color=BASE, size=26, h=.5, fill=.08, text_color=None):
    t = label(text, size, text_color or color)
    box = RoundedRectangle(width=max(.5, t.width + .32), height=h, corner_radius=.12,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def stamp(score, r=.25):
    c = Circle(r, stroke_color=INTERMEDIATE, stroke_width=2.5, fill_color=INTERMEDIATE, fill_opacity=1)
    return VGroup(c, label(str(score), 24, WHITE).move_to(c))


def fmt(x):
    return f'{x:+.1f}'.replace('-', '−')


USER = '#475569'   # the user's turn: slate; the assistant's turn: BASE blue


def person_icon(color=USER, s=1.0):
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


def bubble(content, color, width, fill, stroke=2.2, pad=(.24, .14)):
    """A chat bubble of fixed width with its content left-aligned inside."""
    box = RoundedRectangle(width=width, height=content.height + 2 * pad[1], corner_radius=.16,
                           stroke_color=color, stroke_width=stroke, fill_color=color, fill_opacity=fill)
    content.move_to(box).align_to(box, LEFT).shift(RIGHT * pad[0])
    assert content.width < width - pad[0], (content.width, width)
    return VGroup(box, content)


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


# ---------------------------------------------------------------- 05
ANSWER = dict(open='It\'s in the name.', closed='', partly='some open weights')


class ValueQAdvantage(Loop):
    """The state sits at the height of its value. Ten samples run along the branches (thickness =
    probability) and land at their own judge scores. All ten averaged give V, a line through the
    state; each branch's samples averaged give its Q, next to that branch; Q − V is the advantage."""

    def construct(self):
        self.begin('Value, Q and advantage', 'illustrative numbers')
        y0, k, ax_x = -3.1, .58, 2.65
        Y = lambda v: y0 + k * v

        # the state: the user's question + the assistant's answer so far, placed at the height of V
        ux = -6.25                                                   # left edge of every bubble
        q_txt = label('Is OpenAI open?', 26, FOREGROUND)
        w = q_txt.width + .5
        question = bubble(q_txt, USER, w, .1)
        answer = bubble(label('OpenAI is', 28, WHITE), BASE, w, 1)
        chat_ = VGroup(question, answer).arrange(DOWN, buff=.2)
        chat_.shift([ux - chat_.get_left()[0], Y(V) - answer.get_y(), 0])
        icons = VGroup(person_icon().next_to(question, LEFT, buff=.16), bot_icon().next_to(answer, LEFT, buff=.16))
        node = Dot(answer.get_right() + RIGHT * .2, radius=.11, color=BASE)
        state_l = label('state', 30, BASE).next_to(chat_, DOWN, buff=.25)
        self.play(FadeIn(icons[0]), FadeIn(question, shift=RIGHT * .2), run_time=.6)
        self.play(FadeIn(icons[1]), FadeIn(answer, shift=RIGHT * .2), run_time=.6)
        self.play(FadeIn(state_l), FadeIn(node), run_time=.5)
        self.mark('state')

        # three possible next tokens, each continued to an answer, at the height of its Q
        cards, branches, pct = {}, {}, VGroup()
        card_x = node.get_x() + .95
        body = {t: VGroup(label(t, 30, BASE)) for t in TOKENS}
        for t in TOKENS:
            if ANSWER[t]:
                body[t].add(label(ANSWER[t], 24, BASE).next_to(body[t][0], DOWN, buff=.1, aligned_edge=LEFT))
        cw = max(b.width for b in body.values()) + .5                 # one width for the column
        for t in TOKENS:
            card = bubble(body[t], BASE, cw, .1, stroke=2)
            card.shift([card_x - card.get_left()[0], Y(SCORE[t]) - body[t][0].get_y(), 0])   # token line at its Q
            cards[t] = card
            a, b = node.get_center(), card.get_left() + UP * (Y(SCORE[t]) - card.get_y()) + LEFT * .04
            branches[t] = CubicBezier(a, a + RIGHT * .5, b + LEFT * .5, b, stroke_color=BASE,
                                      stroke_width=26 * PROB[t], stroke_opacity=.55)
            pct.add(label(f'{PROB[t]:.0%}', 24, BASE).next_to(b, LEFT, buff=.0).shift((DOWN if t == 'open' else UP) * .3))
        assert cards['closed'].get_bottom()[1] > Y(V) + .08, 'closed card touches the V line'
        self.play(LaggedStart(*[AnimationGroup(Create(branches[t]), FadeIn(cards[t], shift=RIGHT * .15)) for t in TOKENS],
                              lag_ratio=.3), run_time=1.4)
        self.play(FadeIn(pct), run_time=.4)
        self.mark('answers', prob=PROB)

        # the score axis
        axis = Line([ax_x, Y(0), 0], [ax_x, Y(10), 0], color=GRID, stroke_width=3.5)
        ticks = VGroup(*[VGroup(Line([ax_x - .09, Y(v), 0], [ax_x + .09, Y(v), 0], color=GRID, stroke_width=3.5),
                                label(str(v), 24, MUTED).next_to([ax_x - .12, Y(v), 0], LEFT, buff=.08))
                         for v in (0, 10)])
        cols = dict(open=3.8, closed=5.3, partly=6.6)
        self.play(Create(axis), FadeIn(ticks), run_time=.5)

        # ten samples: each runs along its branch, and the judge scores the finished answer
        slots = {t: iter(np.arange(SAMPLES.count(t)) - (SAMPLES.count(t) - 1) / 2) for t in TOKENS}
        scores = {t: iter(SAMPLE_SCORES[t]) for t in TOKENS}
        dots, runs, runners = {t: VGroup() for t in TOKENS}, [], []
        for t in SAMPLES:
            target = Dot([cols[t] + next(slots[t]) * .37, Y(next(scores[t])), 0], radius=.155, color=INTERMEDIATE)
            dots[t].add(target)
            d = Dot(node.get_center(), radius=.12, color=BASE)
            runners.append(d)
            runs.append(Succession(MoveAlongPath(d, branches[t].copy(), run_time=.45),
                                   Transform(d, target, path_arc=-.35, run_time=.55)))
        self.play(LaggedStart(*runs, lag_ratio=.22), run_time=3.2)
        self.remove(*runners)                                         # the landed dots take their place
        self.add(*[dots[t] for t in TOKENS])
        self.mark('sampled', n=10, counts={t: SAMPLES.count(t) for t in TOKENS}, scores=SAMPLE_SCORES)

        # V: all ten scores fall to their average, a line through the state
        ghosts = VGroup(*[d.copy() for t in TOKENS for d in dots[t]])
        vline = DashedLine(node.get_center(), [6.95, Y(V), 0], color=ACCENT, stroke_width=6, dash_length=.18)
        v_l = label(f'V = {V:.1f}', 34, ACCENT).next_to([ax_x + .2, Y(V) + .12, 0], UP + RIGHT, buff=.05)
        self.play(*[g.animate.move_to([g.get_x(), Y(V), 0]).set_color(ACCENT).scale(.6) for g in ghosts], run_time=.9)
        self.play(Create(vline), FadeOut(ghosts), FadeIn(v_l, shift=UP * .1), node.animate.set_color(ACCENT), run_time=.9)
        self.mark('value', V=V)

        # Q: each branch's scores fall to their own average, written next to that branch
        merged, qbars, q_l = VGroup(), VGroup(), VGroup()
        for t in TOKENS:
            m = VGroup(*[d.copy() for d in dots[t]])
            merged.add(m)
            qbars.add(Line([cols[t] - m.width / 2 - .14, Y(SCORE[t]), 0], [cols[t] + m.width / 2 + .14, Y(SCORE[t]), 0],
                           color=FOREGROUND, stroke_width=3))
            ql = label(f'Q = {SCORE[t]}', 28, FOREGROUND).next_to(cards[t], RIGHT, buff=.15)
            ql.shift(UP * (Y(SCORE[t]) - ql.get_y()))
            assert ql.get_right()[0] < ax_x - .1, (t, ql.get_right())
            q_l.add(ql)
        self.add(merged)
        self.play(*[d.animate.set_opacity(.22) for t in TOKENS for d in dots[t]],
                  *[d.animate.move_to([d.get_x(), Y(SCORE[t]), 0]) for t, m in zip(TOKENS, merged) for d in m], run_time=.9)
        self.play(LaggedStart(*[AnimationGroup(Create(b), FadeIn(q, shift=LEFT * .1)) for b, q in zip(qbars, q_l)],
                              lag_ratio=.3), run_time=1.0)
        self.mark('q', Q={t: SCORE[t] for t in TOKENS})

        # A = Q − V as signed arrows from the V line to each branch's average
        arrows, a_l = VGroup(), VGroup()
        for t, m in zip(TOKENS, merged):
            col = COMPOSED if ADV[t] > 0 else NEGATIVE
            end = Y(SCORE[t]) + (-.17 if ADV[t] > 0 else .17)
            ar = Arrow([cols[t], Y(V), 0], [cols[t], end, 0], buff=0, color=col, stroke_width=8,
                       tip_length=.2, max_tip_length_to_length_ratio=.8, max_stroke_width_to_length_ratio=40)
            arrows.add(ar)
            l = pill(fmt(ADV[t]), col, 32)   # opaque, so it may sit on the V line
            if t == 'partly':   # high on its arrow, clear of the closed group's label
                l.next_to([cols[t], Y(V) + .72 * (end - Y(V)), 0], LEFT, buff=.22)
            elif t == 'closed':
                l.next_to(m, UP, buff=.2)
            else:
                l.next_to([cols[t], (Y(V) + end) / 2, 0], RIGHT, buff=.22)
            a_l.add(l)
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(ar), FadeIn(l, scale=1.3)) for ar, l in zip(arrows, a_l)],
                              lag_ratio=.35), run_time=1.4)
        formula_ = formula('A = Q − V', [-4.9, -2.85, 0])
        assert formula_.get_top()[1] < state_l.get_bottom()[1] - .2
        self.play(FadeIn(formula_, shift=UP * .15), run_time=.5)
        self.mark('advantage', A=ADV, weighted_mean=0.0)
        self.finish('05-value-q-advantage', hold=3.0, V=V, Q={t: SCORE[t] for t in TOKENS}, A=ADV)


# ================================================================ shared pieces for 01–12
END = 'end'
TRAJ = ['OpenAI', 'is', 'open', '.', "It's", 'in', 'the', 'name', '.', END]    # the sampled answer
V_TRAJ = [V, V, V] + [SCORE['open']] * 7          # V(s_t) before each token: drops once "open" is written
R_END = SCORE['open']                              # the judge's score, on the last token only


def chat(question, answer, left=-6.25, y=.55, qsize=26, asize=28, width=None):
    """User bubble over assistant bubble, same left edge and width, with role icons."""
    q, a = label(question, qsize, FOREGROUND), label(answer, asize, WHITE)
    w = width or max(q.width, a.width) + .5
    qb, ab = bubble(q, USER, w, .1), bubble(a, BASE, w, 1)
    VGroup(qb, ab).arrange(DOWN, buff=.2).move_to([left + w / 2, y, 0])
    icons = VGroup(person_icon().next_to(qb, LEFT, buff=.16), bot_icon().next_to(ab, LEFT, buff=.16))
    return qb, ab, icons


def token_chip(t, size=26, h=.58):
    if t == END:
        return chip('end', MUTED, size - 2, h=h, fill=.12)
    return chip(t, BASE, size, h=h, fill=1, text_color=WHITE)


def token_row(tokens, size=26, gap=.07):
    return VGroup(*[token_chip(t, size) for t in tokens]).arrange(RIGHT, buff=gap)


def prob_bars(probs, x0, base, unit=4.0, width=.9, gap=.55, ghost=False, size=28):
    """Next-token bars (open, closed, partly). Returns bars, token labels, value labels."""
    bars, names, vals = VGroup(), VGroup(), VGroup()
    for i, t in enumerate(TOKENS):
        x = x0 + i * (width + gap)
        col = GRID if ghost else BASE
        b = Rectangle(width=width, height=max(unit * probs[i], .001), stroke_width=0, fill_color=col,
                      fill_opacity=.9 if not ghost else .6).move_to([x, base, 0], aligned_edge=DOWN)
        bars.add(b)
        names.add(label(t, size, BASE).move_to([x, base - .35, 0]))
        vals.add(label(f'{probs[i]:.2f}'.lstrip('0'), 24, BASE).next_to(b, UP, buff=.1))
    return bars, names, vals


def check_stamp(r=.3):
    c = Circle(r, stroke_width=0, fill_color=INTERMEDIATE, fill_opacity=1)
    m = VMobject(stroke_color=WHITE, stroke_width=6)
    m.set_points_as_corners([c.get_center() + np.array(p) * r for p in [(-.45, 0, 0), (-.12, -.35, 0), (.5, .4, 0)]])
    return VGroup(c, m)


def signed_arrow(x, y_from, y_to, width=8, tip=.2):
    col = COMPOSED if y_to > y_from else NEGATIVE
    return Arrow([x, y_from, 0], [x, y_to, 0], buff=0, color=col, stroke_width=width, tip_length=tip,
                 max_tip_length_to_length_ratio=.8, max_stroke_width_to_length_ratio=40)


def pill(text, col, size=28):
    t = label(text, size, col)
    box = RoundedRectangle(width=t.width + .34, height=t.height + .26, corner_radius=.2, stroke_color=col,
                           stroke_width=2.5, fill_color=WHITE, fill_opacity=1)
    return VGroup(box, t.move_to(box))


def formula(text, at):
    return chip(text, FOREGROUND, 32, h=.7, fill=.04).move_to(at)


# ---------------------------------------------------------------- 01
class OneScore(Loop):
    """DeepSeek-R1-Zero: a rule checks only the final answer, yet AIME 2024 pass@1 rose 15.6% → 71.0%."""

    def construct(self):
        self.begin('One score, thousands of tokens', 'DeepSeek-R1-Zero, 2025')
        cols, cell, gap, x0, ytop = 18, .27, .07, -5.6, 1.45
        q = bubble(label('AIME math problem', 26, FOREGROUND), USER, cols * (cell + gap) - gap, .1)
        q.move_to([x0, 2.25, 0], aligned_edge=LEFT)
        icons = VGroup(person_icon().next_to(q, LEFT, buff=.16), bot_icon().move_to([x0 - .45, ytop - .05, 0]))
        self.play(FadeIn(icons[0]), FadeIn(q, shift=RIGHT * .2), FadeIn(icons[1]), run_time=.6)

        def cell_at(i):
            r, c = divmod(i, cols)
            return Square(cell, stroke_width=0, fill_color=BASE, fill_opacity=.85).move_to(
                [x0 + cell / 2 + c * (cell + gap), ytop - r * (cell + gap), 0])
        first = VGroup(*[cell_at(i) for i in range(cols * 5 + 7)])
        think = label('reasoning tokens', 26, BASE).next_to(first, DOWN, buff=.25).align_to(first, LEFT)
        self.play(LaggedStart(*[FadeIn(c, scale=.5) for c in first], lag_ratio=.02), run_time=2.2)
        self.play(FadeIn(think), run_time=.3)
        ans = chip('answer', COMPOSED, 24, h=.44, fill=.12).next_to(first[-1], RIGHT, buff=.12)
        stamp_ = check_stamp().next_to(ans, RIGHT, buff=.15)
        rw = label('1 reward', 28, INTERMEDIATE).next_to(stamp_, RIGHT, buff=.15)
        rw.add_updater(lambda m: m.next_to(stamp_, RIGHT, buff=.15))
        self.play(FadeIn(ans, shift=RIGHT * .1), run_time=.35)
        self.play(GrowFromCenter(stamp_), FadeIn(rw), run_time=.5)
        self.mark('rollout', tokens='schematic', reward='final answer only')

        # AIME 2024 pass@1, before and after RL
        base, unit, bx = -2.55, .045, 4.6
        axis = Line([2.6, base, 0], [6.6, base, 0], color=GRID, stroke_width=3)
        pct = ValueTracker(15.6)
        bar = always_redraw(lambda: Rectangle(width=1.5, height=unit * pct.get_value(), stroke_width=0,
                                              fill_color=COMPOSED, fill_opacity=.9).move_to([bx, base, 0], aligned_edge=DOWN))
        num = always_redraw(lambda: label(f'{pct.get_value():.1f}%', 40, COMPOSED).next_to(bar, UP, buff=.15))
        cap = label('AIME 2024 pass@1', 26, MUTED).next_to(axis, DOWN, buff=.2)
        when = label('before RL', 26, FOREGROUND).move_to([bx, 2.3, 0])
        self.play(Create(axis), FadeIn(bar), FadeIn(num), FadeIn(cap), FadeIn(when), run_time=.7)
        self.mark('before', pct=15.6)
        start = DashedLine([bx - 1.0, base + unit * 15.6, 0], [bx + 1.0, base + unit * 15.6, 0], color=MUTED, stroke_width=3)
        start_l = label('15.6%', 26, MUTED).next_to(start, RIGHT, buff=.12)
        self.add(start, start_l)

        # RL training: accuracy rises, answers grow longer (schematic)
        more = VGroup(*[cell_at(i) for i in range(len(first), cols * 9 + 4)])
        after = label('after RL', 26, FOREGROUND).move_to(when)
        new_ans = ans.copy().next_to(more[-1], RIGHT, buff=.12)
        self.play(pct.animate.set_value(71.0), FadeTransform(when, after),
                  LaggedStart(*[FadeIn(c, scale=.5) for c in more], lag_ratio=.02),
                  think.animate.next_to(more, DOWN, buff=.25).align_to(first, LEFT),
                  ans.animate.move_to(new_ans), stamp_.animate.next_to(new_ans, RIGHT, buff=.15), run_time=3.0)
        self.mark('after', pct=71.0)
        self.finish('01-one-score', hold=3.0, before=15.6, after=71.0)


# ---------------------------------------------------------------- 02
class RlLoop(Loop):
    """The LLM (agent) gets a prompt from the environment, writes an answer, and the judge in the
    environment returns a reward; the reward updates the policy."""

    def construct(self):
        self.begin('The RL loop')
        robot = bot_icon(s=3.4).move_to([-4.2, .2, 0])
        agent_l = label('agent', 30, BASE).next_to(robot, DOWN, buff=.3)
        pol = chip('policy π', BASE, 28, h=.56, fill=.08).next_to(agent_l, DOWN, buff=.15)
        env = DashedVMobject(RoundedRectangle(width=4.6, height=5.6, corner_radius=.3, stroke_color=MUTED,
                                              stroke_width=2.5), num_dashes=70).move_to([4.3, -.15, 0])
        env_l = label('environment', 28, MUTED).next_to(env, UP, buff=.12)
        front = bubble(label('Is OpenAI open?', 26, FOREGROUND), USER, 3.1, 1)
        front[0].set_fill(WHITE, 1)
        behind = [RoundedRectangle(width=3.1, height=front.height, corner_radius=.16, stroke_color=USER, stroke_width=2.2,
                                   fill_color=WHITE, fill_opacity=1).shift(UP * .14 * i + RIGHT * .14 * i) for i in (2, 1)]
        front[0].set_fill(USER, .1)
        prompts = VGroup(*behind, VGroup(Rectangle(width=3.1, height=front.height, stroke_width=0, fill_color=WHITE,
                                                   fill_opacity=1).move_to(front), front)).move_to([4.3, 1.45, 0])
        judge = Circle(.62, stroke_width=0, fill_color=INTERMEDIATE, fill_opacity=1).move_to([4.3, -1.55, 0])
        judge_l = label('judge', 28, INTERMEDIATE).next_to(judge, DOWN, buff=.15)
        self.play(FadeIn(robot, scale=.8), FadeIn(agent_l), FadeIn(pol), run_time=.7)
        self.play(Create(env), FadeIn(env_l), FadeIn(prompts), FadeIn(judge), FadeIn(judge_l), run_time=.9)
        self.mark('pieces')

        top = CurvedArrow(prompts.get_left() + LEFT * .1, robot.get_top() + UP * .15 + RIGHT * .3, angle=.5,
                          color=USER, stroke_width=4, tip_length=.22)
        mid = Arrow(robot.get_right() + RIGHT * .15, judge.get_left() + LEFT * .15, buff=0, color=BASE,
                    stroke_width=5, tip_length=.22)
        bot = CurvedArrow(judge.get_bottom() + DOWN * .55 + LEFT * .3, robot.get_bottom() + DOWN * 1.35 + RIGHT * .6,
                          angle=-.45, color=INTERMEDIATE, stroke_width=4, tip_length=.22)
        tl = label('prompt', 26, USER).next_to(top, UP, buff=.08)
        ml = label('answer', 26, BASE).next_to(mid, UP, buff=.1)
        bl = label('reward', 26, INTERMEDIATE).next_to(bot, DOWN, buff=.08)
        self.play(Create(top), FadeIn(tl), run_time=.5)
        self.play(Create(mid), FadeIn(ml), run_time=.5)
        self.play(Create(bot), FadeIn(bl), run_time=.5)
        self.mark('arrows')

        for k, (score, speed) in enumerate([(9, 1.0), (2, .6)]):
            p = RoundedRectangle(width=.7, height=.38, corner_radius=.12, stroke_width=0, fill_color=USER,
                                 fill_opacity=.5).move_to(prompts)
            a = RoundedRectangle(width=.9, height=.38, corner_radius=.12, stroke_width=0, fill_color=BASE,
                                 fill_opacity=1).move_to(robot)
            self.play(MoveAlongPath(p, top.copy()), run_time=.8 * speed)
            self.play(FadeOut(p), robot.animate.scale(1.08), run_time=.25 * speed)
            self.play(robot.animate.scale(1 / 1.08), FadeIn(a), run_time=.2 * speed)
            self.play(a.animate.move_to(judge), run_time=.7 * speed)
            s = label(str(score), 48, WHITE).move_to(judge)
            self.play(FadeOut(a), FadeIn(s, scale=1.5), run_time=.35 * speed)
            dot = Dot(judge.get_bottom(), radius=.16, color=INTERMEDIATE)
            self.play(MoveAlongPath(dot, bot.copy()), FadeOut(s), run_time=.8 * speed)
            up = label('update', 28, COMPOSED).next_to(robot, UP, buff=.2)
            self.play(FadeOut(dot), robot[0].animate.set_fill(COMPOSED), robot[2].animate.set_color(COMPOSED),
                      robot[3].animate.set_color(COMPOSED), FadeIn(up), run_time=.3 * speed)
            self.play(robot[0].animate.set_fill(BASE), robot[2].animate.set_color(BASE), robot[3].animate.set_color(BASE),
                      FadeOut(up), run_time=.3 * speed)
            self.mark(f'cycle_{k + 1}', score=score)
        self.finish('02-rl-loop', hold=2.2, cycles=2)


# ---------------------------------------------------------------- 03
class StateAction(Loop):
    """The prompt plus the tokens so far is the state; the policy's next-token distribution
    is sampled for the action; the finished answer is the trajectory."""

    def construct(self):
        self.begin('State, action, trajectory', 'illustrative numbers')
        row_y, left = .85, -5.85
        q = bubble(label('Is OpenAI open?', 30, FOREGROUND), USER, 3.6, .1).move_to([left, 2.05, 0], aligned_edge=LEFT)
        row = token_row(TRAJ[:2], size=30).move_to([left, row_y, 0], aligned_edge=LEFT)
        icons = VGroup(person_icon().next_to(q, LEFT, buff=.16), bot_icon().next_to(row, LEFT, buff=.16))
        self.play(FadeIn(icons[0]), FadeIn(q, shift=RIGHT * .2), run_time=.5)
        self.play(FadeIn(icons[1]), LaggedStart(*[FadeIn(c, shift=RIGHT * .1) for c in row], lag_ratio=.4), run_time=.7)
        state_l = label('state', 30, BASE).next_to(row, DOWN, buff=.22).align_to(row, LEFT)
        self.play(FadeIn(state_l), run_time=.4)
        self.mark('state')

        # the policy: probabilities for the next token
        probs = [PROB[t] for t in TOKENS]
        bars, names, vals = prob_bars(probs, .2, -2.85, unit=3.4, width=1.0, gap=.7, size=30)
        pi_l = formula('π(next token | state)', [-3.6, -1.9, 0])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=.15), FadeIn(names), FadeIn(vals),
                  FadeIn(pi_l), run_time=1.0)
        self.mark('policy', probs=PROB)

        # sample: a marker hops over the bars and lands on "open"
        hop = Dot(radius=.13, color=INTERMEDIATE).next_to(bars[0], UP, buff=.55)
        self.play(FadeIn(hop), run_time=.2)
        for i in [1, 2, 1, 0]:
            self.play(hop.animate.next_to(bars[i], UP, buff=.55), run_time=.22)
        self.play(bars[0].animate.set_fill(INTERMEDIATE), Flash(hop, color=INTERMEDIATE, line_length=.18), run_time=.4)
        new = token_chip('open', 30).next_to(row, RIGHT, buff=.07)
        fly = names[0].copy()
        self.play(Transform(fly, new), FadeOut(hop), run_time=.7)
        self.remove(fly); self.add(new); row.add(new)
        act = label('action', 30, INTERMEDIATE).next_to(new, DOWN, buff=.22)
        self.play(FadeIn(act, shift=UP * .1), run_time=.4)
        self.mark('action', token='open')

        # the rest of the answer, token by token, then the whole answer is a trajectory
        self.play(FadeOut(VGroup(bars, names, vals, pi_l)), bars[0].animate.set_fill(BASE), run_time=.5)
        for t in TRAJ[3:]:
            c = token_chip(t, 30).next_to(row, RIGHT, buff=.07)
            self.play(FadeIn(c, shift=RIGHT * .1), run_time=.22)
            row.add(c)
        self.play(FadeOut(act), VGroup(icons, q, row, state_l).animate.shift(DOWN * 1.7), run_time=.5)
        brace = Brace(row, DOWN, buff=.75, color=BASE)
        traj_l = label('trajectory: one full answer', 30, BASE).next_to(brace, DOWN, buff=.15)
        self.play(GrowFromCenter(brace), FadeIn(traj_l), run_time=.6)
        self.mark('trajectory', tokens=len(TRAJ))
        self.finish('03-state-action', hold=2.6, tokens=len(TRAJ), action='open')


# ---------------------------------------------------------------- 04
class RewardReturn(Loop):
    """The judge scores the finished answer once: reward 0 on every token, the score on the last.
    With γ = 1 the return from every position is that score."""

    def construct(self):
        self.begin('Reward and return', 'illustrative numbers')
        row_y, left = 1.25, -5.0
        q = bubble(label('Is OpenAI open?', 30, FOREGROUND), USER, 3.6, .1).move_to([left, 2.35, 0], aligned_edge=LEFT)
        row = token_row(TRAJ, size=30).move_to([left, row_y, 0], aligned_edge=LEFT)
        icons = VGroup(person_icon().next_to(q, LEFT, buff=.16), bot_icon().next_to(row, LEFT, buff=.16))
        self.play(FadeIn(icons), FadeIn(q), LaggedStart(*[FadeIn(c, shift=RIGHT * .1) for c in row], lag_ratio=.08), run_time=1.2)
        self.mark('trajectory')

        st = stamp(R_END, r=.32).next_to(row, RIGHT, buff=.25)
        self.play(GrowFromCenter(st), run_time=.45)
        self.mark('judged', R=R_END)

        def cells(y, col):
            return VGroup(*[RoundedRectangle(width=c.width, height=.62, corner_radius=.1, stroke_color=col, stroke_width=2,
                                             fill_color=col, fill_opacity=.06).move_to([c.get_x(), y, 0]) for c in row])
        rew_y, ret_y = -.35, -2.05
        rc, gc = cells(rew_y, INTERMEDIATE), cells(ret_y, INTERMEDIATE)
        rl = label('reward', 30, INTERMEDIATE).next_to(rc, LEFT, buff=.3)
        gl = label('return', 30, INTERMEDIATE).next_to(gc, LEFT, buff=.3)
        zeros = VGroup(*[label('0', 28, MUTED).move_to(c) for c in rc[:-1]])
        self.play(FadeIn(rc), FadeIn(rl), LaggedStart(*[FadeIn(z, shift=DOWN * .15) for z in zeros], lag_ratio=.08), run_time=1.1)
        last = label(str(R_END), 30, WHITE).move_to(rc[-1])
        self.play(st.copy().animate.move_to(rc[-1]).scale(.85), rc[-1].animate.set_fill(INTERMEDIATE, .9), run_time=.6)
        self.add(last)
        sparse = label('sparse', 28, INTERMEDIATE).next_to(rc[-1], DOWN, buff=.18)
        self.play(FadeIn(sparse), run_time=.4)
        self.mark('reward', r=[0] * (len(TRAJ) - 1) + [R_END])

        # the return: everything still to come; the score copies back to every position
        self.play(FadeIn(gc), FadeIn(gl), run_time=.5)
        runner = label(str(R_END), 30, INTERMEDIATE).move_to(rc[-1])
        g_labels = VGroup()
        for c in reversed(gc):
            g = label(str(R_END), 30, INTERMEDIATE).move_to(c)
            self.play(Transform(runner.copy(), g), run_time=.16)
            g_labels.add(g)
        gamma = formula('γ = 1', [-4.9, -3.35, 0])
        self.play(FadeIn(gamma), run_time=.4)
        self.mark('return', G=[R_END] * len(TRAJ))
        self.finish('04-reward-return', hold=2.6, R=R_END, G=[R_END] * len(TRAJ))


FY0, FK, FAX = -3.1, .58, 2.65                       # 05's frame: score v sits at height FY(v); score axis at FAX
FY = lambda v: FY0 + FK * v
FCOLS = dict(open=3.7, closed=5.35, partly=6.6)      # 08–12: one column per branch on the score axis


def chat_state(Y, v):
    """User bubble over assistant bubble, the assistant's line at height Y(v). Returns the group
    (icons, bubbles, the word state) and the branch point at the end of the assistant bubble."""
    ux = -6.25
    q_txt = label('Is OpenAI open?', 26, FOREGROUND)
    w = q_txt.width + .5
    question, answer = bubble(q_txt, USER, w, .1), bubble(label('OpenAI is', 28, WHITE), BASE, w, 1)
    chat_ = VGroup(question, answer).arrange(DOWN, buff=.2)
    chat_.shift([ux - chat_.get_left()[0], Y(v) - answer.get_y(), 0])
    icons = VGroup(person_icon().next_to(question, LEFT, buff=.16), bot_icon().next_to(answer, LEFT, buff=.16))
    state_l = label('state', 30, BASE).next_to(chat_, DOWN, buff=.25)
    return VGroup(icons, chat_, state_l), answer.get_right() + RIGHT * .2


def answer_cards(Y, node_x):
    """The three answers, each at the height of its Q, with Q written beside it. Returns the cards,
    the branch end point of each card, and the Q labels, all keyed by token."""
    body = {t: VGroup(label(t, 30, BASE)) for t in TOKENS}
    for t in TOKENS:
        if ANSWER[t]:
            body[t].add(label(ANSWER[t], 24, BASE).next_to(body[t][0], DOWN, buff=.1, aligned_edge=LEFT))
    cw = max(b.width for b in body.values()) + .5
    cards, ends, qs = {}, {}, {}
    for t in TOKENS:
        card = bubble(body[t], BASE, cw, .1, stroke=2)
        card.shift([node_x + .95 - card.get_left()[0], Y(SCORE[t]) - body[t][0].get_y(), 0])
        cards[t] = card
        ends[t] = np.array([card.get_left()[0] - .04, Y(SCORE[t]), 0])
        ql = label(f'Q = {SCORE[t]}', 28, FOREGROUND).next_to(card, RIGHT, buff=.15)
        qs[t] = ql.shift(UP * (Y(SCORE[t]) - ql.get_y()))
    return cards, ends, qs


def branch_set(node_pt, ends, p):
    """Branches from the state to each card, thickness = probability, with the probability written."""
    lines, pct = VGroup(), VGroup()
    for t in TOKENS:
        a, b = np.array(node_pt), ends[t]
        lines.add(CubicBezier(a, a + RIGHT * .5, b + LEFT * .5, b, stroke_color=BASE,
                              stroke_width=max(26 * p[t], 1.5), stroke_opacity=.55))
        pct.add(label(f'{p[t]:.0%}', 24, BASE).next_to(b, LEFT, buff=0).shift((DOWN if t == 'open' else UP) * .3))
    return VGroup(lines, pct)


def state_tree(Y):
    """05's left side, built without animation: the state at the height of V, three branches
    (thickness = probability) to answer cards at the height of their Q, Q written beside each."""
    chat_, node_pt = chat_state(Y, V)
    cards, ends, qs = answer_cards(Y, node_pt[0])
    lines, pct = branch_set(node_pt, ends, PROB)
    return dict(chat=chat_, node=Dot(node_pt, radius=.11, color=BASE), cards=VGroup(*cards.values()),
                branches=lines, pct=pct, q=VGroup(*qs.values()), state_l=chat_[2])


def score_axis(Y, ax_x=FAX):
    axis = Line([ax_x, Y(0), 0], [ax_x, Y(10), 0], color=GRID, stroke_width=3.5)
    ticks = VGroup(*[VGroup(Line([ax_x - .09, Y(v), 0], [ax_x + .09, Y(v), 0], color=GRID, stroke_width=3.5),
                            label(str(v), 24, MUTED).next_to([ax_x - .12, Y(v), 0], LEFT, buff=.08)) for v in (0, 10)])
    return VGroup(axis, ticks)


def value_line(node_pt, card_left, v_y, ax_x=FAX):
    """The amber V line through the state; it skips the cards so it never crosses one."""
    return VGroup(DashedLine(node_pt, [card_left - .15, v_y, 0], color=ACCENT, stroke_width=6, dash_length=.18),
                  DashedLine([ax_x, v_y, 0], [6.95, v_y, 0], color=ACCENT, stroke_width=6, dash_length=.18))


def value_label(v, v_y, ax_x=FAX):
    return label(f'V = {v:.1f}', 34, ACCENT).next_to([ax_x + .2, v_y + .12, 0], UP + RIGHT, buff=.05)


def adv_arrows(Y, v, cols=FCOLS, size=28, rows=None):
    """A = Q − V: an arrow from the V line to each branch's Q, with its value. With sample rows
    (05's layout) the closed tag sits above its row; without them it sits right of its arrow.
    The group always holds one (arrow, tag) pair per branch, hidden ones transparent, because
    always_redraw copies each frame into the previous frame's structure."""
    g = VGroup()
    for t in TOKENS:
        a = SCORE[t] - v
        end = Y(SCORE[t]) + (-.17 if a > 0 else .17)
        show = abs(a) * FK > .3
        ar = signed_arrow(cols[t], Y(v), end if show else Y(v) + (.4 if a > 0 else -.4), width=8)
        tag = pill(fmt(a), COMPOSED if a > 0 else NEGATIVE, size)
        if t == 'closed' and rows is not None:
            tag.next_to(rows[1], UP, buff=.2)
        elif t == 'partly' and rows is not None:
            tag.next_to([cols[t], Y(v) + .72 * (end - Y(v)), 0], LEFT, buff=.22)
        else:
            tag.next_to([cols[t], (Y(v) + end) / 2, 0], LEFT if t == 'partly' else RIGHT, buff=.2)
            if t == 'closed':          # keep clear of the partly arrow to its right
                tag.scale(.9).next_to([cols[t], (Y(v) + end) / 2, 0], RIGHT, buff=.12)
        if not show:
            ar.set_opacity(0)
        if abs(a) < .15:
            tag.set_opacity(0)
        g.add(VGroup(ar, tag))
    return g


def sample_runs(node_pt, lines, Y, cols=FCOLS, samples=SAMPLES, scores=SAMPLE_SCORES):
    """Each sample runs along its branch and lands at its own judge score in its branch's column."""
    slots = {t: iter(np.arange(samples.count(t)) - (samples.count(t) - 1) / 2) for t in TOKENS}
    sc = {t: iter(scores[t]) for t in TOKENS}
    dots, runs, runners = {t: VGroup() for t in TOKENS}, [], []
    for t in samples:
        target = Dot([cols[t] + next(slots[t]) * .37, Y(next(sc[t])), 0], radius=.155, color=INTERMEDIATE)
        dots[t].add(target)
        d = Dot(node_pt, radius=.12, color=BASE)
        runners.append(d)
        runs.append(Succession(MoveAlongPath(d, lines[TOKENS.index(t)].copy(), run_time=.45),
                               Transform(d, target, path_arc=-.35, run_time=.55)))
    return dots, runs, runners


def to_averages(dots, Y, keep=.22):
    """Copies of each branch's dots fall to that branch's average; the originals stay faint.
    Returns the copies and the animations."""
    merged, anims = VGroup(), []
    for t in TOKENS:
        if not len(dots[t]):
            continue
        m = VGroup(*[d.copy() for d in dots[t]])
        merged.add(m)
        q = np.mean([Y_inv(d.get_y(), Y) for d in dots[t]])
        anims += [d.animate.set_opacity(keep) for d in dots[t]]
        anims += [c.animate.move_to([c.get_x(), Y(q), 0]) for c in m]
    return merged, anims


def Y_inv(y, Y):
    return (y - Y(0)) / (Y(1) - Y(0))


# ---------------------------------------------------------------- 06
class Baseline(Loop):
    """Judge scores are all positive, so measured from 0 every answer is pushed up. Raise the
    reference to V, the state's own value, and only better-than-usual answers go up."""

    def construct(self):
        self.begin('Baseline', 'illustrative numbers')
        y0, k, ax_x = -3.1, .58, 2.65                     # the same frame as 05
        Y = lambda v: y0 + k * v
        cols = dict(open=3.8, closed=5.3, partly=6.6)
        s = state_tree(Y)
        assert s['q'].get_right()[0] < ax_x - .1
        self.play(FadeIn(s['chat']), FadeIn(s['node']), run_time=.6)
        self.play(LaggedStart(*[AnimationGroup(Create(b), FadeIn(c, shift=RIGHT * .15))
                                for b, c in zip(s['branches'], s['cards'])], lag_ratio=.25),
                  FadeIn(s['pct']), run_time=1.1)
        self.play(FadeIn(s['q']), run_time=.4)
        self.mark('answers', Q=SCORE)

        # the score axis and each branch's average, as in 05
        axis = Line([ax_x, Y(0), 0], [ax_x, Y(10), 0], color=GRID, stroke_width=3.5)
        ticks = VGroup(*[VGroup(Line([ax_x - .09, Y(v), 0], [ax_x + .09, Y(v), 0], color=GRID, stroke_width=3.5),
                                label(str(v), 24, MUTED).next_to([ax_x - .12, Y(v), 0], LEFT, buff=.08))
                         for v in (0, 10)])
        rows, qbars = VGroup(), VGroup()
        for t in TOKENS:
            n = SAMPLES.count(t)
            r = VGroup(*[Dot([cols[t] + (i - (n - 1) / 2) * .37, Y(SCORE[t]), 0], radius=.155, color=INTERMEDIATE)
                         for i in range(n)])
            rows.add(r)
            qbars.add(Line([cols[t] - r.width / 2 - .14, Y(SCORE[t]), 0], [cols[t] + r.width / 2 + .14, Y(SCORE[t]), 0],
                           color=FOREGROUND, stroke_width=3))
        zero = Line([ax_x, Y(0), 0], [6.95, Y(0), 0], color=FOREGROUND, stroke_width=4)
        self.play(Create(axis), FadeIn(ticks), FadeIn(rows), FadeIn(qbars), Create(zero), run_time=.8)

        # measured from 0: every arrow points up
        base = ValueTracker(0.0)
        def arrows():
            b = base.get_value()
            return VGroup(*[signed_arrow(cols[t], Y(b), Y(SCORE[t]) + (-.17 if SCORE[t] > b else .17))
                            for t in TOKENS if abs(SCORE[t] - b) > .35])
        def tags():
            b = base.get_value(); g = VGroup()
            for t, r in zip(TOKENS, rows):
                a = SCORE[t] - b
                p = pill(fmt(a), COMPOSED if a > 0 else NEGATIVE, 30)
                if t == 'partly':        # fixed height between the closed tag and the partly row
                    p.next_to([cols[t], Y(7.3), 0], LEFT, buff=.22)
                elif t == 'closed':
                    p.next_to(r, UP, buff=.2)
                else:
                    p.next_to([cols[t], (Y(b) + Y(SCORE[t])) / 2, 0], RIGHT, buff=.22)
                g.add(p)
            return g
        ar, tg = always_redraw(arrows), always_redraw(tags)
        self.play(FadeIn(ar), FadeIn(tg), run_time=.8)
        self.mark('raw', push={t: SCORE[t] for t in TOKENS})
        self.wait(.8)

        # raise the reference line from 0 to V: it lands on the state
        vline = always_redraw(lambda: DashedLine([ax_x, Y(base.get_value()), 0], [6.95, Y(base.get_value()), 0],
                                                 color=ACCENT, stroke_width=6, dash_length=.18))
        self.play(FadeOut(zero), FadeIn(vline), run_time=.3)
        self.play(base.animate.set_value(V), run_time=2.0)
        reach = DashedLine(s['node'].get_center(), [ax_x, Y(V), 0], color=ACCENT, stroke_width=6, dash_length=.18)
        v_l = label(f'V = {V:.1f}', 34, ACCENT).next_to([ax_x + .2, Y(V) + .12, 0], UP + RIGHT, buff=.05)
        self.play(Create(reach), s['node'].animate.set_color(ACCENT), FadeIn(v_l, shift=UP * .1), run_time=.7)
        f = formula('A = Q − V', [-4.9, -2.85, 0])
        assert f.get_top()[1] < s['state_l'].get_bottom()[1] - .2
        self.play(FadeIn(f, shift=UP * .15), run_time=.4)
        self.mark('baseline', V=V, A=ADV)
        self.finish('06-baseline', hold=3.0, V=V, A=ADV)


# ---------------------------------------------------------------- 07
class CreditAssignment(Loop):
    """V along the sampled answer drops right after "open". The TD error puts the blame there
    (λ = 0); λ = 1 spreads it over every token up to the drop."""

    def construct(self):
        self.begin('Credit assignment', 'illustrative numbers')
        ux = -6.25
        question = bubble(label('Is OpenAI open?', 30, FOREGROUND), USER, 3.4, .1)
        text = label("OpenAI is open. It's in the name.", 42, WHITE)
        sizes = [len(t) for t in TRAJ[:-1]]                         # glyphs per token (spaces are not glyphs)
        assert sum(sizes) == len(text)
        cuts = np.cumsum([0] + sizes)
        words = [VGroup(*text[cuts[i]:cuts[i + 1]]) for i in range(len(sizes))]
        answer = bubble(text, BASE, text.width + .6, 1)
        question.move_to([ux, 2.45, 0], aligned_edge=LEFT)
        answer.move_to([ux, -.45, 0], aligned_edge=LEFT)
        icons = VGroup(person_icon().next_to(question, LEFT, buff=.16), bot_icon().next_to(answer, LEFT, buff=.16))
        st = stamp(R_END, r=.32).next_to(answer, RIGHT, buff=.25)
        self.play(FadeIn(icons[0]), FadeIn(question, shift=RIGHT * .2), run_time=.5)
        self.play(FadeIn(icons[1]), FadeIn(answer[0]), LaggedStart(*[FadeIn(w) for w in words], lag_ratio=.12), run_time=1.0)
        self.play(GrowFromCenter(st), run_time=.4)
        self.mark('trajectory', R=R_END)

        # the value line above the answer: V before each token; it drops once "open" is written
        vy = lambda v: answer.get_top()[1] + .3 + .45 * (v - R_END)
        cut_x = (words[2].get_right()[0] + words[3].get_left()[0]) / 2
        x0, x1 = words[0].get_left()[0] - .1, words[-1].get_right()[0] + .1
        hi = DashedLine([x0, vy(V), 0], [cut_x, vy(V), 0], color=ACCENT, stroke_width=6, dash_length=.16)
        lo = DashedLine([cut_x, vy(R_END), 0], [x1, vy(R_END), 0], color=ACCENT, stroke_width=6, dash_length=.16)
        hl = label(f'V = {V:.1f}', 30, ACCENT).next_to(hi, UP, buff=.1)
        ll = label(f'V = {R_END}', 30, ACCENT).next_to(lo, UP, buff=.1)
        self.play(Create(hi), FadeIn(hl), run_time=.7)
        self.play(Create(lo), FadeIn(ll), run_time=.7)
        self.mark('value', V=V_TRAJ)

        # the drop is the TD error, charged to "open"
        drop = signed_arrow(cut_x, vy(V), vy(R_END), width=8, tip=.18)
        mark_ = SurroundingRectangle(words[2], color=NEGATIVE, buff=.07, corner_radius=.08, stroke_width=0,
                                     fill_color=NEGATIVE, fill_opacity=1)
        td = pill(f'TD error {fmt(R_END - V)}', NEGATIVE, 28).next_to(drop, LEFT, buff=.15)
        assert td.get_top()[1] < vy(V) - .08 and td.get_bottom()[1] > answer.get_top()[1] + .05
        self.play(GrowArrow(drop), FadeIn(mark_), run_time=.6)
        self.bring_to_front(words[2])
        self.play(FadeIn(td, shift=LEFT * .1), run_time=.4)
        self.mark('td', delta=[0, 0, round(R_END - V, 2)] + [0] * 7)

        # advantage per token: λ = 0 keeps the blame on "open"; λ = 1 spreads it back to the start
        ay = answer.get_bottom()[1] - .3
        lam = ValueTracker(0)
        def blame():
            l_ = lam.get_value()
            g = VGroup()
            for i in range(3):
                a = (R_END - V) * l_ ** (2 - i)                # GAE with γ = 1: Σ λ^k δ_{t+k}
                if abs(a) > .05:
                    g.add(signed_arrow(words[i].get_x(), ay, ay + .45 * a, width=8, tip=.18))
            return g
        bl = always_redraw(blame)
        al = label('A', 34, FOREGROUND).next_to([answer.get_left()[0], ay - .5, 0], LEFT, buff=.3)
        assert st.get_right()[0] < 6.8
        self.play(FadeIn(bl), FadeIn(al), run_time=.5)
        self.mark('lambda0', A=[0, 0, round(R_END - V, 2)] + [0] * 7)

        track = Line([-1.2, -3.0, 0], [3.0, -3.0, 0], color=GRID, stroke_width=6)
        ends = VGroup(label('λ = 0', 28, FOREGROUND).next_to(track, LEFT, buff=.25),
                      label('λ = 1', 28, FOREGROUND).next_to(track, RIGHT, buff=.25))
        knob = always_redraw(lambda: Dot(track.point_from_proportion(lam.get_value()), radius=.17, color=ACCENT))
        self.play(Create(track), FadeIn(ends), FadeIn(knob), run_time=.5)
        self.play(lam.animate.set_value(1), run_time=2.0)
        self.mark('lambda1', A=[round(R_END - V, 2)] * 3 + [0] * 7)   # λ = 1: every token up to "open" gets −2.3
        self.wait(.6)
        self.play(lam.animate.set_value(0), run_time=1.4)
        f = formula('δ = r + V′ − V', [-4.6, -3.0, 0])
        assert f.get_right()[0] < ends[0].get_left()[0] - .2
        self.play(FadeIn(f, shift=UP * .15), run_time=.4)
        self.finish('07-credit-assignment', hold=2.6, V=V_TRAJ, delta_open=round(R_END - V, 2))


# ---------------------------------------------------------------- 08
def pg_steps(n=2):
    p = np.array([PROB[t] for t in TOKENS]); s = np.array([SCORE[t] for t in TOKENS], float)
    out = []
    for _ in range(n + 1):
        v = float(p @ s); a = s - v
        out.append((p.copy(), v, a))
        l = np.log(p) + p * a                      # one exact expected-gradient step on the logits
        p = np.exp(l) / np.exp(l).sum()
    return out


class PolicyGradient(Loop):
    """05's frame in motion: each update moves the branch thicknesses with the advantage. V rises
    and the state rises with it; once V passes 5, "closed" turns from above to below average."""

    def construct(self):
        self.begin('Policy gradient', 'illustrative step size')
        Y = FY
        steps = pg_steps(2)
        logits = [np.log(st[0]) for st in steps]
        s = ValueTracker(0.0)
        def probs():
            x = s.get_value(); i = min(int(x), len(logits) - 2); f = x - i
            l = (1 - f) * logits[i] + f * logits[i + 1]
            e = np.exp(l - l.max())
            return dict(zip(TOKENS, e / e.sum()))
        val = lambda: sum(probs()[t] * SCORE[t] for t in TOKENS)

        chat_, node_pt = chat_state(Y, V)
        node = Dot(node_pt, radius=.11, color=ACCENT)
        state = VGroup(chat_, node)
        state.add_updater(lambda m: m.shift(UP * (Y(val()) - m[1].get_y())))   # the state sits at its value
        cards, ends, qs = answer_cards(Y, node_pt[0])
        card_left = min(c.get_left()[0] for c in cards.values())
        lines = always_redraw(lambda: branch_set(node.get_center(), ends, probs()))
        marks = VGroup(*[Line([FCOLS[t] - .35, Y(SCORE[t]), 0], [FCOLS[t] + .35, Y(SCORE[t]), 0], color=FOREGROUND,
                              stroke_width=4) for t in TOKENS])
        vline = always_redraw(lambda: value_line(node.get_center(), card_left, node.get_y()))
        vlab = always_redraw(lambda: value_label(val(), node.get_y()))
        arrows = always_redraw(lambda: adv_arrows(Y, val()))
        self.add(state)
        self.play(FadeIn(chat_), FadeIn(node), FadeIn(lines), *[FadeIn(c) for c in cards.values()],
                  *[FadeIn(q) for q in qs.values()], FadeIn(score_axis(Y)), FadeIn(marks), run_time=.9)
        self.play(FadeIn(vline), FadeIn(vlab), run_time=.5)
        self.mark('policy', p=steps[0][0].round(3).tolist())
        self.play(FadeIn(arrows), run_time=.6)
        f = formula('Δ log π ∝ A', [-4.9, -2.85, 0])
        self.play(FadeIn(f, shift=UP * .15), run_time=.4)
        self.mark('advantage_0', V=round(steps[0][1], 2), A=steps[0][2].round(2).tolist())
        for it in (1, 2):
            self.wait(.5)
            self.play(s.animate.set_value(it), run_time=2.4)
            self.mark(f'step_{it}', p=steps[it][0].round(3).tolist())
            self.mark(f'advantage_{it}', V=round(steps[it][1], 2), A=steps[it][2].round(2).tolist())
            assert abs(val() - steps[it][1]) < 1e-9
        self.finish('08-policy-gradient', hold=2.8, p=[st[0].round(3).tolist() for st in steps],
                    V=[round(st[1], 2) for st in steps])


# ---------------------------------------------------------------- 09
class Exploration(Loop):
    """Greedy decoding only ever takes the thickest branch: it never sees what the others score,
    its average is its own score, A = 0, and nothing moves. Sampling tries every branch, finds the
    high scores, and the branches move."""

    def construct(self):
        self.begin('Exploration', 'illustrative numbers')
        Y = FY
        chat_, node_pt = chat_state(Y, V)
        node = Dot(node_pt, radius=.11, color=BASE)
        cards, ends, qs = answer_cards(Y, node_pt[0])
        card_left = min(c.get_left()[0] for c in cards.values())
        unknown = {t: label('Q = ?', 28, MUTED).move_to(qs[t], aligned_edge=LEFT) for t in TOKENS}
        p0, p1 = PROB, dict(zip(TOKENS, pg_steps(1)[1][0]))
        mix = ValueTracker(0.0)
        probs = lambda: {t: (1 - mix.get_value()) * p0[t] + mix.get_value() * p1[t] for t in TOKENS}
        tree = always_redraw(lambda: branch_set(node_pt, ends, probs()))
        mode = chip('greedy', FOREGROUND, 30, h=.62, fill=.05).move_to([0, 3.45, 0])
        self.play(FadeIn(chat_), FadeIn(node), FadeIn(tree), *[FadeIn(c) for c in cards.values()],
                  *[FadeIn(u) for u in unknown.values()], FadeIn(score_axis(Y)), FadeIn(mode), run_time=.9)
        self.mark('start', p=[p0[t] for t in TOKENS])

        # greedy: three answers, all down the thickest branch
        g_dots, g_runs, g_run = sample_runs(node_pt, tree[0], Y, samples=['open'] * 3,
                                            scores=dict(open=[3, 1, 2], closed=[], partly=[]))
        self.play(LaggedStart(*g_runs, lag_ratio=.35), run_time=2.0)
        self.remove(*g_run); self.add(g_dots['open'])
        merged, anims = to_averages(g_dots, Y)
        self.add(merged)
        self.play(*anims, run_time=.7)
        g_mean = 2
        g_line = DashedLine([FAX, Y(g_mean), 0], [6.95, Y(g_mean), 0], color=ACCENT, stroke_width=6, dash_length=.18)
        g_vlab = label(f'V = {g_mean:.1f}', 34, ACCENT).next_to([6.95, Y(g_mean) + .12, 0], UP + LEFT, buff=.05)
        g_q = label(f'Q = {g_mean}', 28, FOREGROUND).move_to(unknown['open'], aligned_edge=LEFT)
        zero = pill('A = 0', MUTED, 28).next_to(merged, UP, buff=.62)
        self.play(Create(g_line), FadeIn(g_vlab), FadeTransform(unknown['open'], g_q), run_time=.7)
        self.play(FadeIn(zero, scale=1.3), Wiggle(tree, scale_value=1.02), run_time=1.0)
        self.mark('greedy', seen=['open'], V=g_mean, A=0)
        self.wait(.6)

        # sampling: every branch gets tried
        unknown['open'] = label('Q = ?', 28, MUTED).move_to(qs['open'], aligned_edge=LEFT)
        self.play(*[FadeOut(m) for m in (g_dots['open'], merged, g_line, g_vlab, zero)],
                  FadeTransform(g_q, unknown['open']),
                  Transform(mode, chip('sampling', FOREGROUND, 30, h=.62, fill=.05).move_to(mode)), run_time=.6)
        dots, runs, runners = sample_runs(node_pt, tree[0], Y)
        self.play(LaggedStart(*runs, lag_ratio=.2), run_time=2.8)
        self.remove(*runners); self.add(*dots.values())
        self.mark('draws', counts={t: SAMPLES.count(t) for t in TOKENS})
        merged, anims = to_averages(dots, Y)
        self.add(merged)
        self.play(*anims, run_time=.7)
        self.play(*[FadeTransform(unknown[t], qs[t]) for t in TOKENS], run_time=.5)
        vl = value_line(node_pt, card_left, Y(V))
        self.play(Create(vl), node.animate.set_color(ACCENT), FadeIn(value_label(V, Y(V))), run_time=.7)
        self.play(FadeIn(adv_arrows(Y, V, rows=merged)), run_time=.6)
        self.play(mix.animate.set_value(1), run_time=1.6)
        self.mark('update', greedy=[p0[t] for t in TOKENS], sampling=[round(p1[t], 3) for t in TOKENS])
        self.finish('09-exploration', hold=2.8, greedy=[p0[t] for t in TOKENS], sampling=[round(p1[t], 3) for t in TOKENS])


# ---------------------------------------------------------------- 10
class RatioClip(Loop):
    """The answers were sampled from π_old. As π("partly") grows, ρ = π / π_old tracks the drift;
    with A > 0 the push stops once ρ reaches 1 + ε."""

    def construct(self):
        self.begin('Ratio and clipping', 'ε = 0.2')
        Y = FY
        old, eps = PROB['partly'], .2
        lo, hi = old * (1 - eps), old * (1 + eps)
        chat_, node_pt = chat_state(Y, V)
        node = Dot(node_pt, radius=.11, color=BASE)
        cards, ends, qs = answer_cards(Y, node_pt[0])
        p = ValueTracker(old)
        def probs():
            x = p.get_value()
            return dict(open=PROB['open'] * (1 - x) / (1 - old), closed=PROB['closed'] * (1 - x) / (1 - old), partly=x)
        tree = always_redraw(lambda: branch_set(node_pt, ends, probs()))
        dim = VGroup(*[cards[t] for t in ('open', 'closed')], *[qs[t] for t in ('open', 'closed')])
        self.play(FadeIn(chat_), FadeIn(node), FadeIn(tree), *[FadeIn(c) for c in cards.values()],
                  *[FadeIn(q) for q in qs.values()], run_time=.8)
        self.play(dim.animate.set_opacity(.25), run_time=.4)

        # the ratio meter: ρ from 0.7 to 1.3, the window [1 − ε, 1 + ε] shaded
        mx = lambda r: 2.9 + (r - .7) / .6 * 3.4
        my = .2
        track = Line([mx(.7), my, 0], [mx(1.3), my, 0], color=GRID, stroke_width=6)
        window = Rectangle(width=mx(1 + eps) - mx(1 - eps), height=.7, stroke_width=0, fill_color=ACCENT,
                           fill_opacity=.12).move_to([(mx(1 - eps) + mx(1 + eps)) / 2, my, 0])
        fences = VGroup(*[DashedLine([mx(r), my - .5, 0], [mx(r), my + .5, 0], color=ACCENT, stroke_width=4)
                          for r in (1 - eps, 1 + eps)])
        f_l = VGroup(label('1 − ε', 26, ACCENT).next_to(fences[0], DOWN, buff=.12),
                     label('1', 26, MUTED).move_to([mx(1), my - .62, 0]),
                     label('1 + ε', 26, ACCENT).next_to(fences[1], DOWN, buff=.12))
        rho = lambda: p.get_value() / old
        knob = always_redraw(lambda: Dot([mx(rho()), my, 0], radius=.2, color=BASE))
        r_l = always_redraw(lambda: label(f'ρ = {rho():.2f}', 40, FOREGROUND).move_to([mx(rho()), my + 1.0, 0]))
        self.play(Create(track), FadeIn(window), Create(fences), FadeIn(f_l), FadeIn(knob), FadeIn(r_l), run_time=.8)
        self.mark('setup', old=old, window=[round(lo, 3), round(hi, 3)])

        # A > 0 pushes π("partly") up; the push stops at 1 + ε
        push = always_redraw(lambda: Arrow([mx(rho()) + .25, my, 0], [mx(rho()) + 1.05, my, 0], buff=0, color=COMPOSED,
                                           stroke_width=8, tip_length=.22, max_tip_length_to_length_ratio=.5))
        a_l = pill(fmt(ADV['partly']), COMPOSED, 30).move_to([mx(1.15), my - 1.55, 0])
        f = formula('ρ = π / π_old', [-4.9, -2.85, 0])
        self.play(FadeIn(push), FadeIn(a_l), FadeIn(f), run_time=.5)
        self.mark('ratio', rho=1.0)
        self.play(p.animate.set_value(hi), run_time=2.6, rate_func=linear)
        self.remove(push)
        stopped = Arrow([mx(1 + eps) + .25, my, 0], [mx(1 + eps) + 1.05, my, 0], buff=0, color=MUTED, stroke_width=8,
                        tip_length=.22, max_tip_length_to_length_ratio=.5)
        self.add(stopped)
        self.play(Flash(fences[1].get_center(), color=ACCENT), FadeOut(stopped), a_l.animate.set_opacity(.3), run_time=.8)
        self.mark('clipped', rho=1.2, p=round(hi, 3))
        self.finish('10-ratio-clip', hold=2.8, window=[round(lo, 3), round(hi, 3)], rho_max=1.2)


# ---------------------------------------------------------------- 11
REF_FLATTER = .05
BETA = 1.0


def kl2(q, r=REF_FLATTER):
    return q * np.log(q / r) + (1 - q) * np.log((1 - q) / (1 - r))


class KlPenalty(Loop):
    """A judge that likes flattery scores it 9.6 against 9. As the policy's branch to flattery
    thickens away from the reference (grey), the judge's score rises a little and β·KL a lot,
    so R − β·KL falls below where it started."""

    def construct(self):
        self.begin('KL penalty', 'illustrative numbers')
        ux = -6.25
        q_txt, a_txt = label('Is OpenAI open?', 26, FOREGROUND), label('OpenAI is partly.', 26, WHITE)
        w = max(q_txt.width, a_txt.width) + .5
        question, answer = bubble(q_txt, USER, w, .1), bubble(a_txt, BASE, w, 1)
        chat_ = VGroup(question, answer).arrange(DOWN, buff=.2)
        chat_.shift([ux - chat_.get_left()[0], .45 - answer.get_y(), 0])
        icons = VGroup(person_icon().next_to(question, LEFT, buff=.16), bot_icon().next_to(answer, LEFT, buff=.16))
        node_pt = answer.get_right() + RIGHT * .2
        texts = dict(plain='Some open weights.', flat='What a brilliant question!')
        cw = max(label(s_, 22, BASE).width for s_ in texts.values()) + .45
        cards, ends, stamps = {}, {}, {}
        for k_, y in (('plain', 1.85), ('flat', -1.0)):
            c = bubble(label(texts[k_], 22, BASE), BASE, cw, .1, stroke=2)
            c.move_to([node_pt[0] + .75, y, 0], aligned_edge=LEFT)
            cards[k_], ends[k_] = c, c.get_left() + LEFT * .04
        stamps['plain'] = stamp(9, r=.3).next_to(cards['plain'], RIGHT, buff=.2)
        stamps['flat'] = stamp(9.6, r=.36).next_to(cards['flat'], RIGHT, buff=.2)
        stamps['flat'][1].scale(.8)
        q = ValueTracker(REF_FLATTER)
        def curve(k_, width, col, op):
            a, b = node_pt, ends[k_]
            return CubicBezier(a, a + RIGHT * .5, b + LEFT * .5, b, stroke_color=col, stroke_width=max(width, 1.5),
                               stroke_opacity=op)
        ref = VGroup(curve('plain', 30 * (1 - REF_FLATTER), MUTED, .3), curve('flat', 30 * REF_FLATTER, MUTED, .3))
        pol = always_redraw(lambda: VGroup(curve('plain', 30 * (1 - q.get_value()), BASE, .7),
                                           curve('flat', 30 * q.get_value(), BASE, .7)))
        pct = always_redraw(lambda: VGroup(
            label(f'{1 - q.get_value():.0%}', 24, BASE).next_to(ends['plain'], LEFT, buff=0).shift(UP * .32),
            label(f'{q.get_value():.0%}', 24, BASE).next_to(ends['flat'], LEFT, buff=0).shift(DOWN * .32)))
        legend = VGroup(VGroup(Line(ORIGIN, RIGHT * .5, color=MUTED, stroke_width=10, stroke_opacity=.45), label('π_ref', 26, MUTED))
                        .arrange(RIGHT, buff=.12),
                        VGroup(Line(ORIGIN, RIGHT * .5, color=BASE, stroke_width=10, stroke_opacity=.7), label('π', 26, BASE))
                        .arrange(RIGHT, buff=.12)).arrange(RIGHT, buff=.45).move_to([-4.6, -2.6, 0])
        self.play(FadeIn(icons), FadeIn(chat_), run_time=.5)
        self.play(FadeIn(ref), FadeIn(pol), FadeIn(cards['plain']), FadeIn(cards['flat']), FadeIn(pct), FadeIn(legend),
                  run_time=.8)
        self.play(GrowFromCenter(stamps['plain']), GrowFromCenter(stamps['flat']), run_time=.5)
        assert stamps['flat'].get_right()[0] < 2.7, stamps['flat'].get_right()
        self.mark('answers', judge=dict(plain=9, flattery=9.6))

        # one stacked bar: the judge's average score, with β·KL cut off its top
        sb, su, sx = -3.1, .5, 3.5
        judge = lambda: 9 * (1 - q.get_value()) + 9.6 * q.get_value()
        pen = lambda: BETA * kl2(q.get_value())
        def stack():
            j, k_ = judge(), pen()
            net = Rectangle(width=1.3, height=su * (j - k_), stroke_width=0, fill_color=COMPOSED, fill_opacity=.9) \
                .move_to([sx, sb, 0], aligned_edge=DOWN)
            cut = Rectangle(width=1.3, height=max(su * k_, .01), stroke_width=0, fill_color=FOREGROUND, fill_opacity=.85) \
                .next_to(net, UP, buff=0)
            top = Line(cut.get_corner(UL) + LEFT * .12, cut.get_corner(UR) + RIGHT * .12, color=INTERMEDIATE, stroke_width=6)
            labels = VGroup(label(f'judge {j:.2f}', 26, INTERMEDIATE).next_to(top, RIGHT, buff=.15).shift(UP * .15),
                            label(f'R − β·KL {j - k_:.2f}', 26, COMPOSED).next_to(net, RIGHT, buff=.2).shift(DOWN * .4))
            if k_ > .05:
                labels.add(label(f'β·KL {k_:.2f}', 26, FOREGROUND).next_to(cut, RIGHT, buff=.2).shift(DOWN * .05))
            for l_ in labels:
                assert l_.get_right()[0] < 7.0, l_.get_right()
            return VGroup(net, cut, top, labels)
        st = always_redraw(stack)
        start = DashedLine([sx - 1.0, sb + su * 9.03, 0], [sx + 1.0, sb + su * 9.03, 0], color=MUTED, stroke_width=3)
        axis = Line([sx - 1.1, sb, 0], [sx + 1.1, sb, 0], color=GRID, stroke_width=3)
        self.play(FadeIn(st), Create(axis), run_time=.6)
        self.add(start)
        self.mark('close', q=REF_FLATTER, judge=round(judge(), 2), net=round(judge() - pen(), 2))
        self.play(q.animate.set_value(.9), run_time=3.6)
        self.mark('drift', q=.9, kl=round(float(kl2(.9)), 3), judge=round(judge(), 2), net=round(judge() - pen(), 2))
        self.finish('11-kl-penalty', hold=3.0, beta=BETA, net_close=round(9 + .6 * REF_FLATTER, 2),
                    net_drift=round(judge() - pen(), 2))


# ---------------------------------------------------------------- 12
class TermMap(Loop):
    """The whole loop on 05's frame, each core term placed on the part of the picture it names:
    state, policy, action, reward, V, Q, advantage, then the update that reshapes the branches."""

    def construct(self):
        self.begin('RL terms for LLMs', 'illustrative numbers')
        Y = FY
        chat_, node_pt = chat_state(Y, V)
        node = Dot(node_pt, radius=.11, color=BASE)
        cards, ends, qs = answer_cards(Y, node_pt[0])
        card_left = min(c.get_left()[0] for c in cards.values())
        p1 = dict(zip(TOKENS, pg_steps(1)[1][0]))
        mix = ValueTracker(0.0)
        tree = always_redraw(lambda: branch_set(node_pt, ends, {t: (1 - mix.get_value()) * PROB[t] + mix.get_value() * p1[t]
                                                                for t in TOKENS}))
        term = lambda s_, col: label(s_, 28, col)
        self.play(FadeIn(chat_), FadeIn(node), run_time=.5)
        self.mark('state')
        pol_l = term('policy π', BASE).move_to([-3.3, -1.75, 0])
        act_l = term('action', BASE).next_to(cards['partly'], UP, buff=.12).align_to(cards['partly'], LEFT)
        self.play(FadeIn(tree), *[FadeIn(c) for c in cards.values()], run_time=.8)
        self.play(FadeIn(pol_l), FadeIn(act_l), cards['partly'][1][0].animate.set_color(INTERMEDIATE), run_time=.5)
        self.mark('policy')

        self.play(FadeIn(score_axis(Y)), run_time=.3)
        dots, runs, runners = sample_runs(node_pt, tree[0], Y)
        self.play(LaggedStart(*runs, lag_ratio=.15), run_time=2.2)
        self.remove(*runners); self.add(*dots.values())
        rew_l = term('reward', INTERMEDIATE).next_to(dots['partly'][0], LEFT, buff=.2)
        self.play(FadeIn(rew_l), run_time=.4)
        self.mark('reward')

        merged, anims = to_averages(dots, Y)
        self.add(merged)
        self.play(*anims, FadeOut(rew_l), run_time=.7)
        self.play(*[FadeIn(q) for q in qs.values()], run_time=.4)
        self.play(Create(value_line(node_pt, card_left, Y(V))), node.animate.set_color(ACCENT),
                  FadeIn(value_label(V, Y(V))), run_time=.6)
        self.play(FadeIn(adv_arrows(Y, V, rows=merged)), FadeIn(formula('A = Q − V', [-4.9, -2.85, 0])), run_time=.6)
        self.mark('advantage', V=V, A=ADV)

        upd = CurvedArrow([5.6, -3.45, 0], [-1.3, -2.95, 0], angle=-.25, color=COMPOSED, stroke_width=5, tip_length=.22)
        upd_l = term('update', COMPOSED).next_to(upd.get_start(), UP + LEFT, buff=.12)
        assert upd_l.get_bottom()[1] > -3.98
        self.play(Create(upd), FadeIn(upd_l), run_time=.8)
        self.play(mix.animate.set_value(1), run_time=1.6)
        self.mark('update', p=[round(p1[t], 3) for t in TOKENS])
        self.finish('12-term-map', hold=4.0, terms=['state', 'policy', 'action', 'reward', 'V', 'Q', 'A', 'update'])


# ---------------------------------------------------------------- 13
class TermCards(Loop):
    """Quick reference after 12: the loop in five steps, with every term from the article pinned
    to its step."""

    def construct(self):
        self.begin('RL terms for LLMs')
        stages = [('sample', BASE, ['policy π', 'state', 'action', 'trajectory', 'exploration']),
                  ('score', INTERMEDIATE, ['environment', 'reward', 'return']),
                  ('compare', ACCENT, ['value V', 'Q', 'advantage A', 'baseline']),
                  ('assign credit', NEGATIVE, ['TD error', 'GAE λ']),
                  ('update', COMPOSED, ['policy gradient', 'ratio ρ', 'clipping ε', 'KL β'])]
        xs = np.linspace(-5.6, 5.6, 5)
        cards = VGroup()
        for (name, col, terms), x in zip(stages, xs):
            head = chip(name, col, 30, h=.64, fill=1, text_color=WHITE)
            items = VGroup(*[label(t, 24, col) for t in terms]).arrange(DOWN, buff=.2)
            box = RoundedRectangle(width=2.45, height=3.4, corner_radius=.2, stroke_color=col, stroke_width=2.5,
                                   fill_color=col, fill_opacity=.06)
            for it in items:
                assert it.width < 2.3, (name, it.width)
            card = VGroup(box, head.move_to(box.get_top()), items.move_to(box).shift(DOWN * .15))
            cards.add(card.move_to([x, .1, 0]))
        arrows = VGroup(*[Arrow(cards[i].get_right(), cards[i + 1].get_left(), buff=.06, color=MUTED, stroke_width=4,
                                tip_length=.18) for i in range(4)])
        back = CurvedArrow(cards[-1][0].get_bottom() + DOWN * .1, cards[0][0].get_bottom() + DOWN * .1, angle=-.35,
                           color=MUTED, stroke_width=4, tip_length=.2)
        for i, c in enumerate(cards):
            self.play(FadeIn(c[0]), FadeIn(c[1], scale=1.2), run_time=.35)
            self.play(LaggedStart(*[FadeIn(t, shift=UP * .1) for t in c[2]], lag_ratio=.15), run_time=.6)
            if i < 4:
                self.play(GrowArrow(arrows[i]), run_time=.25)
        self.play(Create(back), run_time=.7)
        self.mark('map', stages=[s[0] for s in stages])
        self.finish('13-term-cards', hold=4.0, stages=[s[0] for s in stages])
