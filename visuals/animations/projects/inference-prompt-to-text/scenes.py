"""CAND-ANIM-040: inference, from a prompt to text.

One running example. The user asks "Is OpenAI open?" and the model answers "Partly. Some
weights are open." Token strings and IDs are real: Qwen3 tokenizer (Qwen/Qwen3-0.6B,
151,669 tokenizer entries, 151,936 embedding rows, 28 layers, tied embeddings). The chat
template is the ChatML core that Qwen uses; Qwen3 with thinking disabled also inserts an empty
<think></think> block, which these figures leave out. Logits and probabilities are illustrative.

Cast, shared by every clip:

- amber: text the chat template adds (special tokens, role names, newlines)
- slate: the user's own text
- blue: tokens the model writes
- violet: weights (embedding table, layers, LM head)
- green: the KV cache

Timelines record explanatory order, not measured time.
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
USER = '#475569'
SOFT = '#BFDBFE'       # leading-space marks on blue chips

# ---------------------------------------------------------------- the running example (real IDs)
QUESTION = 'Is OpenAI open?'
PROMPT = [(151644, '<|im_start|>'), (872, 'user'), (198, '\n'), (3872, 'Is'), (5264, ' Open'), (15469, 'AI'),
          (1787, ' open'), (30, '?'), (151645, '<|im_end|>'), (198, '\n'), (151644, '<|im_start|>'),
          (77091, 'assistant'), (198, '\n')]
ANSWER = [(5800, 'Part'), (398, 'ly'), (13, '.'), (4329, ' Some'), (14324, ' weights'), (525, ' are'),
          (1787, ' open'), (13, '.'), (151645, '<|im_end|>')]
USER_POS = range(3, 8)                       # positions of the user's own text inside PROMPT
LINES = [range(0, 3), range(3, 10), range(10, 13)]   # the template's three text lines (break after each newline)
SPECIAL = {151644, 151645}
TOKENIZER_SIZE, MODEL_ROWS, LAYERS = 151_669, 151_936, 28
REPLY = ''.join(s for i, s in ANSWER if i not in SPECIAL)
assert ''.join(s for _, s in PROMPT[3:8]) == QUESTION and REPLY == 'Partly. Some weights are open.'
assert len(PROMPT) == 13 and len(ANSWER) == 9


def kind(pos):
    """template / user / answer for a position in PROMPT + ANSWER."""
    if pos >= len(PROMPT):
        return 'template' if ANSWER[pos - len(PROMPT)][0] in SPECIAL else 'answer'
    return 'user' if pos in USER_POS else 'template'


COLOR = dict(template=ACCENT, user=USER, answer=BASE)


def shown(s):
    """How a token string is drawn: newline as ↵, a leading space as ␣."""
    if s == '\n':
        return '↵'
    return '␣' + s[1:] if s.startswith(' ') else s


# ---------------------------------------------------------------- primitives
def chip(text, color=BASE, size=26, h=.5, fill=.08, text_color=None, w=None):
    t = label(text, size, text_color or color)
    box = RoundedRectangle(width=w or max(.42, t.width + .3), height=h, corner_radius=.11,
                           stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=fill)
    return VGroup(box, t.move_to(box))


def token_chip(s, k, size=26, h=.52):
    """A token string as a chip, coloured by who wrote it."""
    if k == 'answer':
        c = chip(shown(s), BASE, size, h, fill=1, text_color=WHITE)
    else:
        c = chip(shown(s), COLOR[k], size, h, fill=.1, text_color=ACCENT if k == 'template' else FOREGROUND)
    if s.startswith(' '):
        c[1][0].set_color(SOFT if k == 'answer' else MUTED)
    return c


def cell(k, s=.32):
    return RoundedRectangle(width=s, height=s, corner_radius=.06, stroke_width=0, fill_color=COLOR[k],
                            fill_opacity=1 if k == 'answer' else .85)


def pill(text, col, size=26):
    t = label(text, size, col)
    box = RoundedRectangle(width=t.width + .34, height=t.height + .26, corner_radius=.2, stroke_color=col,
                           stroke_width=2.5, fill_color=WHITE, fill_opacity=1)
    return VGroup(box, t.move_to(box))


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


def bubble(content, color, width, fill, stroke=2.2, pad=(.24, .14), height=None):
    box = RoundedRectangle(width=width, height=height or content.height + 2 * pad[1], corner_radius=.16,
                           stroke_color=color, stroke_width=stroke, fill_color=color, fill_opacity=fill)
    content.move_to(box).align_to(box, LEFT).shift(RIGHT * pad[0])
    assert content.width < width - pad[0], (content.width, width)
    return VGroup(box, content)


def weight_block(w, h, rows=0, cols=0):
    """A violet block of weights; faint grid lines suggest a matrix."""
    box = RoundedRectangle(width=w, height=h, corner_radius=.1, stroke_color=INTERMEDIATE, stroke_width=2.2,
                           fill_color=INTERMEDIATE, fill_opacity=.1)
    g = VGroup(box)
    for i in range(1, rows):
        y = -h / 2 + h * i / rows
        g.add(Line([-w / 2 + .06, y, 0], [w / 2 - .06, y, 0], stroke_color=INTERMEDIATE, stroke_width=1, stroke_opacity=.3))
    for i in range(1, cols):
        x = -w / 2 + w * i / cols
        g.add(Line([x, -h / 2 + .06, 0], [x, h / 2 - .06, 0], stroke_color=INTERMEDIATE, stroke_width=1, stroke_opacity=.3))
    return g


def vector(rng, s=.15, n=4, color=BASE):
    """A hidden vector: a short column of cells whose shades stand for numbers."""
    return VGroup(*[Square(s, stroke_width=0, fill_color=color, fill_opacity=float(rng.uniform(.25, 1)))
                    for _ in range(n)]).arrange(DOWN, buff=.02)


def recolor(v, rng):
    return [sq.animate.set_fill(opacity=float(rng.uniform(.25, 1))) for sq in v]


def kv_cell(s=.24):
    return VGroup(*[Square(s / 2, stroke_width=0, fill_color=COMPOSED, fill_opacity=o) for o in (1, .55)]).arrange(RIGHT, buff=.02)


def bars(heights, x0, base, step, width, color=BASE, opacity=.55, chosen=None):
    g = VGroup()
    for i, h in enumerate(heights):
        b = Rectangle(width=width, height=max(abs(h), .002), stroke_width=0, fill_color=color,
                      fill_opacity=1 if i == chosen else opacity)
        b.move_to([x0 + i * step, base, 0], aligned_edge=DOWN if h >= 0 else UP)
        g.add(b)
    return g


def extend_text(scene, old, new):
    """Swap in the longer reply: its shared prefix replaces the old glyphs in place, and only the
    new glyphs fade in. Returns the fade-in animation."""
    if old is None:
        return FadeIn(new)
    n = len(old)
    scene.remove(old)
    scene.add(VGroup(*new[:n]))
    return FadeIn(VGroup(*new[n:]), shift=LEFT * .1)


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


def chat_pair(left=-6.0, y=2.35, size=26):
    t = label(QUESTION, size, FOREGROUND)
    q = bubble(t, USER, t.width + .5, .1)
    q.move_to([left + q.width / 2, y, 0])
    return q, person_icon().next_to(q, LEFT, buff=.16)


def template_lines(size, left, ys):
    """The templated prompt as three lines of text; template glyphs amber, the user's text dark.
    Returns the lines and, per PROMPT position, the glyphs of that token (none for a space)."""
    lines, groups = VGroup(), []
    for rng_, y in zip(LINES, ys):
        text = ''.join('↵' if PROMPT[p][1] == '\n' else PROMPT[p][1] for p in rng_)
        t = label(text, size, FOREGROUND)
        t.move_to([0, y, 0]).align_to([left, 0, 0], LEFT)
        i = 0
        for p in rng_:
            n = len(shown(PROMPT[p][1]).lstrip('␣'))
            g = VGroup(*t[i:i + n])
            g.set_color(ACCENT if kind(p) == 'template' else FOREGROUND)
            groups.append(g)
            i += n
        assert i == len(t), (text, i, len(t))
        lines.add(t)
    return lines, groups


# ================================================================ 01 overview
class Overview(Loop):
    """The whole loop. The prompt is tokenized into a sequence; the model reads the sequence and
    gives logits; one token is picked and appended, and its text joins the answer. The loop runs
    once per output token and stops at <|im_end|>."""

    def construct(self):
        self.begin('From prompt to text')
        q, pic = chat_pair(left=-6.1, y=2.45, size=28)
        self.play(FadeIn(pic), FadeIn(q, shift=RIGHT * .2), run_time=.5)

        # the token sequence along the bottom
        x0, step, ys, cs = -6.3, .47, -2.9, .42
        slot = lambda i: np.array([x0 + i * step, ys, 0])
        down = Arrow([-5.3, q.get_bottom()[1] - .1, 0], [-5.3, ys + cs / 2 + .1, 0], buff=0, color=USER,
                     stroke_width=5, tip_length=.24)
        tok_l = label('tokenize', 28, USER).next_to(down, RIGHT, buff=.18)
        cells = [cell(kind(i), cs).move_to(slot(i)) for i in range(len(PROMPT))]
        self.play(GrowArrow(down), FadeIn(tok_l), run_time=.5)
        self.play(LaggedStart(*[FadeIn(c, scale=.4) for c in cells], lag_ratio=.12), run_time=.9)
        self.mark('tokenized', n=len(PROMPT))

        # the model, the logits, the pick
        slabs = VGroup(*[weight_block(3.6, .55, cols=8) for _ in range(3)]).arrange(DOWN, buff=.2).move_to([-1.35, .5, 0])
        model_l = label('model', 28, INTERMEDIATE).next_to(slabs, UP, buff=.18)
        up = Arrow([slabs.get_x(), ys + cs / 2 + .1, 0], slabs.get_bottom() + DOWN * .1, buff=0, color=MUTED,
                   stroke_width=5, tip_length=.22)
        bx0, bstep, bw, bbase, nb = 1.45, .21, .15, -.55, 15
        rng = np.random.default_rng(40)
        heights = lambda chosen: [2.0 if i == chosen else float(rng.uniform(.1, 1.25)) for i in range(nb)]
        chart = bars(heights(None), bx0, bbase, bstep, bw)
        right = Arrow(slabs.get_right() + RIGHT * .1, [bx0 - .22, slabs.get_y(), 0], buff=0, color=MUTED,
                      stroke_width=5, tip_length=.2)
        logit_l = label('logits', 28, BASE).next_to(chart, DOWN, buff=.2).set_x(bx0 + (nb - 1) / 2 * bstep)
        pick_at = np.array([5.95, .5, 0])
        pick_l = label('pick', 28, BASE).move_to(pick_at + UP * .8)
        self.play(GrowArrow(up), FadeIn(slabs), FadeIn(model_l), run_time=.6)
        self.play(GrowArrow(right), FadeIn(chart), FadeIn(logit_l), FadeIn(pick_l), run_time=.5)

        # the answer bubble, empty until the first token
        bw_ = 5.6
        bub = RoundedRectangle(width=bw_, height=.68, corner_radius=.16, stroke_color=BASE, stroke_width=2.2,
                               fill_color=BASE, fill_opacity=1).move_to([1.3 + bw_ / 2, 2.45, 0])
        bot = bot_icon().next_to(bub, LEFT, buff=.16)
        loop_back = CurvedArrow([pick_at[0], pick_at[1] - .45, 0], [4.25, ys, 0], angle=-PI / 2, color=BASE,
                                stroke_width=4, tip_length=.22)
        loop_back.set_stroke(opacity=.6)
        append_l = label('append', 28, BASE).move_to([4.8, -1.7, 0])
        assert append_l.get_right()[0] < loop_back.point_from_proportion(.5)[0] - .25
        self.play(FadeIn(bot), FadeIn(bub), run_time=.4)
        self.mark('pieces')

        speeds = [1, .85, .5, .4, .32, .3, .3, .3, .5]
        text_so_far, a_txt = '', None
        for k, ((tid, s), sp) in enumerate(zip(ANSWER, speeds)):
            pos = len(PROMPT) + k
            ghost = VGroup(*[c.copy() for c in cells])
            self.play(ghost.animate.scale(.25).move_to(slabs.get_bottom()).set_opacity(0), run_time=.5 * sp)
            self.remove(ghost)
            self.play(LaggedStart(*[sl[0].animate(rate_func=there_and_back).set_fill(opacity=.55) for sl in slabs[::-1]],
                                  lag_ratio=.35), run_time=.5 * sp)
            ch = int(rng.integers(3, nb - 2))
            new = bars(heights(ch), bx0, bbase, bstep, bw, chosen=ch)
            self.play(Transform(chart, new), run_time=.4 * sp)
            tc = token_chip(s, kind(pos), 26).move_to(pick_at)
            tc.shift(LEFT * max(0, tc.get_right()[0] - 6.92))
            assert tc.get_left()[0] > bx0 + (nb - 1) * bstep + .15, s
            self.play(Indicate(chart[ch], color=BASE, scale_factor=1.15), FadeIn(tc, shift=RIGHT * .4, scale=.7),
                      run_time=.4 * sp)
            if k < 3:
                self.wait(.4 * sp)
            target = cell(kind(pos), cs).move_to(slot(pos))
            if k == 0:
                self.play(Create(loop_back), FadeIn(append_l), run_time=.5)
            anims = [Transform(tc, target, path_arc=-1.1)]
            if tid not in SPECIAL:
                text_so_far += s
                new_txt = label(text_so_far, 26, WHITE).move_to(bub).align_to(bub, LEFT).shift(RIGHT * .24)
                assert new_txt.get_right()[0] < bub.get_right()[0] - .15
                anims.append(FadeOut(tc[1].copy(), target_position=new_txt.get_right(), scale=.6))
                anims.append(extend_text(self, a_txt, new_txt))
                a_txt = new_txt
            self.play(*anims, run_time=.6 * sp)
            self.remove(tc); self.add(target); cells.append(target)
            if a_txt is not None:
                self.remove(*a_txt); self.add(a_txt)
            if k < 2:
                self.mark(f'token_{k + 1}', token=s, id=tid)
        stop = pill('stop', ACCENT, 26).next_to(cells[-1], UP, buff=.22)
        self.play(FadeIn(stop, scale=1.3), Indicate(cells[-1], color=ACCENT), run_time=.6)
        self.finish('01-overview', hold=3.0, prompt_tokens=len(PROMPT), answer_tokens=len(ANSWER),
                    forward_passes=len(ANSWER), text=text_so_far)


# ================================================================ 02 chat template
class ChatTemplate(Loop):
    """The bare question is wrapped in the chat template: role markers and special tokens that the
    model learned during instruction tuning. The open line after "assistant" is where the answer
    begins."""

    def construct(self):
        self.begin('The chat template')
        left, ys = -4.4, [.95, -.25, -1.45]
        lines, groups = template_lines(44, left, ys)
        q, pic = chat_pair(left=left + .45, y=2.3, size=32)
        self.play(FadeIn(pic), FadeIn(q, shift=RIGHT * .2), run_time=.6)
        self.mark('question')

        user_glyphs = VGroup(*[g for p in USER_POS for g in groups[p]])
        src = q[1].copy()
        assert len(src) == len(user_glyphs)
        self.play(q.animate.set_opacity(.3), pic.animate.set_opacity(.3),
                  ReplacementTransform(src, user_glyphs), run_time=.9)
        tmpl = [groups[p] for p in range(len(PROMPT)) if p not in USER_POS]
        card = RoundedRectangle(width=lines.width + 1.9, height=4.45, corner_radius=.2, stroke_color=GRID, stroke_width=2.5,
                                fill_color=GRID, fill_opacity=.12)
        card.move_to([lines.get_x() - .35, -.78, 0])
        self.add(card); self.bring_to_back(card); card.set_opacity(0)
        self.play(card.animate.set_stroke(opacity=1).set_fill(opacity=.12),
                  LaggedStart(*[FadeIn(g, shift=RIGHT * .2) for g in tmpl], lag_ratio=.2), run_time=1.6)
        assert lines.get_right()[0] < 6.9
        self.mark('wrapped', template_tokens=len(tmpl), user_tokens=len(USER_POS))

        # the answer starts on the next line
        cursor = Rectangle(width=.09, height=.62, stroke_width=0, fill_color=BASE, fill_opacity=1)
        cursor.move_to([left + .06, -2.5, 0])
        bot = bot_icon(s=1.4).next_to(cursor, LEFT, buff=.35)
        self.play(FadeIn(bot), FadeIn(cursor), run_time=.4)
        for _ in range(2):
            self.play(cursor.animate.set_opacity(0), run_time=.3)
            self.play(cursor.animate.set_opacity(1), run_time=.3)

        legend = VGroup(*[VGroup(Square(.3, stroke_width=0, fill_color=c, fill_opacity=1),
                                 label(t, 28, c)).arrange(RIGHT, buff=.15)
                          for c, t in [(ACCENT, 'template'), (FOREGROUND, 'your text')]]).arrange(RIGHT, buff=.7)
        legend.move_to([0, -3.4, 0])
        self.play(FadeIn(legend), run_time=.4)
        self.finish('02-chat-template', hold=2.6, lines=3, tokens=len(PROMPT))


# ================================================================ 03 tokenize
VOCAB_ROWS = [(0, '!'), None, (30, '?'), (198, '\n'), (872, 'user'), (1787, ' open'), (3872, 'Is'),
              (5264, ' Open'), (15469, 'AI'), (77091, 'assistant'), None, (151644, '<|im_start|>'),
              (151645, '<|im_end|>')]
ANSWER_ROWS = [(13, '.'), (398, 'ly'), (525, ' are'), (1787, ' open'), (4329, ' Some'), (5800, 'Part'),
               (14324, ' weights'), None, (151645, '<|im_end|>')]


def vocab_panel(rows, x_left, y_top, row_h=.43, w=4.15, size=26):
    """The vocabulary as a two-column table (ID, string); None rows are gaps."""
    head = VGroup(label('vocabulary', 28, FOREGROUND),
                  label(f'{TOKENIZER_SIZE:,}', 24, MUTED)).arrange(RIGHT, buff=.2, aligned_edge=DOWN)
    head.move_to([x_left + w / 2, y_top - .3, 0])
    table, where = VGroup(), {}
    id_right, str_left = x_left + 1.55, x_left + 1.8
    for r, row in enumerate(rows):
        y = y_top - .8 - r * row_h
        if row is None:
            table.add(label('⋮', 24, MUTED).move_to([id_right - .3, y, 0]))
            continue
        i, s = row
        bg = Rectangle(width=w - .12, height=row_h - .05, stroke_width=0, fill_color=BASE, fill_opacity=0)
        bg.move_to([x_left + w / 2, y, 0])
        idl = label(f'{i}', size, MUTED)
        idl.move_to([0, y, 0]).align_to([id_right, 0, 0], RIGHT)
        st = label(shown(s), size, ACCENT if i in SPECIAL else FOREGROUND)
        st.move_to([0, y, 0]).align_to([str_left, 0, 0], LEFT)
        if s.startswith(' '):
            st[0].set_color(MUTED)
        g = VGroup(bg, idl, st)
        table.add(g)
        where[i] = g
    hgt = .8 + len(rows) * row_h - .1
    frame = RoundedRectangle(width=w, height=hgt, corner_radius=.14, stroke_color=GRID,
                             stroke_width=2.2).move_to([x_left + w / 2, y_top - hgt / 2 + .05, 0])
    assert all(m.get_right()[0] < x_left + w - .05 for m in table)
    return VGroup(frame, head, table), where


class Tokenize(Loop):
    """The tokenizer cuts the templated text into vocabulary entries and replaces each by its row
    number. A leading space belongs to the next token. The model receives only the IDs."""

    def construct(self):
        self.begin('Tokenize')
        left, ys = -6.55, [2.0, .3, -1.4]
        lines, groups = template_lines(32, left, ys)
        self.play(FadeIn(lines), run_time=.6)
        self.mark('text')

        # cut marks between tokens
        ticks = VGroup()
        for rng_ in LINES:
            ps = list(rng_)
            for a, b in zip(ps, ps[1:]):
                x = (groups[a].get_right()[0] + groups[b].get_left()[0]) / 2
                y = groups[a].get_y()
                ticks.add(Line([x, y - .36, 0], [x, y + .36, 0], color=INTERMEDIATE, stroke_width=4))
        self.play(LaggedStart(*[Create(t) for t in ticks], lag_ratio=.08), run_time=.9)

        # each piece becomes a chip; a leading space appears as ␣
        chips = []
        for rng_, y in zip(LINES, ys):
            row = [token_chip(PROMPT[p][1], kind(p), 32, h=.66) for p in rng_]
            VGroup(*row).arrange(RIGHT, buff=.22).move_to([0, y, 0]).align_to([left, 0, 0], LEFT)
            chips += row
        anims = []
        for p, c in enumerate(chips):
            glyphs = c[1][1:] if PROMPT[p][1].startswith(' ') else c[1]
            anims.append(ReplacementTransform(groups[p], glyphs))
            anims.append(FadeIn(c[0]))
            if PROMPT[p][1].startswith(' '):
                anims.append(FadeIn(c[1][0]))
        self.play(FadeOut(ticks), *anims, run_time=1.1)
        self.mark('tokens', n=len(chips))

        panel, where = vocab_panel(VOCAB_ROWS, 2.8, 3.05, row_h=.46)
        assert max(c.get_right()[0] for c in chips) < panel.get_left()[0] - .2
        self.play(FadeIn(panel), run_time=.6)

        # look up each token: its row lights up and its ID appears under the chip
        ids = []
        for p, c in enumerate(chips):
            tid = PROMPT[p][0]
            row = where[tid]
            fly = c.copy()
            idl = label(str(tid), 26, MUTED).next_to(c, DOWN, buff=.14)
            ids.append(idl)
            self.play(fly.animate.scale(.6).move_to(row).set_opacity(0),
                      row[0].animate.set_fill(opacity=.2), run_time=.28)
            self.play(FadeIn(idl, shift=DOWN * .1), row[0].animate.set_fill(opacity=.06), run_time=.2)
            self.remove(fly)
        for a, b in zip(ids, ids[1:]):
            if abs(a.get_y() - b.get_y()) < .01:
                assert b.get_left()[0] > a.get_right()[0] + .08, (a, b)
        self.mark('ids', ids=[i for i, _ in PROMPT])

        # what the model receives: the numbers alone
        self.play(*[c.animate.set_opacity(.22) for c in chips],
                  *[i.animate.set_color(FOREGROUND).scale(1.15) for i in ids], run_time=.8)
        self.finish('03-tokenize', hold=2.6, ids=[i for i, _ in PROMPT], vocabulary=TOKENIZER_SIZE)


# ================================================================ 04 prefill
def recolor_all(vecs, rng):
    return [sq.animate.set_fill(opacity=float(rng.uniform(.25, 1))) for v in vecs for sq in v]


class Prefill(Loop):
    """Each ID selects one row of the embedding table. All prompt positions pass through the
    layers together; every layer keeps each position's keys and values (the KV cache). Only the
    last position's output goes through the LM head, giving one logit per vocabulary entry."""

    def construct(self):
        self.begin('Prefill: the whole prompt at once')
        rng = np.random.default_rng(4)
        n = len(PROMPT)
        x0, step = -4.6, .52
        X = lambda i: x0 + i * step
        y_cell, y_vec = -3.4, -2.45
        cells = VGroup(*[cell(kind(i), .42).move_to([X(i), y_cell, 0]) for i in range(n)])
        self.play(LaggedStart(*[FadeIn(c, scale=.4) for c in cells], lag_ratio=.06), run_time=.7)

        # the embedding table: one row per vocabulary entry
        E = weight_block(.9, 4.9, rows=24).move_to([-6.2, -.95, 0])
        E_l = label('embed', 28, INTERMEDIATE).next_to(E, UP, buff=.15)
        self.play(FadeIn(E), FadeIn(E_l), run_time=.5)
        row_y = lambda tid: E.get_top()[1] - .1 - (E.height - .2) * tid / MODEL_ROWS
        vecs, looks = VGroup(), []
        for i, (tid, _) in enumerate(PROMPT):
            v = vector(rng, s=.18).move_to([X(i), y_vec, 0])
            vecs.add(v)
            hl = Line([E.get_left()[0] + .05, row_y(tid), 0], [E.get_right()[0] - .05, row_y(tid), 0],
                      color=BASE, stroke_width=6)
            src = VGroup(*[sq.copy() for sq in v]).arrange(RIGHT, buff=.02).scale(.5).move_to(hl)
            looks.append(Succession(FadeIn(hl, run_time=.15), AnimationGroup(ReplacementTransform(src, v), FadeOut(hl)),
                                    run_time=.55))
        self.play(LaggedStart(*looks, lag_ratio=.12), run_time=2.0)
        self.mark('embedded', rows=MODEL_ROWS)

        # three layers stand for 28; all positions go through together and leave K, V behind
        slab_y = (-1.3, .05, 1.4)
        slabs = VGroup(*[weight_block(X(n - 1) - X(0) + .8, .7, cols=9).move_to([(X(0) + X(n - 1)) / 2, y, 0])
                         for y in slab_y])
        more = label(f'× {LAYERS}', 32, INTERMEDIATE).next_to(slabs[1], RIGHT, buff=.25)
        self.play(FadeIn(slabs), FadeIn(more), run_time=.5)
        for li, sl in enumerate(slabs):
            ty = sl.get_top()[1] + .45
            row = VGroup(*[kv_cell(.3).move_to([X(i), sl.get_y(), 0]) for i in range(n)])
            self.play(*[v.animate.move_to([v.get_x(), sl.get_y(), 0]) for v in vecs], run_time=.45)
            self.play(sl[0].animate(rate_func=there_and_back).set_fill(opacity=.45), *recolor_all(vecs, rng),
                      run_time=.35)
            self.play(*[v.animate.move_to([v.get_x(), ty, 0]) for v in vecs],
                      LaggedStart(*[FadeIn(c, scale=.5) for c in row], lag_ratio=.04), run_time=.5)
            if li == 0:
                kv_l = label('KV cache', 28, COMPOSED).next_to(slabs[0], RIGHT, buff=.25)
                self.play(FadeIn(kv_l), run_time=.3)
        assert vecs.get_top()[1] < self.title.get_bottom()[1] - .1
        self.mark('layers', layers=LAYERS, kv_positions=n)

        # only the last position goes on to the LM head
        last = vecs[-1]
        head = weight_block(1.75, .66, cols=6).move_to([5.25, last.get_y(), 0])
        head_l = label('LM head', 26, INTERMEDIATE).move_to(head)
        self.play(*[v.animate.set_opacity(.12) for v in vecs[:-1]], run_time=.5)
        self.play(last.animate.move_to(head.get_left() + LEFT * .3), FadeIn(head), FadeIn(head_l), run_time=.7)
        chart_rng = np.random.default_rng(11)
        hs = [float(chart_rng.uniform(.1, 1.1)) for _ in range(17)]
        hs[10] = 2.2
        bx0 = 3.75
        chart = bars(hs, bx0, -2.95, .19, .13, chosen=10)
        assert chart.get_left()[0] > kv_l.get_right()[0] - .4 and max(b.get_top()[1] for i, b in enumerate(chart) if i < 3) \
            < kv_l.get_bottom()[1] - .1
        arr = Arrow(head.get_bottom() + DOWN * .05, [5.25, -.55, 0], buff=0, color=MUTED, stroke_width=5, tip_length=.2)
        cap = VGroup(label(f'{MODEL_ROWS:,}', 28, BASE), label('logits', 28, BASE)).arrange(RIGHT, buff=.15)
        cap.next_to(chart, DOWN, buff=.2)
        self.play(FadeOut(last, target_position=head.get_center(), scale=.5), GrowArrow(arr), run_time=.5)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in chart], lag_ratio=.04), FadeIn(cap), run_time=.9)
        self.mark('logits', logits=MODEL_ROWS)
        self.finish('04-prefill', hold=3.0, positions=n, layers=LAYERS, logits=MODEL_ROWS)


# ================================================================ 05 pick a token
PICK_NAMES = {2: ('No', 2753), 9: ('Open', 5002), 12: ('Part', 5800)}        # bar index → token, real ID
_r = np.random.default_rng(5)
PICK_LOGITS = [float(x) for x in _r.uniform(-1.2, 2.6, 22)]
PICK_LOGITS[12], PICK_LOGITS[9], PICK_LOGITS[2] = 6.2, 5.5, 4.8
TOP_P = .9


def softmax(z, T=1.0):
    e = np.exp((np.array(z) - max(z)) / T)
    return e / e.sum()


def nucleus(p, top_p=TOP_P):
    order = np.argsort(-p)
    keep = order[:int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
    q = np.zeros_like(p); q[keep] = p[keep]
    return q / q.sum(), sorted(int(i) for i in keep)


P1 = softmax(PICK_LOGITS)
PNUC, KEEP = nucleus(P1)
SAMPLE_U = [.2, .62, .41, .9, .1]          # uniform draws on the strip, in order


class PickToken(Loop):
    """Logits become probabilities by softmax. Temperature sharpens or flattens them; top-p keeps
    the smallest set of tokens that covers 90%. Greedy always takes the largest; sampling draws
    in proportion to probability, so the same logits can give different tokens."""

    def construct(self):
        self.begin('Pick the next token', 'illustrative numbers')
        n, x0, step, w, base = len(PICK_LOGITS), -6.35, .32, .23, -1.15
        X = lambda i: x0 + i * step
        ks, kp = .42, 4.0                    # height per logit unit, per unit of probability
        axis = Line([X(0) - .25, base, 0], [X(n - 1) + .25, base, 0], color=GRID, stroke_width=3)
        z = bars([ks * v for v in PICK_LOGITS], x0, base, step, w, opacity=.8)
        cap = label('logits', 30, BASE).move_to([X(n // 2), -2.3, 0])
        names = {i: label(nm, 28, BASE) for i, (nm, _) in PICK_NAMES.items()}

        def place_names(bs):
            return [names[i].animate.next_to(bs[i], UP, buff=.1) for i in names]
        for i in names:
            names[i].next_to(z[i], UP, buff=.1)
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(z, PICK_LOGITS)],
                                            lag_ratio=.03), FadeIn(cap), run_time=1.1)
        self.play(*[FadeIn(names[i]) for i in names], run_time=.4)
        self.mark('logits', top={nm: PICK_LOGITS[i] for i, (nm, _) in PICK_NAMES.items()})

        # softmax: positive, sums to one
        sm = pill('softmax', BASE, 30).move_to([X(n // 2), 2.45, 0])
        probs = lambda p: bars([kp * v for v in p], x0, base, step, w, opacity=.8)
        pz = probs(P1)
        cap2 = label('probabilities', 30, BASE).move_to(cap)
        self.play(FadeIn(sm, shift=DOWN * .2), run_time=.4)
        self.play(Transform(z, pz), FadeOut(cap, shift=UP * .2), FadeIn(cap2, shift=UP * .2), *place_names(pz),
                  run_time=1.0)
        self.play(FadeOut(sm), run_time=.3)
        self.mark('softmax', p={nm: round(float(P1[i]), 3) for i, (nm, _) in PICK_NAMES.items()})

        # temperature
        tx0, tx1, ty = 3.2, 6.6, 2.45
        tmin, tmax = .4, 1.6
        TX = lambda T: tx0 + (T - tmin) / (tmax - tmin) * (tx1 - tx0)
        track = Line([tx0, ty, 0], [tx1, ty, 0], color=GRID, stroke_width=6)
        tick1 = Line([TX(1), ty - .14, 0], [TX(1), ty + .14, 0], color=MUTED, stroke_width=3)
        T = ValueTracker(1.0)
        knob = always_redraw(lambda: Dot([TX(T.get_value()), ty, 0], radius=.15, color=BASE))
        t_l = always_redraw(lambda: label(f'T = {T.get_value():.1f}', 32, BASE).next_to([tx0, ty, 0], LEFT, buff=.3))
        self.play(Create(track), FadeIn(tick1), FadeIn(knob), FadeIn(t_l), run_time=.5)
        live = always_redraw(lambda: probs(softmax(PICK_LOGITS, T.get_value())))
        self.remove(z); self.add(live)
        for i in names:
            names[i].add_updater(lambda m, i=i: m.next_to(live[i], UP, buff=.1))
        for v in (.5, 1.5, 1.0):
            self.play(T.animate.set_value(v), run_time=.9)
            self.mark(f'T_{v}', wait=.25, p_part=round(float(softmax(PICK_LOGITS, v)[12]), 3))
        for i in names:
            names[i].clear_updaters()
        z = probs(P1); self.remove(live); self.add(z)
        self.play(*[FadeOut(m) for m in (track, tick1, knob, t_l)], run_time=.4)

        # top-p
        tp = pill(f'top-p {TOP_P}', BASE, 30).move_to([4.6, 2.45, 0])
        dropped = [i for i in range(n) if i not in KEEP]
        self.play(FadeIn(tp), *[z[i].animate.set_fill(GRID) for i in dropped], run_time=.6)
        kept = probs(PNUC)
        self.play(Transform(z, kept), *place_names(kept), run_time=.9)
        self.mark('top_p', keep=len(KEEP), p={nm: round(float(PNUC[i]), 3) for i, (nm, _) in PICK_NAMES.items()})

        # copies of the kept bars lie down end to end: a strip of length 1
        sx0, sx1, sy = 1.55, 6.8, .95
        segs, edges, acc = VGroup(), [], 0.0
        shades = {12: BASE, 9: '#60A5FA', 2: '#93C5FD'}
        for i in sorted(KEEP, key=lambda i: -PNUC[i]):
            a, b = acc, acc + PNUC[i]
            r = Rectangle(width=(b - a) * (sx1 - sx0), height=.75, stroke_color=WHITE, stroke_width=3,
                          fill_color=shades.get(i, SOFT), fill_opacity=1).move_to([sx0 + (a + b) / 2 * (sx1 - sx0), sy, 0])
            segs.add(r); edges.append((i, a, b)); acc = b
        seg_names = VGroup(*[label(PICK_NAMES[i][0], 28, WHITE if i == 12 else FOREGROUND).move_to(segs[k])
                             for k, (i, _, _) in enumerate(edges) if segs[k].width > .7]).set_z_index(3)
        self.play(*[ReplacementTransform(z[i].copy(), s) for (i, _, _), s in zip(edges, segs)], FadeOut(tp), run_time=1.0)
        self.play(FadeIn(seg_names), run_time=.3)
        self.mark('strip', segments=len(edges))

        def token_at(u):
            for i, a, b in edges:
                if a <= u < b:
                    return i
            return edges[-1][0]

        g_l = label('greedy', 30, FOREGROUND).move_to([sx0, -.2, 0], aligned_edge=LEFT)
        s_l = label('sample', 30, FOREGROUND).move_to([sx0, -1.75, 0], aligned_edge=LEFT)
        self.play(FadeIn(g_l), FadeIn(s_l), run_time=.3)
        best = edges[0]
        x = sx0
        for k in range(3):
            ptr = Triangle(color=FOREGROUND, fill_opacity=1, stroke_width=0).scale(.14).rotate(PI)
            ptr.move_to([sx0 + (best[1] + best[2]) / 2 * (sx1 - sx0), sy + .6, 0])
            c = token_chip('Part', 'answer', 26, h=.55)
            c.move_to([x + c.width / 2, -.85, 0])
            x += c.width + .1
            self.play(FadeIn(ptr, shift=DOWN * .15), run_time=.2)
            self.play(ReplacementTransform(segs[0].copy(), c), FadeOut(ptr), run_time=.45)
        self.mark('greedy', picks=['Part'] * 3)
        x, s_row = sx0, []
        for u in SAMPLE_U:
            i = token_at(u)
            dot = Dot([sx0 + u * (sx1 - sx0), sy + 1.1, 0], radius=.12, color=FOREGROUND)
            k = [e[0] for e in edges].index(i)
            c = token_chip(PICK_NAMES[i][0], 'answer', 26, h=.55)
            c.move_to([x + c.width / 2, -2.4, 0])
            x += c.width + .1
            self.play(FadeIn(dot, scale=.5), run_time=.12)
            self.play(dot.animate.move_to([dot.get_x(), sy, 0]), run_time=.25, rate_func=rush_into)
            self.play(segs[k].animate(rate_func=there_and_back).set_fill(FOREGROUND), FadeOut(dot),
                      ReplacementTransform(segs[k].copy(), c), run_time=.45)
            s_row.append(PICK_NAMES[i][0])
        assert x < 7.0, x
        self.mark('sampled', picks=s_row)
        self.finish('05-pick-token', hold=3.0, keep=len(KEEP), greedy=['Part'] * 3, sampled=s_row,
                    p=dict(zip(['No', 'Open', 'Part'], [round(float(P1[i]), 3) for i in (2, 9, 12)])))


# ================================================================ 06 decode with a KV cache
class DecodeLoop(Loop):
    """After prefill, each step feeds only the newest token. In every layer it reads the keys and
    values already cached and adds its own; the LM head gives the next token, which is appended
    and fed back. The loop stops at <|im_end|>."""

    def construct(self):
        self.begin('Decode: one token per pass')
        rng = np.random.default_rng(6)
        N = len(PROMPT) + len(ANSWER)
        x0, step = -5.15, .4
        X = lambda i: x0 + i * step
        y_cell, y_vec, y_out = -3.3, -2.45, 2.15
        slab_y = (-1.3, 0, 1.3)
        slabs = VGroup(*[weight_block(X(N - 1) - X(0) + .62, .68, cols=11).move_to([(X(0) + X(N - 1)) / 2, y, 0])
                         for y in slab_y])
        E = weight_block(.55, 4.6, rows=20).move_to([-6.5, -1.0, 0])
        n0 = len(PROMPT)
        cs = .33
        cells = [cell(kind(i), cs).move_to([X(i), y_cell, 0]) for i in range(n0)]
        kv = [[kv_cell(.26).move_to([X(i), y, 0]) for i in range(n0)] for y in slab_y]
        kv_l = label('KV cache', 28, COMPOSED).next_to(slabs[2], UP, buff=.15).align_to(slabs[2], LEFT)
        E_l = label('embed', 24, INTERMEDIATE).next_to(E, UP, buff=.12)
        assert E_l.get_right()[0] < kv_l.get_left()[0] - .1
        head = weight_block(1.6, .6, cols=5).move_to([5.55, y_out, 0])
        head_l = label('LM head', 24, INTERMEDIATE).move_to(head)
        bx0, bstep, bbase, nb = 4.35, .2, -1.05, 13
        chart = bars([.3] * nb, bx0, bbase, bstep, .13, opacity=.3)
        logit_l = label('logits', 26, BASE).move_to([bx0 + (nb - 1) / 2 * bstep, bbase - .3, 0])
        bub = RoundedRectangle(width=5.0, height=.62, corner_radius=.16, stroke_color=BASE, stroke_width=2.2,
                               fill_color=BASE, fill_opacity=1).move_to([1.9 + 2.5, 3.45, 0])
        bot = bot_icon().next_to(bub, LEFT, buff=.14)
        assert bot.get_left()[0] > self.title.get_right()[0] + .2, 'bubble meets the title'
        assert slabs.get_right()[0] < bx0 - .3
        self.play(FadeIn(VGroup(*cells)), FadeIn(slabs), FadeIn(E), FadeIn(head), FadeIn(head_l), FadeIn(chart),
                  FadeIn(logit_l), FadeIn(E_l), FadeIn(bot), FadeIn(bub), run_time=.6)
        self.play(LaggedStart(*[FadeIn(c, scale=.5) for row in kv for c in row], lag_ratio=.01), FadeIn(kv_l),
                  run_time=.8)
        self.mark('after_prefill', cached=n0)

        text_so_far, a_txt = '', None

        def grow_text(s):
            nonlocal text_so_far, a_txt
            text_so_far += s
            t = label(text_so_far, 24, WHITE).move_to(bub).align_to(bub, LEFT).shift(RIGHT * .22)
            assert t.get_right()[0] < bub.get_right()[0] - .12, text_so_far
            anim = extend_text(self, a_txt, t)
            a_txt = t
            return anim

        def reads(li, pos):
            """Attention reads: arcs from the new position to every cached position in layer li."""
            y = slab_y[li] + .18
            arcs = VGroup()
            for c in kv[li]:
                L = X(pos) - c.get_x()
                arcs.add(ArcBetweenPoints([X(pos), y, 0], [c.get_x(), y, 0], angle=min(2.4, 8 * .4 / L),
                                          color=COMPOSED, stroke_width=2.5, stroke_opacity=.6))
            return arcs

        pick_at = np.array([5.55, -2.05, 0])

        def append(k, sp):
            """Token k of ANSWER appears at the pick point and drops into its slot."""
            pos = n0 + k
            tid, s = ANSWER[k]
            tc = token_chip(s, kind(pos), 26, h=.52).move_to(pick_at)
            assert tc.get_right()[0] < 7.0
            target = cell(kind(pos), cs).move_to([X(pos), y_cell, 0])
            self.play(FadeIn(tc, scale=.6), run_time=.3 * sp)
            anims = [Transform(tc, target, path_arc=-.9)]
            if tid not in SPECIAL:
                anims.append(grow_text(s))
            self.play(*anims, run_time=.6 * sp)
            self.remove(tc); self.add(target); cells.append(target)
            if a_txt is not None:
                self.remove(*a_txt); self.add(a_txt)

        # the first answer token comes from prefill
        append(0, 1)
        self.mark('first_token', token=ANSWER[0][1])

        modes = ['full', 'medium'] + ['fast'] * 6
        for k, mode in enumerate(modes):
            pos = n0 + k                             # the newest token, fed alone
            tid = ANSWER[k][0]
            sp = dict(full=1, medium=.7, fast=.45)[mode]
            v = vector(rng, s=.19).move_to([X(pos), y_vec, 0])
            band = Rectangle(width=.4, height=y_out - y_cell + .75, stroke_width=0, fill_color=BASE, fill_opacity=.1)
            band.move_to([X(pos), (y_out + y_cell) / 2, 0])
            self.add(band); self.bring_to_back(band)
            hl = Line([E.get_left()[0] + .04, 0, 0], [E.get_right()[0] - .04, 0, 0], color=BASE, stroke_width=6)
            hl.set_y(E.get_top()[1] - .1 - (E.height - .2) * tid / MODEL_ROWS)
            self.play(FadeIn(hl, run_time=.2 * sp), FadeIn(v, shift=UP * .2), run_time=.4 * sp)
            self.remove(hl)
            if mode == 'fast':
                new = [kv_cell(.26).move_to([X(pos), y, 0]) for y in slab_y]
                self.play(v.animate.move_to([X(pos), y_out, 0]),
                          LaggedStart(*[FadeIn(c, scale=.5) for c in new], lag_ratio=.3), run_time=.55)
                for li in range(3):
                    kv[li].append(new[li])
            else:
                for li, y in enumerate(slab_y):
                    self.play(v.animate.move_to([X(pos), y, 0]), run_time=.35 * sp)
                    rd = reads(li, pos)
                    new_kv = kv_cell(.26).move_to([X(pos), y, 0])
                    if mode == 'full':
                        self.play(LaggedStart(*[Create(r) for r in rd], lag_ratio=.03), run_time=.6)
                        self.play(FadeOut(rd), FadeIn(new_kv, scale=.5), *recolor(v, rng), run_time=.35)
                    else:
                        self.play(FadeIn(rd), run_time=.2)
                        self.play(FadeOut(rd), FadeIn(new_kv, scale=.5), run_time=.2)
                    kv[li].append(new_kv)
                self.play(v.animate.move_to([X(pos), y_out, 0]), run_time=.3 * sp)
            self.play(v.animate.move_to(head.get_left() + LEFT * .2), run_time=.35 * sp)
            ch = int(rng.integers(2, nb - 2))
            new = bars([1.6 if i == ch else float(rng.uniform(.1, 1.0)) for i in range(nb)], bx0, bbase, bstep, .13,
                       chosen=ch)
            self.play(FadeOut(v, target_position=head.get_center(), scale=.5), Transform(chart, new), FadeOut(band),
                      run_time=.35 * sp)
            if k == 0:
                self.mark('one_pass', reads=n0, token_in=ANSWER[0][1], token_out=ANSWER[1][1])
            append(k + 1, sp)
        stop = pill('stop', ACCENT, 26).next_to(cells[-1], UP, buff=.2)
        self.play(FadeIn(stop, scale=1.3), run_time=.5)
        self.finish('06-decode-loop', hold=3.0, passes=len(modes), cached=len(kv[0]), text=text_so_far)


# ================================================================ 07 detokenize
class Detokenize(Loop):
    """The output IDs are looked up in the same vocabulary, in reverse. The stop token is dropped,
    a ␣ becomes a space, and the pieces join into the reply."""

    def construct(self):
        self.begin('Detokenize')
        y_row = 2.15
        ids = [chip(str(i), ACCENT if i in SPECIAL else BASE, 28, h=.6, fill=.12 if i in SPECIAL else 1,
                    text_color=ACCENT if i in SPECIAL else WHITE) for i, _ in ANSWER]
        VGroup(*ids).arrange(RIGHT, buff=.12).move_to([0, y_row, 0]).align_to([-6.6, 0, 0], LEFT)
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * .2) for c in ids], lag_ratio=.08), run_time=.9)
        self.mark('ids', ids=[i for i, _ in ANSWER])

        panel, where = vocab_panel(ANSWER_ROWS, -2.2, 1.35, row_h=.46, w=4.4, size=28)
        self.play(FadeIn(panel), run_time=.5)

        toks = [token_chip(s, kind(len(PROMPT) + k), 28, h=.6) for k, (_, s) in enumerate(ANSWER)]
        VGroup(*toks).arrange(RIGHT, buff=.1).move_to([0, y_row, 0]).align_to([-6.6, 0, 0], LEFT)
        assert toks[-1].get_right()[0] < 6.95 and ids[-1].get_right()[0] < 6.95
        assert panel.get_top()[1] < min(t.get_bottom()[1] for t in toks) - .1
        for c, t, (tid, _) in zip(ids, toks, ANSWER):
            row = where[tid]
            back = row[2].copy()
            self.play(row[0].animate.set_fill(opacity=.2), run_time=.18)
            self.play(back.animate.move_to(t[1]).scale(1.05), row[0].animate.set_fill(opacity=.05),
                      FadeOut(c[1]), ReplacementTransform(c[0], t[0]), run_time=.38)
            self.remove(back); self.add(t[1])
        self.mark('strings', strings=[s for _, s in ANSWER])

        # the stop token is dropped; the pieces close up into the reply
        self.play(FadeOut(toks[-1], scale=.6), FadeOut(panel), run_time=.5)
        final = label(REPLY, 36, WHITE)
        w = final.width + .6
        ans = bubble(final, BASE, w, 1, pad=(.28, .18))
        ans.move_to([-3.3 + w / 2, -.75, 0])
        q, pic = chat_pair(left=-3.3, y=.7, size=34)
        bot = bot_icon(s=1.2).next_to(ans, LEFT, buff=.16)
        glyph_i, moves, gone = 0, [], []
        for t, (_, s) in zip(toks[:-1], ANSWER[:-1]):
            body = t[1][1:] if s.startswith(' ') else t[1]
            moves.append(ReplacementTransform(body, VGroup(*final[glyph_i:glyph_i + len(body)])))
            glyph_i += len(body)
            gone.append(FadeOut(t[0]))
            if s.startswith(' '):
                gone.append(FadeOut(t[1][0]))
        assert glyph_i == len(final)
        bodies = [t[1][1:] if s.startswith(' ') else t[1] for t, (_, s) in zip(toks[:-1], ANSWER[:-1])]
        self.play(*gone, *[b.animate.set_color(BASE) for b in bodies], run_time=.4)
        self.play(*moves, FadeIn(ans[0]), FadeIn(bot), run_time=1.2)
        self.add(ans)
        self.play(FadeIn(pic), FadeIn(q, shift=RIGHT * .2), run_time=.5)
        self.finish('07-detokenize', hold=3.0, text=REPLY)
