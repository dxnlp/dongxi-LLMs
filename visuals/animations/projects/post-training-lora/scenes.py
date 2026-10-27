"""ANIM-POST-003: figures for X-POST-003 (LoRA and QLoRA), organized around the pattern.

A pattern is one column times one row (a rank-1 matrix), drawn the same way in every GIF.
GIF 01 shows QLoRA's published claims plus a labeled byte-count estimate; GIF 10 shows
labeled byte-count estimates; everything else is an illustration. Timelines record
explanatory order, not timing.
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




# ---------------------------------------------------------------- shared pieces
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]


def cell_grid(values, cell=.62, color=BASE, show_numbers=True, size=26, vmax=None):
    """A matrix drawn as cells whose fill strength follows |value|."""
    vals = np.asarray(values, dtype=float)
    vmax = vmax or max(abs(vals).max(), 1e-9)
    g = VGroup()
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            sq = Square(cell, stroke_color=GRID, stroke_width=1.4, fill_color=color, fill_opacity=.08 + .72 * abs(v) / vmax)
            sq.move_to([j * cell, -i * cell, 0])
            item = VGroup(sq)
            if show_numbers:
                t = label(f'{v:g}', size, WHITE if abs(v) / vmax > .55 else FOREGROUND)
                if t.width > cell * .85:
                    t.scale_to_fit_width(cell * .85)
                item.add(t.move_to(sq))
            g.add(item)
    return g


def vec_cells(values, color, cell=.62, vertical=True, size=26):
    arr = np.array(values, dtype=float).reshape(-1, 1) if vertical else np.array(values, dtype=float).reshape(1, -1)
    return cell_grid(arr, cell, color, True, size)


def lock_icon(color=MUTED, s=1.0):
    body = RoundedRectangle(width=.34 * s, height=.26 * s, corner_radius=.04, stroke_color=color, stroke_width=2.5, fill_color=color, fill_opacity=.25)
    arc = Arc(radius=.11 * s, start_angle=0, angle=PI, color=color, stroke_width=2.5).next_to(body, UP, buff=0)
    return VGroup(body, arc)


# ---------------------------------------------------------------- 01
class OneGpu(Loop):
    def construct(self):
        self.begin('A 65B model on one GPU', 'QLoRA, 2023')
        base = -2.9
        gpu = DashedVMobject(Rectangle(width=2.4, height=2.5, stroke_color=COMPOSED, stroke_width=3), num_dashes=40).move_to([-.4, base + 1.25, 0])
        gpu_l = label('48 GB GPU', 28, COMPOSED).next_to(gpu, DOWN, buff=.2)
        self.play(Create(gpu), FadeIn(gpu_l), run_time=.7)
        self.mark('gpu')
        full = Rectangle(width=1.6, height=.01, stroke_width=0, fill_color=ACCENT, fill_opacity=.85).move_to([-4.4, base, 0], aligned_edge=DOWN)
        full_l = label('full fine-tuning', 26, ACCENT).move_to([-4.4, base - .4, 0])
        self.add(full)
        self.play(full.animate.stretch_to_fit_height(5.0, about_edge=DOWN), FadeIn(full_l), run_time=1.5)
        up = Arrow([-4.4, 2.1, 0], [-4.4, 2.75, 0], buff=0, color=ACCENT, stroke_width=4, tip_length=.2)
        tb = label('~1 TB (estimate)', 26, ACCENT).next_to(full, RIGHT, buff=.2).align_to(full, UP).shift(DOWN * .3)
        self.play(GrowArrow(up), FadeIn(tb), run_time=.5)
        self.mark('full_ft', estimate_tb=1)
        q = Rectangle(width=1.6, height=.01, stroke_width=0, fill_color=COMPOSED, fill_opacity=.85).move_to([-.4, base, 0], aligned_edge=DOWN)
        self.add(q)
        self.play(q.animate.stretch_to_fit_height(2.15, about_edge=DOWN), run_time=1.0)
        ql = label('QLoRA', 28, WHITE).move_to(q)
        self.play(FadeIn(ql), run_time=.3)
        self.mark('qlora_fits', gpu_gb=48)
        stats = VGroup(card('fits in 48 GB', COMPOSED, 28, w=5.2, h=.74, fill=.1), card('24 hours on one GPU', FOREGROUND, 28, w=5.2, h=.74),
                       card('99.3% of ChatGPT on Vicuna', FOREGROUND, 26, w=5.2, h=.74)).arrange(DOWN, buff=.3).move_to([4.0, -.6, 0])
        self.play(LaggedStart(*[FadeIn(st, shift=LEFT * .15) for st in stats], lag_ratio=.3), run_time=1.2)
        self.mark('stats', pct=99.3, hours=24)
        self.finish('01-one-gpu', hold=3.0, pct=99.3, hours=24, gpu_gb=48)


# ---------------------------------------------------------------- 02
COL1, ROW1 = [10, 20, 30], [1, 2, 2, 1]
COL2, ROW2 = [1, 0, 2], [0, 0, 5, 5]


class Pattern(Loop):
    def construct(self):
        self.begin('A pattern: one column × one row')
        cell = .78
        col = vec_cells(COL1, BASE, cell, True, 28)
        rowv = vec_cells(ROW1, INTERMEDIATE, cell, False, 28)
        m1 = np.outer(COL1, ROW1)
        grid = cell_grid(m1, cell, BASE, True, 26, vmax=70)
        grid.move_to([0.6, -.55, 0])
        rowv.next_to(grid, UP, buff=.35)
        col.next_to(grid, LEFT, buff=.35)
        cl = label('store size', 24, BASE).next_to(col, LEFT, buff=.25)
        rl = label('daily rhythm', 24, INTERMEDIATE).next_to(rowv, UP, buff=.15)
        self.play(FadeIn(col), FadeIn(rowv), FadeIn(cl), FadeIn(rl), run_time=.7)
        self.mark('vectors')
        order = [(i, j) for i in range(3) for j in range(4)]
        anims = []
        for (i, j) in order:
            k = i * 4 + j
            anims.append(AnimationGroup(Indicate(col[i], scale_factor=1.05, color=BASE), Indicate(rowv[j], scale_factor=1.05, color=INTERMEDIATE),
                                        FadeIn(grid[k], scale=.6)))
        self.play(LaggedStart(*anims, lag_ratio=.25), run_time=2.4)
        info = label('rank 1 · 3 + 4 = 7 numbers fill 12 cells', 28, FOREGROUND).move_to([0, -3.15, 0])
        self.play(FadeIn(info), run_time=.4)
        self.mark('rank1', rank=1)
        self.wait(.6)
        add = VGroup(label('+', 40, COMPOSED), vec_cells(COL2, COMPOSED, .55, True, 22), label('×', 32, COMPOSED),
                     vec_cells(ROW2, COMPOSED, .55, False, 22)).arrange(RIGHT, buff=.2).move_to([5.3, -.55, 0])
        al = label('promotion', 24, COMPOSED).next_to(add, UP, buff=.2)
        self.play(FadeIn(add), FadeIn(al), run_time=.6)
        m2 = m1 + np.outer(COL2, ROW2)
        grid2 = cell_grid(m2, cell, BASE, True, 26, vmax=70).move_to(grid)
        changed = [k for k in range(12) if m2.flat[k] != m1.flat[k]]
        for k in changed:
            grid2[k][0].set_fill(COMPOSED, grid2[k][0].get_fill_opacity())
        self.play(*[Transform(grid[k], grid2[k]) for k in range(12)], FadeOut(add, shift=LEFT * .6), FadeOut(al), run_time=1.0)
        info2 = label('rank 2 · a matrix is a sum of patterns', 28, FOREGROUND).move_to(info)
        self.play(FadeTransform(info, info2), run_time=.4)
        self.mark('rank2', rank=2, changed=len(changed))
        self.finish('02-pattern', hold=3.0, rank1=m1.tolist(), rank2=m2.tolist())


# ---------------------------------------------------------------- 03
def _photo_ranks():
    from matplotlib import cbook
    from PIL import Image as PImage
    img = np.asarray(PImage.open(cbook.get_sample_data('grace_hopper.jpg', asfileobj=False)).convert('L'), dtype=float) / 255.0
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    out = {}
    for r in (1, 5, 20, 50):
        a = (U[:, :r] * S[:r]) @ Vt[:r]
        out[r] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return out, S


class StrongestPatterns(Loop):
    def construct(self):
        self.begin('Keep the strongest patterns', 'photo: public domain')
        imgs, S = _photo_ranks()
        def pic(r):
            m = ImageMobject(np.stack([imgs[r]] * 3, -1)).set_height(5.0).move_to([-3.4, -.35, 0])
            return m
        n = 50
        bars = VGroup()
        smax = S[0]
        for k in range(n):
            h = 3.6 * (S[k] / smax) ** .5
            b = Rectangle(width=.1, height=max(h, .02), stroke_width=0, fill_color=GRID, fill_opacity=1)
            b.move_to([1.4 + k * .11, -2.6, 0], aligned_edge=DOWN)
            bars.add(b)
        bl = label('pattern strengths (SVD)', 24, MUTED).next_to(bars, DOWN, buff=.2)
        self.play(FadeIn(bars, lag_ratio=.02), FadeIn(bl), run_time=.8)
        cur = pic(1)
        rl = label('r = 1', 34, BASE).move_to([-3.4, 2.55, 0])
        self.add(cur)
        self.play(FadeIn(rl), *[bars[k].animate.set_fill(BASE) for k in range(1)], run_time=.6)
        self.mark('r_1', r=1)
        for r in (5, 20, 50):
            nxt = pic(r)
            rl2 = label(f'r = {r}', 34, BASE).move_to(rl)
            self.remove(cur); self.add(nxt)  # clean cut: crossfading photos bloats the GIF
            self.play(FadeTransform(rl, rl2), *[bars[k].animate.set_fill(BASE) for k in range(min(r, n))], run_time=.6)
            self.wait(.5)
            cur, rl = nxt, rl2
            self.mark(f'r_{r}', r=r)
        claim = VGroup(label('A few strong patterns', 28, FOREGROUND), label('carry the structure', 28, FOREGROUND)).arrange(DOWN, buff=.12).move_to([4.1, 1.9, 0])
        self.play(FadeIn(claim), run_time=.4)
        self.finish('03-strongest-patterns', hold=3.0, ranks=[1, 5, 20, 50])


# ---------------------------------------------------------------- 04
class LowRank(Loop):
    def construct(self):
        self.begin('Low-rank approximation')
        big = Square(3.6, stroke_color=BASE, stroke_width=2.5, fill_color=BASE, fill_opacity=.18).move_to([-3.6, -.3, 0])
        bl = label('d × d', 30, BASE).next_to(big, UP, buff=.2)
        count = label('16,777,216 numbers', 30, FOREGROUND).next_to(big, DOWN, buff=.3)
        self.play(FadeIn(big), FadeIn(bl), FadeIn(count), run_time=.7)
        self.mark('full', d=4096, numbers=4096 * 4096)
        eq = label('≈', 50, FOREGROUND).move_to([-.9, -.3, 0])
        tall = Rectangle(width=.55, height=3.6, stroke_color=BASE, stroke_width=2.5, fill_color=BASE, fill_opacity=.4).move_to([.4, -.3, 0])
        wide = Rectangle(width=3.6, height=.55, stroke_color=INTERMEDIATE, stroke_width=2.5, fill_color=INTERMEDIATE, fill_opacity=.4).move_to([3.4, 1.0, 0])
        times = label('×', 40, FOREGROUND).move_to([1.15, 1.0, 0])
        tl = label('d × r', 26, BASE).next_to(tall, DOWN, buff=.2)
        wl = label('r × d', 26, INTERMEDIATE).next_to(wide, UP, buff=.2)
        self.play(FadeIn(eq), TransformFromCopy(big, tall), TransformFromCopy(big, wide), FadeIn(times), run_time=1.2)
        self.play(FadeIn(tl), FadeIn(wl), run_time=.4)
        stripes = VGroup(*[Line(tall.get_top() + RIGHT * (-.2 + k * .057), tall.get_bottom() + RIGHT * (-.2 + k * .057), color=WHITE, stroke_width=1.2, stroke_opacity=.7) for k in range(8)])
        self.play(Create(stripes), run_time=.5)
        sub = label('r = 8 columns × 8 rows = 8 patterns', 26, FOREGROUND).move_to([3.75, -.7, 0])
        count2 = label('65,536 numbers (0.4%)', 30, COMPOSED).move_to([3.75, -1.7, 0])
        self.play(FadeIn(sub), FadeIn(count2, shift=UP * .15), run_time=.6)
        self.mark('low_rank', r=8, numbers=2 * 4096 * 8)
        note = label('d = 4096', 24, MUTED).next_to(count, DOWN, buff=.15)
        self.play(FadeIn(note), run_time=.3)
        self.finish('04-low-rank', hold=3.0, full=4096 * 4096, low=2 * 4096 * 8)


# ---------------------------------------------------------------- 05
class Lora(Loop):
    def construct(self):
        self.begin('LoRA: add a few patterns')
        x = card('x', FOREGROUND, 32, w=.9, h=.8).move_to([-6.0, .3, 0])
        W = Square(2.4, stroke_color=MUTED, stroke_width=2.5, fill_color=MUTED, fill_opacity=.15).move_to([-1.6, 1.15, 0])
        Wl = label('W', 40, MUTED).move_to(W)
        lk = lock_icon(MUTED, 1.4).next_to(W, UP, buff=.12)
        frozen = label('frozen', 24, MUTED).next_to(lk, RIGHT, buff=.15)
        A = Rectangle(width=2.0, height=.42, stroke_color=INTERMEDIATE, stroke_width=2.5, fill_color=INTERMEDIATE, fill_opacity=.3).move_to([-2.4, -1.6, 0])
        B = Rectangle(width=.42, height=2.0, stroke_color=BASE, stroke_width=2.5, fill_color=BASE, fill_opacity=.08).move_to([.2, -1.6, 0])
        Al = label('A  (r × d)', 24, INTERMEDIATE).next_to(A, DOWN, buff=.2)
        Bl = label('B  (d × r)', 24, BASE).next_to(B, RIGHT, buff=.2)
        plus = Circle(.32, color=FOREGROUND, stroke_width=2.5).move_to([2.6, .3, 0]); pl = label('+', 34, FOREGROUND).move_to(plus)
        out = card('h', FOREGROUND, 32, w=.9, h=.8).move_to([4.6, .3, 0])
        wires = VGroup(
            Line(x.get_right(), [-3.2, .3, 0], color=GRID, stroke_width=2.5),
            Line([-3.2, .3, 0], [-3.2, 1.15, 0], color=GRID, stroke_width=2.5), Line([-3.2, 1.15, 0], W.get_left(), color=GRID, stroke_width=2.5),
            Line(W.get_right(), [2.6, 1.15, 0], color=GRID, stroke_width=2.5), Line([2.6, 1.15, 0], plus.get_top(), color=GRID, stroke_width=2.5),
            Line([-3.2, .3, 0], [-3.2, -1.6, 0], color=GRID, stroke_width=2.5), Line([-3.2, -1.6, 0], A.get_left(), color=GRID, stroke_width=2.5),
            Line(A.get_right(), B.get_left(), color=GRID, stroke_width=2.5),
            Line(B.get_right(), [2.6, -1.6, 0], color=GRID, stroke_width=2.5), Line([2.6, -1.6, 0], plus.get_bottom(), color=GRID, stroke_width=2.5),
            Line(plus.get_right(), out.get_left(), color=GRID, stroke_width=2.5))
        self.play(FadeIn(x), FadeIn(W), FadeIn(Wl), FadeIn(lk), FadeIn(frozen), Create(wires), FadeIn(plus), FadeIn(pl), FadeIn(out), run_time=1.0)
        self.play(FadeIn(A), FadeIn(B), FadeIn(Al), FadeIn(Bl), run_time=.6)
        b0 = label('B = 0 at start', 24, BASE).next_to(B, UP, buff=.2)
        self.play(FadeIn(b0), run_time=.4)
        self.mark('setup', B_zero=True)
        branch = VMobject().set_points_as_corners([[2.6, -1.6, 0], B.get_right(), B.get_left(), A.get_right(), A.get_left(), [-3.2, -1.6, 0]])
        for k in range(3):
            self.play(ShowPassingFlash(branch.copy().set_stroke(ACCENT, 6), time_width=.5), run_time=.7)
            self.play(B.animate.set_fill(BASE, .15 + .2 * (k + 1)), A.animate.set_fill(INTERMEDIATE, .3 + .15 * (k + 1)), run_time=.25)
        grad = label('gradients update only A and B', 26, ACCENT).move_to([-.2, -3.25, 0])
        self.play(FadeIn(grad), FadeOut(b0), run_time=.4)
        self.mark('training', trained=['A', 'B'], frozen=['W'])
        self.wait(.4)
        merged = label('W + B·A', 34, BASE).move_to(W)
        self.play(FadeOut(grad), A.animate.move_to(W).scale(.3).set_opacity(0), B.animate.move_to(W).scale(.3).set_opacity(0),
                  FadeOut(Al), FadeOut(Bl), FadeTransform(Wl, merged), W.animate.set_stroke(BASE).set_fill(BASE, .2),
                  *[w.animate.set_stroke(opacity=.25) for w in wires[5:10]], run_time=1.0)
        ml = label('after training: merge, no extra inference cost', 26, FOREGROUND).move_to([-.2, -3.25, 0])
        self.play(FadeIn(ml), run_time=.4)
        self.mark('merged')
        self.finish('05-lora', hold=3.0)


# ---------------------------------------------------------------- 06
WEIGHTS = [0.012, -0.034, 0.051, -0.008, 0.0, 0.027, -0.062, 0.019]


class QuantizeBlock(Loop):
    def construct(self):
        self.begin('4-bit quantization of one block')
        scale = max(abs(w) for w in WEIGHTS)
        normed = [w / scale for w in WEIGHTS]
        codes = [int(np.argmin([abs(v - l) for l in NF4])) for v in normed]
        xs = [-5.25 + 1.5 * k for k in range(8)]
        wl = label('weights', 24, MUTED).move_to([-6.45, 2.25, 0], aligned_edge=LEFT)
        ws = VGroup(*[chip(f'{w:+.3f}', BASE, 24, h=.6) for w in WEIGHTS])
        for c, x in zip(ws, xs):
            c.move_to([x + .55, 2.25, 0])
        self.play(FadeIn(ws, lag_ratio=.08), run_time=.8)
        sc = card(f'block scale = max |w| = {scale:.3f}', ACCENT, 26, h=.66, fill=.08).move_to([0, 1.25, 0])
        self.play(FadeIn(sc), run_time=.5)
        self.mark('weights', scale=scale)
        line_y = -.3
        axis = Line([-6.0, line_y, 0], [6.0, line_y, 0], color=GRID, stroke_width=2)
        def X(v):
            return v * 6.0
        ticks = VGroup(*[Line([X(l), line_y - .15, 0], [X(l), line_y + .15, 0], color=INTERMEDIATE, stroke_width=3) for l in NF4])
        tl = label('16 levels between −1 and 1', 24, INTERMEDIATE).next_to(axis, DOWN, buff=.35)
        self.play(Create(axis), Create(ticks), FadeIn(tl), run_time=.7)
        dots = VGroup(*[Dot([c.get_center()[0], 2.25, 0], radius=.09, color=BASE) for c in ws])
        self.add(dots)
        self.play(*[d.animate.move_to([X(v), line_y + .45, 0]) for d, v in zip(dots, normed)], run_time=1.0)
        self.mark('divided', normed=[round(v, 3) for v in normed])
        self.play(*[d.animate.move_to([X(NF4[c]), line_y, 0]).set_color(COMPOSED) for d, c in zip(dots, codes)], run_time=.8)
        self.mark('snapped', codes=codes)
        cl = label('4-bit codes', 24, MUTED).move_to([-6.45, -2.35, 0], aligned_edge=LEFT)
        cs = VGroup(*[chip(format(c, '04b'), COMPOSED, 24, h=.6, fill=.12) for c in codes])
        for c, x in zip(cs, xs):
            c.move_to([x + .55, -2.35, 0])
        self.play(FadeOut(tl), FadeIn(cs, lag_ratio=.08), run_time=.9)
        rebuild = label('to use: level × scale', 28, FOREGROUND).move_to([0, -3.3, 0])
        self.play(FadeIn(rebuild), run_time=.4)
        self.mark('codes', codes=codes)
        self.finish('06-quantize-block', hold=3.0, codes=codes)


# ---------------------------------------------------------------- 07
class Nf4Levels(Loop):
    def construct(self):
        self.begin('Where to put 16 levels')
        rng = np.random.default_rng(3)
        v = np.clip(rng.normal(0, .33, 20000), -1, 1)
        hist, edges = np.histogram(v, bins=40, range=(-1, 1))
        base = -2.6
        bars = VGroup(*[Rectangle(width=.27, height=max(3.0 * h / hist.max(), .01), stroke_width=0, fill_color=GRID, fill_opacity=1)
                        .move_to([(edges[k] + edges[k + 1]) / 2 * 5.6, base, 0], aligned_edge=DOWN) for k, h in enumerate(hist)])
        bl = label('weights in a block (scaled to −1..1)', 24, MUTED).next_to(bars, DOWN, buff=.25)
        self.play(FadeIn(bars, lag_ratio=.02), FadeIn(bl), run_time=.9)
        self.mark('histogram')
        uni = np.linspace(-1, 1, 16)
        ticks = VGroup(*[Line([l * 5.6, base, 0], [l * 5.6, base + 3.4, 0], color=MUTED, stroke_width=3) for l in uni])
        tl = label('evenly spaced', 30, MUTED).move_to([0, 2.15, 0])
        self.play(Create(ticks), FadeIn(tl), run_time=.8)
        self.mark('uniform')
        self.wait(.6)
        ticks2 = VGroup(*[Line([l * 5.6, base, 0], [l * 5.6, base + 3.4, 0], color=BASE, stroke_width=3) for l in NF4])
        tl2 = label('NF4: dense near 0, exact 0', 30, BASE).move_to(tl)
        self.play(Transform(ticks, ticks2), FadeTransform(tl, tl2), run_time=1.3)
        self.mark('nf4', levels=len(NF4))
        self.finish('07-nf4-levels', hold=3.0, levels=len(NF4))


# ---------------------------------------------------------------- 08
class LowRankVsQuant(Loop):
    def construct(self):
        self.begin('Low rank vs quantization')
        n = 30
        s = [1.0 / (1 + .08 * k) for k in range(n)]
        def spectrum(x0):
            return VGroup(*[Rectangle(width=.16, height=2.8 * v, stroke_width=0, fill_color=BASE, fill_opacity=.85).move_to([x0 + k * .19, -2.2, 0], aligned_edge=DOWN) for k, v in enumerate(s)])
        left, right = spectrum(-6.3), spectrum(.9)
        hl = label('low rank (r = 8)', 30, FOREGROUND).move_to([-3.55, 1.75, 0])
        hr = label('4-bit', 30, FOREGROUND).move_to([3.65, 1.75, 0])
        cap = label('pattern strengths', 22, MUTED).move_to([0, -2.65, 0])
        self.play(FadeIn(left), FadeIn(right), FadeIn(hl), FadeIn(hr), FadeIn(cap), run_time=.8)
        self.mark('same_matrix', patterns=n)
        self.play(*[left[k].animate.stretch_to_fit_height(.01, about_edge=DOWN).set_fill(ACCENT, .5) for k in range(8, n)], run_time=1.1)
        ll = label('removes the weak patterns', 26, ACCENT).move_to([-3.55, -3.25, 0])
        self.play(FadeIn(ll), run_time=.3)
        self.mark('low_rank', kept=8)
        rng = np.random.default_rng(1)
        self.play(*[right[k].animate.stretch_to_fit_height(2.8 * s[k] * (1 + rng.normal(0, .01)), about_edge=DOWN) for k in range(n)], run_time=.5)
        floor = Rectangle(width=n * .19, height=.12, stroke_width=0, fill_color=ACCENT, fill_opacity=.45).move_to([.9 - .08, -2.2, 0], aligned_edge=DOWN + LEFT)
        self.play(FadeIn(floor), run_time=.6)
        lr = label('keeps all + small noise', 26, ACCENT).move_to([3.65, -3.25, 0])
        self.play(FadeIn(lr), run_time=.3)
        self.mark('quantized', kept=n)
        self.finish('08-lowrank-vs-quant', hold=3.0, low_rank_kept=8, quant_kept=n)


# ---------------------------------------------------------------- 09
class Qlora(Loop):
    def construct(self):
        self.begin('QLoRA: 4-bit base + 16-bit patterns')
        x = card('x', FOREGROUND, 32, w=.9, h=.8).move_to([-6.0, .3, 0])
        W = Square(2.4, stroke_color=MUTED, stroke_width=2.5, fill_color=MUTED, fill_opacity=.08).move_to([-1.6, 1.15, 0])
        rng = np.random.default_rng(5)
        stip = VGroup(*[Dot([W.get_center()[0] + rng.uniform(-1.05, 1.05), W.get_center()[1] + NF4[rng.integers(0, 16)] * 1.05, 0], radius=.03, color=MUTED) for _ in range(140)])
        Wl = label('W · 4-bit NF4', 26, MUTED).next_to(W, UP, buff=.15)
        lk = lock_icon(MUTED, 1.2).next_to(Wl, LEFT, buff=.15)
        A = Rectangle(width=2.0, height=.42, stroke_color=INTERMEDIATE, stroke_width=2.5, fill_color=INTERMEDIATE, fill_opacity=.45).move_to([-2.4, -1.6, 0])
        B = Rectangle(width=.42, height=2.0, stroke_color=BASE, stroke_width=2.5, fill_color=BASE, fill_opacity=.45).move_to([.2, -1.6, 0])
        ab = label('A, B · 16-bit, trained', 24, BASE).next_to(A, DOWN, buff=.3)
        plus = Circle(.32, color=FOREGROUND, stroke_width=2.5).move_to([2.6, .3, 0]); pl = label('+', 34, FOREGROUND).move_to(plus)
        out = card('h', FOREGROUND, 32, w=.9, h=.8).move_to([4.6, .3, 0])
        wires = VGroup(
            Line(x.get_right(), [-3.2, .3, 0], color=GRID, stroke_width=2.5),
            Line([-3.2, .3, 0], [-3.2, 1.15, 0], color=GRID, stroke_width=2.5), Line([-3.2, 1.15, 0], W.get_left(), color=GRID, stroke_width=2.5),
            Line(W.get_right(), [2.6, 1.15, 0], color=GRID, stroke_width=2.5), Line([2.6, 1.15, 0], plus.get_top(), color=GRID, stroke_width=2.5),
            Line([-3.2, .3, 0], [-3.2, -1.6, 0], color=GRID, stroke_width=2.5), Line([-3.2, -1.6, 0], A.get_left(), color=GRID, stroke_width=2.5),
            Line(A.get_right(), B.get_left(), color=GRID, stroke_width=2.5),
            Line(B.get_right(), [2.6, -1.6, 0], color=GRID, stroke_width=2.5), Line([2.6, -1.6, 0], plus.get_bottom(), color=GRID, stroke_width=2.5),
            Line(plus.get_right(), out.get_left(), color=GRID, stroke_width=2.5))
        self.play(FadeIn(x), FadeIn(W), FadeIn(stip), FadeIn(Wl), FadeIn(lk), Create(wires), FadeIn(plus), FadeIn(pl), FadeIn(out),
                  FadeIn(A), FadeIn(B), FadeIn(ab), run_time=1.1)
        self.mark('setup')
        bf = Square(2.4, stroke_color=BASE, stroke_width=2.5, fill_color=BASE, fill_opacity=.12).move_to(W)
        bfl = label('bf16, just in time', 24, BASE).move_to(W)
        self.play(FadeIn(bf), FadeIn(bfl), stip.animate.set_opacity(.2), run_time=.6)
        fwd = VMobject().set_points_as_corners([x.get_right(), [-3.2, .3, 0], [-3.2, 1.15, 0], W.get_left(), W.get_right(), [2.6, 1.15, 0], plus.get_top()])
        self.play(ShowPassingFlash(fwd.copy().set_stroke(BASE, 6), time_width=.6), run_time=.9)
        self.play(FadeOut(bf), FadeOut(bfl), stip.animate.set_opacity(1), run_time=.5)
        self.mark('forward', dequantized='bf16')
        back = VMobject().set_points_as_corners([plus.get_bottom(), [2.6, -1.6, 0], B.get_right(), B.get_left(), A.get_right(), A.get_left()])
        for _ in range(2):
            self.play(ShowPassingFlash(back.copy().set_stroke(ACCENT, 6), time_width=.5), run_time=.7)
        gl = label('gradients update only A and B; W stays 4-bit', 26, ACCENT).move_to([0, -3.35, 0])
        self.play(FadeIn(gl), run_time=.4)
        self.mark('backward', trained=['A', 'B'])
        self.finish('09-qlora', hold=3.0)


# ---------------------------------------------------------------- 10
MEM = [('full fine-tuning', 100, ACCENT), ('LoRA', 15, BASE), ('QLoRA', 5, COMPOSED)]


class Memory7b(Loop):
    def construct(self):
        self.begin('GPU memory for a 7B model', 'estimates')
        base = -2.5
        xs = [-3.5, 0, 3.5]
        bars, labs = [], []
        for (name, gb, col), x in zip(MEM, xs):
            b = Rectangle(width=2.0, height=.01, stroke_width=0, fill_color=col, fill_opacity=.85).move_to([x, base, 0], aligned_edge=DOWN)
            self.add(b); bars.append(b)
            labs.append(label(name, 28, col).move_to([x, base - .4, 0]))
        self.play(*[FadeIn(l) for l in labs], run_time=.4)
        for (name, gb, col), b, x in zip(MEM, bars, xs):
            h = 4.6 * gb / 100
            self.play(b.animate.stretch_to_fit_height(max(h, .02), about_edge=DOWN), run_time=.8)
            v = label(f'~{gb} GB', 32, col).next_to(b, UP, buff=.15)
            self.play(FadeIn(v), run_time=.3)
            self.mark(f'bar_{name.split()[0]}', gb=gb)
        note = label('weights + gradients + optimizer state, no activations', 24, MUTED).move_to([0, -3.4, 0])
        self.play(FadeIn(note), run_time=.4)
        self.finish('10-memory-7b', hold=3.0, gb=[m[1] for m in MEM])


# ---------------------------------------------------------------- 11 (static)
LENS = [('Rank', 'how many independent patterns a matrix holds'),
        ('SVD', 'finds the patterns, sorted by strength'),
        ('Low-rank approximation', 'keep only the strongest r patterns'),
        ('LoRA', 'frozen W + r new learned patterns (B·A)'),
        ('Quantization', 'keeps every pattern, fewer bits per number'),
        ('QLoRA', '4-bit W keeps all patterns + 16-bit new patterns')]


class PatternLens(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND
        title = label('The pattern lens', 40, FOREGROUND).to_edge(UP, buff=.45).to_edge(LEFT, buff=.6)
        icon = VGroup(vec_cells([1, 2, 3], BASE, .32, True, 16), cell_grid(np.outer([1, 2, 3], [1, 2, 1]), .32, BASE, False),
                      ).arrange(RIGHT, buff=.12)
        rowv = vec_cells([1, 2, 1], INTERMEDIATE, .32, False, 16).next_to(icon[1], UP, buff=.12)
        pat = VGroup(icon, rowv)
        pl = label('pattern = one column × one row', 26, MUTED)
        head = VGroup(pat, pl).arrange(RIGHT, buff=.35).to_edge(UP, buff=.35).to_edge(RIGHT, buff=.6)
        self.add(title, head)
        rows = VGroup()
        for k, (a, b) in enumerate(LENS):
            ca = label(a, 30, BASE if k < 3 else (COMPOSED if a != 'Quantization' else INTERMEDIATE))
            cb = label(b, 28, FOREGROUND)
            rows.add(VGroup(ca, cb))
        y0 = 1.75
        for k, r in enumerate(rows):
            y = y0 - k * .85
            r[0].move_to([-6.3, y, 0], aligned_edge=LEFT)
            r[1].move_to([-1.6, y, 0], aligned_edge=LEFT)
            self.add(r, Line([-6.4, y - .45, 0], [6.4, y - .45, 0], color=GRID, stroke_width=1.2))
        foot = label('LoRA adds patterns; quantization uses fewer bits; QLoRA does both.', 26, ACCENT).move_to([0, -3.55, 0])
        self.add(foot)
        for m in self.mobjects:
            assert m.get_left()[0] > -7.12 and m.get_right()[0] < 7.12, m
            assert m.get_bottom()[1] > -4 and m.get_top()[1] < 4, m
        self.wait(.1)
