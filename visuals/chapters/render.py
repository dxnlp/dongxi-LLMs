"""Render focused chapter mechanisms with the existing README GIF primitives.

uv run --project visuals/animations python visuals/chapters/render.py
uv run --project visuals/animations python visuals/chapters/render.py --only embedding-gradient assistant-loss-mask ppo-clipping

Requires Pillow and FFmpeg. No model, GPU, external media or video export.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import PIL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from visuals.readme import render as base
from visuals.chapters import fixtures as f

W, H, SCALE, FPS = base.W, base.H, base.SCALE, base.FPS
INK, MUTED, LINE = base.INK, base.MUTED, base.LINE
BLUE, VIOLET, GREEN, AMBER, WHITE = base.BLUE, base.VIOLET, base.GREEN, base.AMBER, base.WHITE
clamp, ease, mix = base.clamp, base.ease, base.mix


@lru_cache(maxsize=64)
def font_at(path, size):
    return ImageFont.truetype(path, size)


class Canvas(base.Canvas):
    """Same palette/layout as README; cache fonts and check actual glyph bounds."""
    def text(self, x, y, text, size=22, fill=INK, anchor="la"):
        font = font_at(str(self.font), round(size * SCALE))
        xy = (round(x * SCALE), round(y * SCALE))
        bounds = self.draw.textbbox(xy, text, font=font, anchor=anchor)
        assert bounds[0] >= 0 and bounds[2] <= W * SCALE, (text, bounds)
        assert bounds[1] >= 0 and bounds[3] <= H * SCALE, (text, bounds)
        self.draw.text(xy, text, font=font, fill=fill, anchor=anchor)

    def arrow2(self, start, end, color=BLUE, progress=1.):
        if progress <= 0:
            return
        a, b = start, [start[j] + (end[j] - start[j]) * clamp(progress) for j in range(2)]
        self.line([a, b], color, 2)
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        self.line([(b[0] - 9 * math.cos(angle - .5), b[1] - 9 * math.sin(angle - .5)),
                   b, (b[0] - 9 * math.cos(angle + .5), b[1] - 9 * math.sin(angle + .5))], color, 2)

    def token(self, x, y, text, color=BLUE, width=108, fill=WHITE):
        self.box((x, y, x + width, y + 48), fill, color, radius=9)
        self.text(x + width / 2, y + 10, text, 23, color, "ma")

    def node(self, x, y, text, color=BLUE, width=160):
        self.box((x, y, x + width, y + 64), mix(WHITE, color, .07), color, radius=13)
        self.text(x + width / 2, y + 20, text, 24, color, "ma")

    def particle(self, a, b, t, color=BLUE, size=7):
        u = clamp(t)
        self.circle(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, size, color)


def scene(title, subtitle, font):
    c = Canvas(font)
    c.title(title, subtitle)
    return c


def progress(t, start=2., stop=8.):
    return ease((t - start) / (stop - start))


def interpolate(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)]


def note(c, text):
    c.text(48, 511, text, 19, MUTED)


def finish(c, t):
    # Editorial blank reset; never reverse a learned update or cache lifecycle.
    return c.finish(ease((t - 11.4) / .6))


def bars(c, x, y, values, labels, color=BLUE, width=340, height=160, maximum=1., signed=False):
    gap = width / len(values)
    zero = y + height / 2 if signed else y + height
    scale = height / (2 * maximum) if signed else height / maximum
    c.line([(x, zero), (x + width, zero)], LINE)
    for i, (value, label) in enumerate(zip(values, labels)):
        center = x + gap * (i + .5)
        top = zero - value * scale
        col = (GREEN if value >= 0 else AMBER) if signed else color
        if abs(value) > 1e-10:
            c.box((center - gap * .25, min(top, zero), center + gap * .25, max(top, zero)), col, radius=3)
        c.text(center, y + height + (35 if signed else 10), label, 22, MUTED, "ma")
        ty = min(top, zero) - 30 if value >= 0 else max(top, zero) + 4
        c.text(center, ty, f"{value:+.2f}" if signed else f"{value:.2f}", 21, col, "ma")


def matrix(c, x, y, values, color=VIOLET, cell=42, focus=None, signed=False):
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            col = AMBER if signed and value < 0 else color
            filled = mix(WHITE, col, .08 + .75 * min(abs(value), 1.))
            edge = GREEN if focus is not None and i == focus else LINE
            c.box((x + j * cell, y + i * cell, x + (j + 1) * cell - 5, y + (i + 1) * cell - 5), filled, edge, radius=4)


def graph(c, rect, xs, series, xrange, yrange, xlabel="", ylabel=""):
    x0, y0, x1, y1 = rect
    c.line([(x0, y0), (x0, y1), (x1, y1)], LINE)
    convert = lambda x, y: (x0 + (x - xrange[0]) / (xrange[1] - xrange[0]) * (x1 - x0),
                             y1 - (y - yrange[0]) / (yrange[1] - yrange[0]) * (y1 - y0))
    for ys, color in series:
        if len(xs) > 1:
            c.line([convert(x, y) for x, y in zip(xs, ys)], color, 3)
    if xlabel:
        c.text((x0 + x1) / 2, y1 + 16, xlabel, 22, MUTED, "ma")
    if ylabel:
        c.text(x0, y0 - 34, ylabel, 22, MUTED)
    return convert


def evidence_gap(t, font):
    c = scene("What does lower loss establish?", "Inspect the measurement and the answer under the same frozen contract.", font)
    u = progress(t)
    xs = [i / 50 for i in range(round(50 * u) + 1)]
    losses = [2.4 - 1.25 * x for x in xs]
    graph(c, (95, 204, 540, 410), xs, [(losses, BLUE)], (0, 1), (0, 3), "Illustrative checkpoints", "Held-out token loss")
    c.text(110, 465, f"Loss: {2.4 - 1.25 * u:.2f}", 25, BLUE)
    c.node(690, 195, "Frozen prompt", BLUE, 270)
    c.arrow2((825, 260), (825, 300), VIOLET)
    c.token(690, 315, "red red red ...", AMBER, 270)
    c.text(690, 393, "Fluent story? Inspect separately.", 22, AMBER)
    note(c, "Constructed example: declining token loss supplies no story-quality verdict.")
    return finish(c, t)


def embedding_gradient(t, font):
    c = scene("An answer loss can train a prompt embedding", "Token IDs select rows. Context carries the answer gradient back.", font)
    data = f.embedding_fixture()
    u = progress(t, 6., 9.)
    table = [[a - .8 * u * g for a, g in zip(row, grad)] for row, grad in zip(f.EMBEDDINGS, data["gradient"])]
    c.text(70, 161, "Embedding table [4, 2]", 23, VIOLET)
    for i in range(4):
        c.text(72, 213 + i * 48, f"ID {i}", 22, MUTED)
    matrix(c, 145, 199, table, cell=48, signed=True)
    c.token(345, 205, "ID 1", BLUE)
    c.token(345, 328, "ID 2", BLUE)
    c.arrow2((245, 263), (345, 229), BLUE)
    c.arrow2((245, 310), (345, 352), BLUE)
    c.node(545, 267, "Context h", VIOLET)
    c.arrow2((455, 229), (545, 290), VIOLET)
    c.arrow2((455, 352), (545, 307), VIOLET)
    c.node(827, 267, "Answer loss", GREEN, 200)
    c.arrow2((706, 299), (825, 299), GREEN)
    if t > 2.:
        c.particle((245, 263), (545, 299), progress(t, 2., 4.), BLUE)
    if t > 4.5:
        c.arrow2((927, 332), (245, 430), GREEN, progress(t, 4.5, 6.))
        c.particle((927, 332), (245, 430), progress(t, 4.5, 7.), GREEN)
        c.arrow2((245, 430), (215, 273), GREEN, progress(t, 6., 8.))
        c.arrow2((245, 430), (215, 321), GREEN, progress(t, 6., 8.))
        c.text(490, 422, "Backward: rows 1 and 2 change", 24, GREEN)
    c.text(344, 166, "Prompt targets ignored", 23, AMBER)
    note(c, "Fixed mixing weights and a three-outcome head; unused rows stay unchanged.")
    return finish(c, t)


def cross_entropy_gradient(t, font):
    c = scene("Cross-entropy corrects the whole distribution", "One independent-logit descent step: gradient = p - q.", font)
    new, g = f.ce_step(f.CE_LOGITS)
    u = progress(t, 4., 8.)
    z = interpolate(f.CE_LOGITS, new, u)
    p = f.softmax(z)
    c.text(90, 157, "Predicted p", 25, BLUE)
    bars(c, 80, 245, p, ["cat", "dog", "fox"], width=340)
    c.text(462, 157, "Target q", 25, GREEN)
    bars(c, 450, 245, [0, 1, 0], ["cat", "dog", "fox"], GREEN, width=230)
    c.text(780, 157, "Initial p - q", 25, VIOLET)
    bars(c, 750, 230, g, ["cat", "dog", "fox"], width=300, height=160, signed=True)
    c.text(90, 458, f"Target probability: {p[1]:.3f}      Loss: {f.ce(z):.3f}", 25, GREEN)
    note(c, "Independent logits; shared model parameters can produce more complicated probability changes.")
    return finish(c, t)


def causal_shift(t, font):
    c = scene("The target is one position ahead", "Predict the next token from the prefix available at this position.", font)
    words = ["The", "cat", "sat", "END"]
    xs = [150, 380, 610, 840]
    c.text(48, 207, "Input", 24, MUTED)
    c.text(48, 376, "Target", 24, MUTED)
    focus = min(int(max(0., t - 4.5) / 1.7), 2)
    for i, (x, word) in enumerate(zip(xs, words)):
        c.token(x, 195, word, BLUE if i <= focus else MUTED, 125)
        if i > 0:
            target_x = x + (xs[i - 1] - x) * progress(t, 1., 4.5)
            c.token(target_x, 365, word, GREEN, 125)
    if t > 4.5:
        c.text(902, 380, "ignored", 22, MUTED)
        x = xs[focus] + 62
        c.arrow2((x, 245), (x, 363), GREEN)
        c.text(150, 465, f"Prefix through position {focus} predicts {words[focus + 1]}", 25, GREEN)
    note(c, "Shift once. A future training label is unavailable to the current causal representation.")
    return finish(c, t)


def residual_stream(t, font):
    c = scene("A residual update preserves an identity path", "Forward: h + F(h). Backward: identity contribution + branch contribution.", font)
    c.node(95, 287, "h", BLUE, 130)
    c.node(420, 174, "F(h)", VIOLET, 170)
    c.circle(760, 320, 28, mix(WHITE, GREEN, .12))
    c.text(760, 300, "+", 31, GREEN, "ma")
    c.node(895, 287, "h + F(h)", GREEN, 175)
    c.arrow2((225, 319), (727, 319), BLUE)
    c.arrow2((225, 298), (420, 207), VIOLET)
    c.arrow2((590, 207), (752, 290), VIOLET)
    c.arrow2((790, 320), (895, 320), GREEN)
    if t < 5.5:
        c.particle((225, 319), (727, 319), progress(t, 1.5, 4.), BLUE)
        c.particle((590, 207), (752, 290), progress(t, 2.5, 4.8), VIOLET)
        c.text(385, 381, "Same position; additive feature update", 24, MUTED)
    else:
        u = progress(t, 5.5, 8.)
        c.particle((727, 319), (225, 319), u, GREEN)
        c.particle((752, 290), (590, 207), progress(t, 5.5, 7.5), VIOLET)
        if t > 7.5:
            c.particle((420, 207), (225, 298), progress(t, 7.5, 10.), VIOLET)
        c.text(94, 423, "Upstream gradient", 23, GREEN)
        c.text(360, 423, "[0.50, -0.70] + [0.15, 0.14] = [0.65, -0.56]", 24, GREEN)
    note(c, "Toy branch F(h) = [0.3 h0, -0.2 h1]; the identity path does not guarantee stable training.")
    return finish(c, t)


def rope_rotation(t, font):
    c = scene("Rotary positions make relative angles matter", "Rotate Q and K pairs; retain their lengths. Values bypass this rotation.", font)
    u = progress(t)
    q, k = [.9, .3], [.2, .8]
    qr, kr = f.rotate(q, .45 * u), f.rotate(k, 3 * .45 * u)
    ox, oy, scale = 318, 357, 140
    c.line([(118, oy), (515, oy)], LINE)
    c.line([(ox, 164), (ox, 467)], LINE)
    c.arrow2((ox, oy), (ox + scale * qr[0], oy - scale * qr[1]), BLUE)
    c.arrow2((ox, oy), (ox + scale * kr[0], oy - scale * kr[1]), VIOLET)
    c.text(130, 162, "Q: position 1", 24, BLUE)
    c.text(130, 200, "K: position 3", 24, VIOLET)
    c.text(622, 203, f"Relative angle: {2 * .45 * u:.2f} rad", 26, VIOLET)
    c.text(622, 256, f"Q dot K: {f.dot(qr, kr):+.3f}", 27, GREEN)
    c.text(622, 314, "Q length: 0.949", 23, BLUE)
    c.text(622, 352, "K length: 0.825", 23, VIOLET)
    c.token(622, 411, "V unchanged", GREEN, 240)
    note(c, "One 2D frequency pair; positive-angle convention. Real RoPE uses many pairs.")
    return finish(c, t)


def grouped_query_attention(t, font):
    c = scene("More query heads, fewer cached key/value heads", "Four independent queries share two K/V groups.", font)
    u = progress(t, 2., 7.)
    for i in range(4):
        y = 160 + i * 75
        c.token(95, y, f"Q{i}", BLUE, 105)
        c.token(870, y, f"O{i}", GREEN, 120)
        group = i // 2
        target = (540, 238 + group * 150)
        c.arrow2((201, y + 24), target, BLUE)
        c.particle((201, y + 24), target, u, BLUE)
        c.arrow2((650, target[1]), (870, y + 24), GREEN, u)
    for i in range(2):
        c.node(480, 207 + i * 150, f"K{i} / V{i}", VIOLET, 170)
    c.text(95, 463, "Query heads: 4", 24, BLUE)
    c.text(480, 463, "Cached K/V heads: 2", 24, VIOLET)
    note(c, "Each query computes its own attention weights. Sharing K/V does not make outputs identical.")
    return finish(c, t)


def gradient_accumulation(t, font):
    c = scene("Accumulate by valid targets", "Unequal micro-batches contribute sums before one optimizer step.", font)
    u = progress(t, 2., 7.)
    c.node(80, 187, "2 targets", BLUE, 195)
    c.node(80, 340, "6 targets", BLUE, 195)
    c.text(88, 264, "Loss sum: 2.0", 23, BLUE)
    c.text(88, 417, "Loss sum: 9.0", 23, BLUE)
    c.node(443, 268, "Sum losses", VIOLET, 195)
    c.arrow2((276, 219), (443, 286), BLUE)
    c.arrow2((276, 372), (443, 312), BLUE)
    c.particle((276, 219), (443, 286), u, BLUE)
    c.particle((276, 372), (443, 312), u, BLUE)
    c.node(830, 268, "One update", GREEN, 195)
    c.arrow2((639, 300), (830, 300), GREEN, progress(t, 6., 8.))
    c.text(442, 368, "11 / 8 = 1.375", 28, GREEN)
    c.text(442, 417, "Mean of means = 1.250", 24, AMBER)
    note(c, "Gradient sums use the same valid-target denominator; batch count is a different clock.")
    return finish(c, t)


def checkpoint_recovery(t, font):
    c = scene("Resume the process, not only its weights", "A checkpoint carries the state needed for the next update.", font)
    labels = ["Weights", "Optimizer", "Data cursor", "RNG state"]
    cols = [BLUE, VIOLET, AMBER, GREEN]
    u = progress(t, 1.5, 5.)
    v = progress(t, 6., 9.)
    for i, (label, col) in enumerate(zip(labels, cols)):
        y = 164 + 74 * i
        c.token(60, y, label, col, 195)
        c.arrow2((256, y + 24), (488, 312), col)
        c.particle((256, y + 24), (488, 312), u, col)
        if t > 5.:
            c.arrow2((630, 312), (840, y + 24), col)
            c.particle((630, 312), (840, y + 24), v, col)
            c.token(840, y, label, col, 195)
    c.node(488, 280, "Snapshot", VIOLET, 142)
    c.text(73, 470, "Before interruption", 24, BLUE)
    c.text(823, 470, "Next update", 24, GREEN)
    note(c, "RNG replay is checked here; exact hardware determinism and recovery budgets require their own evidence.")
    return finish(c, t)


def evaluation_contract(t, font):
    c = scene("An answer becomes a score through a contract", "Keep the raw response. Parsing and grading answer different questions.", font)
    c.token(55, 232, "35 0", BLUE, 150)
    c.node(347, 189, "Strict parser", VIOLET, 222)
    c.node(347, 341, "Substring rule", AMBER, 222)
    c.arrow2((205, 255), (347, 219), VIOLET)
    c.arrow2((205, 255), (347, 372), AMBER)
    u = progress(t, 2., 5.)
    c.particle((205, 255), (347, 219), u, VIOLET)
    c.particle((205, 255), (347, 372), u, AMBER)
    if t > 4.:
        c.arrow2((570, 220), (775, 220), VIOLET, progress(t, 4., 6.))
        c.arrow2((570, 372), (775, 372), AMBER, progress(t, 4., 6.))
        c.token(775, 193, "Invalid", AMBER, 200)
        c.token(775, 345, "Contains 35", GREEN, 200)
    c.text(55, 427, "Same response; different rule, different verdict", 26, INK)
    note(c, "Illustrative grader contrast. Invalid extraction is not automatically a wrong mathematical answer.")
    return finish(c, t)


def assistant_loss_mask(t, font):
    c = scene("Supervise the answer and its ending", "Token ownership belongs to the target after one causal shift.", font)
    labels = ["USER", "copy", "red", "ASSIST", "red", "END"]
    xs = [70 + 164 * i for i in range(6)]
    for i, (x, label) in enumerate(zip(xs, labels)):
        col = GREEN if i >= 4 else BLUE
        c.token(x, 175, label, col, 140)
        c.text(x + 70, 143, "scored" if i >= 4 else "ignored", 22, col if i >= 4 else MUTED, "ma")
    shift = progress(t, 1.5, 5.5)
    for i in [4, 5]:
        x = xs[i] - 164 * shift
        c.token(x, 342, labels[i], GREEN, 140)
    c.text(70, 346, "Targets", 24, MUTED)
    if t > 5.5:
        for i in [3, 4]:
            c.arrow2((xs[i] + 70, 224), (xs[i] + 70, 341), GREEN, progress(t, 5.5, 7.))
        c.text(495, 430, "Assistant header predicts the first answer", 25, GREEN)
    note(c, "Prompt direct losses are ignored. Answer gradients can still train prompt representations.")
    return finish(c, t)


def packed_attention(t, font):
    c = scene("An ending token does not isolate packed examples", "Require both causality and membership in the same packed segment.", font)
    u = progress(t, 3., 7.)
    cell, x0, y0 = 48, 140, 207
    labels = ["A0", "A1", "END", "B0", "B1", "END"]
    for j, label in enumerate(labels):
        c.text(x0 + j * cell + 21, 163, label, 21, BLUE if j < 3 else VIOLET, "ma")
        c.text(116, y0 + j * cell + 7, label, 21, BLUE if j < 3 else VIOLET, "ra")
    for i in range(6):
        for j in range(6):
            allowed = j <= i
            leak = i >= 3 and j < 3
            value = (1 - u) if leak else float(allowed)
            col = AMBER if leak else BLUE if i < 3 else VIOLET
            c.box((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 5, y0 + (i + 1) * cell - 5),
                  mix(WHITE, col, .7 * value), LINE, radius=4)
    c.token(620, 206, "A0 A1 END", BLUE, 295)
    c.token(620, 335, "B0 B1 END", VIOLET, 295)
    if u < .98:
        c.arrow2((620, 359), (440, 353), AMBER, 1 - u)
    label = "Cross-segment access allowed" if u == 0 else "Blocking cross-segment access" if u < 1 else "Cross-segment access removed"
    c.text(620, 428, label, 25, GREEN if u == 1 else AMBER)
    note(c, "Mask structure only; cell shading indicates permission, not learned attention probabilities.")
    return finish(c, t)


def teacher_forcing(t, font):
    c = scene("Teacher forcing and generation use different histories", "During supervised training, the next prefix contains the demonstrated token.", font)
    c.text(55, 169, "Training", 25, BLUE)
    c.text(55, 323, "Generation", 25, VIOLET)
    c.text(548, 175, "Given target", 22, GREEN)
    c.text(548, 329, "Sampled token", 22, AMBER)
    for y in [206, 360]:
        c.token(225, y, "copy red", BLUE, 200)
        c.arrow2((426, y + 24), (548, y + 24), VIOLET)
    u = progress(t, 2., 5.)
    c.token(548, 206, "red", GREEN, 160)
    c.token(548, 360, "green", AMBER, 160)
    for y, col in [(230, GREEN), (384, AMBER)]:
        c.particle((426, y), (548, y), u, col)
        c.arrow2((709, y), (839, y), col, progress(t, 5., 8.))
    if t > 5.:
        c.token(839, 206, "P(END)=.70", GREEN, 228)
        c.token(839, 360, "P(END)=.20", AMBER, 228)
    note(c, "Constructed conditional probabilities and one possible sampled error; no measured SFT result.")
    return finish(c, t)


def lora_update(t, font):
    c = scene("LoRA learns a factorized correction", "The base matrix stays frozen; initially B is zero and A supplies directions.", font)
    u = progress(t, 4., 8.)
    c.text(60, 175, "W0 [2, 2]", 25, BLUE)
    matrix(c, 80, 248, [[.6, -.2], [.1, .5]], BLUE, 60, signed=True)
    c.text(287, 175, "A [1, 2]", 25, VIOLET)
    matrix(c, 300, 275, [[.6, -.2]], VIOLET, 60, signed=True)
    c.text(492, 175, "B [2, 1]", 25, VIOLET)
    b = [.3 * u, -.4 * u]
    matrix(c, 520, 248, [[v] for v in b], VIOLET, 60, signed=True)
    c.text(700, 175, "BA [2, 2]", 25, GREEN)
    matrix(c, 720, 248, [[v * a for a in [.6, -.2]] for v in b], GREEN, 60, signed=True)
    c.arrow2((425, 307), (506, 307), VIOLET)
    c.arrow2((583, 307), (710, 307), GREEN)
    c.text(60, 405, "W = W0 + scale * BA", 28, GREEN)
    c.text(60, 454, "At B=0: gradient(A)=0; gradient(B) can be nonzero", 25, VIOLET)
    note(c, "Rank-one schematic with constructed B values; adapter training still needs the base and activations.")
    return finish(c, t)


def preference_margin(t, font):
    c = scene("A score gap becomes a preference probability", "Fit the judged pair with a Bradley-Terry loss.", font)
    u = progress(t, 3., 8.)
    g = f.sigmoid(1.) - 1
    a, b = .4 - .8 * u * g, -.6 + .8 * u * g
    c.token(80, 195, "Chosen A", GREEN, 210)
    c.token(80, 338, "Rejected B", AMBER, 210)
    c.text(330, 206, f"Score {a:+.3f}", 27, GREEN)
    c.text(330, 349, f"Score {b:+.3f}", 27, AMBER)
    c.arrow2((540, 230), (693, 296), GREEN)
    c.arrow2((540, 374), (693, 321), AMBER)
    c.node(693, 274, "sigmoid(A - B)", VIOLET, 328)
    c.text(692, 374, f"P(A preferred): {f.sigmoid(a - b):.3f}", 26, GREEN)
    c.text(80, 450, f"Pair loss: {f.preference_loss([a, b]):.3f}", 26, BLUE)
    note(c, "One judged hard pair, independent scalar scores. Preference labels are not truth labels.")
    return finish(c, t)


def dpo_reference(t, font):
    c = scene("DPO compares policy changes against a reference", "Use chosen and rejected sequence probabilities at the same prompt.", font)
    u = progress(t)
    p = interpolate([.2, .1, .7], [.3, .08, .62], u)
    labels = ["Chosen", "Rejected", "Other"]
    c.text(95, 161, "Frozen reference", 25, MUTED)
    bars(c, 85, 248, [.2, .1, .7], labels, VIOLET, width=400)
    c.text(610, 161, "Changing policy", 25, BLUE)
    bars(c, 600, 248, p, labels, BLUE, width=400)
    c.text(95, 465, f"beta = 0.5     Relative margin = {f.dpo_margin(p):.3f}", 28, GREEN)
    note(c, "Constructed normalized distributions, not an optimizer trajectory; the reference receives no gradient.")
    return finish(c, t)


def dpo_counterexample(t, font):
    c = scene("A better pair margin can make the chosen answer rarer", "Mass can move to a third answer that was absent from the pair.", font)
    u = progress(t)
    p = interpolate([.2, .1, .7], [.1, .01, .89], u)
    bars(c, 75, 249, p, ["Chosen", "Rejected", "Other"], BLUE, width=520)
    margin = f.dpo_margin(p)
    c.text(705, 214, f"Chosen / rejected: {p[0] / p[1]:.2f}", 26, VIOLET)
    c.text(705, 271, f"DPO margin: {margin:.3f}", 26, GREEN)
    c.text(705, 328, f"Pair loss: {-math.log(f.sigmoid(margin)):.3f}", 26, GREEN)
    c.text(705, 397, f"Chosen probability: {p[0]:.3f}", 25, AMBER)
    note(c, "Chapter counterexample: interpolation between constructed vectors, not a training or capability result.")
    return finish(c, t)


def policy_baseline(t, font):
    c = scene("A detached baseline changes noise, not the mean", "Exact categorical policy: p = [0.5, 0.3, 0.2], rewards = [0, 1, 2].", font)
    baseline = .7 * progress(t, 2., 8.)
    gs, mean, var = f.policy_estimates(baseline)
    colors = [BLUE, VIOLET, AMBER]
    c.text(70, 165, f"Baseline: {baseline:.2f}", 26, VIOLET)
    for i, (g, col) in enumerate(zip(gs, colors)):
        c.text(76, 242 + i * 61, f"Action {i}: R-b = {f.REWARDS[i] - baseline:+.2f}", 25, col)
        for j, value in enumerate(g):
            x = 503 + j * 84
            c.circle(x, 352 - 60 * value, 13 * math.sqrt(f.POLICY_P[i]), col)
    for j, value in enumerate(mean):
        x = 503 + j * 84
        y = 352 - 60 * value
        c.line([(x - 17, y), (x + 17, y)], GREEN, 3)
        c.text(x, 417, f"g{j}", 22, MUTED, "ma")
    c.text(716, 218, "Mean gradient", 25, GREEN)
    c.text(716, 266, "[-0.35, 0.09, 0.26]", 26, GREEN)
    c.text(716, 354, f"Covariance trace: {var:.4f}", 24, VIOLET)
    c.text(70, 454, "Dots: per-action gradients. Green ticks: their exact probability-weighted mean.", 23, MUTED)
    note(c, "Exact expectation over three actions. Baselines must be action-independent; a bad baseline can add noise.")
    return finish(c, t)


def ppo_clipping(t, font):
    c = scene("PPO clips an incentive, with opposite sides for each sign", "Positive advantage saturates above 1.2; negative advantage saturates below 0.8.", font)
    xs = [.4 + i * .012 for i in range(101)]
    r = .4 + 1.2 * progress(t)
    for rect, advantage, color, label in [((95, 235, 503, 408), 1., GREEN, "Positive advantage: +1"),
                                         ((660, 235, 1068, 408), -1., AMBER, "Negative advantage: -1")]:
        convert = graph(c, rect, xs, [([f.ppo(x, advantage) for x in xs], color)], (.4, 1.6),
                        (0., 1.7) if advantage > 0 else (-1.7, 0.), "Policy / old-policy ratio")
        c.text(rect[0], 164, label, 25, color)
        for bound in [.8, 1.2]:
            bx, _ = convert(bound, 0)
            c.line([(bx, rect[1]), (bx, rect[3])], LINE)
            c.text(bx, rect[1] - 31, f"{bound:.1f}", 22, MUTED, "ma")
        x, y = convert(r, f.ppo(r, advantage))
        c.circle(x, y, 9, color)
        c.text(rect[0], 472, f"Ratio: {r:.2f}   Surrogate: {f.ppo(r, advantage):+.2f}", 24, color)
    note(c, "The surrogate stays sloped in harmful directions. Clipping is not a hard probability or KL bound.")
    return finish(c, t)


def response_weighting(t, font):
    c = scene("Equal tokens and equal responses define different objectives", "Two responses have lengths 2 and 6, including their scored endings.", font)
    u = progress(t, 3., 8.)
    for i, count in enumerate([2, 6]):
        y = 208 + i * 139
        c.text(60, y + 7, f"Response {i + 1}", 24, BLUE)
        for j in range(count):
            c.token(235 + j * 67, y, "END" if j == count - 1 else str(j + 1), BLUE, 57)
        per_token = (1 - u) / (2 * count) + u / 8
        c.text(744, y + 7, f"Total weight: {count * per_token:.2f}", 28, GREEN)
        for j in range(count):
            c.box((244 + j * 67, y + 68, 274 + j * 67, y + 68 + 125 * per_token), GREEN, radius=3)
    c.text(60, 454, "Response mean" if u < .5 else "Global token mean", 27, VIOLET)
    note(c, "Per-response totals move from [0.50, 0.50] to [0.25, 0.75]; changing reduction changes the objective.")
    return finish(c, t)


def reward_hacking(t, font):
    c = scene("The proxy improves while strict correctness falls", "Replay of the chapter's finite-policy SGD: 60 updates, learning rate 0.4.", font)
    history = f.hacking_history()
    n = round(60 * progress(t, 1.5, 9.))
    rows = history[:n + 1]
    xs = [row["update"] for row in rows]
    graph(c, (95, 214, 630, 409), xs,
          [([r["proxy_reward"] / 2 for r in rows], VIOLET), ([r["strict_accuracy"] for r in rows], GREEN)],
          (0, 60), (0, 1), "SGD update")
    c.text(95, 160, "Proxy / 2", 25, VIOLET)
    c.text(300, 160, "Strict accuracy", 25, GREEN)
    row = history[n]
    c.text(710, 160, f"Update {n} / 60", 24, MUTED)
    bars(c, 710, 248, row["probabilities"], ["35", "35 0", "0"], BLUE, width=330)
    c.text(95, 465, f"Proxy: {row['proxy_reward']:.3f}     Strict accuracy: {row['strict_accuracy']:.3f}", 27, INK)
    note(c, "Actual deterministic finite-policy recurrence; integer updates only. No language-model capability claim.")
    return finish(c, t)


def candidate_selection(t, font):
    c = scene("An available correct answer can still lose selection", "The oracle ceiling and the real scorer are separate quantities.", font)
    answers = ["35", "35 0", "0"]
    scores = [1., 2., 0.]
    for i, answer in enumerate(answers):
        x = 90 + i * 340
        c.token(x, 191, answer, BLUE, 240)
        c.text(x + 120, 270, f"Scorer: {scores[i]:.0f}", 25, VIOLET, "ma")
        c.text(x + 120, 315, "Strict: pass" if i == 0 else "Strict: fail", 24, GREEN if i == 0 else AMBER, "ma")
    u = progress(t, 3., 7.)
    c.arrow2((550, 352), (550, 405), AMBER, u)
    if t > 3.:
        c.token(430, 417, "Selected: 35 0", AMBER, 240)
    c.text(90, 449, "Oracle: 1", 27, GREEN)
    c.text(830, 449, "Selected: 0", 27, AMBER)
    note(c, "Constructed candidate pool and flawed scorer; oracle correctness is an analysis, not a deployed selector.")
    return finish(c, t)


def distillation(t, font):
    c = scene("A soft teacher supervises more than the winning token", "Teacher and student share a vocabulary and the same conditioning prefix.", font)
    temperature = 1. + 3 * progress(t, 1., 4.)
    q, p, loss, grad = f.distillation(f.STUDENT, temperature)
    u = progress(t, 5., 9.)
    z = [x - .4 * u * g for x, g in zip(f.STUDENT, grad)]
    q, p, loss, _ = f.distillation(z, temperature)
    c.text(85, 163, "Teacher q (fixed logits)", 25, VIOLET)
    bars(c, 75, 248, q, ["0", "1", "2"], VIOLET, width=400)
    c.text(635, 163, "Trainable student p", 25, BLUE)
    bars(c, 625, 248, p, ["0", "1", "2"], BLUE, width=400)
    c.arrow2((480, 314), (613, 314), GREEN, u)
    c.text(85, 465, f"Temperature: {temperature:.1f}     Scaled KL: {loss:.3f}", 28, GREEN)
    note(c, "Constructed logits; gradient is temperature * (p - q) for temperature-squared KL.")
    return finish(c, t)


SCENES = {fn.__name__.replace("_", "-"): fn for fn in [
    evidence_gap, embedding_gradient, cross_entropy_gradient, causal_shift,
    residual_stream, rope_rotation, grouped_query_attention, gradient_accumulation,
    checkpoint_recovery, evaluation_contract, assistant_loss_mask, packed_attention,
    teacher_forcing, lora_update, preference_margin, dpo_reference, dpo_counterexample,
    policy_baseline, ppo_clipping, response_weighting, reward_hacking, candidate_selection, distillation,
]}


def decoded_info(path):
    unique, delay = set(), 0
    with Image.open(path) as gif:
        for i in range(gif.n_frames):
            gif.seek(i)
            delay += gif.info.get("duration", 0)
            unique.add(hashlib.sha256(gif.convert("RGB").tobytes()).hexdigest())
        assert gif.size == (W, H) and gif.info.get("loop") == 0
        assert len(unique) > 40
        assert abs(delay - 12000) < 150
        assert path.stat().st_size < 3 * 1024 * 1024
        return {"dimensions": [W, H], "frames": gif.n_frames, "unique_frames": len(unique),
                "duration_seconds": delay / 1000, "loop": 0, "bytes": path.stat().st_size,
                "sha256": base.digest(path)}


def render(name, font, qa):
    fn = SCENES[name]
    out = HERE / "rendered"
    out.mkdir(exist_ok=True)
    gif = out / f"{name}.gif"
    with tempfile.TemporaryDirectory(prefix=f"dongxi-chapter-{name}-") as scratch:
        scratch = Path(scratch)
        for i in range(12 * FPS):
            fn(i / FPS, font).save(scratch / f"{i:04d}.png")
        palette = scratch / "palette.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(scratch / "%04d.png"), "-vf", "palettegen=max_colors=128:reserve_transparent=1",
                        "-frames:v", "1", str(palette)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(scratch / "%04d.png"), "-i", str(palette), "-lavfi",
                        "paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle", "-loop", "0", str(gif)], check=True)
    poster = out / f"{name}.png"
    fn(9., font).save(poster)
    qa.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (W, 3 * (H // 2 + 30)), "#F8FAFC")
    d = ImageDraw.Draw(sheet)
    times = [.7, 2.5, 4.3, 6.2, 8., 10.7]
    with Image.open(gif) as im:
        for i, at in enumerate(times):
            elapsed = 0
            for frame in range(im.n_frames):
                im.seek(frame)
                elapsed += im.info.get("duration", 0)
                if elapsed > at * 1000:
                    break
            x, y = i % 2 * (W // 2), i // 2 * (H // 2 + 30)
            sheet.paste(im.convert("RGB").resize((W // 2, H // 2)), (x, y + 30))
            d.text((x + 12, y + 7), f"{name} - {at:.1f}s", fill=INK)
    sheet.save(qa / f"{name}.png")
    info = decoded_info(gif)
    info.update({"gif": str(gif.relative_to(ROOT)), "poster": str(poster.relative_to(ROOT)),
                 "poster_sha256": base.digest(poster), "scene": name})
    print(f"{name}: {info['unique_frames']} distinct frames, {info['bytes'] / 1024:.0f} KiB", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=SCENES)
    parser.add_argument("--font")
    parser.add_argument("--qa-dir", type=Path, default=Path(tempfile.gettempdir()) / "dongxi-chapter-qa")
    args = parser.parse_args()
    checks = f.verify()
    font = base.find_font(args.font)
    manifest_path = HERE / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    outputs = {item["scene"]: item for item in previous.get("outputs", [])}
    for name in args.only or SCENES:
        outputs[name] = render(name, font, args.qa_dir)
    manifest = {"purpose": "Approved short chapter mechanisms; illustrative except declared finite-policy replay",
                "command": "uv run --project visuals/animations python visuals/chapters/render.py",
                "last_executed_argv": sys.argv,
                "environment": {"python": platform.python_version(), "pillow": PIL.__version__,
                                "platform": platform.platform(), "font": str(font), "font_sha256": base.digest(font),
                                "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]},
                "sources": {str(p.relative_to(ROOT)): base.digest(p) for p in
                            [Path(__file__), HERE / "fixtures.py", ROOT / "visuals/readme/render.py"]},
                "numeric_checks": checks, "outputs": [outputs[k] for k in SCENES if k in outputs],
                "limitations": ["Animation time is editorial time, not measured compute latency.",
                                "White fade resets the illustration; it is not reversal of an update.",
                                "Constructed logit/probability paths do not claim shared-parameter training behavior.",
                                "The reward-hacking recurrence uses actual integer SGD updates; no interpolated measurements.",
                                "LoRA changes illustrate factorization; only first-step derivatives are asserted.",
                                "No model-scale experiments, publication, video outputs or learner-mastery claims."]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
