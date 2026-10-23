"""Procedural 5:2 cover for X-POST-002 (one English cover shared by both versions).

Series style (white title, gold labels, deep ink, engraved vector lines, stipple dust),
with an instruction-tuning motif: one long token chain. The instruction part is read
but not graded (hollow grey tokens dissolving into text fragments). In the answer part,
each token receives a fan of possible next-token curves from the token before it, and
one bright curve lands exactly on the demonstrated token. The chain ends at a distinct
stop token. Editorial metaphor only, not a diagram of real probabilities.

uv run --project visuals/animations python publications/x-articles/x-post-training-002/build_cover.py
"""
from pathlib import Path

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
W, H, S = 2000, 800, 2
AVENIR = '/System/Library/Fonts/Avenir Next.ttc'
HEAVY, DEMI = 8, 2
GOLD = (251, 191, 36)
BLUE = (0.38, 0.65, 0.98)            # #60A5FA, the series colour for "show" (instruction tuning)
GREY = (0.55, 0.60, 0.68)

N_CONTEXT, N_ANSWER = 13, 8
X0, X_SPLIT, X_STOP = 150, 1060, 1860


def track_y(x):
    """Gentle wave the token chain follows."""
    return 548 + 34 * np.sin((x - 150) / 300.0) - 0.03 * (x - 1000)


def token_xs():
    ctx = np.linspace(X0, X_SPLIT - 70, N_CONTEXT)
    ans = np.linspace(X_SPLIT + 30, X_STOP - 105, N_ANSWER)
    return ctx, ans


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(1450, 560, 20, 1450, 560, 900)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.45)
    g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()

    # atmospheric dust
    for _ in range(4800):
        x = rng.uniform(0, W); y = rng.uniform(0, H)
        near = np.exp(-abs(y - track_y(x)) / 230)
        if rng.random() > .2 + .8 * near:
            continue
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.08, .5) * (.35 + .65 * near))
        ctx.arc(x, y, rng.uniform(.5, 1.4), 0, 2 * np.pi); ctx.fill()

    cxs, axs = token_xs()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    # the chain itself: faint dotted thread through the context, a solid line through the answer
    xs = np.linspace(X0 - 60, X_STOP, 600)
    for i in range(len(xs) - 1):
        x = xs[i]
        if x < X_SPLIT and (i // 3) % 2:
            continue
        a = .22 if x < X_SPLIT else .55
        ctx.move_to(x, track_y(x)); ctx.line_to(xs[i + 1], track_y(xs[i + 1]))
        ctx.set_line_width(1.4 if x < X_SPLIT else 2.0)
        ctx.set_source_rgba(.85, .9, 1, a); ctx.stroke()

    # context region: raw-text fragments drifting around hollow grey tokens (read, not graded)
    for _ in range(1500):
        x = rng.uniform(0, X_SPLIT - 40)
        spread = 210 * (1 - x / X_SPLIT) + 40
        y = track_y(x) + rng.normal(0, spread / 2.2)
        w = rng.uniform(4, 20) * (1.1 - .6 * x / X_SPLIT)
        ctx.set_source_rgba(.88, .92, 1, rng.uniform(.08, .5) * (1 - .55 * x / X_SPLIT))
        ctx.rectangle(x, y, w, rng.uniform(1.5, 2.4)); ctx.fill()
    for x in cxs:
        y = track_y(x)
        ctx.arc(x, y, 11, 0, 2 * np.pi); ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.fill_preserve()
        ctx.set_line_width(2.6); ctx.set_source_rgba(*GREY, 1); ctx.stroke()

    # answer region: each token is graded against the demonstration.
    prev = (cxs[-1], track_y(cxs[-1]))
    for k, x in enumerate(list(axs) + [X_STOP]):
        y = track_y(x)
        x0, y0 = prev
        # fan of possible next tokens: fine engraved curves spraying from the previous token
        for j in range(38):
            dy = rng.normal(0, 80)
            ex, ey = x + rng.normal(0, 14), y + dy
            c1 = (x0 + (ex - x0) * .45, y0 + dy * .15)
            c2 = (x0 + (ex - x0) * .8, ey)
            ctx.move_to(x0, y0); ctx.curve_to(*c1, *c2, ex, ey)
            ctx.set_line_width(.8); ctx.set_source_rgba(.82, .88, 1, .12 + .16 * np.exp(-abs(dy) / 50)); ctx.stroke()
            ctx.arc(ex, ey, rng.uniform(.8, 1.6), 0, 2 * np.pi)
            ctx.set_source_rgba(1, 1, 1, .25 + .35 * np.exp(-abs(dy) / 60)); ctx.fill()
        # the one curve that lands exactly on the demonstrated token
        for width, a in [(7, .08), (3.4, .25), (1.6, .95)]:
            ctx.move_to(x0, y0)
            ctx.curve_to(x0 + (x - x0) * .45, y0, x0 + (x - x0) * .8, y, x, y)
            ctx.set_line_width(width); ctx.set_source_rgba(*BLUE, a); ctx.stroke()
        if x < X_STOP:
            glow = cairo.RadialGradient(x, y, 0, x, y, 30)
            glow.add_color_stop_rgba(0, *BLUE, .75); glow.add_color_stop_rgba(1, *BLUE, 0)
            ctx.set_source(glow); ctx.arc(x, y, 30, 0, 2 * np.pi); ctx.fill()
            ctx.arc(x, y, 9.5, 0, 2 * np.pi); ctx.set_source_rgb(.93, .96, 1); ctx.fill()
        prev = (x, y)

    # the stop token: a bright square inside a glowing ring
    y = track_y(X_STOP)
    glow = cairo.RadialGradient(X_STOP, y, 0, X_STOP, y, 64)
    glow.add_color_stop_rgba(0, 1, .95, .82, .85); glow.add_color_stop_rgba(1, 1, .9, .7, 0)
    ctx.set_source(glow); ctx.arc(X_STOP, y, 64, 0, 2 * np.pi); ctx.fill()
    for width, a in [(8, .12), (2.4, .95)]:
        ctx.arc(X_STOP, y, 26, 0, 2 * np.pi); ctx.set_line_width(width); ctx.set_source_rgba(1, .97, .88, a); ctx.stroke()
    ctx.rectangle(X_STOP - 9, y - 9, 18, 18); ctx.set_source_rgb(1, 1, 1); ctx.fill()


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_text(img):
    d = ImageDraw.Draw(img)
    f = font(HEAVY, 156)
    text = 'Instruction Tuning'
    tw = d.textlength(text, font=f)
    assert tw < W * S * .86
    d.text(((W * S - tw) / 2, 150 * S - f.getbbox('W')[1]), text, font=f, fill=(255, 255, 255))

    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    spaced = ' '.join('POST-TRAINING SERIES  ·  02').replace('   ', '  ')
    d.text((mx, my), spaced, font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)

    # small legend under the chain: read / graded / stop
    lf = font(DEMI, 25)
    cxs, axs = token_xs()
    for x, word, col in [((cxs[0] + cxs[-1]) / 2, 'read', tuple(int(c * 255) for c in GREY)),
                         ((axs[0] + axs[-1]) / 2, 'graded', tuple(int(c * 255) for c in BLUE)),
                         (X_STOP, 'stop', (255, 244, 220))]:
        y = 672
        wd = d.textlength(word, font=lf)
        d.text((x * S - wd / 2, y * S), word, font=lf, fill=col)


def main():
    rng = np.random.default_rng(31)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img)
    out = img.resize((W, H), Image.LANCZOS)
    path = P / 'assets' / 'cover-post-training-02.png'
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
