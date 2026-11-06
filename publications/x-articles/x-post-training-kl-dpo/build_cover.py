"""Procedural 5:2 cover for the DPO article (one English cover shared by both versions).

Series style (white title, gold labels, deep ink, engraved lines, stipple dust) with this article's
own motif: the frozen reference as one straight engraved line, and training splitting two answers away from
it, the chosen one up in green and the rejected one down in violet. The fine hatching between them is the
gap, the quantity DPO widens, and the split follows a sigmoid because the push fades as a pair is won.
Editorial picture, not data: the curves are not the article's numbers.

uv run --project visuals/animations python publications/x-articles/x-post-training-kl-dpo/build_cover.py
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
BLUE = (0.38, 0.65, 0.98)
VIOLET = (0.65, 0.55, 0.98)
GREEN = (0.20, 0.83, 0.60)

TITLE, TITLE_SIZE = 'DPO', 250
TITLE_Y = 340
MID, AMP = 584, 172                # the reference line, and how far each answer ends up from it
XS = np.linspace(-40, W + 40, 900)


def split(x):
    """0 at the left, 1 at the right: a sigmoid, flat where the pair is already won."""
    return 1 / (1 + np.exp(-(x - 1000) / 135))


def path(ctx, xs, ys):
    ctx.move_to(xs[0], ys[0])
    for a, b in zip(xs[1:], ys[1:]):
        ctx.line_to(a, b)


def glow(ctx, xs, ys, col, widths):
    for width, alpha in widths:
        path(ctx, xs, ys)
        ctx.set_source_rgba(*col, alpha); ctx.set_line_width(width); ctx.stroke()


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(1100, MID, 30, 1100, MID, 980)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.46); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()
    for _ in range(3400):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.05, .34) * (1 - .45 * y / H))
        ctx.arc(x, y, rng.uniform(.5, 1.3), 0, 2 * np.pi); ctx.fill()

    s = split(XS)
    # engraved contour lines, fanning out from the reference to each answer
    for k in np.linspace(.05, .95, 22):
        ctx.set_line_width(.8 + .5 * k)
        path(ctx, XS, MID - AMP * s * k)
        ctx.set_source_rgba(*GREEN, .08 + .3 * k); ctx.stroke()
        path(ctx, XS, MID + AMP * s * k)
        ctx.set_source_rgba(*VIOLET, .08 + .3 * k); ctx.stroke()

    # the gap: fine vertical hatching, brightest where it is widest
    for x in np.arange(260, W + 10, 5.0):
        v = float(split(np.array([x]))[0])
        if v < .02:
            continue
        ctx.move_to(x, MID - AMP * v); ctx.line_to(x, MID + AMP * v)
        ctx.set_source_rgba(.85, .9, 1, .025 + .10 * v)
        ctx.set_line_width(1.2); ctx.stroke()

    # the frozen reference: straight, blue, and it never moves
    glow(ctx, XS, MID + 0 * XS, BLUE, [(9, .06), (3.2, .22), (1.5, .95)])
    glow(ctx, XS, MID - AMP * s, GREEN, [(14, .07), (6, .2), (2.6, 1)])
    glow(ctx, XS, MID + AMP * s, VIOLET, [(14, .07), (6, .2), (2.6, 1)])


def draw_text(img):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(AVENIR, TITLE_SIZE * S, index=HEAVY)
    tw = d.textlength(TITLE, font=f)
    assert tw < W * S * .9, tw / S
    asc, _ = f.getmetrics()
    d.text(((W * S - tw) / 2, TITLE_Y * S - asc), TITLE, font=f, fill=(255, 255, 255))
    mx, my = 120 * S, 52 * S
    small = ImageFont.truetype(AVENIR, 25 * S, index=DEMI)
    d.text((mx, my), ' '.join('POST-TRAINING SERIES  ·  07').replace('   ', '  '), font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)


def main():
    rng = np.random.default_rng(7)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img)
    out = img.resize((W, H), Image.LANCZOS)
    (P / 'assets').mkdir(exist_ok=True)
    path_ = P / 'assets' / 'cover-post-training-dpo.png'
    out.save(path_, optimize=True)
    assert out.size == (W, H)
    print(path_.relative_to(P), out.size, path_.stat().st_size)


if __name__ == '__main__':
    main()
