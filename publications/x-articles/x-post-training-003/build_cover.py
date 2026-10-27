"""Procedural 5:2 cover for X-POST-003 (one English cover shared by both versions).

Series style (white title, gold labels, deep ink, engraved lines, stipple dust) with this
article's own motif: a dim grid of dots (frozen weights W); three glowing patterns, each a
column bar times a row bar that lights the grid where both are strong (LoRA's added
patterns, blue); on the right, dots snapping onto a few discrete levels (4-bit, green).
Editorial metaphor, not real data.

uv run --project visuals/animations python publications/x-articles/x-post-training-003/build_cover.py
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
GREEN = (0.20, 0.83, 0.60)
GREY = (0.55, 0.60, 0.68)

GX0, GX1, GY0, GY1 = 230, 1470, 385, 705      # grid region (frozen W)
STEP = 20
QX0, QX1 = 1530, 1880                         # quantized region
LEVELS = 7


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(900, 560, 20, 900, 560, 950)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.42); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()
    for _ in range(4200):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        near = 1.0 if (GY0 - 60 < y < GY1 + 40) else .35
        if rng.random() > .25 + .75 * near:
            continue
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.06, .4) * near)
        ctx.arc(x, y, rng.uniform(.5, 1.3), 0, 2 * np.pi); ctx.fill()

    xs = np.arange(GX0, GX1 + 1, STEP); ys = np.arange(GY0, GY1 + 1, STEP)
    # engraved grid lines, very faint
    ctx.set_line_width(.6)
    for x in xs:
        ctx.move_to(x, GY0); ctx.line_to(x, GY1); ctx.set_source_rgba(.8, .85, 1, .06); ctx.stroke()
    for y in ys:
        ctx.move_to(GX0, y); ctx.line_to(GX1, y); ctx.set_source_rgba(.8, .85, 1, .06); ctx.stroke()

    # three patterns: column profile u(y) times row profile v(x)
    pats = [(gauss(ys, 470, 45), gauss(xs, 520, 150)), (gauss(ys, 640, 55), gauss(xs, 900, 190)),
            (gauss(ys, 540, 70), gauss(xs, 1250, 120))]
    strength = np.zeros((len(ys), len(xs)))
    for u, v in pats:
        strength += np.outer(u, v)
    strength /= strength.max()

    # grid dots: grey where frozen, glowing blue where a pattern lights them
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            s = strength[i, j]
            if s > .08:
                glow = cairo.RadialGradient(x, y, 0, x, y, 11)
                glow.add_color_stop_rgba(0, *BLUE, .55 * s); glow.add_color_stop_rgba(1, *BLUE, 0)
                ctx.set_source(glow); ctx.arc(x, y, 11, 0, 2 * np.pi); ctx.fill()
            r = 1.7 + 2.6 * s
            col = tuple(GREY[k] * (1 - s) + BLUE[k] * s for k in range(3))
            ctx.arc(x, y, r, 0, 2 * np.pi); ctx.set_source_rgba(*col, .45 + .55 * s); ctx.fill()

    # the column and row bars of each pattern (left edge and top edge of the grid)
    for k, (u, v) in enumerate(pats):
        cx = GX0 - 40 - 22 * k
        for i, y in enumerate(ys):
            ctx.rectangle(cx - 5, y - STEP / 2 + 2, 10, STEP - 4)
            ctx.set_source_rgba(*BLUE, .12 + .85 * u[i]); ctx.fill()
        ry = GY0 - 32 - 18 * k
        for j, x in enumerate(xs):
            ctx.rectangle(x - STEP / 2 + 2, ry - 4, STEP - 4, 8)
            ctx.set_source_rgba(.65, .55, .98, .1 + .85 * v[j]); ctx.fill()

    # quantized region: dots snap onto a few discrete levels
    lv = np.linspace(GY0 + 10, GY1 - 10, LEVELS)
    for y in lv:
        ctx.move_to(QX0, y); ctx.line_to(QX1, y); ctx.set_line_width(1.0); ctx.set_source_rgba(*GREEN, .28); ctx.stroke()
    for _ in range(520):
        x = rng.uniform(QX0, QX1)
        t = (x - QX0) / (QX1 - QX0)
        y_cont = rng.uniform(GY0, GY1)
        y_snap = lv[np.argmin(abs(lv - y_cont))]
        y = y_cont * (1 - min(1, t * 1.6)) + y_snap * min(1, t * 1.6)
        col = tuple(GREY[k] * (1 - t) + GREEN[k] * t for k in range(3))
        ctx.arc(x, y, rng.uniform(1.2, 2.3), 0, 2 * np.pi); ctx.set_source_rgba(*col, .55 + .4 * t); ctx.fill()


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_text(img):
    d = ImageDraw.Draw(img)
    f = font(HEAVY, 156)
    text = 'LoRA & QLoRA'
    tw = d.textlength(text, font=f)
    assert tw < W * S * .86
    d.text(((W * S - tw) / 2, 150 * S - f.getbbox('W')[1]), text, font=f, fill=(255, 255, 255))
    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    d.text((mx, my), ' '.join('POST-TRAINING SERIES  ·  03').replace('   ', '  '), font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)
    lf = font(DEMI, 25)
    for x, y, word, col in [((GX0 + GX1) / 2 - 330, GY1 + 14, 'frozen W', tuple(int(c * 255) for c in GREY)),
                            ((GX0 + GX1) / 2 + 250, GY1 + 14, '+ patterns', tuple(int(c * 255) for c in BLUE)),
                            ((QX0 + QX1) / 2, GY1 + 14, '4-bit', tuple(int(c * 255) for c in GREEN))]:
        wd = d.textlength(word, font=lf)
        d.text((x * S - wd / 2, y * S), word, font=lf, fill=col)


def main():
    rng = np.random.default_rng(41)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img)
    out = img.resize((W, H), Image.LANCZOS)
    (P / 'assets').mkdir(exist_ok=True)
    path = P / 'assets' / 'cover-post-training-03.png'
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
