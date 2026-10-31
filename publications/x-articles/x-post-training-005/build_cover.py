"""Procedural 5:2 cover for X-POST-005 (one English cover shared by both versions).

Series style (white title, gold labels, deep ink, engraved lines, stipple dust) with this
article's own motif, set by the learner (2026-10-30): the title Reward Model stands on a still
water line, and its reflection in the water reads Bradley–Terry, the model underneath the
reward model. The reflection keeps its letters upright (the learner's choice, 2026-10-30), broken by
engraved ripple lines and fading with depth, in violet (the series' compare colour). `--mirror`
renders the unused upside-down alternative. Editorial picture, not data.

uv run --project visuals/animations python publications/x-articles/x-post-training-005/build_cover.py
uv run --project visuals/animations python publications/x-articles/x-post-training-005/build_cover.py --mirror
"""
import sys
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

TITLE, TITLE_SIZE = 'Reward Model', 200
REFLECTION = 'Bradley–Terry'
WATER = 446                        # y of the water line; the title's baseline sits just above it


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(1000, 330, 20, 1000, 330, 900)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.42); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()
    # the water is a shade deeper than the air
    wg = cairo.LinearGradient(0, WATER, 0, H)
    wg.add_color_stop_rgba(0, 0.05, 0.08, 0.16, .55); wg.add_color_stop_rgba(1, 0, 0, 0, .0)
    ctx.set_source(wg); ctx.rectangle(0, WATER, W, H - WATER); ctx.fill()
    # stipple dust in the air only
    for _ in range(3600):
        x, y = rng.uniform(0, W), rng.uniform(0, WATER - 6)
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.06, .4) * (.4 + .6 * y / WATER))
        ctx.arc(x, y, rng.uniform(.5, 1.3), 0, 2 * np.pi); ctx.fill()

    # the water line: a bright engraved stroke, strongest under the title
    for width, alpha in [(9, .05), (3.5, .14), (1.2, .7)]:
        ctx.move_to(0, WATER); ctx.line_to(W, WATER)
        lg = cairo.LinearGradient(0, 0, W, 0)
        for t, a in [(0, 0), (.22, alpha * .5), (.5, alpha), (.78, alpha * .5), (1, 0)]:
            lg.add_color_stop_rgba(t, .85, .9, 1, a)
        ctx.set_source(lg); ctx.set_line_width(width); ctx.stroke()

    # engraved ripple lines: gentle waves, wider apart and fainter with depth
    y, k = WATER + 14, 0
    while y < H + 10:
        depth = (y - WATER) / (H - WATER)
        amp, lam, ph = 1.2 + 3.5 * depth, rng.uniform(140, 260), rng.uniform(0, 2 * np.pi)
        x0 = rng.uniform(-200, 300)
        while x0 < W:
            seg = rng.uniform(120, 520)
            xs = np.linspace(x0, x0 + seg, 60)
            ys = y + amp * np.sin(2 * np.pi * xs / lam + ph)
            near = np.exp(-((x0 + seg / 2 - 1000) / 650) ** 2)
            ctx.move_to(xs[0], ys[0])
            for a, b in zip(xs[1:], ys[1:]):
                ctx.line_to(a, b)
            ctx.set_line_width(.9 + .5 * near)
            ctx.set_source_rgba(.78, .85, 1, (.07 + .2 * near) * (1 - .6 * depth))
            ctx.stroke()
            x0 += seg + rng.uniform(30, 160)
        y += 9 + 16 * depth ** 1.3
        k += 1


def reflection_layer(upright):
    """Bradley–Terry as a reflection: mirrored, rippled, broken into bands, fading with depth."""
    f = font(HEAVY, TITLE_SIZE)
    l, t, r, b = f.getbbox(REFLECTION)
    mask = Image.new('L', (r - l + 80 * S, b - t + 20 * S), 0)
    ImageDraw.Draw(mask).text((40 * S - l, 10 * S - t), REFLECTION, font=f, fill=255)
    if not upright:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    m = np.asarray(mask, np.float32) / 255
    m = m[np.any(m > 0, axis=1)]                       # trim empty rows: the reflection starts at the water
    h, w = m.shape
    # a reflection is a little compressed vertically
    rows = np.clip((np.arange(int(h * .97)) / .97).astype(int), 0, h - 1)
    m = m[rows]
    h = m.shape[0]
    out = np.zeros_like(m)
    yy = np.arange(h)
    depth = yy / h
    shift = (1 + 7 * depth ** 1.4) * S * np.sin(yy / (9 * S) + 1.3) + 2.5 * S * depth ** 2 * np.sin(yy / (2.3 * S))
    for i in range(h):
        out[i] = np.roll(m[i], int(round(shift[i])))
    # engraved bands: thin dark gaps between rows, like light on ripples
    gap = .12 + .5 * depth                              # the dark gaps widen with depth
    band = np.where(np.sin(yy / (1.9 * S)) > 1 - 2 * gap, .25, 1.0)
    fade = .95 * (1 - .82 * depth) ** 1.4
    alpha = out * (band * fade)[:, None]
    col = np.zeros((h, w, 3), np.float32)
    for c in range(3):                                  # violet near the surface, toward blue in depth
        col[:, :, c] = 255 * (VIOLET[c] * (1 - .45 * depth) + BLUE[c] * .45 * depth)[:, None]
    return col, alpha


def draw_text(img, upright):
    d = ImageDraw.Draw(img)
    f = font(HEAVY, TITLE_SIZE)
    tw = d.textlength(TITLE, font=f)
    assert tw < W * S * .86
    l, t, r, b = f.getbbox(TITLE)
    base = (WATER - 16) * S                              # baseline (no descenders in the title)
    asc, _ = f.getmetrics()
    tx = (W * S - tw) / 2
    d.text((tx, base - asc), TITLE, font=f, fill=(255, 255, 255))

    col, alpha = reflection_layer(upright)
    h, w = alpha.shape
    x0, y0 = int((W * S - w) / 2), (WATER + 8) * S
    y1 = min(y0 + h, H * S)
    a = alpha[:y1 - y0, :, None]
    region = np.asarray(img, np.float32)
    patch = region[y0:y1, x0:x0 + w]
    region[y0:y1, x0:x0 + w] = patch * (1 - a) + col[:y1 - y0] * a
    img.paste(Image.fromarray(region.clip(0, 255).astype(np.uint8)))

    d = ImageDraw.Draw(img)
    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    d.text((mx, my), ' '.join('POST-TRAINING SERIES  ·  05').replace('   ', '  '), font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)


def main():
    upright = '--mirror' not in sys.argv
    rng = np.random.default_rng(5)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img, upright)
    out = img.resize((W, H), Image.LANCZOS)
    (P / 'assets').mkdir(exist_ok=True)
    path = P / 'assets' / ('cover-post-training-05.png' if upright else 'cover-post-training-05-mirror.png')
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
