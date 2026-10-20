"""Procedural 5:2 cover for X-POST-001 (one English cover shared by both versions).

Editorial metaphor only: a broad, engraved river of raw text narrows through
three colored gates — show (blue), compare (violet), check (green), the same
semantic colors as the article animations — into one bright answer line.

uv run --project visuals/animations python publications/x-articles/x-post-training-001/build_cover.py
"""
from pathlib import Path

import cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
W, H, S = 2000, 800, 2                       # final size, supersampling
AVENIR = '/System/Library/Fonts/Avenir Next.ttc'
HEAVY, DEMI, MEDIUM = 8, 2, 5
GATES = [(1080, (0.38, 0.65, 0.98)),          # show     #60A5FA
         (1300, (0.65, 0.55, 0.98)),          # compare  #A78BFA
         (1500, (0.20, 0.83, 0.60))]          # check    #34D399
CY = 590
GOLD = (251, 191, 36)                       # series/brand highlight, matches the warm answer beads
END = (1905, 500)
BEADS = [1620, 1712, 1800, 1890]


def smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def half_width(x):
    """River half-height: slow narrowing, then a clear step down at each gate."""
    hw = 215 - 70 * smooth(x, 0, 1000)
    hw -= 75 * smooth(x, 1000, 1110)
    hw -= 48 * smooth(x, 1230, 1320)
    hw -= 18 * smooth(x, 1440, 1510)
    return np.maximum(hw, 2.5)


def centerline(x):
    """Gentle S-curve that lands exactly on the answer line after the last gate."""
    base = CY + 22 * np.sin(x / 330.0) * (1 - smooth(x, 900, 1450))
    tail = (x - 1500) * (END[1] - CY) / (END[0] - 1500)
    return base + np.where(x > 1500, tail, 0)


def draw_art(ctx, rng):
    # background: deep ink with a faint cool glow behind the convergence point
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(1350, CY, 20, 1350, CY, 900)
    g.add_color_stop_rgba(0, 0.16, 0.22, 0.40, 0.42)
    g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()

    # atmospheric dust, denser near the river
    for _ in range(5200):
        x = rng.uniform(0, W); y = rng.uniform(0, H)
        near = np.exp(-abs(y - centerline(x)) / 260)
        if rng.random() > .25 + .75 * near:
            continue
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.10, .55) * (.35 + .65 * near))
        ctx.arc(x, y, rng.uniform(.5, 1.5), 0, 2 * np.pi); ctx.fill()

    # engraved ribbons: bundles of fine parallel lines with gaps between bundles
    xs = np.linspace(-30, 1505, 900)
    hw, cl = half_width(xs), centerline(xs)
    bundles = 11
    for b in range(bundles):
        lo = -1 + 2 * b / bundles + .025; hi = -1 + 2 * (b + 1) / bundles - .025
        phase = rng.uniform(0, 2 * np.pi)
        n = 13
        surf = [cl + o * hw + 26 * np.sin(xs / 210 + phase + o * 1.6) * (hw / 215) for o in (lo, hi)]
        fill = cairo.LinearGradient(0, 0, 1500, 0)
        fill.add_color_stop_rgba(0, .8, .86, 1, 0)
        fill.add_color_stop_rgba(.45, .8, .86, 1, .07)
        fill.add_color_stop_rgba(1, .9, .93, 1, .14)
        ctx.move_to(xs[0], surf[0][0])
        for x, y in zip(xs, surf[0]): ctx.line_to(x, y)
        for x, y in zip(xs[::-1], surf[1][::-1]): ctx.line_to(x, y)
        ctx.close_path(); ctx.set_source(fill); ctx.fill()
        for k in range(n):
            o = lo + (hi - lo) * k / (n - 1)
            edge = min(k, n - 1 - k)                       # bundle edges catch the light
            twist = 26 * np.sin(xs / 210 + phase + o * 1.6) * (hw / 215)
            ys = cl + o * hw + twist
            alpha = (.28 + .5 * (edge == 0) + .12 * rng.random())
            fade = smooth(xs, 120, 520)                     # lines dissolve into raw text at the left
            ctx.set_line_width(.75 if edge else 1.25)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            start = None
            for i in range(len(xs)):
                a = alpha * (.25 + .75 * fade[i]) * (.55 + .45 * smooth(xs[i], 200, 1500))
                if start is None:
                    ctx.move_to(xs[i], ys[i]); start = i
                else:
                    ctx.line_to(xs[i], ys[i])
                if i - start >= 6 or i == len(xs) - 1:
                    ctx.set_source_rgba(.88, .92, 1, a); ctx.stroke()
                    ctx.move_to(xs[i], ys[i]); start = i
            # stipple shading along the bundle edges
            if edge == 0:
                for x in rng.uniform(0, 1500, 260):
                    j = np.searchsorted(xs, x)
                    y = ys[min(j, len(xs) - 1)] + rng.normal(0, 3.2)
                    ctx.set_source_rgba(1, 1, 1, rng.uniform(.25, .8) * smooth(x, 60, 400))
                    ctx.arc(x, y, rng.uniform(.5, 1.25), 0, 2 * np.pi); ctx.fill()

    # raw-text fragments breaking off the left edge
    for _ in range(1400):
        x = rng.uniform(0, 560)
        y = centerline(x) + rng.uniform(-1, 1) * (half_width(x) + 70 * (1 - x / 560))
        w = rng.uniform(5, 26) * (1 - .5 * x / 560)
        ctx.set_source_rgba(.9, .93, 1, rng.uniform(.15, .75) * (1 - .7 * x / 560))
        ctx.rectangle(x, y, w, rng.uniform(1.6, 2.6)); ctx.fill()

    # three gates: thin elliptical rings with soft colored glow
    for gx, (r, g_, b) in GATES:
        ry = float(half_width(np.array([gx]))[0]) + 26
        cy = float(centerline(np.array([gx]))[0])
        for width, a in [(16, .06), (9, .12), (4.5, .3), (1.8, .95)]:
            ctx.save(); ctx.translate(gx, cy); ctx.scale(.27, 1)
            ctx.arc(0, 0, ry, 0, 2 * np.pi); ctx.restore()
            ctx.set_line_width(width); ctx.set_source_rgba(r, g_, b, a); ctx.stroke()

    # answer line and beads
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for width, a in [(8, .07), (3.5, .2), (1.6, .95)]:
        ctx.move_to(1500, float(centerline(np.array([1500.]))[0]))
        ctx.line_to(*END); ctx.set_line_width(width); ctx.set_source_rgba(1, .97, .9, a); ctx.stroke()
    for i, bx in enumerate(BEADS):
        by = float(centerline(np.array([float(bx)]))[0])
        big = i == len(BEADS) - 1
        glow = cairo.RadialGradient(bx, by, 0, bx, by, 34 if big else 24)
        glow.add_color_stop_rgba(0, 1, .93, .78, .75)
        glow.add_color_stop_rgba(1, 1, .9, .7, 0)
        ctx.set_source(glow); ctx.arc(bx, by, 34 if big else 24, 0, 2 * np.pi); ctx.fill()
        ctx.set_source_rgb(1, 1, 1); ctx.arc(bx, by, 9 if big else 6.5, 0, 2 * np.pi); ctx.fill()


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_text(img):
    d = ImageDraw.Draw(img)
    # title: solid white, one line
    f = font(HEAVY, 156)
    text = 'Why Post-Training?'
    tw = d.textlength(text, font=f)
    assert tw < W * S * .86
    d.text(((W * S - tw) / 2, 150 * S - f.getbbox('W')[1]), text, font=f, fill=(255, 255, 255))

    # series information inside 6% safe margins
    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    label = 'POST-TRAINING SERIES  ·  01'
    spaced = ' '.join(label).replace('   ', '  ')
    d.text((mx, my), spaced, font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')  # letter-spaced caps, like the series label
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)
    # gate legend under the gates, in their colors
    lf = font(DEMI, 23)
    for (gx, rgb), word in zip(GATES, ['show', 'compare', 'check']):
        cy = float(centerline(np.array([float(gx)]))[0]); ry = float(half_width(np.array([float(gx)]))[0]) + 26
        wd = d.textlength(word, font=lf)
        d.text((gx * S - wd / 2, (cy - ry - 52) * S), word, font=lf, fill=tuple(int(c * 255) for c in rgb))


def main():
    rng = np.random.default_rng(23)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())     # BGRA → RGB
    draw_text(img)
    out = img.resize((W, H), Image.LANCZOS)
    path = P / 'assets' / 'cover-post-training.png'
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
