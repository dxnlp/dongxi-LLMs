"""Procedural 5:2 cover for X-TOKEN-001 (one English cover shared by both versions).

Dongxi cover style (white title, gold labels, deep ink, engraved lines, stipple dust) with this
article's motif. On the right is a field of thin score bars standing for the vocabulary's
logits. One bar rises above the rest and becomes a glowing token, and a dashed arc carries that
token back to the empty slot at the end of the token sequence on the left. Editorial picture,
not data.

uv run --project visuals/animations python publications/x-articles/x-token-001/build_cover.py
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
AMBER = (0.90, 0.58, 0.22)
SLATE = (0.55, 0.60, 0.68)

BASE_Y = 735                       # common baseline of the sequence and the score field
CELL, GAP, X_CELLS = 34, 9, 170    # token cells
FX0, FX1, NBARS = 1040, 1800, 210   # score field
PICK = 112                         # the bar that wins


def rounded(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -np.pi / 2, 0); ctx.arc(x + w - r, y + h - r, r, 0, np.pi / 2)
    ctx.arc(x + r, y + h - r, r, np.pi / 2, np.pi); ctx.arc(x + r, y + r, r, np.pi, 3 * np.pi / 2)
    ctx.close_path()


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(1250, 600, 20, 1250, 600, 1000)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.42); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()
    for _ in range(4200):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        near = 1.0 if y > 330 else .35
        if rng.random() > .25 + .75 * near:
            continue
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.06, .4) * near)
        ctx.arc(x, y, rng.uniform(.5, 1.3), 0, 2 * np.pi); ctx.fill()

    # the score field: one thin engraved bar per (sampled) vocabulary entry
    xs = np.linspace(FX0, FX1, NBARS)
    h = rng.gamma(1.4, 32, NBARS)
    for k in rng.choice(NBARS, 14, replace=False):
        h[k] += rng.uniform(50, 140)
    h[PICK] = 330
    for i, (x, hh) in enumerate(zip(xs, h)):
        if i == PICK:
            continue
        a = .14 + .5 * min(hh / 220, 1)
        ctx.move_to(x, BASE_Y); ctx.line_to(x, BASE_Y - hh)
        ctx.set_line_width(2.2); ctx.set_source_rgba(.78, .85, 1, a); ctx.stroke()
    ctx.move_to(FX0 - 20, BASE_Y + 1); ctx.line_to(FX1 + 20, BASE_Y + 1)
    ctx.set_line_width(1); ctx.set_source_rgba(.85, .9, 1, .3); ctx.stroke()

    # the winning bar glows and becomes a token
    px, top = xs[PICK], BASE_Y - h[PICK]
    for width, alpha in [(24, .07), (11, .18), (4.5, .95)]:
        ctx.move_to(px, BASE_Y); ctx.line_to(px, top)
        ctx.set_line_width(width); ctx.set_source_rgba(*BLUE, alpha); ctx.stroke()
    tok = 56
    ty = top - tok - 18
    glow = cairo.RadialGradient(px, ty + tok / 2, 0, px, ty + tok / 2, 90)
    glow.add_color_stop_rgba(0, *BLUE, .55); glow.add_color_stop_rgba(1, *BLUE, 0)
    ctx.set_source(glow); ctx.arc(px, ty + tok / 2, 90, 0, 2 * np.pi); ctx.fill()
    rounded(ctx, px - tok / 2, ty, tok, tok, 11); ctx.set_source_rgb(*BLUE); ctx.fill()

    # the token sequence: template (amber), the user's text (slate), the reply so far (blue)
    kinds = ['a'] * 3 + ['s'] * 5 + ['a'] * 5 + ['b'] * 4
    col = dict(a=AMBER, s=SLATE, b=BLUE)
    for i, k in enumerate(kinds):
        x = X_CELLS + i * (CELL + GAP)
        rounded(ctx, x, BASE_Y - CELL, CELL, CELL, 6); ctx.set_source_rgba(*col[k], .9 if k != 'b' else 1); ctx.fill()
    slot_x = X_CELLS + len(kinds) * (CELL + GAP)
    ctx.set_dash([6, 5]); rounded(ctx, slot_x, BASE_Y - CELL, CELL, CELL, 6)
    ctx.set_line_width(2.4); ctx.set_source_rgba(*BLUE, .9); ctx.stroke()

    # the dashed arc from the new token back to the empty slot
    sx, sy = px - tok / 2 - 14, ty + tok / 2
    ex, ey = slot_x + CELL / 2, BASE_Y - CELL - 16
    ctx.move_to(sx, sy)
    ctx.curve_to(sx - 260, sy - 40, ex + 120, ey - 300, ex, ey)
    ctx.set_dash([14, 10]); ctx.set_line_width(3.2); ctx.set_source_rgba(*BLUE, .85); ctx.stroke()
    ctx.set_dash([])
    # arrow head pointing down into the slot
    ctx.move_to(ex, ey + 4); ctx.line_to(ex - 11, ey - 14); ctx.line_to(ex + 11, ey - 14); ctx.close_path()
    ctx.set_source_rgba(*BLUE, .95); ctx.fill()
    return dict(field=(FX0, FX1), cells=(X_CELLS, slot_x + CELL))


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_text(img, art):
    d = ImageDraw.Draw(img)
    f = font(HEAVY, 150)
    text = 'The Birth of a Token'
    tw = d.textlength(text, font=f)
    assert tw < W * S * .86
    d.text(((W * S - tw) / 2, 140 * S - f.getbbox('W')[1]), text, font=f, fill=(255, 255, 255))
    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    d.text((mx, my), ' '.join('INSIDE AN LLM').replace('   ', '  '), font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)
    lf = font(DEMI, 25)
    for (x0, x1), word, col in [(art['field'], '151,936 scores', BLUE), (art['cells'], 'tokens', SLATE)]:
        wd = d.textlength(word, font=lf)
        d.text(((x0 + x1) / 2 * S - wd / 2, (BASE_Y + 14) * S), word, font=lf, fill=tuple(int(c * 255) for c in col))


def main():
    rng = np.random.default_rng(7)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    art = draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img, art)
    out = img.resize((W, H), Image.LANCZOS)
    (P / 'assets').mkdir(exist_ok=True)
    path = P / 'assets' / 'cover-token-birth.png'
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
