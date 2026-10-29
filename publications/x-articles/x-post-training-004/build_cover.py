"""Procedural 5:2 cover for X-POST-004 (one English cover shared by both versions).

Series style (white title, gold labels, deep ink, engraved lines, stipple dust) with this
article's own motif: a tree of token paths. One glowing state node on the left branches into
many engraved paths, line width = path probability; each path ends in a score dot (green high,
dim violet low). One sampled path is lit in blue, and a violet dashed value line runs above it,
stepping down where the path takes a low-scoring branch. Editorial picture, not real data.

uv run --project visuals/animations python publications/x-articles/x-post-training-004/build_cover.py
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
GREY = (0.55, 0.60, 0.68)

X0, X1, Y0, Y1 = 250, 1720, 330, 745          # tree region
FAN = [3, 2, 2, 2]                             # children per level: 24 leaves


def build_tree(rng):
    """Nodes as dicts: level, children, path probability, leaf score; tidy y positions."""
    root = dict(level=0, p=1.0, kids=[])
    frontier = [root]
    for lv, k in enumerate(FAN):
        nxt = []
        for n in frontier:
            w = rng.dirichlet(np.full(k, 1.3))
            for i in range(k):
                c = dict(level=lv + 1, p=n['p'] * w[i], kids=[], parent=n)
                n['kids'].append(c); nxt.append(c)
        frontier = nxt
    leaves = frontier
    for i, l in enumerate(leaves):
        l['y'] = Y0 + (Y1 - Y0) * (i + .5) / len(leaves)
    def place(n):
        if n['kids']:
            for c in n['kids']:
                place(c)
            n['y'] = float(np.mean([c['y'] for c in n['kids']]))
    place(root)
    xs = np.linspace(X0, X1, len(FAN) + 1)
    def setx(n):
        n['x'] = xs[n['level']]
        for c in n['kids']:
            setx(c)
    setx(root)
    return root, leaves


def scores(root, leaves, rng, bad):
    """Leaf scores 0–10; the subtree under the bad branch scores low. V = probability-weighted mean."""
    def under(n, a):
        while n is not None:
            if n is a:
                return True
            n = n.get('parent')
        return False
    for l in leaves:
        l['score'] = float(np.clip(rng.normal(1.5, .6) if under(l, bad) else rng.normal(7.5, 1.5), 0, 10))
    def value(n):
        if not n['kids']:
            n['v'] = n['score']; return n['v']
        tot = sum(c['p'] for c in n['kids'])
        n['v'] = sum(c['p'] / tot * value(c) for c in n['kids'])
        return n['v']
    value(root)


def curve(ctx, a, b):
    mx = (a['x'] + b['x']) / 2
    ctx.move_to(a['x'], a['y']); ctx.curve_to(mx, a['y'], mx, b['y'], b['x'], b['y'])


def draw_art(ctx, rng):
    ctx.set_source_rgb(0.012, 0.016, 0.03); ctx.paint()
    g = cairo.RadialGradient(820, 560, 20, 820, 560, 1000)
    g.add_color_stop_rgba(0, 0.13, 0.22, 0.42, 0.42); g.add_color_stop_rgba(1, 0, 0, 0, 0)
    ctx.set_source(g); ctx.paint()
    for _ in range(4200):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        near = 1.0 if (Y0 - 50 < y < Y1 + 30) else .35
        if rng.random() > .25 + .75 * near:
            continue
        ctx.set_source_rgba(1, 1, 1, rng.uniform(.06, .4) * near)
        ctx.arc(x, y, rng.uniform(.5, 1.3), 0, 2 * np.pi); ctx.fill()

    root, leaves = build_tree(rng)
    # the lit path: most likely child at each level, except one low-value turn at level 2
    path, n = [root], root
    while n['kids']:
        kids = sorted(n['kids'], key=lambda c: -c['p'])
        n = kids[0] if n['level'] != 1 else kids[min(1, len(kids) - 1)]
        path.append(n)
    bad = path[2]
    scores(root, leaves, rng, bad)

    # engraved paths, width = path probability
    def edges(n):
        for c in n['kids']:
            yield n, c
            yield from edges(c)
    for a, b in edges(root):
        w = .6 + 9 * np.sqrt(b['p'])
        curve(ctx, a, b); ctx.set_line_width(w); ctx.set_source_rgba(.78, .85, 1, .10 + .25 * np.sqrt(b['p'])); ctx.stroke()
        curve(ctx, a, b); ctx.set_line_width(.7); ctx.set_source_rgba(.85, .9, 1, .22); ctx.stroke()

    # the lit path: a soft blue glow under a bright core
    for width, alpha in [(22, .07), (11, .16), (4.5, .95)]:
        for a, b in zip(path, path[1:]):
            curve(ctx, a, b)
        ctx.set_line_width(width); ctx.set_source_rgba(*BLUE, alpha); ctx.stroke()

    # leaf score dots: green for high scores, dim violet for low
    for l in leaves:
        s = l['score'] / 10
        col = tuple(VIOLET[k] * (1 - s) + GREEN[k] * s for k in range(3))
        r = 2.5 + 4.5 * s + (3 if l is path[-1] else 0)
        if s > .6:
            glow = cairo.RadialGradient(l['x'] + 14, l['y'], 0, l['x'] + 14, l['y'], 16)
            glow.add_color_stop_rgba(0, *GREEN, .5 * s); glow.add_color_stop_rgba(1, *GREEN, 0)
            ctx.set_source(glow); ctx.arc(l['x'] + 14, l['y'], 16, 0, 2 * np.pi); ctx.fill()
        ctx.arc(l['x'] + 14, l['y'], r, 0, 2 * np.pi); ctx.set_source_rgba(*col, .6 + .35 * s); ctx.fill()

    # the value line runs parallel above the lit path, offset by V: it steps down at the bad turn
    off = lambda v: 22 + 11 * v
    def bez(a, b, u):
        mx = (a['x'] + b['x']) / 2
        x = (1 - u) ** 3 * a['x'] + 3 * (1 - u) ** 2 * u * mx + 3 * (1 - u) * u ** 2 * mx + u ** 3 * b['x']
        y = (1 - u) ** 3 * a['y'] + 3 * (1 - u) ** 2 * u * a['y'] + 3 * (1 - u) * u ** 2 * b['y'] + u ** 3 * b['y']
        return x, y
    pts = []
    for a, b in zip(path, path[1:]):
        for u in np.linspace(0, 1, 60):
            x, y = bez(a, b, u)
            pts.append((x, y - off(a['v'])))
        pts.append((b['x'], b['y'] - off(b['v'])))          # a vertical step at the node
    ctx.set_dash([12, 9]); ctx.set_line_width(3.2); ctx.set_source_rgba(*VIOLET, .9)
    ctx.move_to(*pts[0])
    for q in pts[1:]:
        ctx.line_to(*q)
    ctx.stroke(); ctx.set_dash([])

    # the state node
    for rad, alpha in [(34, .10), (20, .22)]:
        ctx.arc(root['x'], root['y'], rad, 0, 2 * np.pi); ctx.set_source_rgba(*BLUE, alpha); ctx.fill()
    ctx.arc(root['x'], root['y'], 9, 0, 2 * np.pi); ctx.set_source_rgb(*BLUE); ctx.fill()
    return root, leaves


def font(index, size):
    return ImageFont.truetype(AVENIR, size * S, index=index)


def draw_text(img, root):
    d = ImageDraw.Draw(img)
    f = font(HEAVY, 156)
    text = 'RL Terms for LLMs'
    tw = d.textlength(text, font=f)
    assert tw < W * S * .86
    d.text(((W * S - tw) / 2, 150 * S - f.getbbox('W')[1]), text, font=f, fill=(255, 255, 255))
    mx, my = 120 * S, 52 * S
    small = font(DEMI, 25)
    d.text((mx, my), ' '.join('POST-TRAINING SERIES  ·  04').replace('   ', '  '), font=small, fill=GOLD)
    brand = ' '.join('DONGXI LABS').replace('   ', '  ')
    d.text((W * S - mx - d.textlength(brand, font=small), my), brand, font=small, fill=GOLD)
    lf = font(DEMI, 25)
    for x, y, word, col in [(root['x'], root['y'] + 44, 'state', BLUE), (X1 + 14, Y1 + 12, 'reward', GREEN)]:
        wd = d.textlength(word, font=lf)
        d.text((x * S - wd / 2, y * S), word, font=lf, fill=tuple(int(c * 255) for c in col))


def main():
    rng = np.random.default_rng(4)
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * S, H * S)
    ctx = cairo.Context(surf); ctx.scale(S, S); ctx.set_antialias(cairo.ANTIALIAS_BEST)
    root, _ = draw_art(ctx, rng)
    buf = np.ndarray((H * S, W * S, 4), np.uint8, surf.get_data())
    img = Image.fromarray(buf[:, :, [2, 1, 0]].copy())
    draw_text(img, root)
    out = img.resize((W, H), Image.LANCZOS)
    (P / 'assets').mkdir(exist_ok=True)
    path = P / 'assets' / 'cover-post-training-04.png'
    out.save(path, optimize=True)
    assert out.size == (W, H)
    print(path.relative_to(P), out.size)


if __name__ == '__main__':
    main()
