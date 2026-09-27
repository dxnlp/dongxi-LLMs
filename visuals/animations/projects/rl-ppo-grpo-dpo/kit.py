"""Visual vocabulary for the PPO / GRPO / DPO film."""
from __future__ import annotations

import re
from pathlib import Path
import sys

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, PI, RIGHT, UP, Arc, CurvedArrow, DashedVMobject, Dot, Line,
    Mobject, RoundedRectangle, SVGMobject, Text, VGroup, NORMAL, MEDIUM, config,
)
from svgelements import SVG, Shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mathsvg import math_svg  # noqa: E402

# Palette: one meaning per color, held fixed for the whole film.
BG = "#0B1120"
PANEL = "#111A2E"
TEXT = "#E2E8F0"
SUB = "#94A3B8"
DIM = "#475569"
LINE = "#334155"
POLICY = "#60A5FA"     # the model being trained
REF = "#94A3B8"        # frozen reference copy
REWARD = "#C084FC"     # reward model / verifier score
CRITIC = "#2DD4BF"     # learned value model
POS = "#4ADE80"        # better than expected / chosen
NEG = "#FB7185"        # worse than expected / rejected
CLIP = "#FBBF24"       # constraints: clipping and the KL leash

FONT = "Avenir Next"


def T(s: str, size: float = 26, color: str = TEXT, weight=NORMAL) -> Text:
    """Text shaped near 24pt, then scaled: avoids Pango advance rounding at tiny sizes.

    Pango wraps at the frame's pixel width (in pt, font_size / 4.8), which bounds the
    oversampling for long strings.
    """
    fit = 6.0 * (config.pixel_width - 60) / (max(len(s), 1) * size)
    k = max(1.0, min(115 / size, fit))
    return Text(s, font=FONT, font_size=size * k, color=color, weight=weight).scale(1 / k)


# ---------------------------------------------------------------- equations
PT = 0.0142          # Manim units per SVG point at scale 1 (fontsize 48 source)
PAD = 1.44           # bbox pad written by mathsvg (0.02 in)


def _svg_size(path: Path) -> tuple[float, float]:
    s = path.read_text()
    w = float(re.search(r'width="([\d.]+)pt"', s).group(1))
    h = float(re.search(r'height="([\d.]+)pt"', s).group(1))
    return w, h


def _ink_height_pt(path: Path) -> float:
    ys = []
    for el in SVG.parse(str(path)).elements():
        if isinstance(el, Shape) and el.bbox():
            ys += el.bbox()[1::2]
    return (max(ys) - min(ys)) * 72 / 96


def M(expr: str, scale: float = 1.0, color: str = TEXT) -> SVGMobject:
    """Typeset math at one font scale: the ink is sized from its measured height,
    so a lone 'r' and a full fraction share the same glyph size."""
    path = math_svg(expr)
    mob = SVGMobject(str(path), height=_ink_height_pt(path) * PT * scale)
    mob.set_fill(color, 1).set_stroke(width=0)
    return mob


def Mc(parts: list[tuple[str, str]], scale: float = 1.0) -> SVGMobject:
    """One typeset expression, colored piecewise by the horizontal extent of each part."""
    full = "".join(e for e, _ in parts)
    mob = M(full, scale)
    w_full, _ = _svg_size(math_svg(full))
    unit = mob.width / (w_full - 2 * PAD)
    bounds, acc = [], ""
    for expr, _ in parts[:-1]:
        acc += expr
        w, _ = _svg_size(math_svg(acc))
        bounds.append((w - 2 * PAD) * unit)
    left = mob.get_left()[0]
    groups = [[] for _ in parts]
    for sub in mob.family_members_with_points():
        i = sum(sub.get_center()[0] - left > b + 1e-3 for b in bounds)
        sub.set_fill(parts[i][1], 1)
        groups[i].append(sub)
    mob.parts = groups
    return mob


# ---------------------------------------------------------------- objects
def tok(color: str, s: float = 0.52, fill: float = 0.16, sw: float = 2.4) -> RoundedRectangle:
    return RoundedRectangle(corner_radius=0.18 * s, width=s, height=s, stroke_color=color,
                            stroke_width=sw, fill_color=color, fill_opacity=fill)


def lock_icon(color: str, s: float = 0.2) -> VGroup:
    body = RoundedRectangle(corner_radius=0.03, width=s, height=0.8 * s, stroke_width=0,
                            fill_color=color, fill_opacity=1)
    shackle = Arc(radius=0.3 * s, start_angle=0, angle=PI, stroke_color=color, stroke_width=2.4)
    shackle.next_to(body, UP, buff=0)
    return VGroup(shackle, body)


def net_icon(color: str, w: float = 0.78, h: float = 0.5) -> VGroup:
    layers = []
    for x, n in zip(np.linspace(-w / 2, w / 2, 3), (3, 4, 3)):
        layers.append([np.array([x, y, 0.0]) for y in np.linspace(-h / 2, h / 2, n)])
    edges = VGroup(*[Line(a, b, stroke_width=1.1, stroke_color=color, stroke_opacity=0.4)
                     for l1, l2 in zip(layers, layers[1:]) for a in l1 for b in l2])
    nodes = VGroup(*[Dot(p, radius=0.045, color=color) for layer in layers for p in layer])
    return VGroup(edges, nodes)


def block(symbol: str, color: str, frozen: bool = False, w: float = 1.45, h: float = 1.2,
          symbol_scale: float = 1.0) -> VGroup:
    """A model: network glyph plus its symbol; dashed outline and lock when frozen."""
    fill = RoundedRectangle(corner_radius=0.2, width=w, height=h, stroke_width=0,
                            fill_color=color, fill_opacity=0.08 if frozen else 0.16)
    edge = RoundedRectangle(corner_radius=0.2, width=w, height=h, stroke_color=color,
                            stroke_width=2.6, fill_opacity=0)
    if frozen:
        edge = DashedVMobject(edge, num_dashes=34, dashed_ratio=0.55)
    icon = net_icon(color).move_to(fill.get_center() + UP * 0.2 * h)
    sym = M(symbol, symbol_scale, color).move_to(fill.get_center() + DOWN * 0.27 * h)
    parts = [fill, edge, icon, sym]
    if frozen:
        parts.append(lock_icon(color, 0.18).move_to(fill.get_corner(RIGHT + UP) + np.array([-0.2, -0.2, 0])))
    return VGroup(*parts)


def chip(name: str, symbol: str, color: str, frozen: bool) -> VGroup:
    label = VGroup(M(symbol, 0.74, color), T(name, 21, TEXT)).arrange(RIGHT, buff=0.14)
    extra = 0.34 if frozen else 0
    w, h = label.width + 0.46 + extra, 0.6
    fill = RoundedRectangle(corner_radius=0.16, width=w, height=h, stroke_width=0,
                            fill_color=color, fill_opacity=0.06 if frozen else 0.2)
    edge = RoundedRectangle(corner_radius=0.16, width=w, height=h, stroke_color=color,
                            stroke_width=2, fill_opacity=0)
    if frozen:
        edge = DashedVMobject(edge, num_dashes=30, dashed_ratio=0.55)
    label.move_to(fill).shift(LEFT * extra / 2)
    parts = [fill, edge, label]
    if frozen:
        parts.append(lock_icon(color, 0.15).next_to(label, RIGHT, buff=0.14))
    return VGroup(*parts)


CHIPS = {
    "policy": ("policy", r"\pi_\theta", POLICY, False),
    "critic": ("critic", r"V_\phi", CRITIC, False),
    "reward": ("reward", r"r", REWARD, True),
    "reference": ("reference", r"\pi_{\mathrm{ref}}", REF, True),
}


def make_chip(key: str) -> VGroup:
    return chip(*CHIPS[key])


def row_positions(mobs: list[Mobject], y: float, buff: float = 0.32, x_center: float = 0.0):
    total = sum(m.width for m in mobs) + buff * (len(mobs) - 1)
    x, out = x_center - total / 2, []
    for m in mobs:
        out.append(np.array([x + m.width / 2, y, 0.0]))
        x += m.width + buff
    return out


LOOP_ANGLE = -0.62


def tracker(labels: list[str], loop: bool = True):
    """Four-step process strip; an online method closes the loop underneath.

    Returns (strip, words, loop_arrow_or_None).
    """
    words = [T(s, 21, DIM) for s in labels]
    items = []
    for i, w in enumerate(words):
        items.append(w)
        if i < len(words) - 1:
            items.append(T("›", 24, DIM))
    strip = VGroup(*items).arrange(RIGHT, buff=0.2)
    strip.to_corner(UP + RIGHT, buff=0.42)
    arrow = None
    if loop:
        arrow = CurvedArrow(*loop_ends(words), angle=LOOP_ANGLE, color=DIM, stroke_width=2,
                            tip_length=0.14)
    return strip, words, arrow


def loop_ends(words):
    return (words[-1].get_bottom() + DOWN * 0.17, words[0].get_bottom() + DOWN * 0.17)


def header(name: str, subtitle: str) -> VGroup:
    n = T(name, 46, TEXT, MEDIUM)
    s = T(subtitle, 22, SUB)
    g = VGroup(n, s).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    return g.to_corner(UP + LEFT, buff=0.42)


def spring(a: np.ndarray, b: np.ndarray, coils: int = 7, amp: float = 0.11, color: str = CLIP):
    """Zig-zag leash between two points; stretches with their distance."""
    a, b = np.array(a, float), np.array(b, float)
    d = b - a
    length = np.linalg.norm(d)
    u = d / length
    n = np.array([-u[1], u[0], 0.0])
    lead = min(0.12, length * 0.15)
    pts = [a, a + u * lead]
    k = 2 * coils
    for i in range(1, k):
        pts.append(a + u * (lead + (length - 2 * lead) * i / k) + n * amp * (1 if i % 2 else -1))
    pts += [b - u * lead, b]
    from manim import VMobject
    v = VMobject(stroke_color=color, stroke_width=2.6)
    v.set_points_as_corners([np.array(p) for p in pts])
    return v


__all__ = [name for name in dir() if not name.startswith("_")] + ["ORIGIN"]
