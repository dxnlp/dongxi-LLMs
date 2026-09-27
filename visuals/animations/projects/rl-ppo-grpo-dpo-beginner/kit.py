"""Characters and props for the beginner film: robots, word tiles, prize wheel, balance."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, PI, RIGHT, UP, WHITE, Arc, Circle, Dot, Line, Polygon, Rectangle,
    RoundedRectangle, Sector, Star, Text, VGroup, VMobject, NORMAL, config,
)
from manim.constants import CapStyleType, LineJointType

# Paper theme; one meaning per color for the whole film.
BG = "#FAF7F0"
INK = "#1E293B"
SOFT = "#64748B"
FAINT = "#CBD5E1"
CARD = "#FFFFFF"
SHADOW = "#0F172A"
STUDENT, STUDENT_FILL = "#2563EB", "#DBEAFE"
COACH, COACH_FILL = "#0D9488", "#CCFBF1"
JUDGE, JUDGE_FILL = "#7C3AED", "#EDE9FE"
SNAP, SNAP_FILL = "#64748B", "#E2E8F0"
GOOD, GOOD_FILL = "#16A34A", "#DCFCE7"
BAD, BAD_FILL = "#DC2626", "#FEE2E2"
LEASH, LEASH_FILL = "#D97706", "#FEF3C7"

FONT = "Avenir Next"
HAND = "Noteworthy"
ROUND = "Arial Rounded MT Bold"

DESCENDERS = set("gjpqy,;()")


def T(s: str, size: float = 26, color: str = INK, weight=NORMAL, font: str = FONT,
      t2c: dict | None = None) -> Text:
    """Text shaped near 24pt, then scaled: avoids Pango advance rounding at small sizes.

    Pango wraps at the frame's pixel width (in pt, font_size / 4.8), which bounds the
    oversampling for long strings.
    """
    longest = max(len(line) for line in s.split("\n"))
    fit = 6.0 * (config.pixel_width - 60) / (max(longest, 1) * size)
    k = max(1.0, min(115 / size, fit))
    return Text(s, font=font, font_size=size * k, color=color, weight=weight,
                t2c=t2c or {}).scale(1 / k)


_metrics: dict = {}


def _font_metrics(size: float, font: str, weight) -> tuple[float, float]:
    key = (size, font, str(weight))
    if key not in _metrics:
        x = T("x", size, font=font, weight=weight).height
        p = T("p", size, font=font, weight=weight).height
        cap = T("H", size, font=font, weight=weight).height
        _metrics[key] = (p - x, cap)
    return _metrics[key]


def place_on_baseline(mob: Text, s: str, center_y: float, size: float, font: str = FONT,
                      weight=NORMAL) -> Text:
    """Vertically place a word so words with and without descenders share a baseline."""
    desc, cap = _font_metrics(size, font, weight)
    baseline = center_y - 0.5 * cap
    bottom = baseline - (desc if any(c in DESCENDERS for c in s) else 0.0)
    mob.shift(UP * (bottom - mob.get_bottom()[1]))
    return mob


# ------------------------------------------------------------------ labels
def tag(text: str, color: str = SOFT, size: float = 17) -> VGroup:
    """A small pill that names the technical term behind a metaphor."""
    t = T(text, size, color)
    box = RoundedRectangle(corner_radius=0.15, width=t.width + 0.34, height=0.36,
                           fill_color=CARD, fill_opacity=1, stroke_color=FAINT, stroke_width=1.5)
    place_on_baseline(t, text, box.get_center()[1], size)
    t.set_x(box.get_center()[0])
    return VGroup(box, t)


def shadowed(shape: VMobject, dx: float = 0.05, dy: float = -0.07, opacity: float = 0.10):
    sh = shape.copy().set_fill(SHADOW, opacity).set_stroke(width=0).shift(RIGHT * dx + UP * dy)
    return VGroup(sh, shape)


def card(width: float, height: float, stroke: str = FAINT, fill: str = CARD, radius=0.22,
         stroke_width: float = 2) -> VGroup:
    box = RoundedRectangle(corner_radius=radius, width=width, height=height, fill_color=fill,
                           fill_opacity=1, stroke_color=stroke, stroke_width=stroke_width)
    return shadowed(box)


# ------------------------------------------------------------------ robots
def robot(color: str, fill: str, kind: str = "plain", s: float = 1.0) -> VGroup:
    head = RoundedRectangle(corner_radius=0.32, width=1.44, height=1.18, fill_color=fill,
                            fill_opacity=1, stroke_color=color, stroke_width=5)
    visor = RoundedRectangle(corner_radius=0.22, width=1.06, height=0.56, fill_color=color,
                             fill_opacity=1, stroke_width=0).shift(UP * 0.1)
    eyes = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.15, height=0.23,
                                     fill_color=WHITE, fill_opacity=1, stroke_width=0)
                    .move_to(visor.get_center() + RIGHT * dx) for dx in (-0.23, 0.23)])
    mouth = Arc(radius=0.16, start_angle=-0.8 * PI, angle=0.6 * PI, stroke_color=color,
                stroke_width=4).move_to(head.get_center() + DOWN * 0.37)
    ears = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.13, height=0.42,
                                     fill_color=color, fill_opacity=1, stroke_width=0)
                    .move_to(head.get_center() + RIGHT * dx) for dx in (-0.76, 0.76)])
    parts = [ears, head, visor, eyes, mouth]
    top = head.get_top()
    if kind == "student":
        board = Polygon([-0.66, 0, 0], [0, 0.21, 0], [0.66, 0, 0], [0, -0.21, 0],
                        fill_color=INK, fill_opacity=1, stroke_width=0)
        base = RoundedRectangle(corner_radius=0.05, width=0.7, height=0.24, fill_color=INK,
                                fill_opacity=1, stroke_width=0).shift(DOWN * 0.13)
        c = board.get_center()
        tassel = VGroup(
            Line(c, c + RIGHT * 0.5 + DOWN * 0.06, stroke_color=LEASH, stroke_width=3.5),
            Line(c + RIGHT * 0.5 + DOWN * 0.06, c + RIGHT * 0.5 + DOWN * 0.34,
                 stroke_color=LEASH, stroke_width=3.5),
            Dot(c + RIGHT * 0.5 + DOWN * 0.38, radius=0.055, color=LEASH))
        cap = VGroup(base, board, tassel)
        cap.shift(top + UP * 0.1 - base.get_bottom() + DOWN * 0.02)
        parts.append(cap)
    elif kind == "coach":
        c = head.get_center()
        band = Arc(radius=0.84, start_angle=0.1 * PI, angle=0.8 * PI, arc_center=c + UP * 0.02,
                   stroke_color=INK, stroke_width=7)
        cups = VGroup(*[RoundedRectangle(corner_radius=0.07, width=0.2, height=0.44,
                                         fill_color=INK, fill_opacity=1, stroke_width=0)
                        .move_to(c + RIGHT * dx + UP * 0.02) for dx in (-0.8, 0.8)])
        boom = VMobject(stroke_color=INK, stroke_width=4)
        boom.set_points_smoothly([c + RIGHT * 0.8 + DOWN * 0.18, c + RIGHT * 0.66 + DOWN * 0.46,
                                  c + RIGHT * 0.36 + DOWN * 0.5])
        mic = Dot(c + RIGHT * 0.33 + DOWN * 0.5, radius=0.07, color=INK)
        parts += [band, cups, boom, mic]
    elif kind == "judge":
        c = head.get_bottom() + DOWN * 0.13
        bow = VGroup(Polygon(c, c + LEFT * 0.3 + UP * 0.14, c + LEFT * 0.3 + DOWN * 0.14),
                     Polygon(c, c + RIGHT * 0.3 + UP * 0.14, c + RIGHT * 0.3 + DOWN * 0.14))
        bow.set_fill(color, 1).set_stroke(color, 2, 1)
        knot = RoundedRectangle(corner_radius=0.03, width=0.1, height=0.12, fill_color=color,
                                fill_opacity=1, stroke_width=0).move_to(c)
        parts += [bow, knot]
    else:
        stalk = Line(top, top + UP * 0.22, stroke_color=color, stroke_width=5)
        ball = Circle(radius=0.08, fill_color=color, fill_opacity=1, stroke_width=0)
        ball.move_to(stalk.get_end() + UP * 0.05)
        parts += [stalk, ball]
    g = VGroup(*parts)
    g.eyes = eyes
    g.body = head
    return g.scale(s)


def student(s=1.0):
    return robot(STUDENT, STUDENT_FILL, "student", s)


def coach(s=1.0):
    return robot(COACH, COACH_FILL, "coach", s)


def judge(s=1.0):
    return robot(JUDGE, JUDGE_FILL, "judge", s)


def polaroid(s: float = 1.0, label: str = "original model") -> VGroup:
    """The frozen reference: a snapshot of the model before training."""
    w, h = 1.9, 2.25
    frame = RoundedRectangle(corner_radius=0.06, width=w, height=h, fill_color=CARD,
                             fill_opacity=1, stroke_color=FAINT, stroke_width=2)
    photo = Rectangle(width=w - 0.26, height=w - 0.26, fill_color="#EEF2F7", fill_opacity=1,
                      stroke_width=0).move_to(frame.get_top() + DOWN * (0.13 + (w - 0.26) / 2))
    face = robot(SNAP, SNAP_FILL).scale_to_fit_width(photo.width * 0.66).move_to(photo)
    face.shift(DOWN * 0.04)
    cap = T(label, 19, SOFT, font=HAND)
    cap.move_to((frame.get_bottom() + photo.get_bottom()) / 2)
    tape = Rectangle(width=0.56, height=0.18, fill_color=LEASH_FILL, fill_opacity=0.95,
                     stroke_width=0).move_to(frame.get_top()).rotate(0.1)
    g = VGroup(*shadowed(frame), photo, face, cap, tape).rotate(-0.06)
    g.photo = face
    return g.scale(s)


# ------------------------------------------------------------------ words
def tile(word: str, color: str = STUDENT, fill: str = CARD, size: float = 24,
         height: float = 0.62) -> VGroup:
    t = T(word, size, INK)
    box = RoundedRectangle(corner_radius=0.14, width=t.width + 0.38, height=height,
                           fill_color=fill, fill_opacity=1, stroke_color=color, stroke_width=2.5)
    place_on_baseline(t, word, box.get_center()[1], size)
    t.set_x(box.get_center()[0])
    g = VGroup(box, t)
    g.box, g.word = box, t
    return g


def tile_row(words: list[str], left: float, y: float, gap: float = 0.14, **kw) -> VGroup:
    row = VGroup()
    x = left
    for w in words:
        t = tile(w, **kw)
        t.move_to([x + t.width / 2, y, 0])
        x += t.width + gap
        row.add(t)
    return row


def check_icon(color: str = GOOD, s: float = 0.32, width: float = 7) -> VMobject:
    v = VMobject(stroke_color=color, stroke_width=width, joint_type=LineJointType.ROUND,
                 cap_style=CapStyleType.ROUND)
    v.set_points_as_corners([np.array([-0.5, 0.0, 0]), np.array([-0.15, -0.38, 0]),
                             np.array([0.55, 0.45, 0])])
    return v.scale(s)


def cross_icon(color: str = BAD, s: float = 0.26, width: float = 7) -> VGroup:
    return VGroup(*[Line(a, b, cap_style=CapStyleType.ROUND)
                    for a, b in (([-0.5, -0.5, 0], [0.5, 0.5, 0]),
                                 ([-0.5, 0.5, 0], [0.5, -0.5, 0]))]
                  ).set_stroke(color, width).scale(s)


def scorecard(value: str, color: str = JUDGE, s: float = 1.0) -> VGroup:
    stick = Line([0, -0.95, 0], [0, -0.3, 0], stroke_color="#A16207", stroke_width=8)
    board = RoundedRectangle(corner_radius=0.12, width=0.92, height=0.78, fill_color=CARD,
                             fill_opacity=1, stroke_color=color, stroke_width=4)
    num = T(value, 40, color, font=ROUND).move_to(board)
    g = VGroup(stick, *shadowed(board), num)
    g.num, g.s = num, s
    return g.scale(s)


def clipboard(title: str, value: str, s: float = 1.0) -> VGroup:
    board = RoundedRectangle(corner_radius=0.12, width=1.5, height=1.9, fill_color="#B45309",
                             fill_opacity=1, stroke_width=0)
    paper = RoundedRectangle(corner_radius=0.06, width=1.24, height=1.5, fill_color=CARD,
                             fill_opacity=1, stroke_width=0).move_to(board).shift(DOWN * 0.1)
    clip = RoundedRectangle(corner_radius=0.06, width=0.6, height=0.22, fill_color=SOFT,
                            fill_opacity=1, stroke_width=0).move_to(board.get_top() + DOWN * 0.1)
    t = T(title, 18, SOFT, font=HAND).move_to(paper.get_top() + DOWN * 0.3)
    v = T(value, 36, INK, font=ROUND).move_to(paper.get_center() + DOWN * 0.18)
    return VGroup(*shadowed(board), paper, clip, t, v).scale(s)


# ------------------------------------------------------------------ wheel
def wheel(shares, colors, radius: float = 1.3, rot: float = 0.0) -> VGroup:
    """A prize wheel: slice size is the chance of each next word. Slices run clockwise
    from the pointer at the top."""
    slices = VGroup()
    a = PI / 2 + rot
    for sh, col in zip(shares, colors):
        ang = 2 * PI * sh
        slices.add(Sector(radius=radius, angle=-ang, start_angle=a, fill_color=col,
                          fill_opacity=1, stroke_color=CARD, stroke_width=3))
        a -= ang
    rim = Circle(radius=radius, stroke_color=INK, stroke_width=5, fill_opacity=0)
    hub = Circle(radius=0.11, fill_color=INK, fill_opacity=1, stroke_width=0)
    return VGroup(slices, rim, hub)


def slice_mid_angles(shares, rot: float = 0.0):
    out, a = [], PI / 2 + rot
    for sh in shares:
        ang = 2 * PI * sh
        out.append(a - ang / 2)
        a -= ang
    return out


def pointer(radius: float = 1.3, color: str = INK) -> Polygon:
    tip = np.array([0, radius - 0.2, 0])
    return Polygon(tip, tip + np.array([-0.17, 0.42, 0]), tip + np.array([0.17, 0.42, 0]),
                   fill_color=color, fill_opacity=1, stroke_color=CARD, stroke_width=2)


WHEEL_COLORS = ["#F87171", "#4ADE80", "#A5B4FC", "#E2E8F0"]


def wheel_labels(shares, names, radius: float, center=ORIGIN, size: float = 19,
                 rot: float = 0.0, gap: float = 0.07) -> VGroup:
    """Labels outside the rim, on the side of their slice; labels near the top or bottom
    step aside of the pointer, and labels on one side are pushed apart if they collide."""
    center = np.array(center, dtype=float)
    placed = []
    for ang, name, sh in zip(slice_mid_angles(shares, rot), names, shares):
        d = np.array([np.cos(ang), np.sin(ang), 0])
        pct = f"{round(100 * sh)}%"
        lab = T(f"{name} {pct}", size, INK, t2c={pct: SOFT})
        side = 1 if d[0] >= 0 else -1
        anchor = center + d * (radius + 0.2)
        x = anchor[0] if abs(d[0]) > 0.3 else center[0] + side * 0.3
        lab.move_to([x + side * lab.width / 2, anchor[1], 0])
        placed.append((side, lab))
    for side in (1, -1):
        col = sorted([lb for s, lb in placed if s == side], key=lambda m: -m.get_y())
        for a, b in zip(col, col[1:]):
            need = (a.height + b.height) / 2 + gap
            if a.get_y() - b.get_y() < need:
                b.set_y(a.get_y() - need)
    return VGroup(*[lb for _, lb in placed])


# ------------------------------------------------------------------ balance
def balance(tilt: float, pivot, arm: float = 1.75, drop: float = 1.05, left=None, right=None,
            post: float = 1.6, lw: float = 7) -> VGroup:
    """A beam balance; positive tilt puts the left pan down."""
    pivot = np.array([*pivot, 0.0][:3], dtype=float)
    c, s = np.cos(tilt), np.sin(tilt)
    le = pivot + np.array([-arm * c, -arm * s, 0])
    re = pivot + np.array([arm * c, arm * s, 0])
    stand = VGroup(
        Line(pivot, pivot + DOWN * post, stroke_color=INK, stroke_width=lw),
        RoundedRectangle(corner_radius=0.08 * arm / 1.75, width=0.74 * arm,
                         height=0.11 * arm, fill_color=INK, fill_opacity=1, stroke_width=0
                         ).move_to(pivot + DOWN * (post + 0.05 * arm)))
    beam = Line(le, re, stroke_color=INK, stroke_width=lw)
    g = VGroup(stand, beam)
    pr = 0.35 * arm
    for end, item in ((le, left), (re, right)):
        pan_c = end + DOWN * drop
        pan = Arc(radius=pr, start_angle=PI, angle=PI, stroke_color=INK, stroke_width=lw * 0.72)
        pan.stretch(0.38, 1).move_to(pan_c + DOWN * pr * 0.19)
        strings = VGroup(Line(end, pan_c + LEFT * pr), Line(end, pan_c + RIGHT * pr)
                         ).set_stroke(SOFT, max(1.5, lw * 0.3))
        g.add(strings, pan)
        if item is not None:
            g.add(item.copy().move_to(pan_c + UP * (item.height / 2 + 0.02)))
    g.add(Dot(pivot, radius=0.1 * arm / 1.75, color=INK))
    g.ends = (le, re)
    return g


# ------------------------------------------------------------------ small props
def bowl(good: bool = True, s: float = 1.0) -> VGroup:
    body = Arc(radius=0.9, start_angle=PI, angle=PI).stretch(0.62, 1)
    body = VMobject().set_points(body.points)
    body.add_line_to(body.points[0])
    body.set_fill("#F1F5F9", 1).set_stroke(INK, 4)
    soup = Circle(radius=0.9).stretch(0.16, 1).move_to(body.get_top())
    soup.set_fill("#F59E0B" if good else "#A8A08A", 1).set_stroke(INK, 4)
    g = VGroup(body, soup)
    if not good:
        for dx, dy, r in ((-0.3, 0.01, 0.09), (0.28, -0.02, 0.07)):
            g.add(Circle(radius=r).stretch(0.5, 1).move_to(soup.get_center() + np.array(
                [dx, dy, 0])).set_fill("#6B6454", 1).set_stroke(width=0))
    if good:
        for dx, dy in ((-0.35, 0.02), (0.1, -0.03), (0.42, 0.03)):
            g.add(Dot(soup.get_center() + np.array([dx, dy, 0]), radius=0.06, color=GOOD))
        for dx in (-0.35, 0.0, 0.35):
            st = VMobject(stroke_color=SOFT, stroke_width=3.5)
            st.set_points_smoothly([soup.get_center() + np.array([dx + 0.06 * np.sin(k), 0.25
                                                                  + 0.16 * k, 0])
                                    for k in range(5)])
            g.add(st)
    return g.scale(s)


def star(color: str = LEASH, s: float = 0.35) -> Star:
    return Star(n=5, outer_radius=1, inner_radius=0.45, fill_color=color, fill_opacity=1,
                stroke_color=color, stroke_width=2).scale(s)


def memory_box(members: list[VMobject], width: float = 4.6, height: float = 1.9,
               label: str = "computer memory") -> VGroup:
    box = RoundedRectangle(corner_radius=0.2, width=width, height=height, fill_color=CARD,
                           fill_opacity=1, stroke_color=INK, stroke_width=3)
    lab = T(label, 17, SOFT).next_to(box.get_corner(UP + LEFT), DOWN + RIGHT, buff=0.12)
    row = VGroup(*members).arrange(RIGHT, buff=0.12)
    row.scale_to_fit_height(min(row.height, height - 0.62))
    if row.width > width - 0.3:
        row.scale_to_fit_width(width - 0.3)
    row.move_to(box.get_center() + DOWN * 0.14)
    return VGroup(*shadowed(box), lab, row)


CAST = {"student": (student, "student"), "coach": (coach, "coach"),
        "judge": (judge, "judge"), "original": (polaroid, "original")}


def memory_panel(keys: list[str], slot: float = 2.1, icon_h: float = 1.55) -> VGroup:
    """The cast members that must sit in (GPU) memory at the same time."""
    members = VGroup()
    for k in keys:
        make, name = CAST[k]
        icon = make().scale_to_fit_height(icon_h)
        lab = T(name, 21, SOFT)
        members.add(VGroup(icon, lab.next_to(icon, DOWN, buff=0.16)))
    members.arrange(RIGHT, buff=slot - icon_h * 0.9, aligned_edge=DOWN)
    width = max(members.width + 1.0, 5.0)
    box = RoundedRectangle(corner_radius=0.24, width=width, height=icon_h + 1.5,
                           fill_color=CARD, fill_opacity=1, stroke_color=INK, stroke_width=3)
    members.move_to(box.get_center() + DOWN * 0.2)
    lab = T("computer memory", 20, SOFT).next_to(box.get_corner(UP + LEFT), DOWN + RIGHT,
                                                buff=0.18)
    count = T(f"{len(keys)} models", 25, INK, weight="BOLD")
    count.next_to(box.get_corner(UP + RIGHT), DOWN + LEFT, buff=0.16)
    g = VGroup(*shadowed(box), lab, count, members)
    g.members = members
    return g


# ------------------------------------------------------------------ layout pieces
def section_header(num, title: str, method: str | None, sub: str) -> VGroup:
    t = T(title, 30, INK, weight="BOLD")
    parts = VGroup()
    if num is not None:
        badge = Circle(radius=0.27, fill_color=INK, fill_opacity=1, stroke_width=0)
        parts.add(VGroup(badge, T(str(num), 22, CARD, font=ROUND).move_to(badge)))
    parts.add(t)
    if method:
        mt = T(method, 19, CARD, weight="BOLD")
        pill = RoundedRectangle(corner_radius=0.17, width=mt.width + 0.36, height=0.38,
                                fill_color=STUDENT, fill_opacity=1, stroke_width=0)
        parts.add(VGroup(pill, mt.move_to(pill)))
    parts.arrange(RIGHT, buff=0.24)
    parts.move_to([-6.85 + parts.width / 2, 3.45, 0])
    s = T(sub, 18, SOFT).next_to(t, DOWN, buff=0.14, aligned_edge=LEFT)
    return VGroup(parts, s)


def stepper(active: int | None) -> VGroup:
    pills = VGroup()
    for i, word in enumerate(["What", "Why", "How"]):
        on = i == active
        box = RoundedRectangle(corner_radius=0.18, width=1.02, height=0.4,
                               fill_color=INK if on else CARD, fill_opacity=1,
                               stroke_color=INK if on else FAINT, stroke_width=1.5)
        t = T(word, 18, CARD if on else SOFT, weight="BOLD")
        place_on_baseline(t, word, box.get_center()[1], 18, weight="BOLD")
        t.set_x(box.get_center()[0])
        pills.add(VGroup(box, t))
    pills.arrange(RIGHT, buff=0.1)
    return pills.move_to([6.85 - pills.width / 2, 3.45, 0])


def leash_curve(a, b, bend=0.35, color: str = LEASH, width: float = 5) -> VGroup:
    """A rope from the frozen original (a) to the student (b), bowed sideways by `bend`."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = b - a
    n = np.array([-d[1], d[0], 0]) / (np.linalg.norm(d) + 1e-9)
    rope = VMobject(stroke_color=color, stroke_width=width, cap_style=CapStyleType.ROUND)
    rope.set_points_smoothly([a, (a + b) / 2 + n * bend, b])
    ring = Circle(radius=0.075, stroke_color=color, stroke_width=4, fill_color=CARD,
                  fill_opacity=1).move_to(b)
    return VGroup(rope, ring)


def person(color: str = SOFT, fill: str = "#E2E8F0", s: float = 1.0) -> VGroup:
    head = Circle(radius=0.2, fill_color=fill, fill_opacity=1, stroke_color=color,
                  stroke_width=3).shift(UP * 0.27)
    arc = Arc(radius=0.36, start_angle=0, angle=PI)
    body = VMobject().set_points(arc.points)
    body.add_line_to(arc.points[0])
    body.set_fill(fill, 1).set_stroke(color, 3).shift(DOWN * 0.34)
    return VGroup(body, head).scale(s)


def dashed_tile(width: float = 1.3, height: float = 0.62) -> VGroup:
    from manim import DashedVMobject
    box = RoundedRectangle(corner_radius=0.14, width=width, height=height, stroke_color=SOFT,
                           stroke_width=2.5)
    return VGroup(DashedVMobject(box, num_dashes=28), T("?", 24, SOFT).move_to(box))


def prompt_card(text: str, size: float = 24) -> VGroup:
    badge = Circle(radius=0.23, fill_color=INK, fill_opacity=1, stroke_width=0)
    q = T("Q", 20, CARD, weight="BOLD").move_to(badge)
    t = T(text, size, INK)
    row = VGroup(VGroup(badge, q), t).arrange(RIGHT, buff=0.22)
    box = RoundedRectangle(corner_radius=0.2, width=row.width + 0.5, height=0.8,
                           fill_color=CARD, fill_opacity=1, stroke_color=FAINT, stroke_width=2)
    row.move_to(box)
    return VGroup(*shadowed(box), row)


def answer_card(label: str, text: str, good: bool, width: float = 8.2, size: float = 22
                ) -> VGroup:
    col = GOOD if good else BAD
    box = RoundedRectangle(corner_radius=0.16, width=width, height=0.78, fill_color=CARD,
                           fill_opacity=1, stroke_color=col, stroke_width=3)
    badge = Circle(radius=0.23, fill_color=col, fill_opacity=1, stroke_width=0)
    badge.move_to(box.get_left() + RIGHT * 0.45)
    lab = T(label, 20, CARD, weight="BOLD").move_to(badge)
    t = T(text, size, INK).next_to(badge, RIGHT, buff=0.26)
    icon = check_icon(GOOD, 0.3) if good else cross_icon(BAD, 0.22)
    icon.move_to(box.get_right() + LEFT * 0.45)
    return VGroup(*shadowed(box), badge, lab, t, icon)


def eye_panel(num: str, sharp: bool, size: float = 2.4) -> VGroup:
    frame = RoundedRectangle(corner_radius=0.24, width=size, height=size, fill_color=CARD,
                             fill_opacity=1, stroke_color=FAINT, stroke_width=2)
    base = T("E", 120, INK, font=ROUND).move_to(frame)
    if sharp:
        letter = base
    else:
        letter = VGroup(*[base.copy().shift(0.085 * np.array([np.cos(a), np.sin(a), 0]))
                          .set_fill(INK, 0.13) for a in np.linspace(0, 2 * PI, 12,
                                                                   endpoint=False)])
    n = T(num, 24, SOFT, weight="BOLD").move_to(frame.get_corner(UP + LEFT) + np.array(
        [0.32, -0.32, 0]))
    return VGroup(*shadowed(frame), letter, n)


def mini_tries(n: int = 4, w: float = 1.0) -> VGroup:
    return VGroup(*[RoundedRectangle(corner_radius=0.06, width=w, height=0.17,
                                     fill_color=STUDENT_FILL, fill_opacity=1,
                                     stroke_color=STUDENT, stroke_width=2) for _ in range(n)]
                  ).arrange(DOWN, buff=0.08)


def family_icon(i: int) -> VGroup:
    if i == 0:
        return VGroup(coach(0.5), student(0.5)).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    if i == 1:
        return VGroup(student(0.5), mini_tries()).arrange(RIGHT, buff=0.25)
    a = tile("A", GOOD, GOOD_FILL, size=16, height=0.34)
    b = tile("B", BAD, BAD_FILL, size=16, height=0.34)
    return balance(0.2, ORIGIN, arm=0.95, drop=0.5, left=a, right=b, post=0.8, lw=4.5)


def family_card(i: int, title: str, method: str, sub: str, extra: VMobject | None = None,
                width: float = 4.3, height: float = 4.3) -> VGroup:
    body = card(width, height)
    box = body[1]
    icon = family_icon(i)
    icon.scale_to_fit_height(1.3).move_to(box.get_top() + DOWN * 1.0)
    t = T(title, 24, INK, weight="BOLD").next_to(icon, DOWN, buff=0.36)
    s = T(sub, 18, SOFT).next_to(t, DOWN, buff=0.16)
    mt = T(method, 22, CARD, weight="BOLD")
    pill = RoundedRectangle(corner_radius=0.2, width=mt.width + 0.5, height=0.46,
                            fill_color=STUDENT, fill_opacity=1, stroke_width=0)
    chip = VGroup(pill, mt.move_to(pill)).next_to(s, DOWN, buff=0.3)
    g = VGroup(body, icon, t, s, chip)
    if extra is not None:
        g.add(extra.next_to(chip, DOWN, buff=0.28))
    return g


__all__ = [n for n in dir() if not n.startswith("_")] + ["ORIGIN"]
