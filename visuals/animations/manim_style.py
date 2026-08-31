"""Reusable visual primitives for Dongxi Manim course animations."""

from __future__ import annotations

from manim import RoundedRectangle, Text, VGroup, config


BACKGROUND = "#FFFFFF"
FOREGROUND = "#111827"
MUTED = "#64748B"
GRID = "#CBD5E1"
BASE = "#2563EB"
INTERMEDIATE = "#7C3AED"
COMPOSED = "#047857"
ACCENT = "#B45309"
TOKEN_TEXT = "#FFFFFF"

LATIN_FONT = "Arial"
CJK_FONT = "Songti SC"

config.background_color = BACKGROUND


def label(text: str, size: int, color: str = FOREGROUND, *, cjk: bool = False) -> Text:
    """Create naturally kerned, normal-weight English or Chinese text."""

    return Text(
        text,
        font=CJK_FONT if cjk else LATIN_FONT,
        font_size=size,
        color=color,
        weight="NORMAL",
    )


def token(text: str, color: str, *, width: float | None = None) -> VGroup:
    """Create a large, phone-legible token with a stable color identity."""

    token_width = width or max(1.45, 0.44 * len(text) + 0.78)
    box = RoundedRectangle(
        corner_radius=0.16,
        height=1.12,
        width=token_width,
        stroke_color=GRID,
        stroke_width=2.2,
        fill_color=color,
        fill_opacity=0.94,
    )
    text_label = label(
        text,
        38,
        TOKEN_TEXT,
        cjk=any(ord(char) > 127 for char in text),
    )
    text_label.move_to(box)
    return VGroup(box, text_label)


def centered_positions(
    objects: list[VGroup], y: float = 0.25, buff: float = 0.34
) -> list[list[float]]:
    """Return a centered row from measured object widths without moving it."""

    total_width = sum(item.width for item in objects) + buff * (len(objects) - 1)
    cursor = -total_width / 2
    positions = []
    for item in objects:
        positions.append([cursor + item.width / 2, y, 0.0])
        cursor += item.width + buff
    return positions


def pill(text: str, color: str) -> VGroup:
    """Create a compact outlined structural label."""

    text_label = label(text, 22, color)
    box = RoundedRectangle(
        corner_radius=0.2,
        width=text_label.width + 0.48,
        height=0.52,
        stroke_color=color,
        stroke_width=1.5,
        fill_color=color,
        fill_opacity=0.08,
    )
    text_label.move_to(box)
    return VGroup(box, text_label)
