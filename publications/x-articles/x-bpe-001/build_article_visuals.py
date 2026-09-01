#!/usr/bin/env python3
"""Build the static X-BPE-001 visual package.

The figures follow visuals/animations/STYLE_GUIDE.md: white canvas, Arial for
English, Songti SC for Chinese, semantic color roles, and minimal explanatory
text. Image 03 reuses the approved still from the Manim BPE animation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "assets"
ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
SONGTI = Path("/System/Library/Fonts/Supplemental/Songti.ttc")

W, H = 1600, 900
COVER_W, COVER_H = 2000, 800
WHITE = "#FFFFFF"
FG = "#111827"
MUTED = "#64748B"
GRID = "#CBD5E1"
BASE = "#2563EB"
INTERMEDIATE = "#7C3AED"
COMPOSED = "#047857"
ACCENT = "#B45309"
SOFT = "#F8FAFC"


def font(size: int, *, bold: bool = False, cjk: bool = False) -> ImageFont.FreeTypeFont:
    # The approved system uses normal weight. Visual hierarchy comes from size
    # and color instead of switching to a bold font file.
    path = SONGTI if cjk else ARIAL
    return ImageFont.truetype(str(path), size=size, index=0)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), WHITE)
    return image, ImageDraw.Draw(image)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str,
                size: int, color: str = FG, *, bold: bool = False,
                cjk: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold=bold, cjk=cjk), fill=color,
              anchor="mm")


def text_left(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str,
              size: int, color: str = FG, *, bold: bool = False,
              cjk: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold=bold, cjk=cjk), fill=color,
              anchor="lm")


def header(draw: ImageDraw.ImageDraw, title: str, kicker: str) -> None:
    text_left(draw, (96, 62), kicker.upper(), 24, BASE, bold=True)
    text_left(draw, (96, 116), title, 48, FG, bold=True)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
            *, fill: str = SOFT, outline: str = GRID, width: int = 2,
            radius: int = 24) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          color: str = GRID, width: int = 5) -> None:
    draw.line((start, end), fill=color, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 10), (x - 16, y + 10)], fill=color)


def pill(draw: ImageDraw.ImageDraw, center: tuple[int, int], value: str,
         color: str, width: int = 180, *, cjk: bool = False) -> None:
    x, y = center
    rounded(draw, (x - width // 2, y - 38, x + width // 2, y + 38),
            fill=color, outline=color, radius=20)
    text_center(draw, center, value, 30, WHITE, cjk=cjk)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, format="PNG", optimize=True)


def fit_font(value: str, max_width: int, start_size: int) -> ImageFont.FreeTypeFont:
    """Fit one Arial line without changing its natural character spacing."""

    size = start_size
    while size > 16:
        candidate = font(size)
        if candidate.getlength(value) <= max_width:
            return candidate
        size -= 1
    return font(size)


def build_cover() -> None:
    image = Image.new("RGB", (COVER_W, COVER_H), WHITE)
    draw = ImageDraw.Draw(image)
    text_left(draw, (120, 76), "DONGXI · TOKENIZATION", 26, BASE)

    title = "WHAT IS A TOKEN, REALLY?"
    title_font = fit_font(title, max_width=1760, start_size=122)
    draw.text((1000, 238), title, font=title_font, fill=FG, anchor="mm")
    text_center(draw, (1000, 355), "From Unicode bytes to BPE and token IDs", 38, MUTED)

    # Abstract mechanism: base units are grouped into variable-width tokens,
    # then mapped to fixed integer addresses. No illustrative ID is presented as
    # an observed tokenizer result.
    y = 570
    unit_x = 270
    for index in range(9):
        x = unit_x + index * 36
        draw.rounded_rectangle(
            (x, y - 16, x + 28, y + 16), radius=8, fill=BASE
        )
    text_center(draw, (428, 650), "base units", 20, MUTED)

    arrow(draw, (620, y), (760, y), INTERMEDIATE, 4)

    groups = [(2, 90), (3, 124), (4, 158)]
    cursor = 820
    group_centers: list[int] = []
    for count, group_width in groups:
        x1 = cursor
        x2 = cursor + group_width
        draw.rounded_rectangle(
            (x1, y - 40, x2, y + 40),
            radius=20,
            fill=COMPOSED,
            outline=COMPOSED,
            width=2,
        )
        inner_width = 18
        gap = 8
        total = count * inner_width + (count - 1) * gap
        inner_x = (x1 + x2 - total) / 2
        for offset in range(count):
            left = inner_x + offset * (inner_width + gap)
            draw.rounded_rectangle(
                (left, y - 12, left + inner_width, y + 12),
                radius=6,
                fill=WHITE,
            )
        group_centers.append((x1 + x2) // 2)
        cursor = x2 + 22
    text_center(draw, (sum(group_centers) / 3, 650), "tokens", 20, MUTED)

    arrow(draw, (1270, y), (1430, y), ACCENT, 4)

    id_x = 1480
    for index in range(3):
        x = id_x + index * 84
        draw.rounded_rectangle(
            (x, y - 34, x + 62, y + 34),
            radius=14,
            fill="#FFF7ED",
            outline=ACCENT,
            width=3,
        )
        draw.ellipse((x + 25, y - 12, x + 37, y), fill=ACCENT)
        draw.rounded_rectangle(
            (x + 18, y + 9, x + 44, y + 15), radius=3, fill=ACCENT
        )
    text_center(draw, (1595, 650), "token IDs", 20, MUTED)
    save(image, "cover.png")


def build_units() -> None:
    image, draw = canvas()
    header(draw, "Six different units", "Name the unit before counting")
    cards = [
        ("ORTHOGRAPHIC WORD", "språkmodellen", BASE, False),
        ("GRAPHEME CLUSTER", "é", BASE, False),
        ("CODE POINT", "U+00E9", INTERMEDIATE, False),
        ("UTF-8 BYTES", "C3 A9", INTERMEDIATE, False),
        ("SUBWORD / BYTE TOKEN", "下一个", COMPOSED, True),
        ("TOKEN ID", "108725", ACCENT, False),
    ]
    left, top = 96, 210
    cw, ch, gapx, gapy = 448, 230, 30, 32
    for idx, (label, value, color, cjk) in enumerate(cards):
        row, col = divmod(idx, 3)
        x1 = left + col * (cw + gapx)
        y1 = top + row * (ch + gapy)
        rounded(draw, (x1, y1, x1 + cw, y1 + ch), fill=SOFT)
        text_center(draw, (x1 + cw / 2, y1 + 62), label, 22, MUTED, bold=True)
        text_center(draw, (x1 + cw / 2, y1 + 145), value, 42, color, cjk=cjk)
    text_center(draw, (800, 812), "Different units · no universal one-to-one mapping", 27, MUTED)
    save(image, "01-six-units.png")


def build_tiny_bpe() -> None:
    image, draw = canvas()
    header(draw, "BPE training is a measured loop", "Tiny executable corpus")
    text_left(draw, (102, 190), "hug × 5    hugs × 3    hugging × 2", 31, MUTED)

    rows = [
        ("01", ("h", "u"), "hu", "10", BASE),
        ("02", ("hu", "g"), "hug", "10", INTERMEDIATE),
        ("03", ("hug", "s"), "hugs", "3", COMPOSED),
    ]
    ys = [340, 510, 680]
    for (rank, pair, merged, count, color), y in zip(rows, ys):
        text_center(draw, (135, y), rank, 30, MUTED)
        pill(draw, (330, y), pair[0], color, width=max(150, 44 * len(pair[0]) + 76))
        text_center(draw, (465, y), "+", 36, MUTED)
        pill(draw, (600, y), pair[1], color, width=150)
        arrow(draw, (700, y), (850, y), color, 5)
        pill(draw, (1035, y), merged, color, width=max(190, 44 * len(merged) + 80))
        text_left(draw, (1240, y), f"count {count}", 30, MUTED)

    rounded(draw, (1110, 168, 1498, 244), fill="#FFF7ED", outline="#FED7AA", radius=20)
    text_center(draw, (1304, 206), "round 1: tied maximum", 25, ACCENT)
    save(image, "02-tiny-bpe-training.png")


def build_preprocessing_effects() -> None:
    image, draw = canvas()
    header(draw, "Two stages before model input", "Normalization and chat templates")
    rounded(draw, (96, 200, 770, 760), fill=SOFT)
    rounded(draw, (830, 200, 1504, 760), fill=SOFT)

    text_center(draw, (433, 260), "NORMALIZATION", 23, BASE, bold=True)
    text_center(draw, (433, 340), "NFC  café", 38, FG)
    text_center(draw, (433, 405), "4 code points · 5 bytes", 26, MUTED)
    text_center(draw, (433, 500), "NFD  café", 38, FG)
    text_center(draw, (433, 565), "5 code points · 6 bytes", 26, MUTED)
    arrow(draw, (326, 650), (540, 650), INTERMEDIATE, 5)
    text_center(draw, (433, 704), "same IDs · decoded NFC", 27, INTERMEDIATE)

    text_center(draw, (1167, 260), "CHAT TEMPLATE", 23, COMPOSED, bold=True)
    pill(draw, (1040, 410), "Hello", BASE, width=220)
    text_center(draw, (1040, 480), "1 token", 27, MUTED)
    arrow(draw, (1160, 410), (1285, 410), COMPOSED, 5)
    rounded(draw, (1310, 330, 1450, 490), fill=COMPOSED, outline=COMPOSED, radius=24)
    for yy in (365, 410, 455):
        draw.rounded_rectangle((1340, yy - 9, 1420, yy + 9), radius=9, fill=WHITE)
    text_center(draw, (1167, 590), "1 visible message", 30, FG)
    text_center(draw, (1167, 650), "9 model positions", 34, COMPOSED, bold=True)
    text_center(draw, (800, 825), "Normalization can rewrite text · chat templates add control tokens", 27, MUTED)
    save(image, "04-interface-surprises.png")


def build_multilingual() -> None:
    image, draw = canvas()
    header(draw, "One tokenizer configuration · three segmentations", "Pinned Qwen3 worked example")
    rows = [
        ("Chinese", 9, BASE),
        ("English", 11, INTERMEDIATE),
        ("Swedish", 20, ACCENT),
    ]
    for idx, (name, count, color) in enumerate(rows):
        y = 280 + idx * 135
        text_left(draw, (105, y), name, 34, FG)
        for dot in range(count):
            x = 330 + dot * 48
            draw.rounded_rectangle((x, y - 18, x + 34, y + 18), radius=10, fill=color)
        text_left(draw, (1335, y), f"{count} tokens", 31, color, bold=True)

    rounded(draw, (160, 680, 690, 790), fill="#EFF6FF", outline="#BFDBFE", radius=24)
    text_center(draw, (335, 735), "下一个", 36, BASE, cjk=True)
    arrow(draw, (430, 735), (505, 735), BASE, 4)
    text_left(draw, (535, 735), "one token", 31, BASE)
    rounded(draw, (850, 680, 1440, 790), fill="#FFF7ED", outline="#FED7AA", radius=24)
    text_center(draw, (1145, 735), "spr · å · k · mod · ellen", 31, ACCENT)
    text_center(draw, (800, 845), "A measured example · not a ranking of languages", 27, MUTED)
    save(image, "05-multilingual-measurement.png")


def main() -> None:
    build_cover()
    build_units()
    build_tiny_bpe()
    build_preprocessing_effects()
    build_multilingual()
    source = ROOT / "visuals/animations/rendered/bpe-byte-merges-manim-still.png"
    target = OUT / "03-byte-coverage-compression.png"
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    files = [
        "cover.png",
        "01-six-units.png",
        "02-tiny-bpe-training.png",
        "03-byte-coverage-compression.png",
        "04-interface-surprises.png",
        "05-multilingual-measurement.png",
    ]
    metadata = {
        "visual_system": "visuals/animations/STYLE_GUIDE.md",
        "fonts": {"english": "Arial Regular", "chinese": "Songti SC Regular"},
        "colors": {
            "background": WHITE,
            "foreground": FG,
            "base": BASE,
            "intermediate": INTERMEDIATE,
            "composed": COMPOSED,
            "accent": ACCENT,
        },
        "known_limitations": [
            "The Chinese merge ladder is illustrative, not Qwen training history.",
            "The multilingual counts describe three fixed strings under one pinned tokenizer.",
            "The cover is separate from the five inline-image positions.",
        ],
        "files": {},
    }
    for name in files:
        path = OUT / name
        with Image.open(path) as built:
            dimensions = list(built.size)
        metadata["files"][name] = {
            "dimensions": dimensions,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote visual package to {OUT}")


if __name__ == "__main__":
    main()
