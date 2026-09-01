#!/usr/bin/env python3
"""Build localized Chinese visuals for the X-BPE-001 article package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def font(size: int, *, cjk: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(SONGTI if cjk else ARIAL), size=size, index=0)


def canvas(width: int = W, height: int = H) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    return image, ImageDraw.Draw(image)


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    color: str = FG,
    *,
    cjk: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, cjk=cjk), fill=color, anchor="mm")


def text_left(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    color: str = FG,
    *,
    cjk: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, cjk=cjk), fill=color, anchor="lm")


def mixed_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    spans: list[tuple[str, bool, str]],
    size: int,
) -> None:
    widths = [font(size, cjk=cjk).getlength(value) for value, cjk, _ in spans]
    x = xy[0] - sum(widths) / 2
    for (value, cjk, color), width in zip(spans, widths):
        draw.text((x, xy[1]), value, font=font(size, cjk=cjk), fill=color, anchor="lm")
        x += width


def mixed_left(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    spans: list[tuple[str, bool, str]],
    size: int,
) -> None:
    x = xy[0]
    for value, cjk, color in spans:
        selected = font(size, cjk=cjk)
        draw.text((x, xy[1]), value, font=selected, fill=color, anchor="lm")
        x += selected.getlength(value)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = SOFT,
    outline: str = GRID,
    width: int = 2,
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = GRID,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    x, y = end
    draw.polygon([(x, y), (x - 16, y - 10), (x - 16, y + 10)], fill=color)


def token(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    value: str,
    color: str,
    width: int,
    *,
    cjk: bool = False,
) -> None:
    x, y = center
    rounded(
        draw,
        (x - width // 2, y - 38, x + width // 2, y + 38),
        fill=color,
        outline=color,
        radius=20,
    )
    text_center(draw, center, value, 30, WHITE, cjk=cjk)


def header(draw: ImageDraw.ImageDraw, title: list[tuple[str, bool, str]], kicker: str) -> None:
    text_left(draw, (96, 62), kicker, 24, BASE, cjk=True)
    mixed_left(draw, (96, 122), title, 48)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, format="PNG", optimize=True)


def fit_mixed_size(spans: list[tuple[str, bool, str]], max_width: int, start: int) -> int:
    size = start
    while size > 16:
        width = sum(font(size, cjk=cjk).getlength(value) for value, cjk, _ in spans)
        if width <= max_width:
            return size
        size -= 1
    return size


def abstract_token_flow(draw: ImageDraw.ImageDraw, *, y: int) -> None:
    unit_x = 270
    for index in range(9):
        x = unit_x + index * 36
        draw.rounded_rectangle((x, y - 16, x + 28, y + 16), radius=8, fill=BASE)
    text_center(draw, (428, y + 80), "基础单元", 20, MUTED, cjk=True)
    arrow(draw, (620, y), (760, y), INTERMEDIATE, 4)

    groups = [(2, 90), (3, 124), (4, 158)]
    cursor = 820
    centers: list[int] = []
    for count, group_width in groups:
        x1, x2 = cursor, cursor + group_width
        rounded(draw, (x1, y - 40, x2, y + 40), fill=COMPOSED, outline=COMPOSED, radius=20)
        inner_width, gap = 18, 8
        total = count * inner_width + (count - 1) * gap
        inner_x = (x1 + x2 - total) / 2
        for offset in range(count):
            left = inner_x + offset * (inner_width + gap)
            draw.rounded_rectangle((left, y - 12, left + inner_width, y + 12), radius=6, fill=WHITE)
        centers.append((x1 + x2) // 2)
        cursor = x2 + 22
    mixed_center(draw, (sum(centers) / 3, y + 80), [("Token", False, MUTED)], 20)

    arrow(draw, (1270, y), (1430, y), ACCENT, 4)
    id_x = 1480
    for index in range(3):
        x = id_x + index * 84
        rounded(draw, (x, y - 34, x + 62, y + 34), fill="#FFF7ED", outline=ACCENT, width=3, radius=14)
        draw.ellipse((x + 25, y - 12, x + 37, y), fill=ACCENT)
        draw.rounded_rectangle((x + 18, y + 9, x + 44, y + 15), radius=3, fill=ACCENT)
    mixed_center(draw, (1595, y + 80), [("Token ID", False, MUTED)], 20)


def build_cover() -> None:
    image, draw = canvas(COVER_W, COVER_H)
    text_left(draw, (120, 76), "DONGXI · TOKENIZATION", 26, BASE)
    title = [("TOKEN ", False, FG), ("到底是什么？", True, FG)]
    title_size = fit_mixed_size(title, 1760, 122)
    mixed_center(draw, (1000, 238), title, title_size)
    mixed_center(
        draw,
        (1000, 355),
        [("从 ", True, MUTED), ("Unicode ", False, MUTED), ("字节到 ", True, MUTED),
         ("BPE ", False, MUTED), ("与 ", True, MUTED), ("Token ID", False, MUTED)],
        38,
    )
    abstract_token_flow(draw, y=570)
    save(image, "cover.png")


def build_units() -> None:
    image, draw = canvas()
    header(draw, [("六种不同单位", True, FG)], "计数之前，先说清单位")
    cards = [
        ([("正字法词", True, MUTED)], "språkmodellen", False, BASE),
        ([("字素簇", True, MUTED)], "é", False, BASE),
        ([("Unicode ", False, MUTED), ("码点", True, MUTED)], "U+00E9", False, INTERMEDIATE),
        ([("UTF-8 ", False, MUTED), ("字节", True, MUTED)], "C3 A9", False, INTERMEDIATE),
        ([("子词 / 字节 ", True, MUTED), ("Token", False, MUTED)], "下一个", True, COMPOSED),
        ([("Token ID", False, MUTED)], "108725", False, ACCENT),
    ]
    left, top = 96, 210
    cw, ch, gapx, gapy = 448, 230, 30, 32
    for index, (label, value, cjk, color) in enumerate(cards):
        row, col = divmod(index, 3)
        x1, y1 = left + col * (cw + gapx), top + row * (ch + gapy)
        rounded(draw, (x1, y1, x1 + cw, y1 + ch))
        mixed_center(draw, (x1 + cw / 2, y1 + 62), label, 22)
        text_center(draw, (x1 + cw / 2, y1 + 145), value, 42, color, cjk=cjk)
    text_center(draw, (800, 812), "六种单位不同 · 没有普遍的一一对应", 27, MUTED, cjk=True)
    save(image, "01-six-units.png")


def build_tiny_bpe() -> None:
    image, draw = canvas()
    header(draw, [("BPE ", False, FG), ("训练是一轮轮计数与合并", True, FG)], "微型可执行语料")
    text_left(draw, (102, 190), "hug × 5    hugs × 3    hugging × 2", 31, MUTED)
    rows = [
        ("01", ("h", "u"), "hu", "10", BASE),
        ("02", ("hu", "g"), "hug", "10", INTERMEDIATE),
        ("03", ("hug", "s"), "hugs", "3", COMPOSED),
    ]
    for (rank, pair, merged, count, color), y in zip(rows, (340, 510, 680)):
        text_center(draw, (135, y), rank, 30, MUTED)
        token(draw, (330, y), pair[0], color, max(150, 44 * len(pair[0]) + 76))
        text_center(draw, (465, y), "+", 36, MUTED)
        token(draw, (600, y), pair[1], color, 150)
        arrow(draw, (700, y), (850, y), color, 5)
        token(draw, (1035, y), merged, color, max(190, 44 * len(merged) + 80))
        text_left(draw, (1240, y), f"计数 {count}", 30, MUTED, cjk=True)
    rounded(draw, (1110, 168, 1498, 244), fill="#FFF7ED", outline="#FED7AA", radius=20)
    text_center(draw, (1304, 206), "第 1 轮：最高频并列", 25, ACCENT, cjk=True)
    save(image, "02-tiny-bpe-training.png")


def build_coverage() -> None:
    image, draw = canvas()
    header(draw, [("字节保证覆盖，合并提升压缩", True, FG)], "BYTE-LEVEL BPE")
    text_center(draw, (800, 205), "E6 95 B0", 34, MUTED)

    token(draw, (290, 440), "E6", BASE, 145)
    token(draw, (470, 440), "95", BASE, 145)
    token(draw, (650, 440), "B0", BASE, 145)
    arrow(draw, (745, 440), (875, 440), INTERMEDIATE, 5)
    token(draw, (980, 440), "数", COMPOSED, 180, cjk=True)
    arrow(draw, (1080, 440), (1190, 440), COMPOSED, 5)
    token(draw, (1330, 440), "数据库", COMPOSED, 300, cjk=True)

    text_center(draw, (470, 555), "基础字节", 24, BASE, cjk=True)
    mixed_center(draw, (980, 555), [("多字节 ", True, INTERMEDIATE), ("Token", False, INTERMEDIATE)], 24)
    mixed_center(draw, (1330, 555), [("多字符 ", True, COMPOSED), ("Token", False, COMPOSED)], 24)
    rounded(draw, (290, 690, 1310, 778), fill=SOFT, outline=GRID, radius=22)
    text_center(draw, (800, 734), "能编码 · 压得短 · 能理解，是三项不同结论", 28, MUTED, cjk=True)
    save(image, "03-byte-coverage-compression.png")


def build_preprocessing_effects() -> None:
    image, draw = canvas()
    header(draw, [("模型输入前的两个阶段", True, FG)], "规范化与 CHAT TEMPLATE")
    rounded(draw, (96, 200, 770, 760))
    rounded(draw, (830, 200, 1504, 760))

    text_center(draw, (433, 260), "规范化", 25, BASE, cjk=True)
    mixed_center(draw, (433, 340), [("NFC  café", False, FG)], 38)
    text_center(draw, (433, 405), "4 个码点 · 5 个字节", 26, MUTED, cjk=True)
    mixed_center(draw, (433, 500), [("NFD  café", False, FG)], 38)
    text_center(draw, (433, 565), "5 个码点 · 6 个字节", 26, MUTED, cjk=True)
    arrow(draw, (326, 650), (540, 650), INTERMEDIATE, 5)
    mixed_center(draw, (433, 704), [("ID ", False, INTERMEDIATE), ("相同 · 解码为 ", True, INTERMEDIATE), ("NFC", False, INTERMEDIATE)], 27)

    mixed_center(draw, (1167, 260), [("CHAT TEMPLATE", False, COMPOSED)], 25)
    token(draw, (1040, 410), "Hello", BASE, 220)
    mixed_center(draw, (1040, 480), [("1 ", False, MUTED), ("个 ", True, MUTED), ("Token", False, MUTED)], 27)
    arrow(draw, (1160, 410), (1285, 410), COMPOSED, 5)
    rounded(draw, (1310, 330, 1450, 490), fill=COMPOSED, outline=COMPOSED, radius=24)
    for yy in (365, 410, 455):
        draw.rounded_rectangle((1340, yy - 9, 1420, yy + 9), radius=9, fill=WHITE)
    text_center(draw, (1167, 590), "1 条可见消息", 30, FG, cjk=True)
    text_center(draw, (1167, 650), "9 个模型位置", 34, COMPOSED, cjk=True)
    mixed_center(draw, (800, 825), [("规范化可改写文本 · ", True, MUTED), ("Chat Template ", False, MUTED), ("会加入控制 ", True, MUTED), ("Token", False, MUTED)], 27)
    save(image, "04-interface-surprises.png")


def build_multilingual() -> None:
    image, draw = canvas()
    header(draw, [("同一 ", True, FG), ("Tokenizer ", False, FG), ("配置，三种切分", True, FG)], "固定 QWEN3 的实测示例")
    rows = [("中文", 9, BASE), ("英文", 11, INTERMEDIATE), ("瑞典语", 20, ACCENT)]
    for index, (name, count, color) in enumerate(rows):
        y = 280 + index * 135
        text_left(draw, (105, y), name, 34, FG, cjk=True)
        for dot in range(count):
            x = 330 + dot * 48
            draw.rounded_rectangle((x, y - 18, x + 34, y + 18), radius=10, fill=color)
        mixed_left(draw, (1335, y), [(str(count), False, color), (" 个 ", True, color), ("Token", False, color)], 31)

    rounded(draw, (160, 680, 690, 790), fill="#EFF6FF", outline="#BFDBFE", radius=24)
    text_center(draw, (335, 735), "下一个", 36, BASE, cjk=True)
    arrow(draw, (430, 735), (505, 735), BASE, 4)
    mixed_left(draw, (535, 735), [("一个 ", True, BASE), ("Token", False, BASE)], 31)
    rounded(draw, (850, 680, 1440, 790), fill="#FFF7ED", outline="#FED7AA", radius=24)
    text_center(draw, (1145, 735), "spr · å · k · mod · ellen", 31, ACCENT)
    text_center(draw, (800, 845), "一组实测示例 · 无法代表语言排名", 27, MUTED, cjk=True)
    save(image, "05-multilingual-measurement.png")


def main() -> None:
    build_cover()
    build_units()
    build_tiny_bpe()
    build_coverage()
    build_preprocessing_effects()
    build_multilingual()

    files = [
        "cover.png",
        "01-six-units.png",
        "02-tiny-bpe-training.png",
        "03-byte-coverage-compression.png",
        "04-interface-surprises.png",
        "05-multilingual-measurement.png",
    ]
    metadata = {
        "language": "zh-CN",
        "cover_rule": "5:2",
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
            "中文合并阶梯用于解释机制，没有重建 Qwen Tokenizer 的训练历史。",
            "多语言计数只描述三个固定字符串与一个固定 Tokenizer 修订版本。",
            "封面独立于五张行内图。",
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
    print(f"wrote Chinese visual package to {OUT}")


if __name__ == "__main__":
    main()
