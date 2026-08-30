"""Animate byte-level BPE from UTF-8 bytes to a Chinese phrase token."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


BACKGROUND = "#0B1020"
FOREGROUND = "#F4F7FF"
MUTED = "#AAB6D3"
BYTE_COLOR = "#3B82F6"
CHAR_COLOR = "#8B5CF6"
WORD_COLOR = "#10B981"
ACCENT = "#F59E0B"

STATES = [
    {
        "tokens": [("E6", BYTE_COLOR), ("95", BYTE_COLOR), ("B0", BYTE_COLOR)],
        "headline": "One visible character, three UTF-8 bytes",
        "rule": "数  →  E6 95 B0",
        "level": 0,
    },
    {
        "tokens": [("E6 95", BYTE_COLOR), ("B0", BYTE_COLOR)],
        "headline": "BPE replays a learned byte-pair merge",
        "rule": "E6 + 95  →  E6 95",
        "level": 0,
    },
    {
        "tokens": [("数", CHAR_COLOR)],
        "headline": "Another learned merge completes the character",
        "rule": "E6 95 + B0  →  数",
        "level": 1,
    },
    {
        "tokens": [("数", CHAR_COLOR), ("据", CHAR_COLOR), ("库", CHAR_COLOR)],
        "headline": "Frequent neighboring characters can merge too",
        "rule": "Start:  数 | 据 | 库",
        "level": 1,
    },
    {
        "tokens": [("数据", WORD_COLOR), ("库", CHAR_COLOR)],
        "headline": "A frequent pair becomes a reusable token",
        "rule": "数 + 据  →  数据",
        "level": 2,
    },
    {
        "tokens": [("数据库", WORD_COLOR)],
        "headline": "A later merge can create the complete expression",
        "rule": "数据 + 库  →  数据库",
        "level": 2,
    },
]

HOLD_FRAMES = 15
FADE_FRAMES = 7
FPS = 12


def parse_args() -> argparse.Namespace:
    default = Path(__file__).parent / "rendered" / "bpe-byte-merges.gif"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default)
    return parser.parse_args()


def cjk_font() -> FontProperties:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return FontProperties(fname=candidate)
    return FontProperties(family="sans-serif")


def draw_tokens(ax: plt.Axes, tokens: list[tuple[str, str]], alpha: float, font: FontProperties) -> None:
    widths = [max(1.15, 0.52 * len(label) + 0.72) for label, _ in tokens]
    gap = 0.25
    total = sum(widths) + gap * (len(widths) - 1)
    x = 5.0 - total / 2

    for (label, color), width in zip(tokens, widths, strict=True):
        patch = FancyBboxPatch(
            (x, 3.55),
            width,
            1.12,
            boxstyle="round,pad=0.04,rounding_size=0.13",
            facecolor=color,
            edgecolor=FOREGROUND,
            linewidth=1.2,
            alpha=0.90 * alpha,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            4.11,
            label,
            ha="center",
            va="center",
            color=FOREGROUND,
            fontsize=20,
            fontweight="bold",
            fontproperties=font,
            alpha=alpha,
        )
        x += width + gap


def draw_state(ax: plt.Axes, index: int, alpha: float, font: FontProperties) -> None:
    state = STATES[index]
    draw_tokens(ax, state["tokens"], alpha, font)
    ax.text(
        5,
        3.02,
        state["headline"],
        ha="center",
        color=FOREGROUND,
        fontsize=15,
        fontproperties=font,
        alpha=alpha,
    )
    ax.text(
        5,
        2.38,
        state["rule"],
        ha="center",
        color=ACCENT,
        fontsize=16,
        fontweight="bold",
        fontproperties=font,
        alpha=alpha,
    )


def build_animation(output: Path) -> None:
    font = cjk_font()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    frames_per_state = HOLD_FRAMES + FADE_FRAMES
    frame_count = len(STATES) * frames_per_state

    def update(frame: int):
        ax.clear()
        ax.set_facecolor(BACKGROUND)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        state_index = min(frame // frames_per_state, len(STATES) - 1)
        within = frame % frames_per_state

        ax.text(
            5,
            5.56,
            "BYTE-LEVEL BPE: COVERAGE → COMPRESSION",
            ha="center",
            color=FOREGROUND,
            fontsize=17,
            fontweight="bold",
        )

        if within < HOLD_FRAMES or state_index == len(STATES) - 1:
            draw_state(ax, state_index, 1.0, font)
        else:
            progress = (within - HOLD_FRAMES + 1) / FADE_FRAMES
            draw_state(ax, state_index, 1.0 - progress, font)
            draw_state(ax, state_index + 1, progress, font)

        level_labels = ["bytes", "characters", "words / phrases"]
        for i, label in enumerate(level_labels):
            x = 2.3 + i * 2.7
            active = i <= STATES[state_index]["level"]
            ax.scatter(
                [x],
                [1.18],
                s=115,
                color=WORD_COLOR if active else MUTED,
                alpha=1.0 if active else 0.35,
                zorder=3,
            )
            ax.text(x, 0.76, label, ha="center", color=FOREGROUND, fontsize=11)
            if i < 2:
                ax.plot([x + 0.18, x + 2.52], [1.18, 1.18], color=MUTED, alpha=0.35)

        ax.text(
            5,
            0.20,
            "The base bytes guarantee coverage; learned merges reduce token count.",
            ha="center",
            color=MUTED,
            fontsize=11,
        )
        return []

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / FPS, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=FPS), dpi=120)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    build_animation(parse_args().output)
