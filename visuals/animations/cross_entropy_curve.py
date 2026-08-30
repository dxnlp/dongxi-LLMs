"""Animate the negative-log penalty for the correct next token."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


BACKGROUND = "#0B1020"
FOREGROUND = "#F4F7FF"
MUTED = "#AAB6D3"
CORRECT = "#10B981"
OTHER = "#3B82F6"
CURVE = "#F59E0B"
GRID = "#334155"
FPS = 18
FRAME_COUNT = 144


def parse_args() -> argparse.Namespace:
    default = Path(__file__).parent / "rendered" / "cross-entropy-curve.gif"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default)
    return parser.parse_args()


def probability_at(frame: int) -> float:
    phase = frame / (FRAME_COUNT - 1)
    return 0.05 + 0.90 * (0.5 - 0.5 * np.cos(2 * np.pi * phase))


def build_animation(output: Path) -> None:
    fig = plt.figure(figsize=(8, 4.5), dpi=120, facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, left=0.09, right=0.96, bottom=0.17, top=0.75, wspace=0.34)
    bars_ax = fig.add_subplot(grid[0, 0])
    curve_ax = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "CROSS-ENTROPY: HOW CONFIDENCE CHANGES THE PENALTY",
        x=0.5,
        y=0.94,
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
    )
    formula_text = fig.text(
        0.5,
        0.83,
        "",
        ha="center",
        color=CURVE,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.055,
        "More probability on the correct token → smaller loss",
        ha="center",
        color=MUTED,
        fontsize=11,
    )

    x_curve = np.linspace(0.02, 1.0, 400)
    y_curve = -np.log(x_curve)

    def style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor(BACKGROUND)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.7)

    style_axis(bars_ax)
    style_axis(curve_ax)

    initial_p = probability_at(0)
    initial_probabilities = [initial_p, (1 - initial_p) * 0.62, (1 - initial_p) * 0.38]
    bars = bars_ax.bar(
        ["correct", "other A", "other B"],
        initial_probabilities,
        color=[CORRECT, OTHER, OTHER],
        width=0.68,
    )
    bars_ax.set_ylim(0, 1.05)
    bars_ax.set_ylabel("predicted probability", color=FOREGROUND, fontsize=10)
    bars_ax.set_title("Next-token distribution", color=FOREGROUND, fontsize=12, pad=10)
    bar_labels = [
        bars_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.2f}",
            ha="center",
            color=FOREGROUND,
            fontsize=10,
            fontweight="bold",
        )
        for bar, value in zip(bars, initial_probabilities, strict=True)
    ]

    initial_loss = -np.log(initial_p)
    curve_ax.plot(x_curve, y_curve, color=CURVE, linewidth=2.6)
    marker = curve_ax.scatter(
        [initial_p],
        [initial_loss],
        s=95,
        color=CORRECT,
        edgecolor=FOREGROUND,
        zorder=4,
    )
    (guide,) = curve_ax.plot(
        [initial_p, initial_p],
        [0, initial_loss],
        color=CORRECT,
        linestyle="--",
        alpha=0.65,
    )
    curve_ax.set_xlim(0, 1.0)
    curve_ax.set_ylim(0, 4.05)
    curve_ax.set_xlabel("p(correct token)", color=FOREGROUND, fontsize=10)
    curve_ax.set_ylabel("loss", color=FOREGROUND, fontsize=10)
    curve_ax.set_title("Negative log-likelihood", color=FOREGROUND, fontsize=12, pad=10)
    loss_label = curve_ax.text(
        min(initial_p + 0.05, 0.72),
        min(initial_loss + 0.25, 3.75),
        f"loss = {initial_loss:.2f}",
        color=FOREGROUND,
        fontsize=10,
        fontweight="bold",
    )

    def update(frame: int):
        p_correct = probability_at(frame)
        loss = -np.log(p_correct)
        probabilities = [p_correct, (1 - p_correct) * 0.62, (1 - p_correct) * 0.38]

        for bar, label, value in zip(bars, bar_labels, probabilities, strict=True):
            bar.set_height(value)
            label.set_y(value + 0.035)
            label.set_text(f"{value:.2f}")

        marker.set_offsets(np.array([[p_correct, loss]]))
        guide.set_data([p_correct, p_correct], [0, loss])
        loss_label.set_position((min(p_correct + 0.05, 0.72), min(loss + 0.25, 3.75)))
        loss_label.set_text(f"loss = {loss:.2f}")

        formula_text.set_text(
            rf"$L = -\log p_{{correct}} = -\log({p_correct:.2f}) = {loss:.2f}$"
        )
        return [*bars, *bar_labels, marker, guide, loss_label, formula_text]

    animation = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=1000 / FPS, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=FPS), dpi=120)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    build_animation(parse_args().output)
