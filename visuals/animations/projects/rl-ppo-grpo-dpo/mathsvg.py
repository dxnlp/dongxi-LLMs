"""Vector equations without a LaTeX -> SVG toolchain.

Matplotlib's mathtext typesets a TeX subset in Computer Modern and exports glyph
outlines as SVG paths, which Manim imports as ordinary vector shapes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["svg.hashsalt"] = "rl-ppo-grpo-dpo"

CACHE = Path(__file__).resolve().parent / "media" / "math"


def math_svg(expr: str, size: float = 48) -> Path:
    """Return the path of an SVG containing the typeset expression."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{size}|{expr}".encode()).hexdigest()[:16]
    out = CACHE / f"{key}.svg"
    if not out.exists():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${expr}$", fontsize=size, color="black")
        fig.savefig(out, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        # The figure background is an invisible path; once recolored it would
        # become an opaque box, so drop it and keep only glyphs and rules.
        svg = re.sub(r'<g id="patch_\d+">.*?</g>', "", out.read_text(), flags=re.S)
        out.write_text(svg)
    return out
