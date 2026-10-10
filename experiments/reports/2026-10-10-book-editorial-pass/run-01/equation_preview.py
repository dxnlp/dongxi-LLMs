"""Local editorial proof of four shared equations; no measured plot or animation.

Run in the isolated course environment. GitHub Markdown syntax is checked
separately by check_book_math.py; this MathText proof does not claim a browser run.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [
    ("Stored vocabulary head: [V,D]; row state: [D]",
     r"$z_t=h_tW_{\mathrm{out}}^\top$"),
    ("Query/source scores: [n,n]; head width: d_h",
     r"$A=\mathrm{softmax}_{\mathrm{row}}(QK^\top/\sqrt{d_h}+M)$"),
    ("Verifier reward; response advantage; population spread",
     r"$\hat A_i=(R_i-\bar R)/(s+\epsilon_{\mathrm{adv}})$"),
    ("Frozen teacher; temperature tau; student-logit derivative",
     r"$\partial L/\partial z_i=\tau(p_i^{(\tau)}-q_i^{(\tau)})$"),
]
fig, axes = plt.subplots(4, 1, figsize=(12, 7.5))
for ax, (guide, equation) in zip(axes, rows):
    ax.axis("off")
    ax.text(.02, .84, guide, fontsize=12, color="#385269", transform=ax.transAxes)
    ax.text(.02, .3, equation, fontsize=22, transform=ax.transAxes)
fig.suptitle("Editorial notation proof — schematic equations, no model measurement", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(Path(__file__).with_name("equation-preview.png"), dpi=120)
plt.close(fig)
