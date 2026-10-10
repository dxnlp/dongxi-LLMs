"""Local MathText proof of representative foundations equations.

Adapted from the course's existing editorial equation-preview source. Run in
this goal's isolated CPU environment. This is a local readability proof, not
GitHub/browser rendering, a measured model plot or produced animation.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [
    ("Prediction: normalized target q; gradient over vocabulary logits [V]",
     r"$\partial L/\partial z_i=p_i-q_i$"),
    ("Reward: exact finite-response ascent; fixed R; J is expected reward",
     r"$\partial J/\partial z_i=p_i(R_i-J)$"),
    ("Baseline: pre-action b is fixed in the actor derivative",
     r"$\mathbb{E}[(R-b)\nabla\log\pi]=\nabla J$"),
    ("PPO: the minimum preserves sign-dependent incentives",
     r"$\min(\rho A,\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A)$"),
    ("Entropy and mismatch: normalized target q, model p; nats",
     r"$H(q,p)=H(q)+D_{\mathrm{KL}}(q\Vert p)$"),
    ("AdamW: decoupled shrinkage and corrected moment update",
     r"$\theta_t=(1-\eta_t\lambda_{\mathrm{wd}})\theta_{t-1}-\eta_t\widehat m_t/(\sqrt{\widehat v_t}+\epsilon_{\mathrm{Adam}})$"),
    ("RoPE: one column pair; c=cos(m omega), s=sin(m omega)",
     r"$R_{m\omega}u=(cu_1-su_2,\;su_1+cu_2)^\top$"),
]
fig, axes = plt.subplots(len(rows), 1, figsize=(12, 12))
for ax, (guide, equation) in zip(axes, rows):
    ax.axis("off")
    ax.text(.02, .84, guide, fontsize=12, color="#385269", transform=ax.transAxes)
    ax.text(.02, .28, equation, fontsize=20, transform=ax.transAxes)
fig.suptitle("Foundations equation proof — local typesetting of illustrative mechanisms", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(Path(__file__).with_name("equation-preview.png"), dpi=130)
plt.close(fig)
