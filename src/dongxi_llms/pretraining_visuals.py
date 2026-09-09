"""Data-backed Chapter 6 figures; schematic maps explicitly labeled."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREEN = "#2563eb", "#d97706", "#16806a"


def canvas(rows=1, cols=1, size=(10, 4)):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    return plt.subplots(rows, cols, figsize=size, layout="constrained", squeeze=False)


def process_map(active):
    fig, axes = canvas(size=(12, 4))
    ax = axes[0, 0]
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")
    labels = ["Documents\nsplit + provenance", "Token windows\nIDs / labels [B,T]",
              "Modern decoder\nlogits [B,T,V]", "Token loss\nsum / valid count",
              "Backward + AdamW\naccumulate → clip", "Validate + save\nmodel AND run state"]
    for i, label in enumerate(labels):
        x, y = (i % 3)*4+.1, 2.35 if i < 3 else .55
        selected = i in active
        ax.add_patch(FancyBboxPatch((x, y), 3.5, 1.05, boxstyle="round,pad=.12",
                                   facecolor="#e8f0ff" if selected else "#f3f4f6",
                                   edgecolor=BLUE if selected else "#a1a1aa"))
        ax.text(x+1.75, y+.525, label, ha="center", va="center")
        if i not in (2, 5):
            ax.annotate("", (x+3.9, y+.525), (x+3.55, y+.525), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (.4, 1.72), (10.5, 2.15), arrowprops={"arrowstyle": "->", "connectionstyle": "angle,angleA=180,angleB=90,rad=10"})
    ax.set_title("Training system schematic • left to right, then next row • not a timing diagram")
    return fig


def window_map(windows, indices=(0, 1, 2)):
    x, y = windows["x"][list(indices)].numpy(), windows["y"][list(indices)].numpy()
    fig, axes = canvas(2, 1, (12, 5))
    for ax, values, title in zip(axes[:, 0], (x, y), ("Input IDs: every row is one independent window", "Target IDs: shifted once; × means ignored padding")):
        ax.imshow(values != -100, cmap="Blues", vmin=0, vmax=2, aspect="auto")
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, "×" if value == -100 else str(value), ha="center", va="center", fontsize=9)
        ax.set_yticks(range(len(indices)), [windows["documents"][i] for i in indices])
        ax.set_xticks(range(x.shape[1]))
        ax.set(title=title, xlabel="Position in window (not character index)")
    return fig


def accumulation_plot(audit):
    fig, axes = canvas(1, 2, (10, 4))
    counts = np.array(audit["counts"])
    axes[0, 0].bar([0, 1], counts/counts.sum(), color=BLUE, label="Correct token weighting")
    axes[0, 0].axhline(.5, color=ORANGE, linestyle="--", label="Wrong equal batch weighting")
    axes[0, 0].set(xticks=[0, 1], xticklabels=[f"{c} targets" for c in counts], ylim=(0, 1), ylabel="Microbatch contribution", title="A short tail must not count as a full window")
    axes[0, 0].legend(fontsize=9)
    errors = [audit["correct_error"], audit["wrong_error"]]
    axes[0, 1].bar(["Correct", "Mean of means"], errors, color=[BLUE, ORANGE])
    axes[0, 1].set(ylabel="Maximum absolute gradient difference", title="Measured against one full-batch backward")
    for i, e in enumerate(errors):
        axes[0, 1].text(i, e, f"{e:.3g}", ha="center", va="bottom")
    axes[0, 1].margins(y=.2)
    return fig


def schedule_plot(rates):
    fig, axes = canvas(size=(10, 3.5))
    ax = axes[0, 0]
    ax.plot(range(1, len(rates)+1), rates, "o-", color=BLUE)
    ax.axvspan(.5, 3.5, alpha=.12, color=ORANGE, label="3 warmup updates")
    ax.set(xlabel="Optimizer update (one-based)", ylabel="Learning rate", title="One schedule tick per parameter update, not per microbatch")
    ax.legend()
    return fig


def optimizer_plot(audit):
    fig, axes = canvas(1, 2, (11, 4))
    axes[0, 0].bar(range(16), audit["gradient"].numpy(), color=BLUE)
    axes[0, 0].set(title="Measured decoder embedding gradient", xlabel="First 16 parameter coordinates", ylabel="d loss / d parameter")
    ax = axes[0, 1]
    ax.plot(audit["sgd_delta"].numpy(), "o-", label="SGD, η = .01", color=BLUE)
    ax.plot(audit["adamw_delta"].numpy(), "s-", label="AdamW, η = .01", color=ORANGE)
    ax.set(title="Same gradient; different update rule", xlabel="Same coordinates", ylabel="Parameter change")
    ax.legend()
    return fig


def stability_plot(clip):
    fig, axes = canvas(1, 2, (10, 4))
    axes[0, 0].bar(["Injected gradient", "After clipping"], [clip["before"], clip["after"]], color=[ORANGE, BLUE])
    axes[0, 0].set(ylabel="Global L2 norm", title="Clipping the accumulated gradient")
    axes[0, 1].bar(["Weights", "Gradients", "Adam moments"], [4, 4, 8], color=[BLUE, ORANGE, GREEN])
    axes[0, 1].set(ylabel="Bytes per unique parameter", title="FP32 persistent-state ledger (estimate)", ylim=(0, 10))
    axes[0, 1].text(1, 9, "Not total memory: activations/workspaces omitted", ha="center", fontsize=9)
    return fig


def validation_plot(history):
    fig, axes = canvas(size=(10, 4))
    ax = axes[0, 0]
    ax.plot([r["step"] for r in history], [r["loss"] for r in history], ".-", color=BLUE, label="Current training update (before step)")
    ax.plot([r["step"] for r in history], [r["validation"] for r in history], "o-", color=ORANGE, label="Fixed held-out split (after step)")
    ax.set(xlabel="Completed optimizer updates", ylabel="Mean NLL (nats / byte-or-EOS target)", title="Different samples and measurement times: do not overread the gap")
    ax.legend(fontsize=9)
    return fig


def recovery_plot(audit):
    names = ["complete", "no_optimizer", "no_cursor"]
    fig, axes = canvas(size=(10, 4))
    ax = axes[0, 0]
    errors = [audit[n]["parameter_error"] for n in names]
    ax.bar(["Complete state", "Adam moments missing", "Data cursor missing"], errors, color=[GREEN, ORANGE, ORANGE])
    for i, error in enumerate(errors):
        ax.text(i, error, f"{error:.3g}", ha="center", va="bottom")
    ax.set(ylabel="Maximum |resumed − uninterrupted| parameter", title="Same saved weights do not imply the same continuation")
    ax.margins(y=.2)
    return fig
