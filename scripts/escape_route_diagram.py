"""Diagram: why Latent Feedback's Transformer cannot give up on its fed-back state but an H-Net can.
Left, their model: the top state that is fed back is also the only route to the prediction. Right, the H-Net: the
decoder also reads the encoder's byte states directly, so the main network (and the loop through it) can be dropped
for a small loss. Writes docs/escape_route.png."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import plt, SURFACE, INK, INK2, MUTED, BLUE, ORANGE, GREY  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.05, wspace=0.08)


def box(ax, x, y, w, h, text, color=GREY, fill="#f1f0ec", lw=1.6, size=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=color, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, weight=weight)


def arrow(ax, a, b, color=INK2, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


for ax in axes:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

# Left: Latent Feedback's Transformer
ax = axes[0]
ax.text(5, 9.6, "Latent Feedback's Transformer", ha="center", fontsize=12.5, weight="bold")
box(ax, 4.2, 1.4, 3.6, 1.0, "token")
box(ax, 4.2, 4.6, 3.6, 2.6, "the whole network", color=BLUE, fill="#e6effa", lw=2.2, weight="bold")
box(ax, 4.2, 8.0, 3.6, 1.0, "prediction")
arrow(ax, (4.2, 1.9), (4.2, 3.3))
arrow(ax, (4.2, 5.9), (4.2, 7.5))
ax.text(4.45, 6.7, "top state", fontsize=9.5, color=INK2, va="center")
# loop: top state back to the next position's input
arrow(ax, (6.0, 6.7), (6.0, 2.6), color=ORANGE, lw=2.2, rad=-0.9)
ax.text(8.75, 4.6, "fed back\nto the next\nposition", fontsize=9.5, color=ORANGE, ha="center", va="center")
ax.text(4.2, 0.25, "One route to the prediction: if the fed-back state went\nflat, the loss would blow up, so it can't.",
        ha="center", va="center", fontsize=9.5, color=INK2)

# Right: H-Net with the loop
ax = axes[1]
ax.text(5, 9.6, "H-Net with the loop", ha="center", fontsize=12.5, weight="bold")
X = 4.6
box(ax, X, 1.4, 3.0, 1.0, "bytes")
box(ax, X, 3.0, 3.0, 1.0, "encoder")
box(ax, X, 5.0, 3.0, 1.4, "main network", color=BLUE, fill="#e6effa", lw=2.2, weight="bold")
box(ax, X, 7.0, 3.0, 1.0, "decoder")
box(ax, X, 8.6, 3.0, 0.8, "prediction")
arrow(ax, (X, 1.9), (X, 2.5))
arrow(ax, (X, 3.5), (X, 4.3))
arrow(ax, (X, 5.7), (X, 6.5))
arrow(ax, (X, 7.5), (X, 8.2))
# loop around the main network
arrow(ax, (X + 1.5, 5.4), (X + 1.5, 4.6), color=ORANGE, lw=2.2, rad=-1.6)
ax.text(X + 2.45, 5.0, "fed back to\nthe next\nchunk", fontsize=9.5, color=ORANGE, ha="left", va="center")
# escape route: encoder straight to decoder, outside the main network
arrow(ax, (X - 1.5, 3.0), (X - 1.5, 7.0), color=INK, lw=2.4, rad=-0.35, ls="--")
ax.text(1.35, 5.0, "byte\nstates\nskip the\nmain\nnetwork", fontsize=9.5, color=INK, ha="right", va="center")
ax.text(5, 0.25, "Two routes: the decoder can drop the main network and the loop,\nlosing only ~0.01–0.02 bits per byte.",
        ha="center", va="center", fontsize=9.5, color=INK2)

fig.text(0.02, 0.95, "Their model has no way around the fed-back state; an H-Net does",
         fontsize=15, weight="bold", ha="left", va="top")
fig.text(0.02, 0.895, "Same loop, different escape: the H-Net decoder also reads the byte states directly, so going "
         "flat is cheap.\nTheir code has no extra anti-collapse measure beyond what we already tested.",
         fontsize=10.5, color=INK2, ha="left", va="top")
out = Path(__file__).resolve().parent.parent / "docs" / "escape_route.png"
fig.savefig(out, dpi=150)
print(out)
