"""Schematic: where the latents sit in the two set-ups (Coconut thoughts; hidden-state feedback).
usage: python scripts/diagram_latents.py   ->  logs/figs/how_latents_are_separated.png"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

LIGHT, BLUE_L = "#eceae4", "#d6e6fa"


def box(ax, x, y, w, h, text="", fc=LIGHT, ec="none", size=8.5, color=fs.INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec,
                                lw=1.0, zorder=2))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight, zorder=3)


def arrow(ax, a, b, color=fs.MUTED, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def coconut(ax):
    cols = ["Q: a=17;", " b=a+25;", " c=b*3;", " c=?", "<start>", None, None, "<stop>", "c=42", "*3=26.", "#### 26"]
    ax.set_xlim(-1.75, len(cols) - 0.4)
    ax.set_ylim(-1.05, 5.95)
    ax.axis("off")
    for y, name in ((0, "bytes"), (1, "byte encoder"), (2.05, "inputs"), (3, "main network"), (3.95, "outputs"), (5, "byte decoder")):
        ax.text(-1.7, y, name, ha="left", va="center", fontsize=9, color=fs.INK2)
    box(ax, 2.0, 1, 5.0, 0.5, "reads the question's bytes", fc=LIGHT, color=fs.INK2)
    box(ax, 8.5, 1, 4.0, 0.5, "reads the bytes written so far", fc=LIGHT, color=fs.INK2)
    ax.text(5.5, 1, "idle", ha="center", va="center", fontsize=8.5, color=fs.MUTED, style="italic")
    box(ax, 5.0, 3, 11.0, 0.55, "", fc=LIGHT)
    ax.text(1.5, 3, "one sequence: every position can look back at all earlier ones", ha="center", va="center", fontsize=8.5,
            color=fs.INK2, zorder=3)
    box(ax, 8.5, 5, 4.0, 0.5, "writes the next bytes", fc=LIGHT, color=fs.INK2)
    for i, c in enumerate(cols):
        if c is None:  # a thought: no bytes underneath
            ax.text(i, 0, "no bytes", ha="center", va="center", fontsize=8, color=fs.MUTED, style="italic")
            box(ax, i, 2.05, 0.74, 0.46, f"thought {i - 4}", fc=fs.BLUE, color="#ffffff", size=8, weight="bold")
            box(ax, i, 3.95, 0.74, 0.46, "", fc=BLUE_L)
        else:
            ax.text(i, 0, c, ha="center", va="center", fontsize=8.5, color=fs.INK, family="DejaVu Sans Mono")
            box(ax, i, 2.05, 0.74, 0.46, "chunk", fc="#ffffff", ec=fs.BASE, size=8, color=fs.INK2)
            box(ax, i, 3.95, 0.74, 0.46, "", fc=BLUE_L if i == 4 else "#ffffff", ec="none" if i == 4 else fs.BASE)
            arrow(ax, (i, 1.27), (i, 1.8))
        arrow(ax, (i, 2.3), (i, 2.71))
        arrow(ax, (i, 3.29), (i, 3.7))
        if c is not None and i >= 7:
            arrow(ax, (i, 4.2), (i, 4.73))
    # the feedback of thoughts: output at one position becomes the next position's input
    for i in (4, 5):
        arrow(ax, (i + 0.2, 3.72), (i + 0.8, 2.3), color=fs.BLUE, rad=-0.0, lw=1.8)
    ax.text(5.0, 4.45, "the output is fed straight back as the\nnext input; the decoder never reads it", ha="center", va="bottom",
            fontsize=8.5, color=fs.INK, linespacing=1.25)
    ax.plot([3.5, 3.5], [-0.95, 5.6], color=fs.BASE, lw=1.0, zorder=1)
    ax.text(3.45, 5.62, "question", ha="right", va="bottom", fontsize=9, color=fs.MUTED)
    ax.text(3.55, 5.62, "thinking, then the written answer", ha="left", va="bottom", fontsize=9, color=fs.MUTED)
    ax.text(3.55, -0.72, "cut-off variant: the byte encoder and decoder restart here,\nso only the main network connects the two sides",
            ha="left", va="center", fontsize=8.5, color=fs.INK2, linespacing=1.25)


def feedback(ax):
    cols = ["Q: a=17;", " b=a+25;", " c=b*3;", " c=?", "c=42", "*3=26.", "#### 26"]
    ax.set_xlim(-1.75, len(cols) - 0.4 + 4.0)
    ax.set_ylim(-0.4, 6.1)
    ax.axis("off")
    rows = ((0, "bytes"), (1, "byte encoder"), (2, "main network,\nfirst pass"), (3.05, "its outputs"),
            (4.1, "main network,\nsecond pass"), (5.2, "byte decoder"))
    for y, name in rows:
        ax.text(-1.7, y, name, ha="left", va="center", fontsize=9, color=fs.INK2, linespacing=1.15)
    n = len(cols)
    box(ax, (n - 1) / 2, 1, n, 0.5, "reads all bytes", fc=LIGHT, color=fs.INK2)
    box(ax, (n - 1) / 2, 2, n, 0.5, "", fc=LIGHT)
    box(ax, (n - 1) / 2, 4.1, n, 0.5, "", fc=LIGHT)
    box(ax, (n - 1) / 2, 5.2, n, 0.5, "writes the next bytes", fc=LIGHT, color=fs.INK2)
    for i, c in enumerate(cols):
        ax.text(i, 0, c, ha="center", va="center", fontsize=8.5, color=fs.INK, family="DejaVu Sans Mono")
        arrow(ax, (i, 0.3), (i, 0.74))
        arrow(ax, (i, 1.26), (i, 1.74))
        box(ax, i, 3.05, 0.7, 0.42, "", fc=fs.BLUE)
        arrow(ax, (i, 2.26), (i, 2.83))
        arrow(ax, (i - 0.22, 1.26), (i - 0.22, 3.84), color=fs.BASE, lw=1.0)  # the chunk input also goes to the second pass
        if i + 1 < n:
            arrow(ax, (i + 0.2, 3.27), (i + 0.82, 3.84), color=fs.BLUE, lw=1.8)
        arrow(ax, (i, 4.36), (i, 4.94))
    ax.text(n - 0.1, 3.05, "the output at one chunk is added\nto the next chunk's input", ha="left", va="center", fontsize=8.5,
            color=fs.INK, linespacing=1.25)
    ax.text(n - 0.1, 0.9, "no extra positions and no markers:\nthe latent rides along at every chunk,\nfor all text\n\n"
            "thin grey lines: each chunk's own input,\nas in the plain H-Net", ha="left", va="center", fontsize=8.5,
            color=fs.INK2, linespacing=1.25)


def main():
    fig = plt.figure(figsize=(11.6, 9.6), dpi=200)
    used = fs.titles(fig, "Where the latents sit in the two set-ups",
                     "Blue = a main-network output that is handed on as an input. Grey = the byte-level modules and the normal chunk path.")
    top = 1 - used - 0.035
    h1 = 0.44
    fig.text(0.012, top, "Coconut: thoughts are positions that exist only in the main network", fontsize=11, fontweight="bold",
             color=fs.INK, va="top")
    ax1 = fig.add_axes([0.012, top - 0.03 - h1, 0.976, h1])
    coconut(ax1)
    y2 = top - 0.03 - h1 - 0.035
    fig.text(0.012, y2, "Hidden-state feedback (your version): the previous chunk's output is added to every chunk's input",
             fontsize=11, fontweight="bold", color=fs.INK, va="top")
    ax2 = fig.add_axes([0.012, 0.01, 0.976, y2 - 0.045])
    feedback(ax2)
    out = ROOT / "logs" / "figs" / "how_latents_are_separated.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
