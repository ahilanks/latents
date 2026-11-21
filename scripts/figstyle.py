"""Shared look of the result figures (light surface): colours, axes, columns with a rounded top and the value on the cap.
Colours are the validated categorical order blue, orange, aqua plus a grey for reference bars; text is always ink."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b3ab"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def axes_style(ax, ymax, yticks=None, ylabel=None, pct=True):
    ax.set_ylim(0, ymax)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:g}%" if pct else f"{t:g}" for t in yticks])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="both", length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)


def columns(ax, xs, hs, color, width=0.34, labels=None, fmt=None, label_color=INK, fontsize=10, weight="bold"):
    """Columns that grow from the baseline with a rounded top; the value sits on the cap. Call after axes_style and
    after the x-limits are set (the rounding is computed in pixels)."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px_x, px_y = bb.width / (x1 - x0), bb.height / (y1 - y0)
    r = 4.0 * fig.dpi / 100 / px_x  # ~4 px corner radius, in x data units
    for i, (x, h) in enumerate(zip(xs, hs)):
        c = color[i] if isinstance(color, (list, tuple)) else color
        hh = max(h, 0.0)
        if hh * px_y > 2 * r * px_x:
            p = FancyBboxPatch((x - width / 2, -y1), width, hh + y1, boxstyle=f"round,pad=0,rounding_size={r}",
                               mutation_aspect=px_x / px_y, fc=c, ec="none", clip_on=True, zorder=3)
            ax.add_patch(p)
        else:
            ax.bar([x], [hh], width=width, color=c, zorder=3)
        txt = labels[i] if labels is not None else (fmt.format(h) if fmt else num(h))
        ax.text(x, hh + (y1 - y0) * 0.015, txt, ha="center", va="bottom", color=label_color, fontsize=fontsize,
                fontweight=weight, zorder=4)


def num(v, unit="%"):
    """98.5 -> "98.5%", 29.0 -> "29%" (no rounding of halves to a wrong whole number)."""
    return (f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}") + unit


def titles(fig, title, subtitle=None, x=0.012, size=12.5):
    """Bold takeaway title and a secondary line under it, both wrapped to the figure's width. Returns the share of
    the figure's height they take (use it for subplots_adjust(top=1 - that - a margin))."""
    import textwrap
    w, h = fig.get_figwidth(), fig.get_figheight()
    t = textwrap.fill(title, int(w * 104 / size))
    y = 1 - 0.07 / h
    fig.text(x, y, t, ha="left", va="top", fontsize=size, fontweight="bold", color=INK, linespacing=1.25)
    used = 0.07 / h + (t.count("\n") + 1) * size * 1.3 / 72 / h
    if subtitle:
        sub = textwrap.fill(subtitle, int(w * 13.4))
        fig.text(x, 1 - used - 0.04 / h, sub, ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.3)
        used += 0.04 / h + (sub.count("\n") + 1) * 9.5 * 1.35 / 72 / h
    return used


def legend(ax, items, **kw):
    """items: [(colour, label)]; a legend with square swatches and ink text."""
    from matplotlib.patches import Patch
    return ax.legend(handles=[Patch(fc=c, ec="none", label=l) for c, l in items], frameon=False, fontsize=9.5,
                     labelcolor=INK2, handlelength=1.0, handleheight=1.0, **kw)
