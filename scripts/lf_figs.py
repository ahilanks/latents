"""Figures of the Latent Feedback diagnosis (2026-11-24, scripts/lf_diag.txt and the early-phase runs HE-*):
  state     how much the loop's source state differs between positions over training (log scale), their Transformer
            with / without their loop vs the H-Net with / without it, by bytes seen
  healthy   their loop attached to the finished (healthy) normal H-Net vs from the start of training: main-network
            output variation, against the same continuation without the loop
  bpb       their Transformer on our bytes: bits per byte with their loop (4 passes = exact decoding) vs without
usage: python scripts/lf_figs.py state|healthy|bpb
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

H_BYTES, LF_BYTES = 262144, 131072  # bytes per update: H-Net runs, their Transformer runs


def rows(run):
    return [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]


def series(runs, key, bps, max_step=None):
    pts = {}
    for r in runs:
        for x in rows(r):
            if key in x and (max_step is None or x["step"] <= max_step):
                pts[x["step"] * bps / 1e6] = x[key]
    xs = sorted(pts)
    return xs, [pts[k] for k in xs]


def fig_state():
    lines = [  # (label, runs, key, bytes/step, colour, style)
        ("their Transformer, no loop", ["LF-plain-s0"], "val/out_variation", LF_BYTES, fs.BLUE, "--"),
        ("their Transformer + their loop", ["LF-fb-s0"], "val/out_variation", LF_BYTES, fs.BLUE, "-"),
        ("normal H-Net", ["HE-plain-s0", "HP-plain-s0"], "val/main_variation", H_BYTES, fs.ORANGE, "--"),
        ("H-Net + their loop", ["HE-lf-s0", "HL-L-s0"], "val/main_variation", H_BYTES, fs.ORANGE, "-"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=0.08, right=0.68, top=0.80, bottom=0.11)
    ends = []
    for label, runs, key, bps, c, ls in lines:
        xs, ys = series(runs, key, bps)
        ys = [max(y, 1e-5) * 100 for y in ys]
        ax.plot(xs, ys, ls, color=c, lw=2.2, marker="o", ms=3.5)
        ends.append([xs[-1], ys[-1], label, c])
    ends.sort(key=lambda e: -e[1])
    last = None
    for e in ends:  # direct labels right of the plot at each line's last value, nudged apart (downwards) on the log axis
        y = e[1] if last is None else min(e[1], last / 2.4)
        ax.plot([e[0], 618], [e[1], y], ":", color=e[3], lw=0.9, clip_on=False)
        ax.text(625, y, f"{e[2]}: {e[1]:.2g}%", fontsize=9.5, color=fs.INK, va="center", clip_on=False)
        last = y
    ax.set_yscale("log")
    ax.set_ylim(0.005, 600)
    ax.set_xlim(-10, 605)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.grid(axis="y", color=fs.GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("web text seen (millions of bytes)", color=fs.INK2)
    ax.set_ylabel("how much the output differs between positions", color=fs.INK2)
    fig.text(0.02, 0.96, "In an H-Net the loop's source state turns into one shared vector early; their loop locks it in",
             fontsize=13.5, weight="bold", va="top")
    fig.text(0.02, 0.905, "Variation between positions of the state Latent Feedback feeds back (main network's output in the "
             "H-Net, top state in their\nTransformer). Their full recipe in both (H-Net: not end to end). Same 0.6B bytes, "
             "similar size (27M). Single seed each.", fontsize=10, color=fs.INK2, va="top")
    out = ROOT / "logs/figs/lf_state.png"
    fig.savefig(out, dpi=150)
    print(out)


def fig_healthy():
    last = lambda r: [x for x in rows(r) if "val/main_variation" in x][-1]
    names = ["normal H-Net,\n500 more updates", "+ their loop from the\nfinished normal H-Net", "+ their loop from the\nstart of training"]
    xs = [last("HC-plain-s0"), last("HC-lf-s0"), last("HL-L-s0")]
    vals = [100 * x["val/main_variation"] for x in xs]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.16)
    fs.axes_style(ax, 24, yticks=[0, 5, 10, 15, 20], ylabel="how much the main network's output\ndiffers between chunks")
    ax.set_xlim(-0.6, 2.6)
    ax.bar(range(3), vals, color=[fs.GREY, fs.BLUE, fs.ORANGE], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.1f}%" if v >= 0.1 else f"{v:.2f}%", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    fig.text(0.02, 0.96, "Their loop keeps a healthy H-Net healthy; it only goes flat from the start of training",
             fontsize=13, weight="bold", va="top")
    fig.text(0.02, 0.895, f"Latent Feedback's full recipe (not end to end). Bits per byte after 500 more updates: "
             f"{json.load(open(ROOT / 'runs/HC-lf-s0/summary.json'))['val/bpb']:.4f} with the loop vs "
             f"{json.load(open(ROOT / 'runs/HC-plain-s0/summary.json'))['val/bpb']:.4f} without.\n"
             "From the start: stopped at step 600, flat since step 100. Single seed each.", fontsize=10, color=fs.INK2,
             va="top")
    out = ROOT / "logs/figs/lf_healthy.png"
    fig.savefig(out, dpi=150)
    print(out)


def fig_bpb():
    fb = [x for x in rows("LF-fb-s0") if "val/bpb_p4" in x][-1]
    pl = [x for x in rows("LF-plain-s0") if "val/bpb_p1" in x][-1]
    names = ["their Transformer,\nno loop", "with their loop,\nloop off (1 pass)", "with their loop\n(4 passes = exact)"]
    vals = [pl["val/bpb_p1"], fb["val/bpb_p1"], fb["val/bpb_p4"]]
    cols = [fs.GREY, fs.BLUE, fs.BLUE]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.17)
    lo = min(vals) - 0.05
    hi = max(vals) + 0.03
    fs.axes_style(ax, hi, ylabel="bits per byte (lower is better)", pct=False)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.bar(range(len(vals)), [v - lo for v in vals], bottom=lo, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    d = vals[2] - vals[0]
    fig.text(0.02, 0.96, f"Their loop {'helps' if d < 0 else 'hurts'} their Transformer on our bytes: {d:+.3f} bits per byte",
             fontsize=13, weight="bold", va="top")
    fig.text(0.02, 0.895, "Validation web text, after 0.6B bytes. Their recipe, with their settings for their smallest\n"
             "model and data budget; 27M parameters. Single seed each.", fontsize=10, color=fs.INK2, va="top")
    out = ROOT / "logs/figs/lf_bpb.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    {"state": fig_state, "healthy": fig_healthy, "bpb": fig_bpb}[sys.argv[1]]()
