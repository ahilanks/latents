"""Summary figures of the 2026-11-20 session: one figure per finding, the main number readable at a glance.
Reads the logged results (runs/*/metrics.jsonl, summary.json, coconut_eval.json, logs/nextlat/probe.json,
logs/geometry/geometry.json, logs/coconut/exchange_*.json) and writes logs/figs/*.png.
usage: python scripts/summary_figs.py [coconut] [exchange] [nextlat] [geometry] [carry] [thoughts]      (default: the first four)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "logs" / "figs"


def last_eval(run, stage):
    rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
    ev = [r for r in rows if "eval" in r and r["stage"] == stage]
    tr = [r for r in rows if "loss" in r and r["stage"] == stage]
    c = ev[-1]["eval"]["this_stage"]["carried"]
    return 100 * c["units"], 100 * c["tens"], ev[-1]["step"] - tr[0]["step"]


def coconut():
    """Ones vs tens digit of the hidden value in the three settings."""
    a = last_eval("coco-c1q", 2)
    b = last_eval("coco-c1q-long", 3)
    b = (b[0], b[1], b[2] + 3000)  # continued from 3000 steps of the same stage (coco-c1q-rest)
    c = last_eval("coco-fresh2", 2)  # the finished 8000-step stage (its continuation coco-fresh2-long is still training)
    groups = [("one hidden calculation", a), ("two hidden calculations", b),
              ("one hidden calculation,\nbyte-level modules cut off\nfrom the question", c)]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    used = fs.titles(fig, "Hidden calculations: ones digit learned, tens digit mostly not",
                     "Coconut on the byte H-Net. Share of 200 test problems where that digit of the number the model "
                     "needs from its thoughts is right. Single seed.")
    fig.subplots_adjust(left=0.08, right=0.985, top=1 - used - 0.115, bottom=0.215)
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, 2.85)
    fs.columns(ax, [i - 0.155 for i in range(3)], [g[1][0] for g in groups], fs.BLUE, width=0.22)
    fs.columns(ax, [i + 0.155 for i in range(3)], [g[1][1] for g in groups], fs.ORANGE, width=0.22)
    ax.axhline(10, color=fs.MUTED, lw=0.9, zorder=2)
    ax.text(2.83, 11.5, "guessing: 10%", ha="right", va="bottom", fontsize=8.5, color=fs.MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5, color=fs.INK, linespacing=1.25)
    for i, g in enumerate(groups):
        ax.text(i, -0.265, f"after {g[1][2]:,} training steps", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color=fs.MUTED)
    fs.legend(ax, [(fs.BLUE, "ones digit of the hidden value right"), (fs.ORANGE, "tens digit right")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.03), ncol=2, columnspacing=2.0)
    fig.savefig(OUT / "coconut_digits.png")
    print("wrote", OUT / "coconut_digits.png", [(g[0].replace("\n", " "), [round(v, 1) for v in g[1]]) for g in groups])


def exchange():
    d = json.load(open(ROOT / "logs/coconut/exchange_coco-rt-stage2.json"))
    vals = [100 * d["own_state_own_value"], 100 * d["other_state_own_value"], 100 * d["other_state_other_value"]]
    names = ["its own state:\nwrites its own value", "another problem's state:\nwrites its own value",
             "another problem's state:\nwrites that problem's value"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    used = fs.titles(fig, "The state read after thinking does not carry the value alone",
                     "Coconut, one hidden calculation. Only the state the decoder reads after thinking is exchanged "
                     "between two problems; share of 512 problems where the number then written is exactly right. "
                     "Single seed.")
    fig.subplots_adjust(left=0.085, right=0.985, top=1 - used - 0.05, bottom=0.15)
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, 2.6)
    fs.columns(ax, range(3), vals, [fs.BLUE, fs.BLUE, fs.BLUE], width=0.22)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=9.5, color=fs.INK, linespacing=1.25)
    fig.savefig(OUT / "coconut_exchange.png")
    print("wrote", OUT / "coconut_exchange.png", vals)


def nextlat():
    probe = json.load(open(ROOT / "logs/nextlat/probe.json"))
    models = [("normal\nH-Net", "side-p06-plain-s0", "Cc-p06", fs.GREY),
              ("NextLat:\nbyte-space\nterm only", "NL-klonly-s0", "NLc-p06kl", fs.ORANGE),
              ("NextLat:\nlatent-space\nloss", "NL-h1-s0", "NLc-p06nl", fs.BLUE)]
    bpb = [json.load(open(ROOT / f"runs/{m[1]}/summary.json"))["val/bpb"] for m in models]
    var = [100 * probe[f"runs/{m[1]}"]["variation_share"] for m in models]
    acc = [100 * json.load(open(ROOT / f"runs/{m[2]}/coconut_eval.json"))["written"]["acc"] for m in models]
    panels = [("Web compression\nbits per byte, lower is better", bpb, 1.75, [0, 0.5, 1.0, 1.5], "{:.3f}", False),
              ("How much the main network's\noutput varies, % of its size", var, 13.2, [0, 4, 8, 12], "{:.1f}%", True),
              ("Written reasoning\n% of 2,000 test problems solved", acc, 112, [0, 25, 50, 75, 100], None, True)]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.7), dpi=200)
    used = fs.titles(fig, "NextLat on the H-Net: compression unchanged, main network flattened, reasoning worse",
                     "0.6B-byte models, one seed each. The latent-space loss (the paper's main term) is met by making the "
                     "main network's outputs nearly constant.")
    fig.subplots_adjust(left=0.045, right=0.99, top=1 - used - 0.17, bottom=0.17, wspace=0.2)
    for ax, (ttl, vals, ymax, ticks, fmt, pct) in zip(axes, panels):
        fs.axes_style(ax, ymax, yticks=ticks, pct=pct)
        ax.set_xlim(-0.6, 2.6)
        fs.columns(ax, range(3), vals, [m[3] for m in models], width=0.26, fmt=fmt)
        ax.set_xticks(range(3))
        ax.set_xticklabels([m[0] for m in models], fontsize=9, color=fs.INK, linespacing=1.2)
        ax.set_title(ttl, fontsize=10, color=fs.INK2, loc="left", pad=10, linespacing=1.3)
    fig.savefig(OUT / "nextlat.png")
    print("wrote", OUT / "nextlat.png", [round(v, 3) for v in bpb], [round(v, 2) for v in var], [round(v, 1) for v in acc])


def geometry():
    g = json.load(open(ROOT / "logs/geometry/geometry.json"))
    models = [("word cuts,\nlatent only\n(seed 0)", "word cuts s0"), ("word cuts,\nlatent only\n(seed 1)", "word cuts s1"),
              ("fixed 4-byte\nchunks,\nlatent only", "fixed 4-byte"), ("joint: bytes\nalso train\nthe trunk", "joint")]
    r2 = [100 * g[k]["web"]["r2_head"] for _, k in models]
    top1 = [100 * g[k]["web"]["next_word_top1"] for _, k in models]
    freq = [100 * g[k]["web"]["most_frequent_top1"] for _, k in models]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), dpi=200)
    used = fs.titles(fig, "Latent-only word cuts: little of the target is explained, yet the next word is known",
                     "3B-byte models on web text, no training involved. Fixed chunks and byte-trained targets are far "
                     "more predictable.")
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - used - 0.17, bottom=0.19, wspace=0.14)
    names = [m[0] for m in models]
    ax = axes[0]
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, 3.6)
    fs.columns(ax, range(4), r2, fs.BLUE, width=0.26, fmt="{:.0f}%")
    ax.set_title("How much of the next chunk's latent\nthe prediction explains", fontsize=10,
                 color=fs.INK2, loc="left", pad=10, linespacing=1.3)
    ax = axes[1]
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, 3.6)
    fs.columns(ax, [i - 0.16 for i in range(4)], top1, fs.BLUE, width=0.24, fmt="{:.0f}%")
    fs.columns(ax, [i + 0.16 for i in range(4)], freq, fs.GREY, width=0.24, fmt="{:.0f}%", weight="normal", label_color=fs.INK2)
    ax.set_title("How often the true next word is ranked first\n(among the 300 most frequent words)",
                 fontsize=10, color=fs.INK2, loc="left", pad=10, linespacing=1.3)
    fs.legend(ax, [(fs.BLUE, "the model's predicted latent"), (fs.GREY, "always guessing the most frequent word")],
              loc="upper right", bbox_to_anchor=(1.0, 1.02))
    for ax in axes:
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=8.8, color=fs.INK, linespacing=1.2)
    fig.savefig(OUT / "geometry.png")
    print("wrote", OUT / "geometry.png", [round(v, 1) for v in r2], [round(v, 1) for v in top1], [round(v, 1) for v in freq])


def thoughts():
    """Do the thoughts matter? Hidden value right when the trained model is given 0, 1, 2 (as trained) or 3 thoughts."""
    d = json.load(open(ROOT / "logs/coconut/thought_count.json"))
    series = [("byte-level modules cut off from the question", "runs/coco-fresh2-in-long-stage2", fs.BLUE),
              ("as designed (byte-level modules also see the question)", "runs/coco-c1q", fs.ORANGE)]
    ns = ["0", "1", "2", "3"]
    fig, ax = plt.subplots(figsize=(8.0, 4.9), dpi=200)
    used = fs.titles(fig, "The model uses its thoughts: without them the hidden value is almost never right",
                     "Coconut, one hidden calculation, trained with two thoughts. Share of 400 test problems where the "
                     "number needed from the thoughts is exactly right. Single seed.")
    fig.subplots_adjust(left=0.08, right=0.985, top=1 - used - 0.14, bottom=0.14)
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, 3.6)
    for (name, run, col), off in zip(series, (-0.155, 0.155)):
        fs.columns(ax, [i + off for i in range(4)], [100 * d[run][n]["exact"] for n in ns], col, width=0.22, fmt="{:.0f}%")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["no thoughts", "1 thought", "2 thoughts\n(as trained)", "3 thoughts"], fontsize=9.5, color=fs.INK)
    fs.legend(ax, [(c, n) for n, _, c in series], loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=1)
    fig.savefig(OUT / "coconut_thoughts.png")
    print("wrote", OUT / "coconut_thoughts.png")


def stage_evals(runs, stage):
    """(steps trained on this stage, ones %, tens %) over a chain of runs that continue one another."""
    xs, ones, tens, off = [], [], [], 0
    for run in runs:
        rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
        tr = [r["step"] for r in rows if "loss" in r and r["stage"] == stage]
        for r in rows:
            if "eval" in r and r["stage"] == stage:
                c = r["eval"]["this_stage"]["carried"]
                xs.append(off + r["step"] - tr[0])
                ones.append(100 * c["units"])
                tens.append(100 * c["tens"])
        off += max(tr) - tr[0] + 1
    return xs, ones, tens


def carry():
    """Tens digit of the first hidden value over training, byte-level modules cut off: the two thought formats."""
    series = [("thought = only the part of the output the decoder reads", ["coco-fresh2-in", "coco-fresh2-in-long"], fs.BLUE),
              ("thought = the main network's whole output", ["coco-fresh2", "coco-fresh2-long"], fs.ORANGE)]
    fig, ax = plt.subplots(figsize=(8.0, 4.9), dpi=200)
    used = fs.titles(fig, "Without the byte-level shortcut the hidden value is learned, slowly, and only with the narrower thought",
                     "Coconut, one hidden calculation, byte-level modules cut off from the question. Share of 200 test "
                     "problems with the tens digit of the hidden value right. Single seed.")
    fig.subplots_adjust(left=0.08, right=0.9, top=1 - used - 0.12, bottom=0.12)
    fs.axes_style(ax, 105, yticks=[0, 25, 50, 75, 100])
    xmax = 0
    for name, runs, col in series:
        xs, ones, tens = stage_evals(runs, 2)
        ax.plot(xs, tens, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot(xs[-1:], tens[-1:], "o", ms=7, color=col, mec=fs.SURFACE, mew=1.5, zorder=4)
        ax.annotate(fs.num(tens[-1]), (xs[-1], tens[-1]), textcoords="offset points", xytext=(9, -3.5), fontsize=10,
                    fontweight="bold", color=fs.INK)
        xmax = max(xmax, xs[-1])
    ax.axhline(10, color=fs.MUTED, lw=0.9, zorder=2)
    ax.text(xmax * 1.015, 11.5, "guessing: 10%", ha="right", va="bottom", fontsize=8.5, color=fs.MUTED)
    ax.axvline(8000, color=fs.GRID, lw=0.9, zorder=1)
    ax.text(8000, 99, " continued from here with a fresh optimizer", ha="left", va="top", fontsize=8.5, color=fs.MUTED)
    ax.set_xlim(0, xmax * 1.02)
    ax.set_xlabel("training steps on this calculation")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2, label=n) for n, _, c in series], frameon=False, fontsize=9.5,
              labelcolor=fs.INK2, loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncol=1, handlelength=1.6)
    fig.savefig(OUT / "coconut_carry.png")
    print("wrote", OUT / "coconut_carry.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["coconut", "exchange", "nextlat", "geometry"]
    for w in which:
        globals()[w]()
