"""Results figure: val BPB per recipe at 0.6B bytes (seed dots + mean), compute-matched (~4.2-4.9e16 FLOPs).
usage: python scripts/make_figures.py  -> runs/figures/hillclimb_0p6B.png (+ 3B panel data if available)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, BASE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated: scripts/validate_palette.js)

ROWS = [  # label, family, runs
    ("BPE Transformer (GPT-2 tokenizer)", BASE, ["s0sw_bpe_lr4e-3"]),
    ("Byte Transformer", BASE, ["s0s_bytes_lr4e-3"]),
    ("H-Net 1-stage (baseline)", BASE, ["s0sw_hnet_lr2e-3", "s0s_hnet_seed1"]),
    ("H-Net N=5, T7 (no latent loss)", BASE, ["hcD5_plain_N5_T7"]),
    ("+ latent L1 (EMA 0.99, linear head)", OURS, ["s1b2_tf_ema099", "s1b2_tf_ema099_seed1"]),
    ("+ MLP predictor head", OURS, ["hcA4_ema_mlphead", "hcC3_mlp_seed1"]),
    ("+ N=5 chunks, main T7 (final)", OURS, ["hcD3_mlp_N5_T7", "hcD4_mlp_N5_T7_seed1"]),
]


def bpb(run):
    p = ROOT / "runs" / run / "summary.json"
    return json.load(open(p))["val/bpb"] if p.exists() else None


def main():
    rows = [(lab, col, [v for v in map(bpb, runs) if v is not None]) for lab, col, runs in ROWS]
    rows = [r for r in rows if r[2]]
    fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(rows) + 1.3), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (lab, col, vals) in enumerate(rows):
        y = len(rows) - 1 - i
        mean = sum(vals) / len(vals)
        ax.plot([min(vals), max(vals)], [y, y], color=col, lw=2, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.scatter(vals, [y] * len(vals), s=36, color=col, alpha=0.45, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter([mean], [y], s=80, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        n = f"  {mean:.3f}" + (f"  (n={len(vals)})" if len(vals) > 1 else "")
        ax.text(max(vals) + 0.004, y, n, va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK)
    ax.set_xlabel("validation bits per byte on FineWeb-Edu (lower is better)", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=INK2, length=0)
    lo = min(min(r[2]) for r in rows)
    hi = max(max(r[2]) for r in rows)
    ax.set_xlim(lo - 0.01, hi + 0.045)
    fig.suptitle("Dynamic-chunking H-Net + latent next-chunk loss", fontsize=11, color=INK, x=0.02, ha="left")
    ax.set_title("0.6B training bytes, compute-matched (~4.2–4.9e16 FLOPs); dots = seeds, big dot = mean",
                 fontsize=8.5, color=INK2, loc="left", pad=8)
    ax.scatter([], [], s=60, color=OURS, label="H-Net + latent loss (this work)")
    ax.scatter([], [], s=60, color=BASE, label="baselines")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    out = ROOT / "runs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "hillclimb_0p6B.png", facecolor=SURFACE)
    print(out / "hillclimb_0p6B.png")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- 3B curves (val BPB vs train FLOPs)
CURVES = [  # label, color, run, linestyle   (slots 1-4 in fixed order; every line direct-labelled: aqua/yellow < 3:1)
    ("H-Net + latent (final), seed 0", "#2a78d6", "f3_D3_mlp_N5_T7_s0", "-"),
    ("H-Net + latent (final), seed 1", "#2a78d6", "f3_D3_mlp_N5_T7_s1", "-"),
    ("H-Net 1-stage", "#eb6834", "s0_hnet1s_256-512_m4T9m4", "-"),
    ("H-Net N=5·T7, no latent", "#eb6834", "f3_plain_N5_T7_s0", "--"),
    ("Byte Transformer", "#1baf7a", "s0_bytetf_d384T5", "-"),
    ("BPE Transformer", "#eda100", "s0_bpe_d512T8", "-"),
]


def curves_3b():
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for lab, col, run, ls in CURVES:
        p = ROOT / "runs" / run / "metrics.jsonl"
        sp = ROOT / "runs" / run / "summary.json"
        if not (p.exists() and sp.exists()):
            continue
        rows = [json.loads(l) for l in open(p)]
        pts = [(r["flops"], r["val/bpb"]) for r in rows if "val/bpb" in r and "flops" in r]
        s = json.load(open(sp))
        pts[-1] = (s["train_flops"], s["val/bpb"])  # final = full 8MB val
        x, y = zip(*pts)
        ax.plot(x, y, color=col, lw=2, ls=ls, alpha=0.55 if "seed 1" in lab else 1.0)
        ax.scatter([x[-1]], [y[-1]], s=40, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ends.append((y[-1], x[-1], f"{lab}  {y[-1]:.3f}", col))
    # direct labels at line ends, nudged apart vertically
    ends.sort()
    last = -9
    for yv, xv, txt, col in ends:
        yl = max(yv, last + 0.022)
        ax.text(xv * 1.04, yl, txt, va="center", fontsize=8, color=INK2)
        last = yl
    ax.set_xscale("log")
    ax.set_xlabel("training FLOPs (log)", fontsize=9, color=INK2)
    ax.set_ylabel("val bits per byte", fontsize=9, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8)
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.set_ylim(top=1.55)
    fig.suptitle("3B training bytes: val BPB vs compute", fontsize=11, color=INK, x=0.02, ha="left")
    ax.set_title("FineWeb-Edu, 1×A100; intermediate points = periodic evals (2MB val, stable LR), end = final (8MB val, after cooldown)",
                 fontsize=7.5, color=INK2, loc="left", pad=8)
    fig.tight_layout()
    out = ROOT / "runs" / "figures" / "curves_3B.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    curves_3b()
