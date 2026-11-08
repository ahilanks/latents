"""Compare finished runs: markdown table + val-BPB-vs-FLOPs plot.

usage: python scripts/compare.py <out.png|-> run_name [run_name ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    d = ROOT / "runs" / name
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
    return rows, s


def main():
    out, names = sys.argv[1], sys.argv[2:]
    print("| run | val BPB | L1/L0 | train PFLOPs | params | min | latent tf_l1 / copy_l1 / eff_rank |")
    print("|---|---|---|---|---|---|---|")
    curves = {}
    for n in names:
        rows, s = load(n)
        ev = [r for r in rows if "val/bpb" in r]
        curves[n] = [(r["flops"], r["val/bpb"]) for r in ev if "flops" in r]
        lat = ""
        if "val/lat/tf_l1" in s:
            lat = f"{s['val/lat/tf_l1']:.4f} / {s.get('val/lat/copy_l1', float('nan')):.4f} / {s.get('val/lat/tgt_eff_rank', float('nan')):.1f}"
        comp = s.get("val/compression_L1/L0")
        print(f"| {n} | {s.get('val/bpb', float('nan')):.4f} | {comp:.3f} | " if comp else f"| {n} | {s.get('val/bpb', float('nan')):.4f} | – | ", end="")
        print(f"{s.get('train_flops', 0)/1e15:.0f} | {s.get('n_params', 0)/1e6:.1f}M | {s.get('wall_min', 0):.0f} | {lat} |")
    if out != "-":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        for n, c in curves.items():
            if c:
                x, y = zip(*c)
                ax.plot(x, y, marker="o", ms=3, lw=1.5, label=n)
        ax.set_xlabel("train FLOPs")
        ax.set_ylabel("val BPB (FineWeb-Edu)")
        ax.set_ylim(top=min(2.0, ax.get_ylim()[1]))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
