"""Result figures of the hole recipe (one figure per finding, the main number readable at a glance; style
scripts/figstyle.py). Reads logs/holes/checks_<run>.json (scripts/holes_checks.py) and runs/*/holes_eval.json.
usage: python scripts/holes_figs.py rounds|holes|compression|reasoning [--run HL-A-s0]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "logs" / "figs"


def checks(run):
    return json.load(open(ROOT / "logs/holes" / f"checks_{run}.json"))


def bars(names, vals, colors, title, sub, ylabel, out, fmt="{:.4f}", ref=None, ref_label=None, ymax=None, yticks=None,
         pct=False):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    used = fs.titles(fig, title, sub)
    fig.subplots_adjust(left=0.12, right=0.97, top=1 - used - 0.06, bottom=0.17)
    top = ymax or max(max(vals), ref or 0) * 1.22
    fs.axes_style(ax, top, yticks=yticks, ylabel=ylabel, pct=pct)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    fs.columns(ax, list(range(len(vals))), vals, colors, width=0.5, fmt=fmt)
    if ref is not None:
        ax.axhline(ref, color=fs.MUTED, lw=1.0, ls="--", zorder=2)
        ax.text(len(vals) - 0.42, ref + top * 0.012, ref_label, ha="right", va="bottom", fontsize=8.5, color=fs.MUTED)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=9.5, color=fs.INK, linespacing=1.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / out)
    print("wrote", OUT / out, dict(zip([n.replace("\n", " ") for n in names], [round(v, 5) for v in vals])))


def fig_rounds(run):
    c = checks(run)["check1_rounds_vs_exact"]
    k = checks(run)["trained_rounds"]
    tags = [f"r{k}", "r8", "r16"]
    for tag in ("intact", "holes"):
        d = c[tag]
        vals = [100 * d[t]["z_rel_diff_vs_exact"] for t in tags] + [100 * d["off"]["z_rel_diff_vs_exact"]]
        floor = 100 * d["floor"]["z_rel_diff"]
        ok = d[f"r{k}"]["z_rel_diff_vs_exact"] <= 3 * d["floor"]["z_rel_diff"] + 0.01
        title = (f"{k} rounds {'match' if ok else 'do not match'} the exact loop"
                 f" ({'text with holes' if tag == 'holes' else 'intact text'})")
        sub = (f"How far the main network's outputs are from the exact loop, short rows of web text. Bits per byte: "
               f"{k} rounds {d[f'r{k}']['bpb']:.4f}, 16 rounds {d['r16']['bpb']:.4f}, exact loop {d['exact']['bpb']:.4f}, "
               f"loop off {d['off']['bpb']:.4f}. Single seed.")
        bars([f"{k} rounds\n(as trained)", "8 rounds", "16 rounds", "loop\nswitched off"], vals,
             [fs.BLUE, fs.BLUE, fs.BLUE, fs.GREY], title, sub, "difference from the exact loop", f"holes_rounds_{tag}.png",
             fmt="{:.2f}%", ref=floor, ref_label=f"rounding noise: {floor:.2f}%")


def fig_holes(run):
    c = checks(run)
    k = c["trained_rounds"]
    d = c["check2_holes"][f"r{k}"]
    on, off = d["h/bpb_after"], d["h/bpb_after_off"]
    title = (f"Switching the loop off {'hurts' if off - on > 0.02 else 'barely changes'} the text right after a hole "
             f"({off - on:+.3f} bits per byte)")
    sub = (f"Bits per byte on the word that follows a hole, web validation text with 10% of words deleted, the "
           f"recipe model with the loop on (as trained, {k} rounds) and switched off at test time. Whole holed text: "
           f"{d['h/bpb']:.3f} on, {d['h/bpb_off']:.3f} off. Single seed.")
    bars(["loop on\n(as trained)", "loop switched off"], [on, off], [fs.BLUE, fs.GREY], title, sub,
         "bits per byte after a hole", "holes_after_hole.png", fmt="{:.3f}")


def fig_compression(run):
    c = checks(run)
    k = c["trained_rounds"]
    d = c["check3_compression"]
    plain = d["plain_summary"]
    vals = [plain, d[f"r{k}"], d["r16"]]
    diff = d[f"r{k}"] - plain
    title = f"Compression on intact text: the recipe is {abs(diff):.3f} bits per byte {'worse' if diff > 0 else 'better'} than the normal H-Net"
    sub = (f"Whole web validation set (8 MB), 0.6B bytes of training each. Recipe with the loop off at test time: "
           f"{d['off']:.4f}. Single seed.")
    bars(["normal H-Net", f"recipe\n({k} rounds, as trained)", "recipe\n(16 rounds)"], vals, [fs.GREY, fs.BLUE, fs.BLUE],
         title, sub, "bits per byte (lower is better)", "holes_compression.png", fmt="{:.4f}", ymax=1.75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    ap.add_argument("--run", default="HL-A-s0")
    a = ap.parse_args()
    for w in a.what:
        {"rounds": fig_rounds, "holes": fig_holes, "compression": fig_compression}[w](a.run)


if __name__ == "__main__":
    main()
