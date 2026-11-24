"""Result figures of the hole recipe (one figure per finding, the main number readable at a glance; style
scripts/figstyle.py). Reads logs/holes/checks_<run>.json (scripts/holes_checks.py) and runs/*/holes_eval.json.
usage: python scripts/holes_figs.py rounds|holes|compression|reasoning_arith|reasoning_logic [--run HL-A-s0]
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
         pct=False, figsize=(7.6, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
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
               f"loop off {d['off']['bpb']:.4f}. Rounding noise: {floor:.2f}%. Single seed.")
        bars([f"{k} rounds\n(as trained)", "8 rounds", "16 rounds", "loop\nswitched off"], vals,
             [fs.BLUE, fs.BLUE, fs.BLUE, fs.GREY], title, sub, "difference from the exact loop", f"holes_rounds_{tag}.png",
             fmt="{:.2f}%")


def fig_holes(run):
    c = checks(run)
    k = c["trained_rounds"]
    d = c["check2_holes"][f"r{k}"]
    on, off = d["h/bpb_after"], d["h/bpb_after_off"]
    title = (f"Switching the loop off {'hurts' if off - on > 0.02 else 'barely changes'} the text right after a hole "
             f"({off - on:+.4f} bits per byte)")
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


def acc(run, mode):
    p = ROOT / "runs" / run / "holes_eval.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return 100 * d[mode]["acc"] if mode in d else None


def fig_reasoning(task):
    """Final answers right on 2000 test problems: written reasoning, thoughts as holes, answering directly."""
    name = {"arith": "arithmetic", "logic": "multi-hop English"}[task]
    rows = [("normal H-Net\nwrites every step", f"HPc-{task}", "written", fs.GREY),
            ("recipe\nwrites every step", f"HLc-{task}", "written", fs.BLUE),
            ("recipe, first 3 steps\nas holes (curriculum)", f"HLh-{task}-stage3", "stage3", fs.BLUE),
            ("recipe, every step\nas holes (at once)", f"HLh-{task}-all-all", "hidden", fs.BLUE),
            ("recipe\nanswers directly", f"HLh-{task}-direct-all", "nothoughts", fs.AQUA),
            ("normal H-Net\nanswers directly", f"HPh-{task}-direct-all", "nothoughts", fs.GREY)]
    rows = [(l, r, m, c, acc(r, m)) for l, r, m, c in rows]
    rows = [r for r in rows if r[4] is not None]
    if not rows:
        print("no results for", task)
        return
    vals = [r[4] for r in rows]
    best = max(rows[1:], key=lambda r: r[4]) if len(rows) > 1 else rows[0]
    title = (f"{name.capitalize()}: normal H-Net writing every step {rows[0][4]:.1f}%, "
             f"best recipe variant {best[4]:.1f}% ({best[0].replace(chr(10), ' ')})")
    chance = 50 if task == "logic" else 1
    sub = (f"Final answer right on 2000 test problems ({'2–8 hops' if task == 'logic' else '2–8 steps'}; training "
           f"used 2–6). Recipe = loop + holes, exact loop at test time. Guessing: {chance}%. Single seed.")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
    used = fs.titles(fig, title, sub)
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - used - 0.06, bottom=0.2)
    fs.axes_style(ax, 112, yticks=[0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, len(vals) - 0.4)
    fs.columns(ax, list(range(len(vals))), vals, [r[3] for r in rows], width=0.52)
    ax.axhline(chance, color=fs.MUTED, lw=0.9, ls="--", zorder=2)
    ax.text(len(vals) - 0.42, chance + 1.5, f"guessing: {chance}%", ha="right", va="bottom", fontsize=8.5, color=fs.MUTED)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9, color=fs.INK, linespacing=1.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"holes_reasoning_{task}.png")
    print("wrote", OUT / f"holes_reasoning_{task}.png", [(r[1], r[2], round(r[4], 1)) for r in rows])


def fig_flat():
    """The loop switches the main network off: variation of its outputs between chunks, normal H-Net vs loop models."""
    d = json.load(open(ROOT / "logs/holes/flat_diag.json"))
    rows = [("normal H-Net\n(same-seed rerun)", "HP-plain-s0", fs.GREY), ("normal H-Net\n(original)", "side-p06-plain-s0", fs.GREY),
            ("old two-pass loop\n(no holes, 2026-11-20)", "FB-h-s0", fs.ORANGE), ("hole recipe\n(loop + holes)", "HL-A-s0", fs.BLUE)]
    for extra, lab in (("HL-C-s0", "centred loop\n(fix 1)"), ("HL-S-s0", "slow loop\n(fix 2)")):
        if extra in d:
            rows.append((lab, extra, fs.AQUA))
    rows = [r for r in rows if r[1] in d]
    vals = [100 * d[r[1]]["main_variation"] for r in rows]
    a, p = d.get("HL-A-s0", {}), d.get("HP-plain-s0", {})
    title = ("The loop switches the main network off: its outputs barely differ between chunks"
             if len(rows) <= 4 else "Main-network outputs: how much they differ between chunks")
    sub = (f"Variation between chunks as a share of the outputs' mean square, 16 web windows, 0.6B bytes of training. "
           f"Hole recipe vs normal: effective rank {a.get('main_rank', 0):.1f} vs {p.get('main_rank', 0):.1f}; decoder's loss "
           f"rise when chunk states are swapped {a.get('swap_nats', 0):.4f} vs {p.get('swap_nats', 0):.4f} nats per byte. "
           f"Single seed each.")
    bars([r[0] for r in rows], vals, [r[2] for r in rows], title, sub, "variation between chunks", "holes_flat_main.png",
         fmt="{:.1f}%", ymax=max(vals) * 1.25, pct=True, yticks=[0, 5, 10, 15])


VARIANTS = [("normal\nH-Net", "HP-plain-s0", fs.GREY), ("old two-pass\nloop (Nov 20)", "FB-h-s0", fs.ORANGE),
            ("first\nrecipe", "HL-A-s0", fs.BLUE), ("centred\nloop*", "HL-C-s0", fs.BLUE),
            ("slow\nloop", "HL-S-s0", fs.AQUA), ("paper's form\n(end to end)*", "HL-R-s0", fs.BLUE),
            ("paper's training\n(not end to end)*", "HL-P-s0", fs.BLUE)]
STOPPED = "*stopped early on the health alarm (centred at step 1500, the paper runs at step 750; last eval shown)."


def last_eval(run):
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return None
    ev = [json.loads(l) for l in open(p)]
    ev = [r for r in ev if "val/bpb" in r]
    return ev[-1] if ev else None


def fig_variants():
    """One figure per finding across the loop variants: main network alive?, loop fills holes?, compression."""
    diag = json.load(open(ROOT / "logs/holes/flat_diag.json"))
    rows = []
    for lab, run, col in VARIANTS:
        d = diag.get(run)
        le = last_eval(run)
        killed = (ROOT / "runs" / run / "summary.json").exists() and "killed" in json.load(open(ROOT / "runs" / run / "summary.json"))
        var = d["swap_nats"] if d else (le or {}).get("val/swap_nats")
        ck = ROOT / "logs/holes" / f"checks_{run}.json"
        gain = None
        if ck.exists():
            c = json.load(open(ck))
            gain = c["check2_holes"][f"r{c['trained_rounds']}"]["h/loop_gain_after"]
        elif le is not None and "val_holes/loop_gain_after" in le:
            gain = le["val_holes/loop_gain_after"]
        sm = ROOT / "runs" / run / "summary.json"
        bpb = json.load(open(sm))["val/bpb"] if sm.exists() and not killed else None
        rows.append((lab, run, col, var, gain, bpb))
    ok = [r for r in rows if r[3] is not None]
    bars([r[0] for r in ok], [r[3] for r in ok], [r[2] for r in ok],
         "Only the slow loop keeps the main network in use; every other loop switches it off",
         "How much the decoder uses the main network: rise in its loss (nats per byte) when every chunk's main-network "
         f"state is swapped for another chunk's, 16 web windows, end of training. {STOPPED} The normal H-Net already "
         f"had 0.017 at step 750. 0.6B bytes each, single seed.", "loss rise when states are swapped",
         "holes_variants_main.png", fmt="{:.4f}", ymax=max(r[3] for r in ok) * 1.25, pct=False,
         yticks=[0, 0.01, 0.02, 0.03], figsize=(10.5, 5.0))
    g = [r for r in rows if r[4] is not None]
    bars([r[0] for r in g], [r[4] for r in g], [r[2] for r in g],
         "No loop variant fills the holes: switching the loop off changes the text after a hole by under 0.001 bits per byte",
         "Bits per byte on the word after a hole with the loop switched off minus with it on (higher = the loop helps), "
         f"web validation text with 10% of words deleted. {STOPPED} Single seed.", "bits per byte lost without the loop",
         "holes_variants_gain.png", fmt="{:+.4f}", ymax=0.005, yticks=[0, 0.001, 0.002, 0.003, 0.004, 0.005], pct=False,
         figsize=(10.5, 5.0))
    c = [r for r in rows if r[5] is not None]
    plain0 = json.load(open(ROOT / "runs/side-p06-plain-s0/summary.json"))["val/bpb"]
    bars([r[0] for r in c], [r[5] for r in c], [r[2] for r in c],
         "Compression: the slow loop matches the normal H-Net; the loops that flatten the main network lose 0.006–0.013",
         f"Bits per byte on the whole web validation set (lower is better). Normal H-Net: same-seed rerun shown; the "
         f"original run had {plain0:.4f}. Runs stopped early are left out. Single seed.", "bits per byte",
         "holes_variants_bpb.png", fmt="{:.4f}", ymax=1.6, figsize=(8.5, 4.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+")
    ap.add_argument("--run", default="HL-A-s0")
    a = ap.parse_args()
    for w in a.what:
        if w.startswith("reasoning"):
            fig_reasoning(w.split("_")[1] if "_" in w else "arith")
            continue
        if w == "flat":
            fig_flat()
            continue
        if w == "variants":
            fig_variants()
            continue
        {"rounds": fig_rounds, "holes": fig_holes, "compression": fig_compression}[w](a.run)


if __name__ == "__main__":
    main()
