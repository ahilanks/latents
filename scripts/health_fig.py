"""Health figure: a run against the normal H-Net, step by step (NEXT_STEPS.md "H-Net health during training"; alarm
levels from there). One small panel per number, grey = normal H-Net, blue = the run; a red band marks the alarm zone.
Also prints / writes the alarms found (a JSON next to the figure).

usage: python scripts/health_fig.py --run runs/<recipe> --plain runs/<plain> [--prefix val] [--plain_prefix val]
           [--out logs/figs/health_<run>.png] [--title ...] [--from_step 0]
Prefixes: pretraining "val"; latents.train fine-tunes "val_web" (web text) or "val" (reasoning text); holes_train
fine-tunes "web" (their rows keep health numbers under "health").
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RED = "#d03b3b"

# key, plain-English title, alarm rule (run value, plain value at the same step, plain series) -> True if alarming
PANELS = [
    ("bpb", "bits per byte (web text)", lambda v, p, ps: p is not None and v - p > 0.03, "> normal + 0.03"),
    ("compression_L1/L0", "cut rate", lambda v, p, ps: not 0.15 <= v <= 0.25, "outside 0.15–0.25"),
    ("cuts_after_space", "share of cuts right after a space", lambda v, p, ps: v > 0.5, "> 0.5"),
    ("router_rank_q", "router rank (q)", lambda v, p, ps: v < 3, "< 3"),
    ("p_mid", "cut probabilities at 0.4–0.5", lambda v, p, ps: v > 0.74, "> 74%"),
    ("bnd_moved", "cuts moved since the last eval", lambda v, p, ps: p is not None and v > 2 * p + 0.05, "> 2× normal"),
    ("main_variation", "main network: variation between chunks\n(% of mean square)",
     lambda v, p, ps: p is not None and v < 0.5 * p, "< ½ normal"),
    ("main_rank", "main network: effective rank", lambda v, p, ps: p is not None and v < 0.5 * p, "< ½ normal"),
    ("main_state_const", "main network: share of its state that is\nthe same for every chunk (normal: 0.82 at the end)",
     None, None),  # informational: added 2026-11-24, the normal H-Net's log has no value at the same steps
    ("swap_nats", "decoder's use of the main network\n(loss rise when chunk states are swapped)",
     lambda v, p, ps: p is not None and p > 0.002 and v < 0.3 * p, "< 0.3× normal"),
    ("enc_cos_next", "encoder: similarity of neighbouring bytes", lambda v, p, ps: v > 0.3, "> 0.3"),
    ("grad_norm_max", "largest gradient norm between logs",
     lambda v, p, ps: len(ps) > 0 and v > 3 * max(ps), "> 3× normal's largest"),
    ("round_diff", "loop: 4 vs 8 rounds (relative difference)", lambda v, p, ps: v > 0.05, "> 0.05"),
    ("fb_share", "loop: size of the fed-back vector\n(relative to the chunk input)", None, None),
]


def rows(run):
    out = []
    for l in open(ROOT / run / "metrics.jsonl"):
        r = json.loads(l)
        if "health" in r:
            r = {"step": r["step"], **r["health"]}
        out.append(r)
    return out


def series(rs, prefix, key, from_step=0):
    if key == "grad_norm_max":
        full = ["train/grad_norm_max", "grad_norm_max"]
    elif key == "p_mid":
        full = [f"{prefix}/p_hist/0.4"]
    else:
        full = [f"{prefix}/{key}"]
    pts = [(r["step"], r[k]) for r in rs for k in full if k in r and r["step"] >= from_step]
    if not pts:
        return np.array([]), np.array([])
    s, v = zip(*sorted(pts))
    return np.array(s), np.array(v, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--prefix", default="val")
    ap.add_argument("--plain_prefix", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--run_label", default="hole recipe")
    ap.add_argument("--plain_label", default="normal H-Net")
    ap.add_argument("--from_step", type=int, default=0)
    ap.add_argument("--warmup_steps", type=int, default=250, help="gradient spikes are judged after this step")
    a = ap.parse_args()
    rr, pr = rows(a.run), rows(a.plain)
    pp = a.plain_prefix or a.prefix
    panels = []
    for key, title, rule, rule_txt in PANELS:
        s, v = series(rr, a.prefix, key, a.from_step)
        if not len(s):
            continue
        ps, pv = series(pr, pp, key, a.from_step)
        panels.append((key, title, rule, rule_txt, s, v, ps, pv))
    alarms = []
    for key, title, rule, rule_txt, s, v, ps, pv in panels:
        if rule is None:
            continue
        ref_all = pv[ps >= a.warmup_steps] if key == "grad_norm_max" else pv
        for st, val in zip(s, v):
            if key == "grad_norm_max" and st < a.warmup_steps:
                continue
            p_same = None
            if len(ps):
                j = int(np.argmin(np.abs(ps - st)))
                p_same = pv[j] if abs(ps[j] - st) <= max(50, 0.05 * st) else None
            if rule(val, p_same, ref_all):
                alarms.append({"metric": key, "step": int(st), "value": float(val),
                               "normal": None if p_same is None else float(p_same), "rule": rule_txt})
    panels = [(k, t, r, rt, s, (100 * v if k == "main_variation" else v), ps,
               (100 * pv if k == "main_variation" else pv)) for k, t, r, rt, s, v, ps, pv in panels]
    n = len(panels)
    cols = 5 if n > 12 else 4
    rws = (n + cols - 1) // cols
    fig, axs = plt.subplots(rws, cols, figsize=(3.3 * cols, 2.9 * rws + 1.4), dpi=150)
    axs = np.atleast_2d(axs)
    bad = sorted({al["metric"] for al in alarms})
    title = a.title or ("Health: every number stays in the normal H-Net's range" if not bad else
                        f"Health: {len(bad)} number(s) leave the normal H-Net's range")
    sub = (f"{a.run_label} ({Path(a.run).name}, blue) vs {a.plain_label} ({Path(a.plain).name}, grey) at the same "
           f"steps. Red: alarm zone from the plan. Single seed.")
    used = fs.titles(fig, title, sub)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=fs.BLUE, ec="none", label=a.run_label), Patch(fc=fs.GREY, ec="none", label=a.plain_label)],
               loc="upper left", bbox_to_anchor=(0.008, 1 - used - 0.005), ncol=2, frameon=False, fontsize=9.5,
               labelcolor=fs.INK2, handlelength=1.0, handleheight=1.0)
    fig.subplots_adjust(left=0.045, right=0.975, top=1 - used - 0.11, bottom=0.08, hspace=0.85, wspace=0.32)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis("off")
            continue
        key, title_, rule, rule_txt, s, v, ps, pv = panels[i]
        if len(ps):
            ax.plot(ps, pv, color=fs.GREY, lw=2)
        ax.plot(s, v, color=fs.BLUE, lw=2, marker="o" if len(s) < 4 else None, ms=4)
        vals = np.concatenate([v, pv]) if len(pv) else v
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = 0.15 * (hi - lo) if hi > lo else 0.1 * abs(hi) + 1e-3
        ax.set_ylim(lo - pad, hi + pad)
        y0, y1 = ax.get_ylim()
        thr = {"compression_L1/L0": [(y0, 0.15), (0.25, y1)], "cuts_after_space": [(0.5, y1)],
               "router_rank_q": [(y0, 3)], "p_mid": [(0.74, y1)], "enc_cos_next": [(0.3, y1)],
               "round_diff": [(0.05, y1)]}.get(key, [])
        for a_, b_ in thr:
            if b_ > a_:
                ax.axhspan(max(a_, y0), min(b_, y1), color=RED, alpha=0.08, lw=0)
        ax.set_ylim(y0, y1)
        xs_all = np.concatenate([s, ps]) if len(ps) else s
        x0_, x1_ = float(xs_all.min()), float(xs_all.max())
        ax.set_xlim(x0_ - 0.03 * (x1_ - x0_ + 1), x1_ + 0.22 * (x1_ - x0_ + 1))  # room for the end labels
        ax.yaxis.get_major_formatter().set_useOffset(False)
        flag = key in bad
        ax.set_title(title_ + ("  ⚠" if flag else ""), fontsize=9, color=RED if flag else fs.INK, loc="left")
        ax.grid(axis="y", color=fs.GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, length=0)
        ax.text(s[-1], v[-1], f" {v[-1]:.3g}", color=fs.INK, fontsize=8, va="bottom", ha="left")
        if len(pv):
            ax.text(ps[-1], pv[-1], f" {pv[-1]:.3g}", color=fs.INK2, fontsize=8, va="top", ha="left")
        if i // cols == rws - 1 or i + cols >= n:
            ax.set_xlabel("training step", fontsize=8, color=fs.INK2)
    out = ROOT / (a.out or f"logs/figs/health_{Path(a.run).name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    json.dump({"run": a.run, "plain": a.plain, "alarms": alarms, "bad": bad}, open(out.with_suffix(".json"), "w"), indent=1)
    print("wrote", out, "alarms:", bad, [al for al in alarms if al["metric"] in bad][:8])


if __name__ == "__main__":
    main()
