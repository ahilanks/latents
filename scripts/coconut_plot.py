"""Learning curve of Coconut runs: is the first number the model needs from its thoughts right (units digit, tens
digit, whole number) and are the answers right, at every quick eval.
usage: python scripts/coconut_plot.py <run> [<run> ...] [--out logs/coconut/curve.png] [--labels "a|b"]
Reads runs/<run>/metrics.jsonl (the "eval" rows of scripts/coconut_train.py)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#1f2430", "#6b7280"
COL = {"units": "#2a78d6", "tens": "#d9662b", "answer": "#1f2430"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="logs/coconut/curve.png")
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    labels = a.labels.split("|") if a.labels else a.runs
    fig, axes = plt.subplots(1, len(a.runs), figsize=(6.4 * len(a.runs), 4.3), sharey=True, squeeze=False)
    for ax, run, lab in zip(axes[0], a.runs, labels):
        rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
        ev = [r for r in rows if "eval" in r]
        xs = [r["step"] for r in ev]
        ts = [r["eval"]["this_stage"] for r in ev]
        series = (("units", "ones digit of the hidden value", [t["carried"]["units"] for t in ts]),
                  ("tens", "tens digit of the hidden value", [t["carried"]["tens"] for t in ts]),
                  ("answer", "final answer", [t["acc"] for t in ts]))
        stages = {r["stage"] for r in ev}
        ends = []
        for key, name, ys in series:
            ax.plot(xs, [100 * y for y in ys], color=COL[key], lw=2, marker="o", ms=4, label=name)
            ends.append([100 * ys[-1], key])
        ends.sort()
        ypos = []
        for v, key in ends:  # end labels at least 6 points apart
            ypos.append(max(v, ypos[-1] + 6) if ypos else v)
        for (v, key), y in zip(ends, ypos):
            ax.annotate(f"{v:.1f}%", (xs[-1], y), textcoords="offset points", xytext=(7, -3), color=COL[key], fontsize=9)
        # stage spans from the training rows; stage s hides s - 1 calculations (stage 1: only the restated first value)
        span = {}
        for r in rows:
            if "loss" in r:
                lo, hi = span.get(r["stage"], (r["step"], r["step"]))
                span[r["stage"]] = (min(lo, r["step"]), max(hi, r["step"]))
        for st, (lo, hi) in sorted(span.items()):
            if st not in stages:  # a stage that was started but never evaluated
                continue
            if lo > 0:
                ax.axvline(lo, color=MUTED, lw=0.8, ls=":")
            txt = "no calculation\nhidden" if st == 1 else f"{st - 1} calculation{'s' if st > 2 else ''}\nhidden"
            ax.text((lo + hi) / 2, 113, txt, color=MUTED, fontsize=8, ha="center", va="bottom", linespacing=1.1)
        ax.set_title(lab, fontsize=11, color=INK, loc="left", pad=30)
        ax.set_xlabel("training step", color=INK)
        ax.set_ylim(0, 112)
        ax.set_xlim(0, max(xs) * 1.09)
        ax.grid(axis="y", color="#e5e7eb", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("% right (200 test problems)", color=INK)
    axes[-1][-1].legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
