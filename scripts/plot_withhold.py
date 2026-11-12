"""Accuracy-vs-withheld curves from scripts/think_withhold.py. usage: python scripts/plot_withhold.py <out.png> <L2 think run> [<L1 think run>]
(pools runs/<run>/think_withhold_rows.jsonl into 2-6-step = training range and 7-8-step problems)."""
import json, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out_png, runs = sys.argv[1], sys.argv[2:]
def pool(r):
    agg = defaultdict(lambda: [0, 0])
    for l in open(f"runs/{r}/think_withhold_rows.jsonl"):
        x = json.loads(l)
        g = "in" if x["steps"] <= 6 else "out"
        agg[(x["setting"], x["fill"], g)][0] += x["ok"]; agg[(x["setting"], x["fill"], g)][1] += 1
    return {f"{s}|{f}|{g}": a / n for (s, f, g), (a, n) in agg.items()}
d = {r: pool(r) for r in runs}
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF)
ms = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, "all"]
xs = [m if m != "all" else 19 for m in ms]
S = lambda r, fill, g, names: [100 * d[r].get(f"{n}|{fill}|{g}", float("nan")) for n in names]  # noqa: E731
l2, l1 = runs[0], (runs[1] if len(runs) > 1 else None)
ax = axs[0]
names = [f"m={m}" for m in ms]
drop = S(l2, "drop", "in", names); drop[0] = S(l2, "own", "in", names)[0]
spec = [("L2, 2–6 steps", S(l2, "own", "in", names), BLUE, "-", "o"),
        ("L2, 7–8 steps (beyond training)", S(l2, "own", "out", names), BLUE, "--", "s")]
if l1: spec.append(("L1, 2–6 steps", S(l1, "own", "in", names), ORANGE, "-", "o"))
spec.append(("control: dropped, not filled", drop, GRAY, ":", None))
for lab, ys, c, ls, mk in spec:
    ax.plot(xs, ys, color=c, ls=ls, lw=2, marker=mk, ms=5, label=lab, markeredgecolor=SURF, markeredgewidth=1)
ax.set_xticks(xs, [str(m) for m in ms])
ax.set_xlabel("m = last teacher thoughts withheld (4-byte chunks)")
ax.set_ylabel("answer accuracy (%)")
ax.set_title("Withhold the last m chunks; the model fills them in", color=INK, loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.02, 1.0))
ax = axs[1]
ns = list(range(1, 7)); names = [f"steps={n}" for n in ns]
spec = [("L2", l2, "own", BLUE, "-", "o")] + ([("L1", l1, "own", ORANGE, "-", "o")] if l1 else []) + \
       [("control: dropped (L2)", l2, "drop", GRAY, ":", None)]
for lab, r, fill, c, ls, mk in spec:
    ys = S(r, fill, "in", names)
    ax.plot(ns, ys, color=c, ls=ls, lw=2, marker=mk, ms=5, label=lab, markeredgecolor=SURF, markeredgewidth=1)
    if fill == "own":
        for n, y in zip(ns, ys):
            ax.annotate(f"{y:.0f}", (n, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
ax.set_xticks(ns)
ax.set_xlabel("n = whole steps withheld (2–6-step problems, n ≤ steps)")
ax.set_title("Withhold the last n steps; it computes them in latent space", color=INK, loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
for ax in axs:
    ax.set_facecolor(SURF); ax.set_ylim(0, 106); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Read-then-compute, all 2000 test problems ({' vs '.join(runs)}; trained teacher-forced, own rollouts only at test)",
             color=INK, fontsize=10.5, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out_png, dpi=150, facecolor=SURF)
print("wrote", out_png)
