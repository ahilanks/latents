"""Cut probabilities on the synthetic chains, per byte, for several models (a qualitative look).

Each model reads a stretch of held-out chains (data/fwe_math/val_arith.u8, as in pretraining) followed by a few test
problems written out in full; the picture shows, for those test problems, each character shaded by the model's cut
probability there, with the characters where a new piece starts outlined (H-Net: a piece starts at a cut byte).

usage: python scripts/cut_probs.py out.png runs/RT4n6r-s0 runs/NM0-s0 ... [--problems 1,5] [--context 6000]
Also prints each row as text: the probability (0-9, x=1.0) under every character.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from latents import tasks  # noqa: E402
from latents.models import model_forward  # noqa: E402
from show_cuts import load_any  # noqa: E402

SEP = 254


@torch.no_grad()
def probs(run, x):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, lat = load_any(run, dev)
    xt = torch.from_numpy(x.astype(np.int64))[None].to(dev)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        out = model_forward(model, xt, cfg["model"], lat)
    bp = out.bpred[0]
    p = bp.boundary_prob[..., -1].float().view(-1).cpu().numpy()
    m = bp.boundary_mask.view(-1).cpu().numpy()
    return p, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="runs/<name> or Label=runs/<name>")
    ap.add_argument("--problems", default="1,5", help="indices into data/task_arith/test.jsonl")
    ap.add_argument("--context", type=int, default=6000)
    a = ap.parse_args()
    test = [json.loads(l) for l in open(ROOT / "data/task_arith/test.jsonl")]
    ctx = np.fromfile(ROOT / "data/fwe_math/val_arith.u8", dtype=np.uint8)[:a.context]
    docs = [bytes([SEP]) + tasks.document(test[int(i)]) for i in a.problems.split(",")]
    tail = b"".join(docs)
    x = np.concatenate([ctx, np.frombuffer(tail, dtype=np.uint8)])
    start = len(ctx)
    rows = []
    for spec in a.runs:
        label, run = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        p, m = probs(run, x)
        rows.append((label, p[start:], m[start:]))
    # text: split the tail into lines (one per line of the documents), skip the SEP byte
    chars = [chr(b) if 32 <= b < 127 else ("⏎" if b == 10 else "·") for b in tail]
    lines, cur = [], []
    for i, b in enumerate(tail):
        if b == SEP:
            continue
        cur.append(i)
        if b == 10:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    for label, p, m in rows:
        print(f"== {label}")
        for ln in lines:
            print("   " + "".join(("[" if m[i] else " ") + chars[i] for i in ln))
            print("   " + "".join(" " + ("x" if p[i] >= 0.95 else str(int(p[i] * 10))) for i in ln))
    # picture: one block per model, one text line per document line
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    width = max(len(ln) for ln in lines)
    nl = len(lines)
    cw, lh = 0.16, 0.30
    H = len(rows) * (nl * lh + 0.45) + 0.6
    fig = plt.figure(figsize=(width * cw + 1.8, H), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-11, width + 1)
    ax.set_ylim(H / lh, 0)
    ax.axis("off")
    cmap = plt.get_cmap("Blues")
    y = 1.0
    for label, p, m in rows:
        ax.text(-10.5, y + 0.3, label, fontsize=8, va="center", fontweight="bold")
        for ln in lines:
            for j, i in enumerate(ln):
                ax.add_patch(Rectangle((j, y), 1, 0.9, facecolor=cmap(0.05 + 0.85 * float(p[i])), edgecolor="none"))
                if m[i]:
                    ax.add_patch(Rectangle((j, y), 1, 0.9, fill=False, edgecolor="#eb6834", lw=1.6))
                ax.text(j + 0.5, y + 0.47, chars[i], ha="center", va="center", fontsize=7, family="monospace",
                        color="white" if p[i] > 0.6 else "black")
            y += 1.0
        y += 1.5
    ax.text(-10.5, y - 0.4, "shade = cut probability (white 0 → dark blue 1); orange box = a new piece starts here",
            fontsize=7, color="#555555", va="center")
    fig.savefig(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
