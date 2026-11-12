"""How does each method cut text into pieces? Numbers for learned routers vs fixed pieces vs a standard tokenizer.

Methods (same spec as scripts/show_cuts.py): "Label=fixed:4" | "Label=bpe" | "Label=runs/<name>".
A piece ends at the byte where the method fires (segment S_j = (t_{j-1}, t_j]). Measured on the first --windows val
windows of 8192 bytes:
  per100       pieces per 100 bytes
  len1         share of pieces that are a single byte;  p95 / max = piece-length percentiles
  edge_prec    share of cuts that sit next to whitespace (a cut between two bytes, one of which is whitespace)
  edge_rec     share of word edges (each side of every whitespace run) that have a cut
  whole_words  share of words (>= 4 letters) with no cut strictly inside
  consistent   for words seen >= 10 times: share of occurrences cut in that word's most common way (frequency-weighted).
               A tokenizer scores ~1 by construction; position-based cuts score low.
  f1_bpe       F1 of the cut positions against the GPT-2 BPE cut positions (exact byte)
  bpb          the run's readout bits per byte (models only; from its summary.json)

usage: python scripts/tokenize_compare.py out.json "Label=fixed:4" "Label=bpe" "Label=runs/RT0-s0" ... [--windows 64]
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from show_cuts import ends_bpe, ends_fixed, load_any, model_piece_ends  # noqa: E402

L = 8192
WS = (32, 9, 10, 13)
WORD = re.compile(rb"(?<![A-Za-z])[A-Za-z]{4,}(?![A-Za-z])")


@torch.no_grad()
def model_ends(run, wins):
    from latents.models import model_forward
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, lat = load_any(run, dev)
    out = []
    for i in range(0, len(wins), 16):
        x = torch.from_numpy(np.stack(wins[i:i + 16]).astype(np.int64)).to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model_forward(model, x, cfg["model"], lat)
        out.append(o.bpred[0].boundary_mask.view(x.shape).cpu().numpy())
    return np.concatenate(out, 0)


def measure(ends, wins, bpe):
    """ends, bpe: (W, L) bool, True at the last byte of each piece. wins: list of (L,) uint8."""
    st = {}
    n_cut = int(ends.sum())
    st["per100"] = 100.0 * n_cut / ends.size
    lens = []
    for e in ends:
        pos = np.flatnonzero(e)
        if len(pos) > 1:
            lens.append(np.diff(pos))
    lens = np.concatenate(lens) if lens else np.array([L])
    st["mean_len"], st["len1"] = float(lens.mean()), float((lens == 1).mean())
    st["p95"], st["max"] = float(np.percentile(lens, 95)), int(lens.max())
    X = np.stack(wins)
    ws = np.isin(X, WS)
    nxt_ws = np.concatenate((ws[:, 1:], np.zeros((len(wins), 1), bool)), 1)
    at_edge = (ws ^ nxt_ws)  # a cut after byte t is at a word edge if exactly one of t, t+1 is whitespace
    st["edge_prec"] = float((ends & at_edge).sum() / max(n_cut, 1))
    st["edge_rec"] = float((ends & at_edge).sum() / max(int(at_edge.sum()), 1))
    tp = int((ends & bpe).sum())
    pr, rc = tp / max(n_cut, 1), tp / max(int(bpe.sum()), 1)
    st["f1_bpe"] = 2 * pr * rc / max(pr + rc, 1e-9)
    whole = tot = 0
    pats = defaultdict(Counter)
    for w, e in zip(wins, ends):
        b = bytes(w)
        for m in WORD.finditer(b):
            s, t = m.span()
            inside = e[s:t - 1]  # cuts strictly inside the word
            tot += 1
            whole += not inside.any()
            pats[b[s:t].lower()][inside.tobytes()] += 1
    st["whole_words"] = whole / max(tot, 1)
    num = den = 0
    for wd, c in pats.items():
        n = sum(c.values())
        if n >= 10:
            num += c.most_common(1)[0][1]
            den += n
    st["consistent"] = num / max(den, 1)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--windows", type=int, default=64)
    a = ap.parse_args()
    v = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    wins = [v[i * L:(i + 1) * L] for i in range(a.windows)]
    bpe = np.stack([ends_bpe(bytes(w)) for w in wins])
    res = {}
    cols = ["per100", "len1", "p95", "max", "edge_prec", "edge_rec", "whole_words", "consistent", "f1_bpe", "bpb"]
    print(f"{'method':44s}" + "".join(f"{c:>12s}" for c in cols))
    for spec in a.methods:
        label, what = spec.split("=", 1)
        if what.startswith("fixed:"):
            ends = np.stack([ends_fixed(L, int(what.split(":")[1])) for _ in wins])
        elif what == "bpe":
            ends = bpe
        else:
            ends, conv = model_piece_ends(model_ends(what, wins), np.stack(wins))
            label += " [start]" if conv == "start" else ""
        st = measure(ends, wins, bpe)
        sj = ROOT / what / "summary.json"
        st["bpb"] = json.load(open(sj))["val/bpb"] if what.startswith("runs/") and sj.exists() else float("nan")
        res[label] = dict(st, source=what)
        print(f"{label:44s}" + "".join(f"{st[c]:12.3f}" for c in cols), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
