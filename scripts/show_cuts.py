"""Show how different methods cut the same passage into pieces (a picture with one row per method + plain text).

A piece ends at the byte where the method fires (router brief convention: segment S_j = (t_{j-1}, t_j]). Methods:
  fixed:K       a cut every K bytes
  bpe           GPT-2 BPE tokens (the tokenizer of the similar-size BPE baseline)
  runs/<name>   a trained model's own boundaries on this passage (run on the whole 8192-byte val window)

usage: python scripts/show_cuts.py out.png "Label=fixed:4" "Label=bpe" "Label=runs/RT0-s0" ... [--start 39] [--chars 264]
       [--window 0] [--cols 88] [--title "..."]
Also prints every row as text with | between pieces, and the pieces-per-100-bytes of each method on the passage.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))


def ends_fixed(n, k, offset=0):
    """True at the last byte of each piece."""
    e = np.zeros(n, dtype=bool)
    e[[i for i in range(n) if (i + offset) % k == k - 1]] = True
    return e


def ends_bpe(data: bytes):
    """GPT-2 BPE token ends, in byte positions of `data`. Documents are separated by byte 254 (its own piece); bytes of
    a multi-byte character cut off at the window edges are their own pieces too, so positions never shift."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    e = np.zeros(len(data), dtype=bool)
    i = 0
    while i < len(data):
        if data[i] == 254:
            e[i] = True
            i += 1
            continue
        j = data.find(b"\xfe", i)
        j = len(data) if j < 0 else j
        lo, hi = i, j
        while lo < hi:  # trim to valid UTF-8
            try:
                text = data[lo:hi].decode("utf-8")
                break
            except UnicodeDecodeError as err:
                if err.start == 0:
                    e[lo] = True
                    lo += 1
                else:
                    hi = lo + err.start
        else:
            text = ""
        pos = lo
        for t in enc.encode(text, disallowed_special=()):
            pos += len(enc.decode_single_token_bytes(t))
            e[pos - 1] = True
        e[hi:j] = True
        i = j
    return e


def load_any(run, dev):
    """A trained run, with or without a latent objective (normal H-Nets have none)."""
    from latents import latent as latent_lib
    from latents.models import build_model
    ck = torch.load(ROOT / run / "final.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], device=dev)
    model.load_state_dict(ck["model"])
    lat = None
    if cfg.get("latent"):
        lat = latent_lib.build(cfg["latent"], model, cfg["model"])
        lat.load_state_dict(ck["latent"], strict=False)
    return cfg, model.eval(), lat


@torch.no_grad()
def ends_model(run, window: np.ndarray):
    from latents.models import model_forward
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, lat = load_any(run, dev)
    x = torch.from_numpy(window.astype(np.int64))[None].to(dev)
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        out = model_forward(model, x, cfg["model"], lat)
    return out.bpred[0].boundary_mask.view(-1).cpu().numpy()


def model_piece_ends(bnd, win):
    """A model fires at boundary bytes. Read them the way that puts more cuts at word edges: the boundary byte is the
    last byte of a piece ("end") or the first byte of the next piece ("start", the H-Net convention). Returns
    (piece ends, convention)."""
    ws = np.isin(win, (32, 9, 10, 13))
    best = None
    for conv in ("end", "start"):
        e = bnd.copy() if conv == "end" else np.concatenate((bnd[..., 1:], np.zeros_like(bnd[..., :1])), -1)
        nxt = np.concatenate((ws[..., 1:], np.zeros_like(ws[..., :1])), -1)
        score = (e & (ws ^ nxt)).sum() / max(int(e.sum()), 1)
        if best is None or score > best[0]:
            best = (score, e, conv)
    return best[1], best[2]


def as_text(chars, ends):
    out = []
    for c, e in zip(chars, ends):
        out.append(c)
        if e:
            out.append("|")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("methods", nargs="+", help='"Label=fixed:4" | "Label=bpe" | "Label=runs/<name>"')
    ap.add_argument("--start", type=int, default=39)
    ap.add_argument("--chars", type=int, default=264)
    ap.add_argument("--window", type=int, default=0, help="index of the 8192-byte val window")
    ap.add_argument("--cols", type=int, default=88)
    ap.add_argument("--title", default="How each method cuts the same passage (colour changes at every cut)")
    a = ap.parse_args()
    L = 8192
    v = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    win = v[a.window * L:(a.window + 1) * L]
    s, n = a.start, a.chars
    chars = [chr(b) if 32 <= b < 127 else ("↵" if b == 10 else "·") for b in win[s:s + n]]
    rows = []
    for spec in a.methods:
        label, what = spec.split("=", 1)
        if what.startswith("fixed:"):
            e = ends_fixed(L, int(what.split(":")[1]))
        elif what == "bpe":
            e = ends_bpe(bytes(win))
        else:
            e, conv = model_piece_ends(ends_model(what, win), win)
            print(f"[{label}: the firing byte {'ends a piece' if conv == 'end' else 'starts a piece'}]")
        rate = 100.0 * e.mean()
        seg = e[s:s + n]
        rows.append((label, seg, rate, int(seg.sum())))
        print(f"\n== {label}  ({rate:.1f} pieces per 100 bytes over the window; {int(seg.sum())} cuts in this passage)")
        print(as_text(chars, seg))
    # ---- picture
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    TINT = ("#cfe2f9", "#fbdcc9")
    cols = a.cols
    nlines = -(-n // cols)
    block = nlines + 1.9  # label row + text rows + gap
    H = len(rows) * block + 1.2
    cw = 13.0 / (cols + 2)  # inches per column
    rh = cw * 2.0
    fig = plt.figure(figsize=(13.0, H * rh), facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, cols + 1)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fs = cw * 72 / 0.62 * 0.80
    ax.text(0, 0.65, a.title, fontsize=fs * 1.05, color=INK, va="center", family="DejaVu Sans", weight="bold")
    y0 = 1.5
    for label, seg, rate, ncut in rows:
        ax.text(0, y0 + 0.45, f"{label}", fontsize=fs * 0.95, color=INK, va="center", family="DejaVu Sans", weight="bold")
        ax.text(cols, y0 + 0.45, f"{ncut} cuts here · {rate:.1f} pieces per 100 bytes", fontsize=fs * 0.85, color=INK2,
                va="center", ha="right", family="DejaVu Sans")
        piece = 0
        for i, c in enumerate(chars):
            r, col = divmod(i, cols)
            y = y0 + 1.0 + r
            ax.add_patch(Rectangle((col, y), 1, 0.92, facecolor=TINT[piece % 2], edgecolor="none"))
            ax.text(col + 0.5, y + 0.47, c, fontsize=fs, color=INK if c not in "↵·" else INK2, ha="center", va="center",
                    family="DejaVu Sans Mono")
            if seg[i]:
                piece += 1
        y0 += block
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
