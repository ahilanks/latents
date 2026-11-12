"""Word probes that do not depend on where a model fires inside a word.

The eval suite's probes label a chunk by the word at its firing byte. A model that fires on the first letter of the next
word (the learned-cut latent models) is then asked about the wrong word. Here a chunk is the segment of bytes it
covers, S_j = (t_{j-1}, t_j], and its label is the word that has the most bytes inside it:
  segment_word        linear probe from the chunk's input summary (out.chunks)      -> the word chunk j is mostly about
  next_segment_word   linear probe from the model's prediction at chunk j           -> the word chunk j+1 is mostly about
                      (latent models: head(main-net output) = the predicted next-chunk latent; normal H-Nets: the
                      main-net output itself)
Words = maximal runs of non-whitespace bytes, lower-cased; the 1000 most frequent labels; 80/20 split; accuracy and the
majority baseline. Same probe training as scripts/eval_suite.py.

usage: python scripts/segment_probe.py out.json "Label=runs/<name>" ... [--windows 48]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from show_cuts import load_any  # noqa: E402

L = 8192
WS = (32, 10, 9, 13)


def word_ids(text, words):
    """Per byte: id of the word it belongs to (-1 for whitespace)."""
    ids = np.full(len(text), -1, dtype=np.int64)
    i = 0
    n = len(text)
    while i < n:
        if text[i] in WS:
            i += 1
            continue
        j = i
        while j < n and text[j] not in WS:
            j += 1
        ids[i:j] = words.setdefault(bytes(text[i:j]).lower(), len(words))
        i = j
    return ids


def fit_probe(X, Y, n_cls=1000):
    Y = torch.tensor(Y)
    top = torch.bincount(Y[Y >= 0]).argsort(descending=True)[:n_cls]
    keep = torch.isin(Y, top)
    Xk, Yk = torch.stack(X)[keep], Y[keep]
    remap = {int(w): i for i, w in enumerate(top)}
    Yk = torch.tensor([remap[int(y)] for y in Yk])
    n = len(Yk)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    clf = torch.nn.Linear(Xk.shape[1], len(top))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    mu, sd = Xk[tr].mean(0), Xk[tr].std(0) + 1e-6
    Z = (Xk - mu) / sd
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(clf(Z[tr]), Yk[tr]).backward()
        opt.step()
    acc = float((clf(Z[te]).argmax(-1) == Yk[te]).float().mean())
    maj = float((Yk[te] == torch.bincount(Yk[tr]).argmax()).float().mean())
    return acc, maj, n


@torch.no_grad()
def collect(run, wins):
    from latents.models import model_forward
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, lat = load_any(run, dev)
    feats, nfeats, cur, nxt = [], [], [], []
    words = {}
    for w0 in range(0, len(wins), 4):
        xs = np.stack(wins[w0:w0 + 4]).astype(np.int64)
        x = torch.from_numpy(xs).to(dev)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model_forward(model, x, cfg["model"], lat)
            pred = lat.head(o.z_full).float().cpu() if lat is not None else o.z.float().cpu()
        ch = o.chunks.float().cpu()
        mask = o.bpred[0].boundary_mask.view(x.shape).cpu().numpy()
        k = 0
        for r in range(xs.shape[0]):
            wid = word_ids(xs[r], words)
            ends = np.nonzero(mask[r])[0]
            labs, prev = [], -1
            for t in ends:
                seg = wid[prev + 1: t + 1]
                seg = seg[seg >= 0]
                labs.append(int(np.bincount(seg).argmax()) if len(seg) else -1)
                prev = t
            for i in range(len(ends)):
                feats.append(ch[k + i])
                cur.append(labs[i])
                if i + 1 < len(ends):
                    nfeats.append(pred[k + i])
                    nxt.append(labs[i + 1])
            k += len(ends)
    return feats, cur, nfeats, nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--windows", type=int, default=48)
    a = ap.parse_args()
    v = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    wins = [v[i * L:(i + 1) * L] for i in range(a.windows)]
    res = {}
    for spec in a.methods:
        label, run = spec.split("=", 1)
        feats, cur, nfeats, nxt = collect(run, wins)
        a1, m1, n1 = fit_probe(feats, cur)
        a2, m2, n2 = fit_probe(nfeats, nxt)
        res[label] = dict(segment_word=a1, segment_word_majority=m1, next_segment_word=a2, next_segment_word_majority=m2,
                          n=n1, source=run)
        print(f"{label:40s} this chunk's word {100 * a1:5.1f}% (majority {100 * m1:4.1f})   "
              f"next chunk's word {100 * a2:5.1f}% (majority {100 * m2:4.1f})   n={n1}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
