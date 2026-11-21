"""Is the predicted next-chunk latent a meaningful summary of what could come next, or close to a constant? (no training)

Squared error makes the best prediction the average latent of the possible next chunks. That average is useful only if
the target space is arranged by meaning (mat / rug / floor close together). This measures, for each latent model, on
web text and on the arithmetic chains (whose next step is fully determined):

  spread_ratio      typical spread of the predictions / spread of the targets (1 = as varied as the targets)
  r2_head           share of the targets' variation the prediction explains (1 = perfect, 0 = no better than the mean)
  r2_rescaled       the same after the best single rescaling of the prediction (high here but low r2_head = the
                    prediction points the right way but is shrunk toward the mean)
  r2_copy           the "next chunk = this chunk" baseline
  r2_linear         a fresh linear map from the predictor's features to the target, fit on half and scored on the
                    other half: how much of the target is predictable from the context representation at all
  dirs_r2_gt_0.1    how many of the targets' top-64 principal directions the prediction explains above 0.1, and
  predictable_share the share of the targets' variation that lies in those directions
  digit_r2_head     (chains) the same r2 on chunks that contain a digit (the computed values)
Web text only, word geometry of the target space (a chunk's word = the word with most bytes in its segment; the 300 most
frequent words that occur at least 5 times):
  same_word_cos / diff_word_cos   mean cosine between targets of the same word vs different words (centred)
  next_word_top1 / top10          rank the 300 word centroids by cosine to the prediction: is the true next word first /
                                  in the first ten? (chance 0.3% / 3.3%; next to: the same ranking with the most frequent
                                  words, and with a prediction from another position)
  neighbours                      the nearest word centroids of a few probe words (are interchangeable words close?)
  examples                        a few contexts: the prediction's nearest words vs the word that came

usage: python scripts/latent_geometry.py out.json "Label=runs/<run>" ... [--windows 32] [--plot out.png]
Models without a latent head (normal H-Nets) are skipped.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents import tasks  # noqa: E402
from show_cuts import load_any  # noqa: E402

WS = (32, 10, 9, 13)
BOS = 254
PROBE_WORDS = ["the", "and", "water", "people", "said", "three", "school", "children", "however", "because"]


def word_ids(text, words):
    ids = np.full(len(text), -1, dtype=np.int64)
    i, n = 0, len(text)
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


@torch.no_grad()
def collect(cfg, model, lat, wins, dev, words):
    """Pairs (prediction at chunk j, target of chunk j+1, predictor features at j) within each row, with chunk j+1's
    word label, whether its segment has a digit, and the text around it."""
    from latents.models import model_forward
    P, Tg, Z, lab, dig, ctx, nxt_txt = [], [], [], [], [], [], []
    for w0 in range(0, len(wins), 4):
        xs = np.stack(wins[w0:w0 + 4]).astype(np.int64)
        x = torch.from_numpy(xs).to(dev)
        with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                            enabled=dev.startswith("cuda")):
            o = model_forward(model, x, cfg["model"], lat)
            pred = lat.head(o.z_full).float()
            tgt = lat.targets(model, o, x).float()
        if getattr(lat, "target_norm", "none") == "ln":
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        pred, tgt, z = pred.cpu(), tgt.cpu(), o.z_full.float().cpu()
        mask = o.bpred[0].boundary_mask.view(x.shape).cpu().numpy()
        k = 0
        for r in range(xs.shape[0]):
            text = xs[r]
            wid = word_ids(text, words)
            ends = np.nonzero(mask[r])[0]
            starts = np.concatenate(([0], ends[:-1] + 1))
            for i in range(len(ends) - 1):
                s, e = starts[i + 1], ends[i + 1]
                seg = wid[s:e + 1]
                seg = seg[seg >= 0]
                P.append(pred[k + i])
                Tg.append(tgt[k + i + 1])
                Z.append(z[k + i])
                lab.append(int(np.bincount(seg).argmax()) if len(seg) else -1)
                piece = bytes(text[s:e + 1])
                dig.append(any(48 <= b <= 57 for b in piece))
                ctx.append(bytes(text[max(0, ends[i] - 60):ends[i] + 1]))
                nxt_txt.append(piece)
            k += len(ends)
    return dict(P=torch.stack(P), T=torch.stack(Tg), Z=torch.stack(Z), lab=np.array(lab), dig=np.array(dig),
                ctx=ctx, nxt=nxt_txt)


def r2(pred, tgt):
    mu = tgt.mean(0)
    return float(1 - (pred - tgt).pow(2).sum() / (tgt - mu).pow(2).sum().clamp(min=1e-12))


def predictability(d):
    P, T, Z = d["P"].double(), d["T"].double(), d["Z"].double()
    n = len(T)
    mu = T.mean(0)
    res = dict(n=n, spread_ratio=float(P.std(0).mean() / T.std(0).mean().clamp(min=1e-12)), r2_head=r2(P, T))
    pc, tc = P - P.mean(0), T - mu
    a = float((pc * tc).sum() / (pc * pc).sum().clamp(min=1e-12))
    res["rescale"] = a
    res["r2_rescaled"] = r2(mu + a * pc, T)
    res["r2_copy"] = r2(torch.roll(T, 1, 0), T)  # previous chunk's target (row ends add a little noise)
    # a fresh ridge map from the predictor's features: fit on the first half, score on the second
    h = n // 2
    Xc, Yc = Z[:h] - Z[:h].mean(0), T[:h] - T[:h].mean(0)
    lam = 1e-3 * float((Xc * Xc).sum() / h)
    W = torch.linalg.solve(Xc.T @ Xc / h + lam * torch.eye(Z.shape[1], dtype=Z.dtype), Xc.T @ Yc / h)
    Yh = (Z[h:] - Z[:h].mean(0)) @ W + T[:h].mean(0)
    res["r2_linear"] = r2(Yh, T[h:])
    # per principal direction of the targets
    ev, V = torch.linalg.eigh(tc.T @ tc / n)
    V, ev = V[:, ev.argsort(descending=True)], ev.sort(descending=True).values
    K = min(64, V.shape[1])
    pt, pp = tc @ V[:, :K], (P - mu) @ V[:, :K]
    r2d = 1 - (pp - pt).pow(2).sum(0) / pt.pow(2).sum(0).clamp(min=1e-12)
    res["r2_by_direction"] = [float(v) for v in r2d]
    good = r2d > 0.1
    res["dirs_r2_gt_0.1"] = int(good.sum())
    res["predictable_share"] = float(ev[:K][good].sum() / ev.sum().clamp(min=1e-12))
    res["top_direction_share"] = float(ev[0] / ev.sum().clamp(min=1e-12))
    if d["dig"].any():
        m = torch.from_numpy(d["dig"])
        res["digit_r2_head"] = r2(P[m], T[m])
        res["digit_n"] = int(m.sum())
    return res


def word_geometry(d, words, n_words=300, min_count=5, seed=0):
    inv = {v: k.decode("utf-8", "replace") for k, v in words.items()}
    lab = d["lab"]
    ok = lab >= 0
    cnt = np.bincount(lab[ok])
    top = [w for w in np.argsort(-cnt)[:n_words] if cnt[w] >= min_count]
    T = d["T"].double()
    mu = T.mean(0)
    Tc = F.normalize(T - mu, dim=-1)
    labt = torch.from_numpy(lab)
    cent = torch.stack([F.normalize((T[labt == int(w)] - mu).mean(0), dim=0) for w in top])
    pos = {w: i for i, w in enumerate(top)}
    keep = np.array([l in pos for l in lab])
    idx = np.nonzero(keep)[0]
    y = np.array([pos[l] for l in lab[keep]], dtype=np.int64)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(seed)
    # same word vs different words (sampled pairs)
    a, b = rng.integers(0, len(idx), 20000), rng.integers(0, len(idx), 20000)
    cos = (Tc[torch.from_numpy(idx[a])] * Tc[torch.from_numpy(idx[b])]).sum(-1).numpy()
    same = y[a] == y[b]
    by_word = {}
    for i, w in zip(idx, y):
        by_word.setdefault(w, []).append(i)
    sa, sb = [], []
    for _ in range(20000):
        w = rng.integers(0, len(top))
        if len(by_word.get(w, [])) >= 2:
            i, j = rng.choice(by_word[w], 2, replace=False)
            sa.append(i)
            sb.append(j)
    same_cos = float((Tc[torch.tensor(sa)] * Tc[torch.tensor(sb)]).sum(-1).mean()) if sa else float("nan")
    res = dict(n_words=len(top), same_word_cos=same_cos, diff_word_cos=float(cos[~same].mean()))
    # next-word ranking by the prediction
    Pc = F.normalize(d["P"].double()[torch.from_numpy(idx)] - mu, dim=-1)
    sims = Pc @ cent.T
    ar = torch.arange(len(y))
    rank = (sims > sims[ar, yt][:, None]).sum(1).numpy()
    res["next_word_top1"] = float((rank == 0).mean())
    res["next_word_top10"] = float((rank < 10).mean())
    freq = np.argsort(-np.bincount(y, minlength=len(top)))
    res["most_frequent_top1"] = float((y == freq[0]).mean())
    res["most_frequent_top10"] = float(np.isin(y, freq[:10]).mean())
    sh = sims[torch.from_numpy(rng.permutation(len(y)))]
    rank_sh = (sh > sh[ar, yt][:, None]).sum(1).numpy()
    res["other_position_top10"] = float((rank_sh < 10).mean())
    # neighbours of probe words in the target space
    nb = {}
    csim = cent @ cent.T
    for pw in PROBE_WORDS:
        w = words.get(pw.encode())
        if w is not None and w in pos:
            order = torch.argsort(csim[pos[w]], descending=True)[1:9]
            nb[pw] = [inv[top[int(i)]] for i in order]
    res["neighbours"] = nb
    ex = []
    for i in rng.choice(len(idx), size=min(6, len(idx)), replace=False):
        order = torch.argsort(sims[i], descending=True)[:6]
        ex.append(dict(context=d["ctx"][idx[i]].decode("utf-8", "replace").replace("\n", " "),
                       came=d["nxt"][idx[i]].decode("utf-8", "replace"),
                       predicted_near=[inv[top[int(o)]] for o in order]))
    res["examples"] = ex
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("methods", nargs="+", help='"Label=runs/<run>"')
    ap.add_argument("--windows", type=int, default=32, help="windows of each text kind")
    ap.add_argument("--data", default="data/fwe")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(7)
    chains = b"".join(bytes([BOS]) + tasks.document(tasks.problem(rng, rng.randint(2, 6))) for _ in range(20000))
    val = np.fromfile(ROOT / a.data / "val.u8", dtype=np.uint8)
    res = {}
    for spec in a.methods:
        label, run = spec.split("=", 1)
        cfg, model, lat = load_any(run, dev)
        if lat is None:
            print(f"{label}: no latent head, skipped")
            continue
        L = cfg["train"]["seq_len"]
        arr = np.frombuffer(chains, dtype=np.uint8)
        kinds = {"web": [val[i * L:(i + 1) * L] for i in range(min(a.windows, len(val) // L))],
                 "chains": [arr[i * L:(i + 1) * L] for i in range(min(a.windows, len(arr) // L))]}
        res[label] = {"run": run}
        for kind, wins in kinds.items():
            words = {}
            d = collect(cfg, model, lat, wins, dev, words)
            out = predictability(d)
            if kind == "web":
                out |= word_geometry(d, words)
            res[label][kind] = out
            print(f"{label:28s} {kind:6s} spread {out['spread_ratio']:.3f}  r2 {out['r2_head']:+.3f}  rescaled "
                  f"{out['r2_rescaled']:+.3f}  copy {out['r2_copy']:+.3f}  linear {out['r2_linear']:+.3f}  "
                  f"dirs>0.1 {out['dirs_r2_gt_0.1']:2d}/64  share {out['predictable_share']:.3f}"
                  + (f"  digits r2 {out['digit_r2_head']:+.3f}" if "digit_r2_head" in out else "")
                  + (f"  next word top1/top10 {out['next_word_top1']:.3f}/{out['next_word_top10']:.3f}"
                     f" (frequent {out['most_frequent_top1']:.3f}/{out['most_frequent_top10']:.3f})" if kind == "web" else ""),
                  flush=True)
        del model, lat
    json.dump(res, open(a.out, "w"), indent=1)
    if a.plot and res:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        for label, r in res.items():
            for kind, ls in (("web", "-"), ("chains", "--")):
                if kind in r:
                    ax.plot(range(1, len(r[kind]["r2_by_direction"]) + 1), r[kind]["r2_by_direction"], ls,
                            label=f"{label} ({kind})", lw=1.5)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("target direction (1 = the one with the most variation)")
        ax.set_ylabel("share of variation the prediction explains")
        ax.set_ylim(-0.5, 1.0)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
