"""Is the answer in the latents? Probe what the decoder reads at each answer byte (scripts/think_diag.py setup: true
reasoning latents as thoughts, answer teacher-forced).

At each scored position t (predicting answer byte t+1) the decoder reads the latent of the chunk after t's chunk.
An MLP probe gets that latent (+ t's offset in its chunk and the byte at t) and predicts byte t+1. Trained on fresh
training problems, scored as exact match on the test problems:
  true  the true next-chunk latent (EMA target encoder): ~100% = the digits are there, the decoder fails to read them
  pred  the trunk's prediction of it: the ceiling for any decoder reading predicted latents
  z     the main-net output the head maps to pred (is the answer computed but lost by the head?)
  none  offset + byte only (no latent)

usage: python scripts/think_probe.py runs/<run> [--n_train 20000] [--n 2000]
Writes runs/<run>/think_probe.json.
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hnet.modules.dc import RoutingModuleOutput  # noqa: E402,F401

from latents import tasks  # noqa: E402
from latents.think import BOS, Thinker  # noqa: E402
from think_diag import teacher  # noqa: E402
from think_train import load  # noqa: E402


@torch.no_grad()
def features(th, probs):
    """Per example: list of (true latent, pred latent, offset, byte at t, byte t+1) over the scored positions."""
    bb, lat = th.bb, th.lat
    teach = teacher(th, probs)
    qb = [bytes([BOS]) + p["q"].encode() for p in probs]
    ctx = [q + p["answer"].encode() for q, p in zip(qb, probs)]
    q_lens = [len(q) for q in qb]
    x = th.batch(ctx)
    B, L = x.shape
    h, cu, bp = th.encode(x)
    idx = torch.tensor([b * L + q for b, q in enumerate(q_lens)], device=x.device)
    mask = bp.boundary_mask.clone()
    mask[idx] = True
    hc, cu1, _, _ = bb.chunk_layer(h, mask, cu)
    mv = mask.view(B, L)
    qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
    rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
    seqs = [torch.cat((th.pad(rows[b][:qn[b]]), th.pad(teach[b].to(hc.dtype)), th.pad(rows[b][qn[b]:])), 0) for b in range(B)]
    z, zcu = th.run_main(seqs)
    zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
    zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + teach[b].shape[0]:]), 0) for b in range(B)], 0)
    tgt = lat.ema_states(th.m, x)[mask].float()
    if getattr(lat, "target_norm", None) == "ln":
        tgt = F.layer_norm(tgt, tgt.shape[-1:])
    cum = mv.long().cumsum(1) - 1
    starts = [torch.nonzero(mv[b]).squeeze(1).tolist() for b in range(B)]
    out = []
    for b in range(B):
        ex = []
        for t in range(q_lens[b] + 4, len(ctx[b]) - 1):
            ci = int(cum[b, t])
            j = cu1[b] + ci
            ex.append((tgt[j + 1], lat.head(zc[j]).float(), t - starts[b][ci], ctx[b][t], ctx[b][t + 1], zc[j].float()))
        out.append(ex)
    return out


def collect(th, probs, bs, ac):
    T, P, O, Bt, Y, E, Z = [], [], [], [], [], [], []
    for i in range(0, len(probs), bs):
        with torch.autocast(**ac):
            fs = features(th, probs[i:i + bs])
        for e, ex in enumerate(fs):
            for tr, pr, off, cur, nxt, zz in ex:
                Z.append(zz); T.append(tr); P.append(pr); O.append(off); Bt.append(cur); Y.append(nxt); E.append(i + e)  # noqa: E702
    st = lambda v: torch.stack(v).float()  # noqa: E731
    tt = lambda v: torch.tensor(v, device=T[0].device)  # noqa: E731
    return dict(true=st(T), pred=st(P), z=st(Z), off=tt(O), cur=tt(Bt), y=tt(Y), ex=tt(E))


class Probe(torch.nn.Module):
    def __init__(self, d, use_lat):
        super().__init__()
        self.use_lat = use_lat
        self.off, self.cur = torch.nn.Embedding(16, 64), torch.nn.Embedding(256, 64)
        self.net = torch.nn.Sequential(torch.nn.Linear((d if use_lat else 0) + 128, 1024), torch.nn.GELU(),
                                       torch.nn.Linear(1024, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 256))

    def forward(self, lat, off, cur):
        f = [self.off(off), self.cur(cur)]
        if self.use_lat:
            f = [lat] + f
        return self.net(torch.cat(f, -1))


def fit(tr, te, key, epochs=30, bs=1024):
    use = key != "none"
    d = tr[key].shape[-1] if use else 0
    pr = Probe(d, use).to(tr["y"].device)
    opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=0.01)
    n = tr["y"].shape[0]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((n + bs - 1) // bs))
    lat_tr = tr[key] if use else None
    for _ in range(epochs):
        perm = torch.randperm(n, device=tr["y"].device)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(pr(lat_tr[j] if use else None, tr["off"][j], tr["cur"][j]), tr["y"][j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        ok = (pr(te[key] if use else None, te["off"], te["cur"]).argmax(-1) == te["y"]).float()
    n_ex = int(te["ex"].max()) + 1
    wrong = torch.zeros(n_ex, device=ok.device).index_add_(0, te["ex"], 1 - ok)
    return dict(exact=float((wrong == 0).float().mean()), byte_acc=float(ok.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    cfg, model, lat = load(run, device)
    model.eval()
    th = Thinker(model, lat, cfg["model"])
    ac = dict(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda")
    rng = random.Random(777)
    train = [tasks.sample(rng, (2, 8)) for _ in range(a.n_train)]
    test = [json.loads(l) for l in open(ROOT / cfg.get("data_dir", "data/task_arith") / "test.jsonl")][:a.n]
    tr, te = collect(th, train, a.batch, ac), collect(th, test, a.batch, ac)
    print("features", tr["true"].shape, te["true"].shape, flush=True)
    res = {"run": run.name, "n_train": a.n_train, "n": a.n}
    for key in ("z", "true", "pred", "none"):
        res[key] = fit(tr, te, key)
        print(key, res[key], flush=True)
    json.dump(res, open(run / "think_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
