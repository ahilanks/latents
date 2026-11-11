"""Where in the trunk is the answer value lost? (true reasoning latents as thoughts, like scripts/think_diag.py)
A probe predicts the answer value (0-99) from:
  thoughts   the last 4 thought inputs (true reasoning-chunk latents, concatenated): is the value in the inputs?
  z_last     main-net output at the last thought (it predicts the "####" chunk)
  z_ans      main-net output at the "####" chunk (it predicts the chunk with the first digit)
  z_read     main-net output at the chunk the decoder reads for the first digit (the "####" chunk or the next)
  pred_read  head(z_read): the predicted latent the decoder reads there
usage: python scripts/think_probe_value.py runs/<run> [--n_train 20000] [--n 2000]
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

from latents import tasks  # noqa: E402
from latents.think import BOS, Thinker  # noqa: E402
from think_diag import teacher  # noqa: E402
from think_train import load  # noqa: E402


@torch.no_grad()
def feats(th, probs):
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
    hc, cu1, _, _ = th.bb.chunk_layer(h, mask, cu)
    mv = mask.view(B, L)
    qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
    rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
    seqs = [torch.cat((th.pad(rows[b][:qn[b]]), th.pad(teach[b].to(hc.dtype)), th.pad(rows[b][qn[b]:])), 0) for b in range(B)]
    z, zcu = th.run_main(seqs)
    out = {"thoughts": [], "z_last": [], "z_ans": [], "z_read": [], "pred_read": []}
    cum = mv.long().cumsum(1) - 1
    for b in range(B):
        k = teach[b].shape[0]
        zb = z[zcu[b]:zcu[b + 1]]
        r = int(cum[b, q_lens[b] + 4]) - qn[b]  # chunk the decoder reads for the first digit (rel. to "####")
        out["thoughts"].append(teach[b][-4:].flatten().float())
        out["z_last"].append(zb[qn[b] + k - 1].float())
        out["z_ans"].append(zb[qn[b] + k].float())
        out["z_read"].append(zb[qn[b] + k + r].float())
        out["pred_read"].append(th.lat.head(zb[qn[b] + k + r]).float())
    return out


def collect(th, probs, bs, ac):
    acc = {}
    for i in range(0, len(probs), bs):
        with torch.autocast(**ac):
            f = feats(th, probs[i:i + bs])
        for k, v in f.items():
            acc.setdefault(k, []).extend(v)
    res = {k: torch.stack(v) for k, v in acc.items()}
    res["y"] = torch.tensor([p["value"] for p in probs], device=res["z_ans"].device)
    return res


def fit(xtr, ytr, xte, yte, epochs=40, bs=512):
    pr = torch.nn.Sequential(torch.nn.LayerNorm(xtr.shape[-1]), torch.nn.Linear(xtr.shape[-1], 1024), torch.nn.GELU(),
                             torch.nn.Linear(1024, 100)).to(xtr.device)
    opt = torch.optim.AdamW(pr.parameters(), lr=1e-3, weight_decay=0.01)
    n = xtr.shape[0]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((n + bs - 1) // bs))
    for _ in range(epochs):
        perm = torch.randperm(n, device=xtr.device)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(pr(xtr[j]), ytr[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        return float((pr(xte).argmax(-1) == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    device = "cuda"
    run = ROOT / a.run
    cfg, model, lat = load(run, device)
    model.eval()
    th = Thinker(model, lat, cfg["model"])
    ac = dict(device_type="cuda", dtype=torch.bfloat16)
    rng = random.Random(777)
    tr = collect(th, [tasks.sample(rng, (2, 8)) for _ in range(a.n_train)], 100, ac)
    te = collect(th, [json.loads(l) for l in open(ROOT / cfg.get("data_dir", "data/task_arith") / "test.jsonl")][:a.n], 100, ac)
    res = {"run": run.name}
    for k in ("thoughts", "z_last", "z_ans", "z_read", "pred_read"):
        res[k] = fit(tr[k], tr["y"], te[k], te["y"])
        print(k, round(res[k], 4), flush=True)
    json.dump(res, open(run / "think_probe_value.json", "w"), indent=1)


if __name__ == "__main__":
    main()
