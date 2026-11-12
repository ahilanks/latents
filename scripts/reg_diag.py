"""Why does the anti-collapse term blow up on the arithmetic text? (no training)

For one pretrained learned-cut model, the regularizer exactly as training computes it (LatentObjective.reg_loss on the
chunk inputs, plus the same term on the online chunk-local states when the model uses chunk-local targets), split
into its variance part and its covariance part, on four kinds of batches:
  web            FineWeb-Edu val windows of 8 KB (what it was calibrated on in pretraining)
  arith_rows     one problem per row, padded, boundaries forced at the ends of the parts (the reasoning stage)
  arith_packed   the same problems written one after another into 8 KB rows, like pretraining windows
  arith_batch    one problem per row, but the statistics over all chunks of the batch instead of per row
For each: chunks per row, mean within-row std vs the target std (gamma), share of dimensions below gamma, the two
parts of the term, and the gradient size the term sends into the router and the encoder (next to the latent loss's
own, for scale).

usage: python scripts/reg_diag.py runs/<run> [--n 512] [--windows 16] [--batch 64]
Writes runs/<run>/reg_diag.json and prints a table.
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

from latents import tasks  # noqa: E402
from latents import train as T  # noqa: E402
from latents.latent import center_rows, row_ids  # noqa: E402
from latents.models import model_forward  # noqa: E402
from latents.think_lc import make_thinker  # noqa: E402
from think_train import load  # noqa: E402

BOS = 254


def parts(lat, c, cu):
    """The regularizer on chunk latents c with rows cu, split into its parts (as LatentObjective.reg_loss)."""
    big = 10 ** 9
    loss, st = lat.reg_loss(c, cu, big, big)
    cov = st.get("lat/reg_cov", torch.zeros(()))
    x = c.float()
    x = center_rows(x, cu) if lat.reg_scope == "within" else x - x.mean(0)
    std = (x.pow(2).mean(0) + 1e-6).sqrt()
    g = float(st.get("lat/reg_gamma", torch.tensor(float("nan"))))
    _, lens = row_ids(cu, c.shape[0])
    return loss, dict(total=float(loss), var_part=float(loss - lat.beta_cov * cov), cov_part=float(lat.beta_cov * cov),
                      std_mean=float(std.mean()), gamma=g, dims_below_gamma=float((std < g).float().mean()),
                      chunks_per_row=float(lens.float().mean()), n_chunks=int(c.shape[0]))


def grad_norms(loss, model):
    rm = model.backbone.routing_module
    groups = {"router": [p for p in rm.parameters() if p.requires_grad],
              "encoder": [p for p in model.backbone.encoder.parameters() if p.requires_grad],
              "embedding": list(model.embeddings.parameters())}
    out = {}
    for name, ps in groups.items():
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        out[f"grad_{name}"] = float(sum((g.float().pow(2).sum() for g in gs if g is not None), torch.zeros(())) ** 0.5)
    return out


def measure(lat, model, out, x, chunk_valid=None, scope=None):
    """Regularizer parts + gradient sizes for one forward output (chunk inputs and, if used, chunk-local states)."""
    old = lat.reg_scope
    if scope:
        lat.reg_scope = scope
    try:
        cu = out.cu1
        c = out.chunks
        if chunk_valid is not None:
            row, _ = row_ids(cu, c.shape[0])
            n = torch.zeros(cu.numel() - 1, dtype=torch.long, device=row.device).index_add_(0, row, chunk_valid.long())
            cu = torch.cat((n.new_zeros(1), n.cumsum(0))).to(out.cu1.dtype)
            c = c[chunk_valid]
        l1, r = parts(lat, c, cu)
        res = {f"in_{k}": v for k, v in r.items()}
        total = lat.beta_reg * l1
        if lat.target_local and lat.reg_local:
            bm = out.bpred[0].boundary_mask
            y = lat.local_states(model.embeddings, model.backbone.encoder, x, bm, lat.target_local_cap)[bm]
            if chunk_valid is not None:
                y = y[chunk_valid]
            l2, r2 = parts(lat, y, cu)
            res |= {f"local_{k}": v for k, v in r2.items()}
            total = total + lat.beta_reg * l2
        res |= {f"reg_{k}": v for k, v in grad_norms(total, model).items()}
        res["cut_rate"] = float(out.bpred[0].boundary_mask.float().mean())
        return res
    finally:
        lat.reg_scope = old


def latent_grad(lat, model, out, x, chunk_valid=None):
    """Gradient size of the whole latent loss (for scale)."""
    loss, _ = lat.loss(model, out, x, step=10 ** 9, total_steps=10 ** 9, chunk_valid=chunk_valid)
    return {f"latent_{k}": v for k, v in grad_norms(loss, model).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=512, help="arithmetic problems")
    ap.add_argument("--windows", type=int, default=16, help="8 KB web / packed windows")
    ap.add_argument("--batch", type=int, default=64, help="problems per reasoning-stage batch")
    ap.add_argument("--data", default="data/fwe")
    ap.add_argument("--seq_len", type=int, default=8192)
    a = ap.parse_args()
    dev = T.DEV
    cfg, model, lat = load(ROOT / a.run if not Path(a.run).is_absolute() else a.run, dev)
    mc = cfg["model"]
    model.train()
    assert lat.reg != "none", "this model has no anti-collapse term"
    ac = dict(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=dev.startswith("cuda"))
    rng = random.Random(7)
    probs = [tasks.problem(rng, rng.randint(2, 6)) for _ in range(a.n)]
    L = a.seq_len
    rows = {}

    def avg(dicts):
        return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}

    # web
    val = np.fromfile(ROOT / a.data / "val.u8", dtype=np.uint8)
    res = []
    for i in range(0, a.windows, 4):
        x = torch.from_numpy(np.stack([val[j * L:(j + 1) * L] for j in range(i, min(a.windows, i + 4))]).astype(np.int64)).to(dev)
        with torch.autocast(**ac):
            out = model_forward(model, x, mc, lat)
            res.append(measure(lat, model, out, x) | latent_grad(lat, model, out, x))
    rows["web"] = avg(res)
    # arith, one problem per row (the reasoning stage), per-row and batch statistics
    th = make_thinker(model, lat, mc)
    for name, scope in (("arith_rows", None), ("arith_batch", "batch")):
        res = []
        for i in range(0, a.n, a.batch):
            with torch.autocast(**ac):
                f = th.full(probs[i:i + a.batch])
                cv = f.real.flatten()[f.out.bpred[0].boundary_mask]
                res.append(measure(lat, model, f.out, f.x, chunk_valid=cv, scope=scope)
                           | latent_grad(lat, model, f.out, f.x, chunk_valid=cv))
        rows[name] = avg(res)
    # arith packed into 8 KB rows like pretraining windows
    stream = b"".join(bytes([BOS]) + tasks.document(p) for p in probs)
    arr = np.frombuffer(stream, dtype=np.uint8)
    nw = min(a.windows, len(arr) // L)
    res = []
    for i in range(0, nw, 4):
        x = torch.from_numpy(np.stack([arr[j * L:(j + 1) * L] for j in range(i, min(nw, i + 4))]).astype(np.int64)).to(dev)
        with torch.autocast(**ac):
            out = model_forward(model, x, mc, lat)
            res.append(measure(lat, model, out, x) | latent_grad(lat, model, out, x))
    if res:
        rows["arith_packed"] = avg(res)
    json.dump(rows, open(Path(a.run) / "reg_diag.json", "w"), indent=1)
    cols = [k for k in rows["web"]]
    print("| quantity | " + " | ".join(rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    for k in cols:
        print(f"| {k} | " + " | ".join("–" if k not in r else f"{r[k]:.4g}" for r in rows.values()) + " |")


if __name__ == "__main__":
    main()
