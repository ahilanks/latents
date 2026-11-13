"""Which loss terms make consecutive encoder states alike? (one backward pass per term, no training)

For each term L, g = dL/dh on the encoder output h (bytes), and the change of C = mean_t cos(h_t, h_t+1) under a small
gradient-descent step on h: dC = -<dC/dh, g> (x1e3; > 0: the term pulls consecutive states together). Also |g|.
Terms: latent prediction loss (as trained), anti-collapse, ratio loss (0.03 / 1.0), next-byte decoder loss (blocked in
the latent recipe: shows whether it reaches the encoder at all), and the per-byte latent loss for runs that have it.
usage: python scripts/smooth_pressure.py runs/A [runs/B ...] [--init]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hnet.utils.train import load_balancing_loss  # noqa: E402
from latents.models import model_forward  # noqa: E402
from router_diag import load  # noqa: E402


def run(cfg, model, lat, x):
    model.train()
    enc = model.backbone.encoder
    keep = {}
    orig = enc.forward

    def fwd(*a, **k):
        h = orig(*a, **k)
        h.retain_grad()
        keep["h"] = h
        return h
    enc.forward = fwd
    rm = model.backbone.routing_module
    ro = rm.forward

    def rfwd(*a, **k):
        o = ro(*a, **k)
        keep["bp"] = o
        return o
    rm.forward = rfwd
    N = cfg["model"]["N"]
    terms = {"latent_loss": lambda out: lat_only(lat, model, out, x, reg=False, byte=False),
             "anti_collapse": lambda out: lat.beta_reg * lat.reg_loss(out.chunks, out.cu1, 10 ** 9, 10 ** 9)[0],
             "ratio_w0.03": lambda out: 0.03 * load_balancing_loss(keep["bp"], N=N),
             "ratio_w1.0": lambda out: load_balancing_loss(keep["bp"], N=N),
             "nextbyte_decoder": lambda out: F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten())}
    if getattr(lat, "byte_loss", False):
        terms["per_byte_latent"] = lambda out: lat.byte_level_loss(model, out, lat.head(out.z_full), {})
    res = {}
    try:
        for name, fn in terms.items():
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model_forward(model, x, cfg["model"], lat)
                loss = fn(out)
            if not loss.requires_grad:
                res[name] = (float(loss), 0.0, 0.0)
                continue
            loss.backward()
            h = keep["h"]
            g = h.grad
            if g is None:
                res[name] = (float(loss), 0.0, 0.0)
                continue
            hv = h.detach().float().requires_grad_(True)
            hh = hv.view(x.shape[0], x.shape[1], -1)
            C = F.cosine_similarity(hh[:, 1:], hh[:, :-1], dim=-1).mean()
            (dC,) = torch.autograd.grad(C, hv)
            res[name] = (float(loss), float(-(dC * g.float()).sum()) * 1e3, float(g.float().norm()))
    finally:
        enc.forward, rm.forward = orig, ro
    return res


def lat_only(lat, model, out, x, reg, byte):
    old = (lat.reg, getattr(lat, "byte_loss", False))
    lat.reg = lat.reg if reg else "none"
    lat.byte_loss = byte
    try:
        loss, _ = lat.loss(model, out, x, step=10 ** 9, total_steps=10 ** 9)
    finally:
        lat.reg, lat.byte_loss = old
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--init", action="store_true", help="also at initialisation")
    a = ap.parse_args()
    val = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    x = torch.from_numpy(val[: 2 * 8192].reshape(2, 8192).astype(np.int64)).cuda()
    for run_ in a.runs:
        for init in ([True, False] if a.init else [False]):
            cfg, model, lat = load(run_, init=init)
            res = run(cfg, model, lat, x)
            print(f"== {Path(run_).name} {'at initialisation' if init else 'trained'}  (pull together x1e3: + = more alike)")
            for k, (l, p, gn) in res.items():
                print(f"   {k:18s} loss {l:8.4f}  pull-together {p:10.4f}  |grad on h| {gn:.4f}")
            del model, lat
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
