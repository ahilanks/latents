"""Which loss terms push the router toward fewer cuts? (one backward pass per term, no training)

For a learned-cut run (rebuilt at initialisation with its seed, and trained), on web val windows: the gradient of each
loss term with respect to every byte's cut probability p_t, summarised as
  push = -mean_t dL/dp_t   (> 0: gradient descent raises p = more cuts; < 0: fewer cuts)
  push_cut / push_noncut   the same over current cut bytes / other bytes
Terms: the latent prediction loss averaged over pieces (as trained) and weighted per byte (latent.loss_weight=len),
the anti-collapse term, and the ratio loss at weight 0.03 and 1.0.
usage: python scripts/cut_pressure.py runs/RT4n6r-s0 [--windows 2]
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

from hnet.utils.train import load_balancing_loss  # noqa: E402
from latents.models import model_forward  # noqa: E402
from router_diag import load  # noqa: E402


def pressures(cfg, model, lat, x):
    model.train()
    rm = model.backbone.routing_module
    keep = {}
    orig = rm.forward

    def fwd(*a, **k):
        o = orig(*a, **k)
        o.boundary_prob.retain_grad()
        keep["bp"] = o
        return o
    rm.forward = fwd
    res = {}
    terms = {
        "latent_per_piece": lambda out: lat_loss(lat, model, out, x, "chunk", reg=False),
        "latent_per_byte": lambda out: lat_loss(lat, model, out, x, "len", reg=False),
        "anti_collapse": lambda out: lat.beta_reg * lat.reg_loss(out.chunks, out.cu1, 10 ** 9, 10 ** 9)[0],
        "ratio_w0.03": lambda out: 0.03 * load_balancing_loss(keep["bp"], N=cfg["model"]["N"]),
        "ratio_w1.0": lambda out: 1.0 * load_balancing_loss(keep["bp"], N=cfg["model"]["N"]),
    }
    try:
        for name, fn in terms.items():
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model_forward(model, x, cfg["model"], lat)
                loss = fn(out)
            loss.backward()
            g = keep["bp"].boundary_prob.grad[..., -1].float().view(-1)
            m = keep["bp"].boundary_mask.view(-1)
            res[name] = dict(loss=float(loss), push=float(-g.mean()) * 1e3, push_cut=float(-g[m].mean()) * 1e3,
                             push_noncut=float(-g[~m].mean()) * 1e3, cut_rate=float(m.float().mean()))
    finally:
        rm.forward = orig
    return res


def lat_loss(lat, model, out, x, weight, reg):
    old_w, old_r = lat.loss_weight, lat.reg
    lat.loss_weight = weight
    lat.reg = lat.reg if reg else "none"
    try:
        loss, _ = lat.loss(model, out, x, step=10 ** 9, total_steps=10 ** 9)
    finally:
        lat.loss_weight, lat.reg = old_w, old_r
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--windows", type=int, default=2)
    a = ap.parse_args()
    val = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    x = torch.from_numpy(val[: a.windows * 8192].reshape(a.windows, 8192).astype(np.int64)).cuda()
    for run in a.runs:
        for init in (True, False):
            cfg, model, lat = load(run, init=init)
            res = pressures(cfg, model, lat, x)
            print(f"== {Path(run).name} {'at initialisation' if init else 'trained'}  (push x1e3: + = more cuts)")
            print(f"   {'term':18s} {'loss':>8s} {'push all':>9s} {'on cuts':>9s} {'on others':>9s}")
            for k, v in res.items():
                print(f"   {k:18s} {v['loss']:8.4f} {v['push']:9.4f} {v['push_cut']:9.4f} {v['push_noncut']:9.4f}")
            print(f"   cut rate {next(iter(res.values()))['cut_rate']:.3f}", flush=True)
            del model, lat
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
