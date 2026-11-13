"""Why are the learned-cut router's probabilities so firm? Checks that need no training (2026-11-12 review).

For each run, on web val windows:
  logit(p) histogram (final, and at initialisation: the same config rebuilt with the run's seed)
  cut rate vs mean p (the ratio loss wants them equal)
  overlap of the final cuts with the cuts at initialisation (F1; ~1 = cuts fixed at init)
  where cuts land: share of cuts on the byte after whitespace / on whitespace; share of bytes that are whitespace
  effective rank (participation ratio) of the encoder's byte vectors h, q(h), k(h) and of their unit-normalised versions
  the characters in a probability band (e.g. the rollout model's 0.2-0.3 bump)
  router gradient norms by path, from one backward pass (latent-objective runs): the latent loss as trained
  (pooling), and with the blocked paths reopened, the next-byte loss through the confidence scaling and through the
  smoothing separately
usage: python scripts/router_diag.py runs/A runs/B ... [--windows 4] [--band 0.2,0.3] [--out json]
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
import torch.nn.functional as F  # noqa: E402

from latents import latent as latent_lib  # noqa: E402
from latents.models import build_model, model_forward  # noqa: E402

DEV = "cuda"
WS = (32, 9, 10, 13)


def load(run, init=False):
    ck = torch.load(ROOT / run / "final.pt", map_location=DEV, weights_only=False)
    cfg = ck["cfg"]
    torch.manual_seed(cfg["train"].get("seed", 0))
    model = build_model(cfg["model"], device=DEV)
    lat = latent_lib.build(cfg["latent"], model, cfg["model"]) if cfg.get("latent") else None
    if not init:
        model.load_state_dict(ck["model"])
        if lat is not None:
            lat.load_state_dict(ck["latent"], strict=False)
    return cfg, model, lat


def part_ratio(x):
    x = x.float() - x.float().mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x / max(x.shape[0] - 1, 1)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))


def router_inputs(model):
    """Capture the encoder output h that the routing module sees."""
    store = {}
    rm = model.backbone.routing_module

    def hook(mod, args, kwargs):
        store["h"] = args[0].detach()
    hd = rm.register_forward_pre_hook(hook, with_kwargs=True)
    return store, hd


@torch.no_grad()
def stats(cfg, model, lat, x):
    store, hd = router_inputs(model)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model_forward(model, x, cfg["model"], lat)
    hd.remove()
    bp = out.bpred[0]
    p = bp.boundary_prob[..., -1].float().view(-1)
    m = bp.boundary_mask.view(-1)
    h = store["h"].reshape(-1, store["h"].shape[-1]).float()
    rm = model.backbone.routing_module
    q, k = rm.q_proj_layer(h.to(rm.q_proj_layer.weight.dtype)).float(), rm.k_proj_layer(h.to(rm.k_proj_layer.weight.dtype)).float()
    sub = torch.randperm(h.shape[0], device=h.device)[:20000]
    ranks = {f"rank_{n}": part_ratio(v[sub]) for n, v in (("h", h), ("q", q), ("k", k))}
    ranks |= {f"rank_{n}_unit": part_ratio(F.normalize(v[sub], dim=-1)) for n, v in (("h", h), ("q", q), ("k", k))}
    return p.cpu().numpy(), m.cpu().numpy(), ranks


def grad_paths(cfg, model, lat, x):
    """Router gradient norms by path (one backward pass each)."""
    rm = model.backbone.routing_module
    params = [rm.q_proj_layer.weight, rm.k_proj_layer.weight]
    bb = model.backbone
    res = {}
    y = x[:, 1:]

    def gnorm(loss):
        gs = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
        return float(sum((g.float().pow(2).sum() for g in gs if g is not None), torch.zeros((), device=DEV)) ** 0.5)

    def ce(out):
        return F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), y.flatten())

    model.train()
    # 1) as trained: the latent loss (pooling path) + ratio loss
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model_forward(model, x, cfg["model"], lat)
        l_lat, _ = lat.loss(model, out, x, step=10 ** 9, total_steps=10 ** 9)
    res["grad_latent_loss_as_trained"] = gnorm(l_lat)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model_forward(model, x, cfg["model"], lat)
    res["grad_nextbyte_as_trained"] = gnorm(ce(out))  # should be 0: blocked
    # 2) reopened: next-byte loss through confidence only / smoothing only
    old_dc, old_rf, old_dt = bb.dechunk_layer.forward, bb.residual_func, lat.cfg.get("detach_trunk")
    lat.cfg["detach_trunk"] = False
    try:
        for name, det_prob, det_sel in (("confidence", True, False), ("smoothing", False, True)):
            def dc(z, mask, prob, cu1, *a, _f=old_dc, _d=det_prob, **kw):
                return _f(z, mask, prob.detach() if _d else prob, cu1, *a, **kw)

            def rf(out_, residual, sel, _f=old_rf, _d=det_sel):
                return _f(out_, residual, sel.detach() if _d else sel)
            bb.dechunk_layer.forward, bb.residual_func = dc, rf
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model_forward(model, x, cfg["model"], lat)
            res[f"grad_nextbyte_via_{name}"] = gnorm(ce(out))
    finally:
        bb.dechunk_layer.forward, bb.residual_func = old_dc, old_rf
        lat.cfg["detach_trunk"] = old_dt
    model.eval()
    return res


def logit_hist(p, edges):
    lp = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    return np.histogram(lp, bins=edges)[0] / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--band", default="0.2,0.3")
    ap.add_argument("--out", default="logs/router_diag.json")
    ap.add_argument("--no_grad", action="store_true")
    a = ap.parse_args()
    val = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    L = 8192
    xb = val[: a.windows * L].reshape(a.windows, L)
    x = torch.from_numpy(xb.astype(np.int64)).to(DEV)
    flat = xb.reshape(-1)
    ws = np.isin(flat, WS)
    after_ws = np.concatenate(([True], ws[:-1]))
    edges = np.arange(-16, 16.01, 1.0)
    lo, hi = (float(v) for v in a.band.split(","))
    results = {}
    for run in a.runs:
        name = Path(run).name
        r = {}
        cfg, model, lat = load(run, init=True)
        model.eval()
        p0, m0, rk0 = stats(cfg, model, lat, x)
        del model, lat
        cfg, model, lat = load(run)
        model.eval()
        p, m, rk = stats(cfg, model, lat, x)
        r["cut_rate"], r["mean_p"] = float(m.mean()), float(p.mean())
        r["cut_rate_init"], r["mean_p_init"] = float(m0.mean()), float(p0.mean())
        both = (m & m0).sum()
        r["f1_final_vs_init"] = float(2 * both / max(m.sum() + m0.sum(), 1))
        r["cuts_after_ws"] = float((m & after_ws).sum() / m.sum())
        r["cuts_on_ws"] = float((m & ws).sum() / m.sum())
        r["cuts_after_or_on_ws"] = float((m & (ws | after_ws)).sum() / m.sum())
        r["bytes_ws"] = float(ws.mean())
        r["ws_followed_by_cut"] = float(m[1:][ws[:-1]].mean())
        r |= {k + "_final": v for k, v in rk.items()} | {k + "_init": v for k, v in rk0.items()}
        r["logit_hist_final"] = logit_hist(p, edges).tolist()
        r["logit_hist_init"] = logit_hist(p0, edges).tolist()
        r["p_quantiles_noncut"] = np.quantile(p[~m], [0.5, 0.9, 0.99]).tolist()
        r["p_quantiles_cut"] = np.quantile(p[m], [0.01, 0.1, 0.5]).tolist()
        band = (p >= lo) & (p < hi)
        r["band_share"] = float(band.mean())
        if band.any():
            vals, cnt = np.unique(flat[band], return_counts=True)
            top = np.argsort(-cnt)[:8]
            r["band_chars"] = {repr(chr(vals[i])): round(float(cnt[i] / band.sum()), 3) for i in top}
            prev = np.concatenate(([0], flat[:-1]))[band]
            vals, cnt = np.unique(prev, return_counts=True)
            top = np.argsort(-cnt)[:5]
            r["band_prev_chars"] = {repr(chr(vals[i])): round(float(cnt[i] / band.sum()), 3) for i in top}
        if lat is not None and not a.no_grad:
            r |= grad_paths(cfg, model, lat, x[:2])
        results[name] = r
        del model, lat
        torch.cuda.empty_cache()
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                if not k.startswith("logit_hist")}), flush=True)
    results["_edges"] = edges.tolist()
    json.dump(results, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
