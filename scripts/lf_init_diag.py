"""Diagnosis (2026-11-24): how much of the state that Latent Feedback's loop feeds back is the same for every position,
in the H-Net's main network vs their Transformer, at initialization and in finished runs. The loop feeds back
W rms(top); if top is mostly one shared vector, the fed-back vector is nearly a constant added to every input.

usage: python scripts/lf_init_diag.py [--runs HP-plain-s0,HL-P-s0,LF-fb-s0,LF-plain-s0] > logs/lf/init_diag.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from latents.lf import LFTransformer  # noqa: E402


def shared_share(x):
    """Share of the mean square that is the same for every row (1 = every row identical)."""
    x = x.float()
    return float(x.mean(0).pow(2).sum() / x.pow(2).sum(-1).mean().clamp(min=1e-12))


def val_windows(n=8, L=2049):
    v = np.memmap(ROOT / "data/fwe/val.u8", dtype=np.uint8, mode="r")
    s = np.linspace(0, len(v) - L - 1, n).astype(np.int64)
    return torch.from_numpy(np.stack([v[i:i + L] for i in s]).astype(np.int64)).cuda()


@torch.no_grad()
def hnet_numbers(model, mc, x):
    """Main network's state before its final norm (the H-Net's 'top' for the loop) and its normalised fed vector."""
    from latents.models import model_forward
    seen = {}
    inner = model.backbone.main_network.main_network
    norm = inner.rmsnorm

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs.get("x", kwargs.get("hidden_states"))
        r = kwargs.get("residual")
        seen["top"] = (h.float() + r.float()) if r is not None else h.float()
    hk = norm.register_forward_pre_hook(pre, with_kwargs=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model_forward(model, x[:, :-1], mc, None)
    hk.remove()
    top = seen["top"]
    pad = model.backbone.main_network.pad_dimension
    inp = out.chunks.float() if pad is None else torch.cat((out.chunks.float(), pad.float().expand(out.chunks.shape[0], -1)), -1)
    return dict(top_shared=shared_share(top), rms_top_shared=shared_share(F.rms_norm(top, (top.shape[-1],))),
                input_shared=shared_share(inp), out_shared=shared_share(out.z_full))


@torch.no_grad()
def lf_numbers(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, st = model(x[:, :-1], passes=1, return_state=True)
    top = st["top"].float().flatten(0, 1)
    return dict(top_shared=shared_share(top), rms_top_shared=shared_share(F.rms_norm(top, (top.shape[-1],))),
                input_shared=shared_share(st["emb"].flatten(0, 1)), out_shared=shared_share(model.norm(st["top"]).flatten(0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="HP-plain-s0,HL-P-s0,LF-fb-s0,LF-plain-s0")
    a = ap.parse_args()
    torch.manual_seed(0)
    x = val_windows()
    res = {}
    from latents.models import build_model
    cfg = yaml.safe_load(open(ROOT / "runs/HP-plain-s0/config.yaml"))
    m = build_model(cfg["model"], device="cuda").eval()
    res["H-Net at init"] = hnet_numbers(m, cfg["model"], x)
    res["their Transformer at init"] = lf_numbers(LFTransformer(feedback=False).cuda().eval(), x)
    for r in a.runs.split(","):
        d = ROOT / "runs" / r
        if not (d / "final.pt").exists():
            continue
        ck = torch.load(d / "final.pt", map_location="cuda", weights_only=False)
        if r.startswith("LF-"):
            c = ck["cfg"]
            mm = LFTransformer(c["dim"], c["layers"], c["heads"], c["hidden"], feedback=bool(c["feedback"])).cuda().eval()
            mm.load_state_dict(ck["model"])
            res[r] = lf_numbers(mm, x)
        else:
            from show_cuts import load_any
            c, mm, _ = load_any(str(d), "cuda")
            res[r] = hnet_numbers(mm.eval(), c["model"], x)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
