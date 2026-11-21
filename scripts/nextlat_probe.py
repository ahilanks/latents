"""Is the main network's state a summary from which the next state can be computed? A control for the next-latent
runs (latents/nextlat.py): on a FROZEN model, fit a fresh transition MLP (state at chunk t, input of chunk t+1 ->
state at chunk t+1) and measure it on validation text. The model is not trained here (a probe, not end to end).

Per run, on --val_windows windows of the validation text:
  kl_zero      bytes read from zeros instead of the main network's states: KL to the normal byte distribution (nats per
               byte) = how much the decoder relies on the main network at all
  kl_mean      ... from the states' mean vector (no information, right scale)
  kl_shuffle   ... from another random chunk's state (no information, right distribution) = what the decoder gets
               from the main network's information
  kl_copy      ... from the previous chunk's state
  kl_own       ... from the run's own trained transition MLP (runs trained with model.nextlat)
  kl_probe     ... from a fresh MLP fitted with the byte-space (KL) loss for --steps steps
  expl_*       share of the next state's variation explained (copy / own / KL probe / regression probe)
usage: python scripts/nextlat_probe.py runs/<a> [runs/<b> ...] [--steps 400] [--out logs/nextlat/probe.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents.models import model_forward  # noqa: E402
from latents.nextlat import NextLat  # noqa: E402
from show_cuts import load_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val_windows", type=int, default=16)
    ap.add_argument("--data_dir", default="data/fwe")
    ap.add_argument("--out", default="logs/nextlat/probe.json")
    a = ap.parse_args()
    dev = "cuda"
    ac = dict(device_type="cuda", dtype=torch.bfloat16)
    train = np.memmap(ROOT / a.data_dir / "train.u8", dtype=np.uint8, mode="r")
    val = np.fromfile(ROOT / a.data_dir / "val.u8", dtype=np.uint8)
    res = {}
    for run in a.runs:
        cfg, model, _ = load_any(run, dev)
        mc = cfg["model"]
        L = cfg["train"]["seq_len"]
        for p in model.parameters():
            p.requires_grad_(False)
        own = getattr(model, "nextlat", None)
        D1, D0 = mc["d_model"][-1], mc["d_model"][0]
        probes = {"kl": NextLat(D1, D0, weight=0.0, kl=1.0).to(dev), "reg": NextLat(D1, D0, weight=1.0, kl=0.0).to(dev)}
        opts = {k: torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=0.0) for k, p in probes.items()}
        rng = np.random.default_rng(0)

        def fwd(x):
            with torch.no_grad(), torch.autocast(**ac):
                return model_forward(model, x, mc)

        def kl_of(probe, out, x, pred):
            for p in model.parameters():  # kl_loss freezes and re-enables the decoder: keep the whole model frozen
                p.requires_grad_(False)
            with torch.autocast(**ac):
                v = probe.kl_loss(model, out, x, mc, pred)
            for p in model.parameters():
                p.requires_grad_(False)
            return v

        for step in range(a.steps):
            idx = rng.integers(0, len(train) - L - 1, size=a.batch)
            x = torch.from_numpy(np.stack([train[i:i + L] for i in idx]).astype(np.int64)).to(dev)
            out = fwd(x)
            lr = 1e-3 * min(1.0, (step + 1) / 20) * (0.5 * (1 + np.cos(np.pi * step / a.steps)))
            for k, probe in probes.items():
                for g in opts[k].param_groups:
                    g["lr"] = lr
                with torch.autocast(**ac):
                    pred = probe.rollout(out)
                if k == "kl":
                    loss = kl_of(probe, out, x, pred)
                else:
                    loss = F.smooth_l1_loss(pred[0][1], out.z_full.float()[pred[0][0]])
                loss.backward()
                opts[k].step()
                opts[k].zero_grad(set_to_none=True)
        acc = {}

        def add(k, v):
            acc[k] = acc.get(k, 0.0) + float(v) / (a.val_windows // a.batch)
        with torch.no_grad():
            for b in range(a.val_windows // a.batch):
                x = torch.from_numpy(val[b * a.batch * L:(b + 1) * a.batch * L].reshape(a.batch, L).astype(np.int64)).to(dev)
                out = fwd(x)
                z = out.z_full.float()
                var = (z - z.mean(0)).pow(2).sum(-1).mean()
                ref = probes["kl"]
                with torch.autocast(**ac):
                    idx = ref.rollout(out)[0][0]

                def expl(p):
                    return 1 - (p.float() - z[idx]).pow(2).sum(-1).mean() / var
                add("variation_share", var / z.pow(2).sum(-1).mean())
                add("kl_zero", kl_of(ref, out, x, [(idx, torch.zeros_like(z[idx]))]))
                add("kl_mean", kl_of(ref, out, x, [(idx, z.mean(0, keepdim=True).expand(idx.numel(), -1))]))
                perm = torch.randperm(z.shape[0], device=z.device)[: idx.numel()]
                add("kl_shuffle", kl_of(ref, out, x, [(idx, z[perm])]))
                add("kl_copy", kl_of(ref, out, x, [(idx, z[idx - 1])]))
                add("expl_copy", expl(z[idx - 1]))
                for k, probe in (("own", own), ("probe_kl", probes["kl"]), ("probe_reg", probes["reg"])):
                    if probe is None:
                        continue
                    with torch.autocast(**ac):
                        pred = probe.rollout(out)
                    add(f"kl_{k}", kl_of(ref, out, x, pred))
                    add(f"expl_{k}", expl(pred[0][1]))
        res[run] = acc
        print(run, {k: round(v, 5) for k, v in acc.items()}, flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
