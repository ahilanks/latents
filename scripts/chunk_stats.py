"""Chunk-boundary statistics of finished H-Net runs on val (first N windows).

word-edge precision: fraction of boundaries at a whitespace byte or right after one
space recall:        fraction of whitespace bytes w with a boundary at w or w+1
usage: python scripts/chunk_stats.py [run_substring ...]
"""
import json
import sys
from pathlib import Path

import latents  # noqa: F401
import numpy as np
import torch

from latents.models import build_model, forward_lm

ROOT = Path(__file__).resolve().parents[1]
WS = torch.tensor([32, 9, 10, 13], device="cuda")


@torch.no_grad()
def stats(run_dir, n_windows=64, L=8192):
    ck = torch.load(run_dir / "final.pt", map_location="cuda", weights_only=False)
    mc = ck["cfg"]["model"]
    if len(mc["d_model"]) == 1 or mc.get("pool_k"):
        return None
    m = build_model(mc)
    m.load_state_dict(ck["model"])
    m.eval()
    v = torch.from_numpy(np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8).astype(np.int64))
    nb = nprec = nsp = nrec = 0
    for i in range(0, n_windows, 16):
        x = torch.stack([v[j * L:(j + 1) * L] for j in range(i, i + 16)]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward_lm(m, x, pool_k=0)
        b = out.bpred[0].boundary_mask.view(x.shape)
        ws = torch.isin(x, WS)
        prev_ws = torch.nn.functional.pad(ws[:, :-1], (1, 0))
        edge = ws | prev_ws
        nb += b.sum().item()
        nprec += (b & edge).sum().item()
        nxt = torch.nn.functional.pad(b[:, 1:], (0, 1))
        nsp += ws.sum().item()
        nrec += (ws & (b | nxt)).sum().item()
    return {"L1/L0": nb / (n_windows * L), "edge_prec": nprec / nb, "space_recall": nrec / nsp}


def main():
    pats = sys.argv[1:] or [""]
    for d in sorted((ROOT / "runs").glob("*/")):
        if not (d / "final.pt").exists() or not any(p in d.name for p in pats) or "smoke" in d.name:
            continue
        s = stats(d)
        if s is None:
            continue
        bpb = json.load(open(d / "summary.json"))["val/bpb"]
        print(f"{d.name:34s} bpb {bpb:.4f}  L1/L0 {s['L1/L0']:.3f}  edge_prec {s['edge_prec']:.3f}  space_recall {s['space_recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
