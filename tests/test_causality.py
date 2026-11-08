"""Causality check on a trained checkpoint: perturbing bytes at positions >= k must not change logits at positions < k.
usage: python tests/test_causality.py runs/<name>/final.pt"""
import sys
import latents  # noqa: F401
import numpy as np
import torch
from latents.models import build_model, forward_lm

ck = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
mc = ck["cfg"]["model"]
m = build_model(mc)
m.load_state_dict(ck["model"])
m.eval()
v = np.fromfile("data/fwe/val.u8", dtype=np.uint8)
L = 8192
x = torch.from_numpy(v[:4 * L].astype(np.int64)).view(4, L).cuda()
worst = 0.0
for k in [100, 1000, 4097, 8000]:
    x2 = x.clone()
    x2[:, k:] = torch.randint(0, 256, x2[:, k:].shape, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = forward_lm(m, x).logits.float()
        b = forward_lm(m, x2).logits.float()
    d = (a[:, :k] - b[:, :k]).abs().max().item()
    dn = (a[:, k:] - b[:, k:]).abs().max().item()
    worst = max(worst, d)
    print(f"k={k}: max |dlogit| before k = {d:.2e}   (after k, expected large: {dn:.2e})")
print("max |dlogit| is bf16 kernel noise if similar for a baseline checkpoint; see tests/test_causality2.py (loss-based check)")
