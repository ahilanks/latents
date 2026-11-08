"""Localised causality check: where do earlier-position logit changes occur, and do they change the loss?"""
import sys
import latents  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from latents.models import build_model, forward_lm

ck = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
mc = ck["cfg"]["model"]
m = build_model(mc); m.load_state_dict(ck["model"]); m.eval()
v = np.fromfile("data/fwe/val.u8", dtype=np.uint8)
L = 8192
x = torch.from_numpy(v[:4 * L].astype(np.int64)).view(4, L).cuda()
k = 4097
x2 = x.clone(); x2[:, k:] = torch.randint(0, 256, x2[:, k:].shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    a = forward_lm(m, x).logits.float(); b = forward_lm(m, x2).logits.float()
d = (a - b).abs().amax(-1)  # (B, L) per-position max diff
y = x[:, 1:k]
ce_a = F.cross_entropy(a[:, :k-1].flatten(0, 1), y.flatten()).item()
ce_b = F.cross_entropy(b[:, :k-1].flatten(0, 1), y.flatten()).item()
print(f"{sys.argv[1].split('/')[1]:28s} mean|d| pos<k-64: {d[:, :k-64].mean():.2e}  max|d| last 8 pos before k: {d[:, k-8:k].amax():.2e}  "
      f"max|d| pos<k-64: {d[:, :k-64].amax():.2e}  CE(pos<k) clean {ce_a:.5f} vs perturbed-future {ce_b:.5f}")
