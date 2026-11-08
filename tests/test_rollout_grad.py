"""Rollout-loss gradients with the mem-efficient SDPA kernel must match the reference math kernel."""
import latents  # noqa: F401
import torch
from torch.nn.attention import SDPBackend
import latents.rollout as R
from latents.models import build_model, forward_lm
from latents.latent import LatentObjective

mc = dict(ssm_cfg=dict(chunk_size=256, d_conv=4, d_state=128, expand=2), N=6, vocab_size_padded=256,
          arch_layout=["m4", ["T2"], "m4"], d_model=[256, 512], d_intermediate=[0, 1408],
          num_heads=[4, 8], rotary_emb_dim=[32, 32], window_size=[1023, -1])

def grads(backend):
    torch.manual_seed(0)
    m = build_model(mc)
    lat = LatentObjective(dict(target="online", beta_tf=0.0, beta_r=1.0), m, mc)
    R.sdpa_kernel_backend = backend
    x = torch.randint(0, 256, (2, 1024), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward_lm(m, x)
        loss, st = lat.loss(m, out, x)
    loss.backward()
    g = {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}
    g["head"] = lat.head.weight.grad.float().clone()
    return loss.item(), g

l1, g1 = grads(SDPBackend.EFFICIENT_ATTENTION)
l2, g2 = grads(SDPBackend.MATH)
print("loss", l1, l2)
worst = 0
for n in g2:
    rel = ((g1[n] - g2[n]).norm() / (g2[n].norm() + 1e-12)).item()
    worst = max(worst, rel)
    if rel > 0.05: print("MISMATCH", n, rel)
print("worst rel grad diff", worst)
