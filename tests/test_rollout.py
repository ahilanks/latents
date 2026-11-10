"""If every rollout input is the true chunk (ĉ_p := c_p), pass 2 must equal the teacher-forced pass 1."""
import latents  # noqa: F401
import torch
from latents.models import build_model, forward_lm
from latents.rollout import Rollout

mc = dict(ssm_cfg=dict(chunk_size=256, d_conv=4, d_state=128, expand=2), N=6, vocab_size_padded=256,
          arch_layout=["m4", ["T3"], "m4"], d_model=[256, 512], d_intermediate=[0, 1408],
          num_heads=[4, 8], rotary_emb_dim=[32, 32], window_size=[1023, -1])
torch.manual_seed(0)
m = build_model(mc)
ro = Rollout.for_main(m)
x = torch.randint(0, 256, (3, 1024), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = forward_lm(m, x)
    z2 = ro.run(None, out.cu1, out.chunks)
rel = ((z2.float() - out.z_full.float()).abs().mean() / out.z_full.float().abs().mean()).item()
print("chunks", out.chunks.shape[0], "rel diff pass2 vs pass1:", rel)
assert rel < 2e-2, rel
z2.float().pow(2).mean().backward()  # gradient must flow into pass-1 graph (cached K/V) and params
g = m.backbone.encoder.layers[0].mixer.in_proj.weight.grad
assert g is not None and g.abs().sum() > 0
print("OK")
