"""forward_lm (exposes chunk internals) must match the official HNetForCausalLM.forward."""
import latents  # noqa: F401  (compat shim)
import torch
from latents.models import build_model, forward_lm, lr_multipliers

def check(mc, B=4, L=2048):
    torch.manual_seed(0)
    m = build_model(mc)
    x = torch.randint(0, mc["vocab_size_padded"], (B, L), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        r = m(x, mask=torch.ones_like(x, dtype=torch.bool))  # official (masked path)
        ref, ref_b = r.logits.float(), r.bpred_output
        out = forward_lm(m, x)
    err = (ref - out.logits.float()).abs().mean().item() / ref.abs().mean().item()
    print(mc["arch_layout"], "mean rel |diff|", err, "lr mults", lr_multipliers(mc))
    assert err < 1e-2, err  # bf16 kernel-path noise (packed varlen vs padded)
    if out.bpred:
        same = ref_b[0].boundary_mask.flatten().equal(out.bpred[0].boundary_mask)
        assert same, "boundary masks differ"

base = dict(ssm_cfg=dict(chunk_size=256, d_conv=4, d_state=128, expand=2), N=6, vocab_size_padded=256)
check(dict(base, arch_layout=["m4", ["T9"], "m4"], d_model=[256, 512], d_intermediate=[0, 1408],
           num_heads=[4, 8], rotary_emb_dim=[32, 32], window_size=[1023, -1]))
check(dict(base, arch_layout=["T5"], d_model=[384], d_intermediate=[1024], num_heads=[6], rotary_emb_dim=[32], window_size=[1023]))
print("OK")
