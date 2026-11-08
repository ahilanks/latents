"""Exact 2-step rollout through the H-Net main network (V-JEPA 2-AC rollout loss on chunks).

Pass 1 is the normal teacher-forced main-network forward (official modules, flash-attn varlen); we
capture each layer's post-rotary qkv via a pre-hook on `inner_attn`, so it stays inside autograd.
Pass 2 runs only the rollout tokens: the token at position p has input ĉ_p = head(P(c≤p-1)) and attends to
the pass-1 keys/values at positions ≤ p-1 *and itself*:

    out_p = P(c_{<p}, ĉ_p)   →   head → compared with sg(c̄_{p+1})

This is P(c≤j, ĉ_{j+1}) with j = p-1, without recomputing pass 1. Attention uses SDPA over
[K1 ; K_rollout] with an explicit mask on padded rows.
Sanity check (tests/test_rollout.py): if ĉ_p := c_p, pass 2 must reproduce pass-1 outputs.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sdpa_kernel_backend = SDPBackend.EFFICIENT_ATTENTION  # overridable in tests


class Rollout:
    def __init__(self, model, predictor_only=False):
        self.predictor_only = predictor_only
        self.inner = model.backbone.main_network  # innermost HNet (stage 1)
        self.iso = self.inner.main_network
        assert all(a in "tT" for a in self.iso.arch_full), "rollout needs an attention-only main network"
        self.cache = []
        self.iso.register_forward_pre_hook(lambda mod, args: self.cache.clear())
        for blk in self.iso.layers:
            blk.mixer.inner_attn.register_forward_pre_hook(self._capture)

    def _capture(self, mod, args):
        self.cache.append(args[0])  # qkv: (T1, 3, H, hd), post-rotary

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def layout(cu1, T1):
        lens = (cu1[1:] - cu1[:-1]).long()
        B, M = lens.numel(), (int(lens.max()) + 15) // 16 * 16  # pad: mem-efficient SDPA needs aligned mask
        row = torch.repeat_interleave(torch.arange(B, device=cu1.device), lens)
        pos = torch.arange(T1, device=cu1.device) - cu1[:-1].long()[row]
        return B, M, lens, row, pos

    @staticmethod
    def rotary(x, inv_freq, rd):
        """Non-interleaved (GPT-NeoX) rotary on the first rd dims; x: (B, M, H, hd), positions = slot index."""
        M = x.shape[1]
        f = torch.outer(torch.arange(M, device=x.device, dtype=torch.float32), inv_freq.float())
        cos, sin = f.cos()[None, :, None, :].to(x.dtype), f.sin()[None, :, None, :].to(x.dtype)
        x1, x2, rest = x[..., : rd // 2], x[..., rd // 2 : rd], x[..., rd:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)

    def run(self, x_in, cu1, rollout_in):
        """x_in unused (pass-1 already ran). rollout_in: (T1, D0) inputs for every slot (slot p's input ĉ_p).
        Returns pass-2 outputs (T1, D1) in packed order."""
        T1 = rollout_in.shape[0]
        B, M, lens, row, pos = self.layout(cu1, T1)
        assert len(self.cache) == len(self.iso.layers), "pass-1 qkv not captured"
        pad = self.inner.pad_dimension
        h = rollout_in if pad is None else torch.cat((rollout_in, pad.expand(T1, -1).to(rollout_in.dtype)), -1)
        D1 = h.shape[-1]
        hp = h.new_zeros(B, M, D1)
        hp[row, pos] = h
        # mask (B, 1, M, 2M): key i<M allowed iff i <= p-1 and i < len; key M+i allowed iff i == p
        p = torch.arange(M, device=h.device)
        hist = (p[None, :] <= p[:, None] - 1)  # (M, M) strictly causal
        hist = hist[None] & (p[None, None, :] < lens[:, None, None])  # (B, M, M)
        selfm = torch.eye(M, dtype=torch.bool, device=h.device)[None].expand(B, -1, -1)
        mask = torch.cat((hist, selfm), dim=-1)[:, None]  # (B,1,M,2M)
        res = None
        hcur = hp
        for li, blk in enumerate(self.iso.layers):
            hcur, res = blk.norm1(hcur, residual=res, prenorm=True, residual_in_fp32=True)
            mx = blk.mixer
            qkv = mx.Wqkv(hcur).view(B, M, 3, -1, mx.head_dim)
            q, k, v = qkv.unbind(2)
            if mx.rotary_emb_dim > 0:
                q = self.rotary(q, mx.rotary_emb.inv_freq, mx.rotary_emb_dim)
                k = self.rotary(k, mx.rotary_emb.inv_freq, mx.rotary_emb_dim)
            qkv1 = self.cache[li].detach() if self.predictor_only else self.cache[li]
            k1 = qkv1.new_zeros(B, M, *qkv1.shape[2:])
            v1 = qkv1.new_zeros(B, M, *qkv1.shape[2:])
            k1[row, pos] = qkv1[:, 1]
            v1[row, pos] = qkv1[:, 2]
            K = torch.cat((k1, k.to(k1.dtype)), 1).transpose(1, 2)
            V = torch.cat((v1, v.to(v1.dtype)), 1).transpose(1, 2)
            with sdpa_kernel(sdpa_kernel_backend):  # mem-efficient: never fall back to the O(M^2)-memory math kernel
                o = F.scaled_dot_product_attention(q.to(k1.dtype).transpose(1, 2), K, V, attn_mask=mask)
            hcur = mx.out_proj(o.transpose(1, 2).reshape(B, M, -1))
            if blk.mlp is not None:
                hcur, res = blk.norm2(hcur, residual=res, prenorm=True, residual_in_fp32=True)
                hcur = blk.mlp(hcur)
        z = self.iso.rmsnorm(hcur, residual=res, prenorm=False, residual_in_fp32=True)
        self.cache.clear()
        return z[row, pos]  # (T1, D1)

    def loss(self, model, lat, out, pred_all, tgt):
        """pred_all: head(P(c≤t)) for every chunk t (T1, D0); tgt: c̄ (T1, D0)."""
        T1 = tgt.shape[0]
        B, M, lens, row, pos = self.layout(out.cu1, T1)
        # slot p gets ĉ_p = pred_all[p-1]; slot 0 of each row has no prediction -> feed its true chunk (masked from loss)
        prev = torch.roll(pred_all, 1, 0).to(out.chunks.dtype)
        rin = torch.where((pos == 0)[:, None], out.chunks, prev)
        if self.predictor_only:
            rin = rin.detach()
        z2 = self.run(None, out.cu1, rin)
        pred2 = lat.head(z2).float()
        valid = (pos >= 1) & (pos + 1 < lens[row])
        idx = torch.nonzero(valid).squeeze(1)
        return (pred2[idx] - tgt[idx + 1]).abs().mean()
