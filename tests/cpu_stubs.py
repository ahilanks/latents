"""Pure-PyTorch stand-ins for the CUDA-only kernels the official H-Net code imports (mamba_ssm, flash_attn),
so the real hnet modules and all of latents/ can run on CPU in tests.

The stubs keep the interfaces and the properties the tests rely on (causality, packed-sequence isolation,
parameter names like mixer.in_proj), not the exact numerics: Mamba2 becomes a gated causal conv + EMA scan.

usage (before importing hnet / latents.models):
    import tests.cpu_stubs as S; S.install()
HNET_SRC (env) points at a goombalab/hnet checkout if it is not pip-installed (default: third_party/hnet).
"""
import math
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


# ----------------------------------------------------------------------------- flash_attn stand-ins
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, **kw):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x.float() + residual.float() if residual_in_fp32 else x + residual
        res = x.float() if residual_in_fp32 else x
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y, res) if prenorm else y


def swiglu(gate, y):
    return F.silu(gate) * y


def _positions(total, cu_seqlens, seqlen_offsets, device):
    if cu_seqlens is None:
        return None
    cu = cu_seqlens.long()
    lens = cu[1:] - cu[:-1]
    row = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens)
    return torch.arange(total, device=device) - cu[:-1][row]


def apply_rotary(x, cos, sin, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None, interleaved=False,
                 inplace=False, conjugate=False):
    """x: (B, S, H, D) or packed (T, H, D) with cu_seqlens; cos/sin: (S_rot, rd/2). NeoX (non-interleaved)."""
    assert not interleaved
    rd = cos.shape[-1] * 2
    if cu_seqlens is not None:
        pos = _positions(x.shape[0], cu_seqlens, seqlen_offsets, x.device)
        c, s = cos[pos][:, None, :], sin[pos][:, None, :]
    else:
        S = x.shape[1]
        off = seqlen_offsets if isinstance(seqlen_offsets, int) else 0
        c, s = cos[off:off + S][None, :, None, :], sin[off:off + S][None, :, None, :]
    if conjugate:
        s = -s
    x1, x2 = x[..., : rd // 2].float(), x[..., rd // 2: rd].float()
    c, s = c.float(), s.float()
    out_r = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)
    if inplace:
        x[..., :rd] = out_r
        return x
    return torch.cat((out_r, x[..., rd:]), dim=-1)


def _varlen_causal_attn(q, k, v, cu_seqlens, window, scale):
    """q,k,v: (T, H, D) packed; causal within each sequence, optional left window (keys >= i - window)."""
    cu = cu_seqlens.long()
    lens = cu[1:] - cu[:-1]
    B, M = lens.numel(), int(lens.max())
    row = torch.repeat_interleave(torch.arange(B, device=q.device), lens)
    pos = torch.arange(q.shape[0], device=q.device) - cu[:-1][row]

    def pad(t):
        o = t.new_zeros(B, M, *t.shape[1:])
        o[row, pos] = t
        return o.transpose(1, 2)  # (B, H, M, D)

    qp, kp, vp = pad(q), pad(k), pad(v)
    i = torch.arange(M, device=q.device)
    m = i[None, :] <= i[:, None]
    if window is not None and window >= 0:
        m = m & (i[None, :] >= i[:, None] - window)
    m = m[None] & (i[None, None, :] < lens[:, None, None])
    o = F.scaled_dot_product_attention(qp.float(), kp.float(), vp.float(), attn_mask=m[:, None], scale=scale)
    return o.transpose(1, 2)[row, pos].to(q.dtype)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=None, causal=True,
                                     window_size=(-1, -1), **kw):
    q, k, v = qkv.unbind(1)
    return _varlen_causal_attn(q, k, v, cu_seqlens, window_size[0], softmax_scale)


def flash_attn_qkvpacked_func(qkv, softmax_scale=None, causal=True, window_size=(-1, -1), **kw):
    B, S = qkv.shape[:2]
    cu = torch.arange(B + 1, device=qkv.device, dtype=torch.int32) * S
    out = flash_attn_varlen_qkvpacked_func(qkv.flatten(0, 1), cu, S, softmax_scale, causal, window_size)
    return out.view(B, S, *out.shape[1:])


def _unsupported(*a, **k):
    raise NotImplementedError("not available in CPU stubs")


class GenerationMixin:
    pass


# ----------------------------------------------------------------------------- mamba_ssm stand-ins
def _segment_scan(x, decay, seq_idx):
    """h_t = decay_t * h_{t-1} + x_t, reset where seq_idx changes. x: (B, L, ...), decay: (B, L, ...)-broadcastable."""
    B, L = x.shape[:2]
    out = []
    h = torch.zeros_like(x[:, 0])
    for t in range(L):
        if seq_idx is not None and t > 0:
            reset = (seq_idx[:, t] != seq_idx[:, t - 1]).view(B, *([1] * (x.dim() - 2)))
            h = torch.where(reset, torch.zeros_like(h), h)
        h = decay[:, t] * h + x[:, t]
        out.append(h)
    return torch.stack(out, 1)


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256, seq_idx=None, **kw):
    """SSD with d_state=1 as used by H-Net's DeChunkLayer. x: (b,l,h,p), dt: (b,l,h), A: (h,), B,C: (b,l,g,1)."""
    dA = torch.exp(dt.float() * A.float()[None, None, :])  # (b,l,h)
    inp = (dt.float()[..., None] * B.float()[..., :1, :] * x.float())  # (b,l,h,p)
    h = _segment_scan(inp, dA[..., None], seq_idx)
    return (h * C.float()[..., :1, :]).to(x.dtype)


class Mamba2(nn.Module):
    """Gated causal depthwise conv + per-channel EMA scan. Causal, isolated per packed sequence (seq_idx)."""

    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, chunk_size=256, layer_idx=None,
                 device=None, dtype=None, **kw):
        super().__init__()
        self.layer_idx = layer_idx
        di = expand * d_model
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False, device=device, dtype=dtype)
        self.conv_weight = nn.Parameter(torch.randn(di, d_conv, device=device) * 0.2)
        self.decay_logit = nn.Parameter(torch.zeros(di, device=device))
        self.out_proj = nn.Linear(di, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, u, seq_idx=None, inference_params=None, **kw):
        B, L, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, -1)
        xs = []
        for i in range(self.d_conv):  # causal conv that never crosses a packed-sequence boundary
            sh = F.pad(x, (0, 0, i, 0))[:, :L]
            if seq_idx is not None and i > 0:
                same = F.pad(seq_idx, (i, 0), value=-1)[:, :L] == seq_idx
                sh = sh * same[..., None]
            xs.append(sh * self.conv_weight[:, i])
        x = F.silu(sum(xs))
        a = torch.sigmoid(self.decay_logit + 2.0)
        h = _segment_scan(x * (1 - a), a.expand_as(x), seq_idx)
        return self.out_proj((h + x) * F.silu(z))


# ----------------------------------------------------------------------------- install
def install():
    if getattr(install, "done", False):
        return
    _mod("flash_attn", flash_attn_kvpacked_func=_unsupported, flash_attn_qkvpacked_func=flash_attn_qkvpacked_func,
         flash_attn_varlen_kvpacked_func=_unsupported, flash_attn_varlen_qkvpacked_func=flash_attn_varlen_qkvpacked_func,
         flash_attn_with_kvcache=_unsupported)
    _mod("flash_attn.ops")
    _mod("flash_attn.ops.triton")
    _mod("flash_attn.ops.triton.layer_norm", RMSNorm=RMSNorm)
    _mod("flash_attn.ops.triton.rotary", apply_rotary=apply_rotary)
    _mod("flash_attn.ops.activations", swiglu=swiglu)
    _mod("flash_attn.utils")
    _mod("flash_attn.utils.generation", GenerationMixin=GenerationMixin)
    _mod("mamba_ssm")
    _mod("mamba_ssm.modules")
    _mod("mamba_ssm.modules.mamba2", Mamba2=Mamba2)
    _mod("mamba_ssm.ops")
    _mod("mamba_ssm.ops.triton")
    _mod("mamba_ssm.ops.triton.ssd_combined", mamba_chunk_scan_combined=mamba_chunk_scan_combined)
    src = os.environ.get("HNET_SRC", str(ROOT / "third_party" / "hnet"))
    if src not in sys.path:
        sys.path.insert(0, src)
    import hnet.modules.mha as mha

    def fwd(self, qkv, cu_seqlens=None, max_seqlen=None):  # official forward minus the is_cuda/bf16 asserts
        if cu_seqlens is not None:
            return flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, softmax_scale=self.softmax_scale,
                                                    causal=True, window_size=(self.window_size, -1))
        return flash_attn_qkvpacked_func(qkv, softmax_scale=self.softmax_scale, causal=True,
                                         window_size=(self.window_size, -1))

    mha.FlashCausalSelfAttention.forward = fwd
    install.done = True


def tiny_mc(**over):
    """A small 1-stage H-Net config for CPU tests (same structure as the real runs)."""
    mc = dict(ssm_cfg=dict(chunk_size=256, d_conv=4, d_state=16, expand=2), N=4, alpha=0.03, pool_k=0,
              vocab_size_padded=256, vocab_size=256, tokenizer="bytes", seq_len=96, bytes_per_token=1.0,
              arch_layout=["m1", ["T2"], "m1"], d_model=[32, 64], d_intermediate=[0, 96],
              num_heads=[2, 4], rotary_emb_dim=[8, 8], window_size=[63, -1])
    mc.update(over)
    return mc
