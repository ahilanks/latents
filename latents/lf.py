"""Latent Feedback's Transformer (arXiv 2611.20404), rewritten from its code (ifm-ai/LoRM @ 2460ca8, Apache License 2.0:
lorm/model/block.py, transformer.py, feedback.py) for bytes instead of Llama-2 tokens. Used to repeat their run on our
web bytes and see whether their loop keeps their Transformer's top state healthy where it flattens the H-Net's main
network (logistics.md, 2026-11-24).

Model as theirs: pre-norm Llama blocks (RMSNorm, attention with per-head q/k RMSNorm and adjacent-pair RoPE, SwiGLU),
no biases, untied embedding and head, every weight N(0, 0.02) and every gain 1 (OLMo 2 initialization).
Feedback: u_t = e_t + W rms(h_{t-1}), h = the top state before the final norm, W without bias; with `passes` p the
first p-1 passes run without gradient and the fed state is detached (their forward).
"""
import torch
import torch.nn.functional as F
from torch import nn


def rotary(n, head_dim, theta, device):
    powers = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    ang = torch.arange(n, device=device).float()[:, None] * theta ** (-powers[None, :])
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    v = x.float()
    even, odd = v[..., 0::2], v[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, eps):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q_norm, self.k_norm = nn.RMSNorm(self.hd, eps=eps), nn.RMSNorm(self.hd, eps=eps)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim, bias=False) for _ in range(4))

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q, k, v = (p(x).view(B, L, self.h, self.hd).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(dim, eps=eps), nn.RMSNorm(dim, eps=eps)
        self.attention = Attention(dim, heads, eps)
        self.gate_proj, self.up_proj = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin)
        h = self.norm2(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

    def matrices(self):
        a = self.attention
        return [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.out_proj.weight,
                self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight]


class LFTransformer(nn.Module):
    """feedback=False: their plain Transformer arm; feedback=True: their latent feedback Transformer."""

    def __init__(self, dim=512, layers=8, heads=4, hidden=1536, vocab=256, feedback=True, eps=1e-5, theta=10000.0):
        super().__init__()
        self.dim, self.heads, self.eps, self.theta = dim, heads, eps, theta
        self.embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, hidden, eps) for _ in range(layers))
        self.norm = nn.RMSNorm(dim, eps=eps)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.fusion = nn.Linear(dim, dim, bias=False) if feedback else None
        for p in self.parameters():  # their initialize(): gains of one, every other weight N(0, 0.02)
            if p.ndim == 1:
                nn.init.ones_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def hidden_matrices(self):
        """What NorMuon trains (theirs: the block matrices and the fusion W); the rest is on AdamW."""
        return [p for b in self.blocks for p in b.matrices()] + ([self.fusion.weight] if self.fusion is not None else [])

    def stack(self, x, cos, sin):
        for b in self.blocks:
            x = b(x, cos, sin)
        return x

    def fed(self, top):
        """W rms(top) of each position (before the shift to the next position)."""
        return self.fusion(F.rms_norm(top, (top.shape[-1],), eps=self.eps))

    def mix(self, e, top):
        f = self.fed(top)
        return torch.cat([e[:, :1], e[:, 1:] + f[:, :-1]], 1)

    def forward(self, ids, passes=1, return_state=False):
        """Logits of the last of `passes` passes; earlier passes without gradient (their forward)."""
        x = self.embedding(ids)
        cos, sin = rotary(ids.shape[1], self.dim // self.heads, self.theta, ids.device)
        passes = passes if self.fusion is not None else 1
        state, top = x, None
        for i in range(passes):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i == passes - 1):
                if i:
                    state = self.mix(x, top.detach())
                top = self.stack(state, cos, sin)
        logits = self.head(self.norm(top))
        return (logits, dict(top=top, emb=x, inp=state)) if return_state else logits
