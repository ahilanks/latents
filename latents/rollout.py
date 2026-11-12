"""Exact k-step rollout through an attention-only predictor (V-JEPA 2-AC rollout loss on chunks).

Pass 1 is the normal teacher-forced predictor forward (official modules, flash-attn varlen); each layer's
post-rotary qkv is captured by a pre-hook on `inner_attn`, so it stays inside autograd.

Rollout pass s (s = 2..k) holds one token per slot p. That token belongs to the chain of anchor j = p - s + 1
and stands for position p: its input is ĉ^{(s-1)}_p, the (s-1)-step prediction of c_p made from c_{<=j}.
It attends to the pass-1 keys at positions <= j, to the earlier tokens of its own chain (pass t at slot
p - (s - t), t = 2..s-1), and to itself:

    out^{(s)}_p = P(c_{<=j}, ĉ_{j+1}, ..., ĉ_{j+s-1})   ->   head  ->  compared with sg(c̄_{j+s}) = sg(c̄_{p+1})

For k = 2 this is exactly the original 2-step rollout. Attention uses SDPA over [K1 ; K^2 ; ... ; K^s] with
an explicit mask (one allowed key per earlier chain pass).
Sanity check (tests): if every rollout input is the true chunk, every pass reproduces pass 1.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sdpa_kernel_backend = SDPBackend.EFFICIENT_ATTENTION  # overridable in tests


class QKVCapture:
    """Captures the post-rotary qkv of every attention layer of one Isotropic stack, per `tag`.
    One capture per stack (shared stacks, e.g. tied levels, set `tag` before each call)."""

    def __init__(self, iso):
        self.tag = None
        self.store = {}
        iso.register_forward_pre_hook(self._start)
        for blk in iso.layers:
            blk.mixer.inner_attn.register_forward_pre_hook(self._capture)

    def _start(self, mod, args):
        self.store[self.tag] = []

    def _capture(self, mod, args):
        self.store.setdefault(self.tag, []).append(args[0])  # qkv: (T, 3, H, hd), post-rotary

    def get(self, tag):
        return self.store.get(tag, [])

    def clear(self, tag=None):
        self.store.pop(tag, None)


def capture_for(iso):
    if getattr(iso, "_qkv_capture", None) is None:
        iso._qkv_capture = QKVCapture(iso)
    return iso._qkv_capture


class Rollout:
    def __init__(self, iso, pad=None, predictor_only=False, k=2, tag=None, in_add=None):
        """iso: the predictor's Isotropic stack (attention only); pad: its pad_dimension parameter (or None);
        in_add: callable -> (D,) tensor added to every rollout input after padding (e.g. a level embedding)."""
        assert all(a in "tT" for a in iso.arch_full), "rollout needs an attention-only predictor"
        assert k >= 2
        self.iso, self.pad, self.k, self.tag, self.in_add = iso, pad, int(k), tag, in_add
        self.predictor_only = predictor_only
        self.cap = capture_for(iso)

    @classmethod
    def for_main(cls, model, **kw):
        inner = model.backbone.main_network  # innermost HNet (stage 1)
        return cls(inner.main_network, pad=inner.pad_dimension, **kw)

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
        x1, x2, rest = x[..., : rd // 2], x[..., rd // 2: rd], x[..., rd:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)

    @staticmethod
    def pass_mask(s, B, M, lens, device):
        """(B, 1, M, s*M): block 0 = pass-1 keys i <= p - s + 1 (and i < len); block t-1 (t = 2..s) = i == p - (s - t)."""
        p = torch.arange(M, device=device)
        hist = (p[None, :] <= p[:, None] - (s - 1))[None] & (p[None, None, :] < lens[:, None, None])
        blocks = [hist]
        for t in range(2, s + 1):
            d = s - t
            blocks.append((p[None, :] == p[:, None] - d)[None].expand(B, -1, -1))
        return torch.cat(blocks, dim=-1)[:, None]

    def _pass1_kv(self, B, M, row, pos):
        caches = self.cap.get(self.tag)
        assert len(caches) == len(self.iso.layers), "pass-1 qkv not captured (call the predictor first)"
        kv = []
        for qkv1 in caches:
            if self.predictor_only:
                qkv1 = qkv1.detach()
            k1 = qkv1.new_zeros(B, M, *qkv1.shape[2:])
            v1 = qkv1.new_zeros(B, M, *qkv1.shape[2:])
            k1[row, pos] = qkv1[:, 1]
            v1[row, pos] = qkv1[:, 2]
            kv.append(([k1], [v1]))
        return kv

    def run_pass(self, s, rin, layout, kv):
        """One rollout pass. rin: (T1, D_in) inputs per slot. kv: per layer ([K1, K2..], [V1, V2..]), appended to."""
        B, M, lens, row, pos = layout
        T1 = rin.shape[0]
        h = rin if self.pad is None else torch.cat((rin, self.pad.expand(T1, -1).to(rin.dtype)), -1)
        if self.in_add is not None:
            h = h + self.in_add().to(h.dtype)
        hp = h.new_zeros(B, M, h.shape[-1])
        hp[row, pos] = h
        mask = self.pass_mask(s, B, M, lens, h.device)
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
            ks, vs = kv[li]
            ks.append(k.to(ks[0].dtype))
            vs.append(v.to(vs[0].dtype))
            K = torch.cat(ks, 1).transpose(1, 2)
            V = torch.cat(vs, 1).transpose(1, 2)
            # mem-efficient on GPU: never fall back to the O(M^2)-memory math kernel (CPU only has the math kernel)
            with sdpa_kernel(sdpa_kernel_backend if K.is_cuda else SDPBackend.MATH):
                o = F.scaled_dot_product_attention(q.to(K.dtype).transpose(1, 2), K, V, attn_mask=mask)
            hcur = mx.out_proj(o.transpose(1, 2).reshape(B, M, -1))
            if blk.mlp is not None:
                hcur, res = blk.norm2(hcur, residual=res, prenorm=True, residual_in_fp32=True)
                hcur = blk.mlp(hcur)
        z = self.iso.rmsnorm(hcur, residual=res, prenorm=False, residual_in_fp32=True)
        return z[row, pos]  # (T1, D1)

    def run(self, x_in, cu1, rollout_in):
        """Single 2-step pass with given inputs (kept for tests / backwards compatibility)."""
        layout = self.layout(cu1, rollout_in.shape[0])
        kv = self._pass1_kv(*layout[:2], layout[3], layout[4])
        z = self.run_pass(2, rollout_in, layout, kv)
        self.cap.clear(self.tag)
        return z

    def losses(self, head, chunks, cu1, pred_all, tgt, feed_all=None, feed_head=None, in_tgt=None, dist=None, w=None,
               valid=None):
        """head: predictor head; chunks: true chunk inputs c (T1, D_in); pred_all: head(P(c<=t)) (T1, D_t);
        tgt: c̄ (T1, D_t). Returns {s: loss of the s-step prediction} and {s: L1 copy baseline at horizon s}.
        What is fed back as the next input: head's prediction (default), or, with feed_all / feed_head, a second head
        that predicts the next INPUT (needed when the targets do not live in the input space, e.g. chunk-local targets);
        its rollout predictions are graded against in_tgt (the next inputs), stored in self.last_in.
        dist: element-wise distance (default |p - t|); w: (T1,) per-target weights (e.g. segment lengths);
        valid: (T1,) bool, chunks that may be targets (padding chunks are not). L1 values for logging: self.last_l1."""
        T1 = tgt.shape[0]
        layout = self.layout(cu1, T1)
        B, M, lens, row, pos = layout
        kv = self._pass1_kv(B, M, row, pos)
        dist = dist or (lambda p, t: (p - t).abs())

        def red(err, idx):  # (n, D) element-wise errors at targets idx + 1 -> scalar
            e = err.mean(-1)
            if w is None:
                return e.mean()
            ww = w[idx + 1].float()
            return (ww * e).sum() / ww.sum().clamp(min=1.0)

        prev = pred_all if feed_all is None else feed_all
        out, copy = {}, {}
        self.last_in, self.last_l1 = {}, {}
        for s in range(2, self.k + 1):
            # slot p gets the (s-1)-step prediction made at slot p-1; slots without an anchor get the true chunk
            fed = torch.roll(prev, 1, 0).to(chunks.dtype)
            rin = torch.where((pos >= s - 1)[:, None], fed, chunks)
            if self.predictor_only:
                rin = rin.detach()
            z = self.run_pass(s, rin, layout, kv)
            pred = head(z).float()
            ok = (pos >= s - 1) & (pos + 1 < lens[row])
            if valid is not None:
                ok = ok & valid[(torch.arange(T1, device=ok.device) + 1).clamp(max=T1 - 1)]
            idx = torch.nonzero(ok).squeeze(1)
            nxt = pred
            if feed_head is not None:
                nxt = feed_head(z).float()
            if idx.numel():
                out[s] = red(dist(pred[idx], tgt[idx + 1]), idx)
                if feed_head is not None and in_tgt is not None:
                    self.last_in[s] = red(dist(nxt[idx], in_tgt[idx + 1]), idx)
                with torch.no_grad():
                    self.last_l1[s] = (pred[idx] - tgt[idx + 1]).abs().mean()
                    copy[s] = (tgt[idx - (s - 1)] - tgt[idx + 1]).abs().mean()
            prev = nxt
        self.cap.clear(self.tag)
        return out, copy
