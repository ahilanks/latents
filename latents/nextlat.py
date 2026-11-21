"""Next-latent prediction (NextLat, Teoh et al. 2025, arXiv:2511.05963) on a byte-level 1-stage H-Net.

NextLat adds one auxiliary loss to next-token training: a small latent transition model p_psi predicts the model's own
next hidden state from the current hidden state and the next input,
    h_hat_{t+1} = h_t + MLP(LayerNorm([h_t, x_{t+1}])),     loss = SmoothL1(h_hat_{t+1}, sg[h_{t+1}]),
so every hidden state has to be a summary of the past from which the next one can be computed (a belief state).
Nothing changes at inference: p_psi is used only in training.

Here the hidden states are the main network's last-layer outputs, one per chunk (full width D1), and the "next token"
is the next chunk's input to the main network (the encoder's chunk vector, D0). Everything else is as in the paper:
a three-layer GELU MLP on the layer-normalised concatenation, a residual update, stop-gradient on the target, and an
optional rollout over `horizon` chunks (teacher-forced inputs, every step supervised, averaged). The next-byte loss
trains the whole model as usual; the next-latent loss sends gradient into the main network (through h_t) and into
the encoder and router inputs (through the chunk vector). Predictions never cross a row.

Optional KL term (`kl` > 0, paper eq. 4): the decoder reads the predicted states instead of the real ones and its
byte distribution is pulled to the normal one. The decoder is frozen for that pass (the gradient goes only through
the predicted states), and it costs one extra decoder pass.

`norm` measures the next-latent loss in units of the states' own spread, so that shrinking the states' variation
does not lower it. Without it the H-Net satisfies the loss by making its states almost constant (NL-h1-s0, 2026-11-20:
variation 0.4% of the states' mean square vs 10.8% in the normal H-Net; the deep decoder reads the small differences
at no cost). The spread is a constant for the gradient.
  scale  one number: the root of the mean per-dimension variance over the batch's chunks
  batch  per dimension (mean and standard deviation). Dimensions with a tiny spread get a huge gradient: NL-h1n-s0
         had gradient norm ~900 and val bpb 2.29 at step 1000 (others 1.62); kept only for the record.

Config (model.nextlat): {weight: 1.0, kl: 0.0, horizon: 1, hidden: 1024, layers: 3, norm: none}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NextLat(nn.Module):
    def __init__(self, D1, D0, weight=1.0, kl=0.0, horizon=1, hidden=1024, layers=3, norm="none"):
        super().__init__()
        assert norm in ("none", "scale", "batch")
        self.weight, self.kl, self.horizon, self.norm_mode = float(weight), float(kl), int(horizon), norm
        hidden, layers = int(hidden), int(layers)
        self.norm = nn.LayerNorm(D1 + D0)
        dims = [D1 + D0] + [hidden] * (layers - 1) + [D1]
        mods = []
        for i in range(layers):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < layers - 1:
                mods.append(nn.GELU())
        self.mlp = nn.Sequential(*mods)
        nn.init.zeros_(self.mlp[-1].weight)  # starts as "the next state = this state"
        nn.init.zeros_(self.mlp[-1].bias)

    def step(self, h, c):
        """(h_t (n, D1), next chunk input (n, D0)) -> predicted h_{t+1} (n, D1)."""
        x = self.norm(torch.cat((h, c.to(h.dtype)), -1).float())
        return h.float() + self.mlp(x)

    def rollout(self, out):
        """Per horizon i = 1..d: (index of the target chunk (n_i,), prediction (n_i, D1)). Chunks are packed row
        after row (out.cu1); a prediction for chunk j + i starts from the real state at chunk j of the same row."""
        z, c, cu1 = out.z_full, out.chunks, out.cu1.long()
        T1 = z.shape[0]
        lens = cu1[1:] - cu1[:-1]
        end = torch.repeat_interleave(cu1[1:], lens)  # row end of every chunk
        src = torch.arange(T1, device=z.device)
        cur, res = z, []
        for i in range(1, self.horizon + 1):
            ok = src + i < end[src]
            src, cur = src[ok], cur[ok]
            if src.numel() == 0:
                break
            cur = self.step(cur, c[src + i])
            res.append((src + i, cur))
        return res

    def loss(self, model, out, x, mc):
        """Weighted next-latent (+ KL) loss and its stats."""
        z = out.z_full
        res = self.rollout(out)
        if not res:
            zero = z.sum() * 0.0
            return zero, {"nl_loss": zero.detach()}
        tgt = z.detach().float()
        if self.norm_mode == "batch":  # in units of each dimension's spread over the batch's chunks
            mu, sd = tgt.mean(0), tgt.std(0).clamp(min=1e-4)
            l_h = sum(F.smooth_l1_loss((p - mu) / sd, (tgt[idx] - mu) / sd) for idx, p in res) / self.horizon
        elif self.norm_mode == "scale":  # in units of the states' overall spread
            sc = tgt.var(0).mean().sqrt().clamp(min=1e-4)
            l_h = sum(F.smooth_l1_loss(p / sc, tgt[idx] / sc) for idx, p in res) / self.horizon
        else:
            l_h = sum(F.smooth_l1_loss(p, tgt[idx]) for idx, p in res) / self.horizon
        st = {"nl_loss": l_h.detach()}
        total = self.weight * l_h
        if self.kl > 0:
            l_kl = self.kl_loss(model, out, x, mc, res)
            st["nl_kl"] = l_kl.detach()
            total = total + self.kl * l_kl
        return total, st

    def kl_loss(self, model, out, x, mc, res):
        """KL(bytes from the real states || bytes from the one-step predicted states), frozen decoder."""
        bb = model.backbone
        bp = out.bpred[0]
        B, L = x.shape
        D0 = out.chunks.shape[-1]
        idx, pred = res[0]
        zhat = out.z.detach().clone()
        zhat[idx] = pred[..., :D0].to(zhat.dtype)  # a row's first chunk has no prediction: the real state
        frozen = [p for mod in (bb.residual_proj, bb.decoder, model.lm_head) for p in mod.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        try:
            residual = bb.residual_proj(out.enc.detach().to(bb.residual_proj.weight.dtype))
            prob = bp.boundary_prob.detach() if mc.get("smoothing", True) else torch.ones_like(bp.boundary_prob)
            d = bb.dechunk_layer(zhat, bp.boundary_mask, prob, out.cu1)
            hd = bb.residual_func(d.to(residual.dtype), residual, bp.selected_probs.detach()).to(d.dtype)
            if getattr(model, "dec_emb", None) is not None:
                hd = hd + model.dec_emb(x).flatten(0, 1).to(hd.dtype).detach()
            hd = bb.decoder(hd, cu_seqlens=out.cu0, max_seqlen=L)
            logq = F.log_softmax(model.lm_head(hd).view(B, L, -1).float(), -1)
        finally:
            for p in frozen:
                p.requires_grad_(True)
        logp = F.log_softmax(out.logits.detach().float(), -1)
        kl = (logp.exp() * (logp - logq)).sum(-1)  # (B, L)
        has_pred = bp.boundary_mask.view(B, L).long().cumsum(1) >= 2  # bytes read from a predicted state
        return (kl * has_pred).sum() / has_pred.sum().clamp(min=1)

    @torch.no_grad()
    def eval_metrics(self, out):
        """How well the next state is predicted, per batch: the loss, the share of the targets' variation explained by
        the one-step prediction, by the last rollout step, and by copying the current state; the states' effective
        rank (participation ratio)."""
        res = self.rollout(out)
        if not res:
            return {}
        z = out.z_full.float()
        var = (z - z.mean(0)).pow(2).sum(-1).mean().clamp(min=1e-12)
        ms = z.pow(2).sum(-1).mean().clamp(min=1e-12)

        def expl(idx, p):
            return float(1 - (p.float() - z[idx]).pow(2).sum(-1).mean() / var)
        idx, pred = res[0]
        m = {"nl/loss": float(sum(F.smooth_l1_loss(p, z[i]) for i, p in res) / self.horizon),
             "nl/explained": expl(idx, pred), "nl/explained_copy": expl(idx, z[idx - 1])}
        if len(res) > 1:
            m[f"nl/explained_h{len(res)}"] = expl(*res[-1])
        m["nl/variation_share"] = float(var / ms)  # the states' variance as a share of their mean square
        m["nl/state_rank"] = state_rank(out)
        return m


def state_rank(out):
    """Effective rank (participation ratio) of the main network's outputs."""
    with torch.autocast(device_type=out.z_full.device.type, enabled=False):
        z = out.z_full.float()[:20000]
        zc = z - z.mean(0)
        ev = torch.linalg.eigvalsh(zc.T @ zc / max(zc.shape[0] - 1, 1)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))
