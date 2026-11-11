"""Latent chain of thought, trained: the reasoning is never decoded, only thought.

For a problem (question Q, written reasoning R, answer A; latents/tasks.py):
  1. the question's chunks go into the main network as usual;
  2. k thoughts follow, generated one after another: thought j = head(main-net output at the last position), fed back
     as the next main-net input (exactly what scripts/latent_cot.py does at test time); k = the number of chunks the
     model's router puts on the written reasoning R;
  3. the answer is decoded from "Q #### " with the thoughts in the main network's sequence (a boundary is forced at
     "####", as in the evaluator).
Loss = byte cross-entropy on the answer, back-propagated through the whole thought chain
     + beta * L1(thought j, latent of the j-th chunk of R)   -- the same JEPA target the latent loss uses in
       pretraining (EMA encoder state at the chunk's boundary), so every thought has a target from step one.
1-level H-Nets only (no model.nest, no surprise router).

loss_pure (the user's design: the decoder never shapes the latents):
  - normal training on the full written examples (byte loss -> decoder; latent loss -> trunk; with detach_trunk the
    byte loss cannot reach the trunk);
  - thoughts trained only by latent targets: thought j vs the latent of the j-th reasoning chunk; its input is the
    true chunk latent with probability p_true (read first), else the model's own prediction (then produce);
  - after the thoughts the trunk predicts the latents of the answer chunks ("#### 86") -- again a latent loss, so the
    thoughts must carry what the answer's latent needs;
  - the decoder is trained to decode the answer from the detached predicted latents (stop-gradient): its loss
    updates the decoder only.
"""
import random

import torch
import torch.nn.functional as F
from hnet.modules.dc import RoutingModuleOutput

from latents.models import fixed_routing, model_forward

BOS = 254


class Thinker:
    def __init__(self, model, lat, mc):
        assert not mc.get("nest"), "1-level models only"
        assert mc.get("router", "learned") in ("learned", "identity") or mc.get("pool_k"), "learned/identity/pool routers only"
        self.m, self.lat, self.mc = model, lat, mc
        self.bb = model.backbone
        self.inner = self.bb.main_network
        self.iso = self.inner.main_network

    # ------------------------------------------------------------------ pieces shared with scripts/latent_cot.py
    def pad(self, v):
        p = self.inner.pad_dimension
        return v if p is None else torch.cat((v, p.expand(v.shape[0], -1).to(v.dtype)), -1)

    def batch(self, seqs):
        L = (max(len(s) for s in seqs) + 63) // 64 * 64
        x = torch.zeros(len(seqs), L, dtype=torch.long, device=self.m.lm_head.weight.device)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(list(s), device=x.device)
        return x

    def encode(self, x):
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        h = self.bb.encoder(self.m.embeddings(x).flatten(0, 1), cu_seqlens=cu, max_seqlen=L)
        bp = fixed_routing(B, L, self.mc["pool_k"], x.device) if self.mc.get("pool_k") else self.bb.routing_module(h, cu_seqlens=cu)
        return h, cu, bp

    def run_main(self, seqs):
        lens = [s.shape[0] for s in seqs]
        cu = torch.tensor([0] + lens, device=seqs[0].device).cumsum(0).to(torch.int32)
        z = self.iso(torch.cat(seqs, 0), cu_seqlens=cu, max_seqlen=max(lens))
        return z, cu

    # ------------------------------------------------------------------ the chain
    def thoughts(self, q_bytes, ks, teacher=None, p_true=0.0, return_inputs=False):
        """q_bytes: per example BOS + question. ks: thoughts per example. Returns per-example (k, D_t) predictions.
        teacher: per example (k, D_t) true chunk latents; each thought input is the true one with probability p_true."""
        x = self.batch(q_bytes)
        B, L = x.shape
        h, cu, bp = self.encode(x)
        mask = bp.boundary_mask.view(B, L)
        hv = h.view(B, L, -1)
        seqs = [self.pad(hv[b, :len(q_bytes[b])][mask[b, :len(q_bytes[b])]]) for b in range(B)]
        outs, ins = [[] for _ in range(B)], [[] for _ in range(B)]
        for t in range(max(ks) if ks else 0):
            act = [b for b in range(B) if t < ks[b]]
            z, zcu = self.run_main([seqs[b] for b in act])
            c = self.lat.head(z[zcu[1:].long() - 1]).to(h.dtype)
            for i, b in enumerate(act):
                outs[b].append(c[i])
                use_true = teacher is not None and p_true > 0 and random.random() < p_true
                nxt = teacher[b][t:t + 1].to(h.dtype) if use_true else c[i:i + 1]
                ins[b].append(nxt[0])
                seqs[b] = torch.cat((seqs[b], self.pad(nxt)), 0)
        d = [m for m in self.lat.head.modules() if isinstance(m, torch.nn.Linear)][-1].out_features
        st = lambda xs: [torch.stack(o) if o else h.new_zeros(0, d) for o in xs]  # noqa: E731
        if return_inputs:
            return st(outs), st(ins), [int(m.sum()) for m in mask]
        return st(outs), [int(m.sum()) for m in mask]

    def answer_logits(self, ctx_bytes, q_lens, thoughts):
        """ctx: per example BOS + question + '#### ' + answer. q_lens: len(BOS + question). Logits (B, L, V)."""
        x = self.batch(ctx_bytes)
        B, L = x.shape
        h, cu, bp = self.encode(x)
        idx = torch.tensor([b * L + q for b, q in enumerate(q_lens)], device=x.device)
        mask, prob, sel = bp.boundary_mask.clone(), bp.boundary_prob.clone(), bp.selected_probs.clone()
        mask[idx] = True
        prob[idx] = torch.tensor([0.0, 1.0], device=x.device, dtype=prob.dtype)
        sel[idx] = 1.0
        bp = RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)
        residual = self.bb.residual_proj(h.to(self.bb.residual_proj.weight.dtype))
        hc, cu1, max1, _ = self.bb.chunk_layer(h, bp.boundary_mask, cu)
        mv = mask.view(B, L)
        qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
        rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
        seqs = [torch.cat((self.pad(rows[b][:qn[b]]), self.pad(thoughts[b].to(hc.dtype)), self.pad(rows[b][qn[b]:])), 0)
                for b in range(B)]
        z, zcu = self.run_main(seqs)
        zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
        zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + thoughts[b].shape[0]:]), 0) for b in range(B)], 0)
        src = self.lat.head(zc).to(hc.dtype) if self.lat.decode_from_pred else zc[..., : hc.shape[-1]]
        out = self.bb.dechunk_layer(src, bp.boundary_mask, bp.boundary_prob, cu1)
        hd = self.bb.residual_func(out.to(residual.dtype), residual, bp.selected_probs).to(out.dtype)
        if getattr(self.m, "dec_emb", None) is not None:
            hd = hd + self.m.dec_emb(x).flatten(0, 1).to(hd.dtype)
        hd = self.bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
        return self.m.lm_head(hd).view(B, L, -1)

    @torch.no_grad()
    def reasoning_targets(self, probs):
        """Per example: (k, D_t) latents of the written reasoning's chunks, from the pretraining target encoder."""
        full = [bytes([BOS]) + (p["q"] + p["cot"] + p["answer"]).encode() for p in probs]
        x = self.batch(full)
        B, L = x.shape
        out = model_forward(self.m, x, self.mc, self.lat)
        tgt = self.lat.targets(self.m, out, x).float()
        if getattr(self.lat, "target_norm", None) == "ln":
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        mv = out.bpred[0].boundary_mask.view(B, L)
        off = torch.cat((torch.zeros(1, device=x.device, dtype=torch.long), mv.sum(1).cumsum(0)))
        res = []
        for b, p in enumerate(probs):
            q = 1 + len(p["q"].encode())
            r = q + len(p["cot"].encode())
            qn, k = int(mv[b, :q].sum()), int(mv[b, q:r].sum())
            res.append(tgt[off[b] + qn: off[b] + qn + k])
        return res

    def loss(self, probs, beta=1.0):
        tg = self.reasoning_targets(probs)
        ks = [t.shape[0] for t in tg]
        qb = [bytes([BOS]) + p["q"].encode() for p in probs]
        th, _ = self.thoughts(qb, ks)
        l_th = torch.cat([(a.float() - t).abs().mean(-1) for a, t in zip(th, tg) if t.shape[0]]).mean()
        ctx = [q + p["answer"].encode() for q, p in zip(qb, probs)]
        lg = self.answer_logits(ctx, [len(q) for q in qb], th)
        ce, n = 0.0, 0
        for b, (q, c) in enumerate(zip(qb, ctx)):
            a0 = len(q) + 5  # first answer byte, after "#### "
            ce = ce + F.cross_entropy(lg[b, a0 - 1: len(c) - 1].float(), torch.tensor(list(c[a0:]), device=lg.device),
                                      reduction="sum")
            n += len(c) - a0
        ce = ce / n
        return ce + beta * l_th, {"answer_ce": ce.detach(), "thought_l1": l_th.detach(), "mean_k": sum(ks) / len(ks)}

    # ------------------------------------------------------------------ the pure loss (decoder on stop-gradient)
    def _answer_pass_pure(self, ctx_bytes, q_lens, inputs):
        """Trunk part with grad (latent loss on the answer chunks), decoder part on detached inputs (CE)."""
        x = self.batch(ctx_bytes)
        B, L = x.shape
        h, cu, bp = self.encode(x)
        idx = torch.tensor([b * L + q for b, q in enumerate(q_lens)], device=x.device)
        mask, prob, sel = bp.boundary_mask.clone(), bp.boundary_prob.clone(), bp.selected_probs.clone()
        mask[idx] = True
        prob[idx] = torch.tensor([0.0, 1.0], device=x.device, dtype=prob.dtype)
        sel[idx] = 1.0
        hc, cu1, max1, _ = self.bb.chunk_layer(h, mask, cu)
        mv = mask.view(B, L)
        qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
        rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
        ks = [t.shape[0] for t in inputs]
        seqs = [torch.cat((self.pad(rows[b][:qn[b]]), self.pad(inputs[b].to(hc.dtype)), self.pad(rows[b][qn[b]:])), 0)
                for b in range(B)]
        z, zcu = self.run_main(seqs)
        zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
        # (a) latent loss on the answer chunks: predicted from the position before each, vs the EMA target latents
        with torch.no_grad():
            tgt = self.lat.ema_states(self.m, x)[mask].float()
            if getattr(self.lat, "target_norm", None) == "ln":
                tgt = F.layer_norm(tgt, tgt.shape[-1:])
        preds, tgts = [], []
        for b in range(B):
            n_ans = rows[b].shape[0] - qn[b]
            if n_ans <= 0 or qn[b] + ks[b] == 0:
                continue
            pos = torch.arange(qn[b] + ks[b] - 1, qn[b] + ks[b] - 1 + n_ans, device=x.device)
            preds.append(self.lat.head(zr[b][pos]).float())
            tgts.append(tgt[cu1[b] + qn[b]: cu1[b] + qn[b] + n_ans])
        l_ans_lat = (torch.cat(preds) - torch.cat(tgts)).abs().mean()
        # (b) the decoder reads the detached predicted latents: its loss updates the decoder only
        zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + ks[b]:]), 0) for b in range(B)], 0).detach()
        with torch.no_grad():
            src = self.lat.head(zc).to(hc.dtype) if self.lat.decode_from_pred else zc[..., : hc.shape[-1]]
        residual = self.bb.residual_proj(h.detach().to(self.bb.residual_proj.weight.dtype))
        out = self.bb.dechunk_layer(src, mask, prob.detach(), cu1)
        hd = self.bb.residual_func(out.to(residual.dtype), residual, sel.detach()).to(out.dtype)
        if getattr(self.m, "dec_emb", None) is not None:
            hd = hd + self.m.dec_emb(x).flatten(0, 1).to(hd.dtype)
        hd = self.bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
        return self.m.lm_head(hd).view(B, L, -1), l_ans_lat

    def loss_pure(self, probs, p_true=0.0, beta=1.0, gamma=1.0, w_lm=1.0, step=0, total_steps=1):
        B = len(probs)
        full = [bytes([BOS]) + (p["q"] + p["cot"] + p["answer"]).encode() for p in probs]
        x = self.batch(full)
        L = x.shape[1]
        # 1. normal training on the full written examples (the model's own recipe)
        out = model_forward(self.m, x, self.mc, self.lat)
        lens = torch.tensor([len(f) for f in full], device=x.device)
        valid = (torch.arange(L - 1, device=x.device)[None, :] < (lens[:, None] - 1)).float()
        ce_tok = F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, L - 1)
        ce_lm = (ce_tok * valid).sum() / valid.sum()
        l_lat, _ = self.lat.loss(self.m, out, x, step=step, total_steps=total_steps)
        # thought targets and teacher inputs: the reasoning chunks of this forward
        with torch.no_grad():
            tgt = self.lat.targets(self.m, out, x).float()
            if getattr(self.lat, "target_norm", None) == "ln":
                tgt = F.layer_norm(tgt, tgt.shape[-1:])
            chunks = out.chunks.detach()
            mv = out.bpred[0].boundary_mask.view(B, L)
            off = torch.cat((torch.zeros(1, device=x.device, dtype=torch.long), mv.sum(1).cumsum(0)))
            tg, teach = [], []
            for b, p in enumerate(probs):
                q = 1 + len(p["q"].encode())
                r = q + len(p["cot"].encode())
                qn_, k = int(mv[b, :q].sum()), int(mv[b, q:r].sum())
                tg.append(tgt[off[b] + qn_: off[b] + qn_ + k])
                teach.append(chunks[off[b] + qn_: off[b] + qn_ + k])
        # 2. thoughts, trained only by latent targets
        ks = [t.shape[0] for t in tg]
        qb = [bytes([BOS]) + p["q"].encode() for p in probs]
        pred, ins, _ = self.thoughts(qb, ks, teacher=teach, p_true=p_true, return_inputs=True)
        l_th = torch.cat([(a.float() - t).abs().mean(-1) for a, t in zip(pred, tg) if t.shape[0]]).mean()
        # 3. answer: latent loss on its chunks (trunk) + decoder CE on stop-gradient latents (decoder only)
        ctx = [q + p["answer"].encode() for q, p in zip(qb, probs)]
        lg, l_ans_lat = self._answer_pass_pure(ctx, [len(q) for q in qb], ins)
        ce, n = 0.0, 0
        for b, (q, c) in enumerate(zip(qb, ctx)):
            a0 = len(q) + 5
            ce = ce + F.cross_entropy(lg[b, a0 - 1: len(c) - 1].float(), torch.tensor(list(c[a0:]), device=lg.device),
                                      reduction="sum")
            n += len(c) - a0
        ce_ans = ce / n
        loss = w_lm * (ce_lm + l_lat) + beta * l_th + gamma * l_ans_lat + ce_ans
        return loss, {"answer_ce": ce_ans.detach(), "thought_l1": l_th.detach(), "answer_lat_l1": l_ans_lat.detach(),
                      "lm_ce": ce_lm.detach(), "lm_lat": l_lat.detach(), "mean_k": sum(ks) / len(ks), "p_true": p_true,
                      "_ce_ans": ce_ans}
