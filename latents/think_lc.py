"""Latent thoughts for learned-cut models (the router-brief variants: model.chunk_pool = mean, learned router).

Same idea as latents/think.py, with what learned cuts need:
  - a chunk's input is the p-weighted mean of its segment (the bytes after the previous boundary up to and including
    this one), targets come from LatentObjective.targets (pooled EMA or chunk-local), as in pretraining;
  - a thought stands for one chunk the model's own router puts on the written reasoning, so the number of thoughts
    for a piece of reasoning = the number of boundaries the router places in it;
  - boundaries are forced where the parts meet, so no chunk straddles them: at the last byte of the question, at the
    last byte of the reasoning, at the last byte of the example, and (when thoughts sit before it) at the first byte
    after the thoughts, so every later byte is decoded from a main-network output that comes after the thoughts;
  - the ratio loss stays on (model.alpha), over the real bytes only: without it the cuts die.
Training rules as before: the decoder is trained on stop-gradient latents and never shapes them.

LCThinker.loss_pure      thought stage A: teacher-forced on the written traces (thought inputs = the true chunks)
LCThinker.loss_coconut   practice stage: the first s written steps are replaced by the model's own thoughts; each own
                         thought keeps its latent target (the chunk it replaces), the rest of the text is predicted
LCThinker.prepare_eval / answer_logits   what scripts/think_withhold.py needs (read, n steps computed, all own)
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from hnet.modules.dc import RoutingModuleOutput

from latents.models import forward_lm, forward_opts, segment_pool
from latents.think import BOS, Thinker


def step_spans(cot):
    """Per written step (the restated assignment first): (start, value start, value end, value), offsets in cot.
    Works for both task formats ("b=17+25=42." and "b = 17 + 25 = 42 .")."""
    import re
    out = []
    for m in re.finditer(r"[a-z] ?=[^.]*\.", cot):
        v = re.search(r"(\d+) ?\.$", m.group())
        out.append((m.start(), m.start() + v.start(1), m.start() + v.end(1), int(v.group(1))))
    return out


def make_thinker(model, lat, mc):
    return LCThinker(model, lat, mc) if mc.get("chunk_pool") == "mean" else Thinker(model, lat, mc)


class LCThinker(Thinker):
    def __init__(self, model, lat, mc):
        super().__init__(model, lat, mc)
        assert not mc.get("pool_k") and mc.get("router", "learned") == "learned", "learned-cut models only"
        self.alpha = float(mc.get("alpha", 0.03))
        self.N = float(mc.get("N", 6))
        # what a thought is made with: the input-predicting head if the model has one, else the target-predicting head
        self.th_head = lat.in_head if getattr(lat, "in_head", None) is not None else lat.head
        self.in_space = getattr(lat, "in_head", None) is not None
        # latent.cut_from = <pretrained run>: NOT end to end. The cuts (mask and probabilities) come from a frozen copy
        # of that run's byte embedding + encoder + router, so reasoning training cannot move them; everything else
        # still trains. A labelled comparison for "what if the cuts stayed as pretrained".
        self.cut = None
        if lat.cfg.get("cut_from"):
            import copy
            from pathlib import Path as _P
            root = _P(__file__).resolve().parents[1]
            ck = torch.load(root / lat.cfg["cut_from"] / "final.pt", map_location="cpu", weights_only=False)
            src = copy.deepcopy(model)
            src.load_state_dict(ck["model"])
            dev = next(model.parameters()).device
            self.cut = SimpleNamespace(emb=src.embeddings.to(dev).eval().requires_grad_(False),
                                       enc=src.backbone.encoder.to(dev).eval().requires_grad_(False),
                                       rm=src.backbone.routing_module.to(dev).eval().requires_grad_(False))

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def force(bp, idx):
        mask, prob, sel = bp.boundary_mask.clone(), bp.boundary_prob.clone(), bp.selected_probs.clone()
        mask[idx] = True
        prob[idx] = torch.tensor([0.0, 1.0], device=prob.device, dtype=prob.dtype)
        sel[idx] = 1.0
        return RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)

    def route(self, x, h, cu, idx):
        """Boundaries (with the forced ones): the model's own router, or the frozen pretrained copy (cut_from)."""
        if self.cut is None:
            return self.force(self.bb.routing_module(h, cu_seqlens=cu), idx)
        with torch.no_grad():
            hc_ = self.cut.enc(self.cut.emb(x).flatten(0, 1), cu_seqlens=cu, max_seqlen=x.shape[1])
            return self.force(self.cut.rm(hc_, cu_seqlens=cu), idx)

    def enc_chunks(self, x, idx):
        """Encoder, router with forced boundaries, pooled chunk inputs. idx: flat byte indices to force."""
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        h = self.bb.encoder(self.m.embeddings(x).flatten(0, 1), cu_seqlens=cu, max_seqlen=L)
        bp = self.route(x, h, cu, idx)
        _, cu1, _, _ = self.bb.chunk_layer(h, bp.boundary_mask, cu)
        hc = segment_pool(h, bp.boundary_prob[..., -1], bp.boundary_mask, cu).to(h.dtype)
        return h, cu, bp, hc, cu1

    @torch.no_grad()
    def targets_for(self, x, bp, cu):
        t = self.lat.targets(self.m, SimpleNamespace(bpred=[bp], cu0=cu, extras={}), x).float()
        return F.layer_norm(t, t.shape[-1:]) if getattr(self.lat, "target_norm", None) == "ln" else t

    def ratio(self, bp, real):
        """The ratio loss (H-Net eq. 10) over the real bytes only (rows are padded). With latent.ratio_window = W it is
        computed in every window of W bytes and averaged, so the cut rate has to be about 1/N everywhere and not only
        on average: a global rate can be met in the question while the reasoning is left as one chunk."""
        N = self.N
        W = int(self.lat.cfg.get("ratio_window", 0) or 0)
        m, pr, rl = bp.boundary_mask.float(), bp.boundary_prob[..., -1].float(), real.float()
        if W:
            n = (rl.numel() // W) * W
            cnt = rl[:n].view(-1, W).sum(1)
            ok = cnt >= W / 2
            Fm = ((m * rl)[:n].view(-1, W).sum(1) / cnt.clamp(min=1))[ok]
            G = ((pr * rl)[:n].view(-1, W).sum(1) / cnt.clamp(min=1))[ok]
        else:
            Fm, G = m[real].mean(), pr[real].mean()
        return (N / (N - 1) * ((N - 1) * Fm * G + (1 - Fm) * (1 - G))).mean()

    def wmean(self, err, w):
        """Mean of per-chunk errors: over bytes (each chunk weighted by its segment length) when the model's latent
        loss is per byte (latent.loss_weight = len), else over chunks."""
        if getattr(self.lat, "loss_weight", "chunk") == "len":
            return (w * err).sum() / w.sum().clamp(min=1.0)
        return err.mean()

    @staticmethod
    def seg_lens(mask, cu1):
        pos = torch.nonzero(mask).squeeze(1)
        seg = (pos - torch.cat((pos.new_full((1,), -1), pos[:-1]))).float()
        seg[cu1[:-1].long()] = 1.0
        return seg

    def flat(self, B, L, pos_lists, device):
        return torch.tensor([b * L + p for b in range(B) for p in pos_lists[b]], device=device, dtype=torch.long)

    def full(self, probs):
        """The full written example (question + reasoning + answer) with boundaries forced at the ends of its parts."""
        qb = [bytes([BOS]) + p["q"].encode() for p in probs]
        full = [q + (p["cot"] + p["answer"]).encode() for q, p in zip(qb, probs)]
        x = self.batch(full)
        B, L = x.shape
        q = [len(v) for v in qb]
        r = [qi + len(p["cot"].encode()) for qi, p in zip(q, probs)]
        idx = self.flat(B, L, [[q[b] - 1, r[b] - 1, len(full[b]) - 1] for b in range(B)], x.device)
        opts = forward_opts(self.mc, self.lat)
        opts["router_fn"] = lambda h, cu: self.route(x, h, cu, idx)
        out = forward_lm(self.m, x, pool_k=0, detach_main=bool(self.lat.detach_main),
                         main_out_fn=(self.lat.head if self.lat.decode_from_pred else None), opts=opts)
        if self.lat.needs_ema_states(self.mc):
            out.extras["ema_h"] = self.lat.ema_states(self.m, x)
        mv = out.bpred[0].boundary_mask.view(B, L)
        lens = torch.tensor([len(v) for v in full], device=x.device)
        real = torch.arange(L, device=x.device)[None, :] < lens[:, None]
        off = torch.cat((torch.zeros(1, device=x.device, dtype=torch.long), mv.sum(1).cumsum(0))).tolist()
        qn = [int(mv[b, :q[b]].sum()) for b in range(B)]
        k = [int(mv[b, q[b]:r[b]].sum()) for b in range(B)]
        return SimpleNamespace(x=x, out=out, qb=qb, full=full, q=q, r=r, mv=mv, real=real, lens=lens, off=off, qn=qn, k=k,
                               B=B, L=L, seg=self.seg_lens(out.bpred[0].boundary_mask, out.cu1))

    def base_terms(self, f, step, total_steps):
        """Normal training on the full written examples: decoder CE, latent loss, ratio loss; targets and predictions."""
        out, x, B, L = f.out, f.x, f.B, f.L
        valid = (torch.arange(L - 1, device=x.device)[None, :] < (f.lens[:, None] - 1)).float()
        ce_tok = F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, L - 1)
        ce_lm = (ce_tok * valid).sum() / valid.sum()
        chunk_valid = f.real.flatten()[out.bpred[0].boundary_mask]
        l_lat, st_lat = self.lat.loss(self.m, out, x, step=step, total_steps=total_steps, chunk_valid=chunk_valid)
        l_ratio = self.ratio(out.bpred[0], f.real.flatten())
        with torch.no_grad():
            tgt = self.lat.targets(self.m, out, x).float()
            if getattr(self.lat, "target_norm", None) == "ln":
                tgt = F.layer_norm(tgt, tgt.shape[-1:])
        f.tgt_in, f.l_in = None, None
        if self.in_space:
            # second head: predict the next chunk's input (every real chunk of the example)
            from latents.latent import next_pairs
            f.tgt_in = self.lat.input_targets(self.m, out, x).float()
            j = next_pairs(out.cu1, tgt.shape[0], 1)
            j = j[chunk_valid[j + 1]]
            if getattr(self.lat, "in_loss", False):  # already part of the latent loss (pretrained with it)
                f.l_in = st_lat["lat/in_l"]
            else:
                f.l_in = self.wmean(self.lat.dist_fn(self.lat.in_head(out.z_full[j]).float(), f.tgt_in[j + 1]).mean(-1),
                                    f.seg[j + 1])
                l_lat = l_lat + f.l_in
        return ce_lm, l_lat, l_ratio, tgt, valid, st_lat

    def answer_ce(self, lg, ctx, a0s):
        ce, n = 0.0, 0
        for b, c in enumerate(ctx):
            a0 = a0s[b]
            ce = ce + F.cross_entropy(lg[b, a0 - 1: len(c) - 1].float(), torch.tensor(list(c[a0:]), device=lg.device),
                                      reduction="sum")
            n += len(c) - a0
        return ce / n

    def after_thoughts(self, qb, tail, thoughts, dec_true=False, grad_thoughts=True):
        """[question chunks; thoughts; chunks of `tail`] where tail = the text after the thoughts (remaining written
        steps + answer, or the answer alone), encoded without the bytes the thoughts stand for. Returns decoder logits
        on stop-gradient predictions, the latent loss on the tail's chunks, logits on true latents (dec_true)."""
        ed = [q + t for q, t in zip(qb, tail)]
        x = self.batch(ed)
        B, L = x.shape
        idx = self.flat(B, L, [[len(qb[b]) - 1, len(qb[b]), len(ed[b]) - 1] for b in range(B)], x.device)
        h, cu, bp, hc, cu1 = self.enc_chunks(x, idx)
        mv = bp.boundary_mask.view(B, L)
        qn = [int(mv[b, :len(qb[b])].sum()) for b in range(B)]
        ks = [t.shape[0] for t in thoughts]
        rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
        seqs = [torch.cat((self.pad(rows[b][:qn[b]]), self.pad(thoughts[b].to(hc.dtype)), self.pad(rows[b][qn[b]:])), 0)
                for b in range(B)]
        z, zcu = self.run_main(seqs)
        zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
        tgt = self.targets_for(x, bp, cu)
        seg = self.seg_lens(bp.boundary_mask, cu1)  # bytes in each chunk's segment
        preds, tgts, wts = [], [], []
        for b in range(B):
            n_tail = int(mv[b, :len(ed[b])].sum()) - qn[b]  # chunks that end inside the example
            pos = torch.arange(qn[b] + ks[b] - 1, qn[b] + ks[b] - 1 + n_tail, device=x.device)
            preds.append(self.lat.head(zr[b][pos]).float())
            tgts.append(tgt[cu1[b] + qn[b]: cu1[b] + qn[b] + n_tail])
            wts.append(seg[cu1[b] + qn[b]: cu1[b] + qn[b] + n_tail])
        P, T = torch.cat(preds), torch.cat(tgts)
        l_tail = self.wmean(self.lat.dist_fn(P, T).mean(-1), torch.cat(wts))
        zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + ks[b]:]), 0) for b in range(B)], 0).detach()
        with torch.no_grad():
            src = self.lat.head(zc).to(hc.dtype)
        mask, prob, sel = bp.boundary_mask, bp.boundary_prob, bp.selected_probs
        lg = self._decode(x, h, mask, prob, sel, cu, cu1, src)
        lg_true = self._decode(x, h, mask, prob, sel, cu, cu1, self._true_src(src, tgt, cu1)) if dec_true else None
        with torch.no_grad():
            l1 = (P - T).abs().mean()
        return SimpleNamespace(lg=lg, lg_true=lg_true, l_tail=l_tail, l1=l1, ed=ed, qn=qn, rows=rows)

    # ------------------------------------------------------------------ stage A: teacher-forced on the traces
    def loss_pure(self, probs, p_true=1.0, beta=1.0, gamma=1.0, w_lm=1.0, step=0, total_steps=1, dec_true=0.0,
                  thought_grad=False):
        assert p_true >= 1.0, "learned-cut thought stage is teacher-forced (use loss_coconut for own thoughts)"
        f = self.full(probs)
        out, B = f.out, f.B
        ce_lm, l_lat, l_ratio, tgt, valid, st_lat = self.base_terms(f, step, total_steps)
        pred_all = self.lat.head(out.z_full)
        # thoughts: with true chunks as inputs, the prediction of reasoning chunk j is the full pass's own prediction
        ip = [i for b in range(B) for i in range(f.off[b] + f.qn[b] - 1, f.off[b] + f.qn[b] - 1 + f.k[b])]
        ip = torch.tensor(ip, device=tgt.device)
        l_th = self.wmean(self.lat.dist_fn(pred_all[ip].float(), tgt[ip + 1]).mean(-1), f.seg[ip + 1])
        with torch.no_grad():
            l_th_l1 = (pred_all[ip].float() - tgt[ip + 1]).abs().mean()
        # answer: decoded from latents only, the reasoning present only as thought inputs
        chunks = out.chunks if thought_grad else out.chunks.detach()
        teach = [chunks[f.off[b] + f.qn[b]: f.off[b] + f.qn[b] + f.k[b]] for b in range(B)]
        ans = [p["answer"].encode() for p in probs]
        a = self.after_thoughts(f.qb, ans, teach, dec_true=dec_true > 0)
        a0s = [len(q) + 5 for q in f.qb]  # first answer byte, after "#### "
        ce_ans = self.answer_ce(a.lg, a.ed, a0s)
        loss = w_lm * (ce_lm + l_lat) + self.alpha * l_ratio + beta * l_th + gamma * a.l_tail + ce_ans
        extra = {k.replace("lat/", "lat_"): v.detach() for k, v in st_lat.items()
                 if k in ("lat/reg", "lat/reg_cov", "lat/reg_std")}
        if dec_true > 0:
            bp0 = out.bpred[0]
            src_pred = self.lat.head(out.z_full).to(out.chunks.dtype)
            lg_full = self._decode(f.x, out.enc, bp0.boundary_mask, bp0.boundary_prob, bp0.selected_probs, out.cu0,
                                   out.cu1, self._true_src(src_pred, tgt, out.cu1))
            ce_t = F.cross_entropy(lg_full[:, :-1].float().flatten(0, 1), f.x[:, 1:].flatten(), reduction="none").view(B, f.L - 1)
            ce_lm_true = (ce_t * valid).sum() / valid.sum()
            ce_ans_true = self.answer_ce(a.lg_true, a.ed, a0s)
            loss = loss + dec_true * (w_lm * ce_lm_true + ce_ans_true)
            extra |= {"lm_ce_true": ce_lm_true.detach(), "answer_ce_true": ce_ans_true.detach()}
        bm = out.bpred[0].boundary_mask
        return loss, {"answer_ce": ce_ans.detach(), "thought_l1": l_th_l1, "answer_lat_l1": a.l1,
                      "thought_loss": l_th.detach(), "answer_lat_loss": a.l_tail.detach(), "lm_ce": ce_lm.detach(),
                      "lm_lat": l_lat.detach(), "ratio_loss": l_ratio.detach(), "mean_k": sum(f.k) / B, "p_true": 1.0,
                      "cut_rate": bm[f.real.flatten()].float().mean().detach(), "_ce_ans": ce_ans} | extra | (
                          {"in_loss": f.l_in.detach()} if f.l_in is not None else {})

    # ------------------------------------------------------------------ practice: own thoughts replace written steps
    def loss_coconut(self, probs, stages, c=0, w_lm=1.0, step=0, total_steps=1, dec_true=0.0, beta_rest=1.0,
                     anchor_w=0.0, bptt="1"):
        """Stage s replaces the first s written steps with the model's own thoughts, one per chunk its router put on
        them. Each own thought keeps its latent target (the chunk it replaces; weight anchor_w); the text after the
        thoughts is encoded without the replaced bytes and predicted (latent loss); the decoder reads stop-gradient
        predictions. bptt "1": a fed-back thought is detached as the next input; "full": gradient through the chain."""
        assert c == 0, "learned cuts: one thought per replaced chunk"
        f = self.full(probs)
        out, B = f.out, f.B
        ce_lm, l_lat, l_ratio, tgt, valid, _ = self.base_terms(f, step, total_steps)
        tails, n_th = [], []
        for b, (p, s) in enumerate(zip(probs, stages)):
            sp_ = step_spans(p["cot"])
            ks_ = min(s, len(sp_))
            removed = sp_[ks_][0] if ks_ < len(sp_) else len(p["cot"])  # the first ks_ written steps are replaced
            rest = p["cot"][removed:]
            n_th.append(int(f.mv[b, f.q[b]: f.q[b] + removed].sum()))
            tails.append(rest.encode() + p["answer"].encode())
        # own thoughts, one after another, from the question's chunks (the same chunks as in the full example)
        seqs = [self.pad(out.chunks[f.off[b]: f.off[b] + f.qn[b]]) for b in range(B)]
        D0 = out.chunks.shape[-1]
        ths = [[] for _ in range(B)]
        for t in range(max(n_th, default=0)):
            act = [b for b in range(B) if t < n_th[b]]
            z, zcu = self.run_main([seqs[b] for b in act])
            cth = self.th_head(z[zcu[1:].long() - 1]).to(out.chunks.dtype)
            for i, b in enumerate(act):
                ths[b].append(cth[i])
                fed = cth[i:i + 1].detach() if bptt == "1" else cth[i:i + 1]
                seqs[b] = torch.cat((seqs[b], self.pad(fed)), 0)
        th_t = [torch.stack(o) if o else out.chunks.new_zeros(0, D0) for o in ths]
        a = self.after_thoughts(f.qb, tails, th_t, dec_true=dec_true > 0)
        a0s = [len(e) - len(p["answer"]) + 5 for e, p in zip(a.ed, probs)]
        ce_ans = self.answer_ce(a.lg, a.ed, a0s)
        loss = w_lm * (ce_lm + l_lat) + self.alpha * l_ratio + beta_rest * a.l_tail + ce_ans
        extra = {"anchor_w": anchor_w}
        if anchor_w > 0 and sum(n_th) > 0:
            pa = torch.cat([th_t[b].float() for b in range(B) if n_th[b]])
            tsrc = f.tgt_in if self.in_space else tgt  # the chunk each thought replaces, in the space thoughts live in
            ta = torch.cat([tsrc[f.off[b] + f.qn[b]: f.off[b] + f.qn[b] + n_th[b]] for b in range(B) if n_th[b]])
            wa = torch.cat([f.seg[f.off[b] + f.qn[b]: f.off[b] + f.qn[b] + n_th[b]] for b in range(B) if n_th[b]])
            l_anchor = self.wmean(self.lat.dist_fn(pa, ta).mean(-1), wa)
            loss = loss + anchor_w * l_anchor
            extra["anchor_loss"] = l_anchor.detach()
        if dec_true > 0:
            bp0 = out.bpred[0]
            src_pred = self.lat.head(out.z_full).to(out.chunks.dtype)
            lg_full = self._decode(f.x, out.enc, bp0.boundary_mask, bp0.boundary_prob, bp0.selected_probs, out.cu0,
                                   out.cu1, self._true_src(src_pred, tgt, out.cu1))
            ce_t = F.cross_entropy(lg_full[:, :-1].float().flatten(0, 1), f.x[:, 1:].flatten(), reduction="none").view(B, f.L - 1)
            ce_lm_true = (ce_t * valid).sum() / valid.sum()
            ce_ans_true = self.answer_ce(a.lg_true, a.ed, a0s)
            loss = loss + dec_true * (w_lm * ce_lm_true + ce_ans_true)
            extra |= {"lm_ce_true": ce_lm_true.detach(), "answer_ce_true": ce_ans_true.detach()}
        bm = out.bpred[0].boundary_mask
        return loss, {"answer_ce": ce_ans.detach(), "rest_l1": a.l1, "rest_loss": a.l_tail.detach(),
                      "lm_ce": ce_lm.detach(), "lm_lat": l_lat.detach(), "ratio_loss": l_ratio.detach(),
                      "mean_thoughts": sum(n_th) / B, "mean_stage": sum(stages) / B, "mean_k": sum(f.k) / B,
                      "cut_rate": bm[f.real.flatten()].float().mean().detach()} | extra | (
                          {"in_loss": f.l_in.detach()} if f.l_in is not None else {})

    # ------------------------------------------------------------------ evaluation (scripts/think_withhold.py)
    @torch.no_grad()
    def prepare_eval(self, probs):
        """Per problem: question chunk inputs, the true reasoning chunks (teacher thoughts), their targets, and the
        byte offset (within the reasoning) of each reasoning chunk's boundary = the last byte that chunk has seen."""
        f = self.full(probs)
        out = f.out
        tgt = (self.lat.input_targets(self.m, out, f.x) if self.in_space else self.lat.targets(self.m, out, f.x)).float()
        ex = []
        for b, p in enumerate(probs):
            o, qn, k = f.off[b], f.qn[b], f.k[b]
            pos = torch.nonzero(f.mv[b, f.q[b]: f.r[b]]).squeeze(1)
            ex.append(dict(qin=out.chunks[o: o + qn], teach=out.chunks[o + qn: o + qn + k], tgt=tgt[o + qn: o + qn + k],
                           tgt_qlast=tgt[o + qn - 1], bpos=pos.tolist(), k=k, qb=f.qb[b], spans=step_spans(p["cot"]), p=p))
        return ex

    def answer_logits(self, ctx_bytes, q_lens, thoughts):
        """ctx: per example BOS + question + '#### ' + answer so far. Logits (B, L, V), decoded from latents only."""
        x = self.batch(ctx_bytes)
        B, L = x.shape
        idx = self.flat(B, L, [[q_lens[b] - 1, q_lens[b]] for b in range(B)], x.device)
        h, cu, bp, hc, cu1 = self.enc_chunks(x, idx)
        mv = bp.boundary_mask.view(B, L)
        qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
        rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
        seqs = [torch.cat((self.pad(rows[b][:qn[b]]), self.pad(thoughts[b].to(hc.dtype)), self.pad(rows[b][qn[b]:])), 0)
                for b in range(B)]
        z, zcu = self.run_main(seqs)
        zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
        zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + thoughts[b].shape[0]:]), 0) for b in range(B)], 0)
        src = self.lat.head(zc).to(hc.dtype)
        return self._decode(x, h, bp.boundary_mask, bp.boundary_prob, bp.selected_probs, cu, cu1, src)
