"""Coconut (Hao et al. 2024, "chain of continuous thought") on a byte-level 1-stage H-Net.

A thought is the main network's output at the previous position, fed straight back as its next input: the byte
encoder and decoder do nothing while the model thinks. Training follows Coconut's curriculum: at stage s the first s
written reasoning steps are deleted from the text and replaced by c*s thoughts; the model is trained with the normal
next-byte loss on what follows (the remaining written steps and the answer), back-propagated through the whole chain
of thoughts. Everything trains (encoder, router, main network, decoder). Picture: see the session notes / logistics.

One training example at stage 2 (c = 1), problem "a=17; b=a+25; c=b*3; d=c-40; d=?":
  bytes the encoder sees   BOS Q: ...; d=?\\n <start> <stop> c=42*3=26. d=26-40=86.\\n#### 86\\n
  main network sequence    [question chunks] [<start> chunk] T1 T2 [<stop> chunk] [chunks of the rest]
  T1 = output at the <start> chunk, T2 = output at T1 (fed back, never decoded)
  loss                     next-byte cross-entropy on every byte after <stop> (no loss on the question or markers)
Boundaries are forced at <start> and <stop>: the thoughts attach right after the question, and the first byte after
thinking is predicted from a main-network position that has seen every thought. The deleted steps' bytes appear
nowhere. Stage 0 (no thoughts, no markers) is plain written-reasoning training with the loss after the question.

feed = "full"   the thought is the main network's full-width output (Coconut's way)
       "input"  the decoder-facing part of the output, padded like a normal chunk input
read = "stop"     the first bytes after thinking are decoded from the main network's output at the <stop> chunk, an
                  extra position after the thoughts (Coconut's <eot>). The last calculation can then be done there,
                  in an output that is never fed back: at the next stage it has to be relearned inside the thoughts
                  (2026-11-20: one hidden calculation learned, the second stuck at the tens digit).
       "thought"  the <stop> chunk is not a main-network position; the decoder reads the output at the last thought
                  instead. The state that writes the next step at stage s is exactly the new thought of stage s+1.
fresh = True     the byte-level encoder and decoder restart at <start>: their own memory of the question is cut, so
                 everything written after thinking has to come through the main network (its attention to the
                 question's chunks and to the thoughts). Without it the hidden value is produced jointly by the main
                 network's output and the byte-level path (2026-11-20: exchanging only the read-out state between
                 problems gives neither problem's value), so no thought ever holds a complete result.
ground > 0       every thought has a target: the latent of the reasoning chunk it replaces. The full written example is
                 chunked by the model itself (a second, plain encoder pass); the chunks that begin inside the deleted
                 steps give both the NUMBER of thoughts and, one by one, their targets (the encoder's chunk vectors,
                 constants for the gradient). Loss: ground x SmoothL1(thought, target) next to the byte loss. At test
                 time the count is still taken from the true reasoning's chunking (an oracle count, to be replaced).
1-level H-Nets with a learned (or fixed-chunk) router, no latent objective needed.
"""
import torch
import torch.nn.functional as F
from hnet.modules.dc import RoutingModuleOutput

from latents.models import fixed_routing, ratio_loss  # noqa: F401
from latents.think_lc import step_spans

BOS, START, STOP = 254, 1, 2


def n_written_steps(q):
    """Written reasoning steps of a problem, read off its question (one per assignment; the restated first value
    counts): "Q: a=17; b=a+25; c=b*3; d=c-40; d=?" -> 4."""
    return q.count(";")


def example(p, stage, c=1):
    """(bytes, index of <stop> or -1, index of the first byte the loss covers, number of thoughts) for stage `stage`."""
    q = bytes([BOS]) + p["q"].encode()
    spans = step_spans(p["cot"])
    k = min(stage, len(spans))
    if k == 0:
        b = q + (p["cot"] + p["answer"]).encode()
        return b, -1, len(q), 0
    cut = spans[k][0] if k < len(spans) else len(p["cot"])
    rest = p["cot"][cut:]
    b = q + bytes([START, STOP]) + (rest + p["answer"]).encode()
    return b, len(q) + 1, len(q) + 2, c * k


class Coconut:
    def __init__(self, model, mc, c=1, feed="full", read="stop", fresh=False, ground=0.0):
        assert not mc.get("nest"), "1-level H-Nets only"
        assert feed in ("full", "input") and read in ("stop", "thought")
        self.m, self.mc, self.c, self.feed, self.read, self.fresh = model, mc, int(c), feed, read, bool(fresh)
        self.ground = float(ground)
        self.thought_byte_grad = True  # False: the byte loss does not reach back into the thoughts (they learn only from
        #                                their chunk targets); a blocked path, i.e. the not-end-to-end comparison
        self.bb = model.backbone
        self.inner = self.bb.main_network
        self.iso = self.inner.main_network
        self.D0 = mc["d_model"][0]

    # ------------------------------------------------------------------ pieces
    def dev(self):
        return self.m.lm_head.weight.device

    def pad(self, v):
        p = self.inner.pad_dimension
        return v if p is None else torch.cat((v, p.expand(v.shape[0], -1).to(v.dtype)), -1)

    def batch(self, seqs):
        L = (max(len(s) for s in seqs) + 63) // 64 * 64
        x = torch.zeros(len(seqs), L, dtype=torch.long, device=self.dev())
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(list(s), device=x.device)
        return x

    def run_main(self, seqs, valid=None):
        lens = [s.shape[0] for s in seqs]
        cu = torch.tensor([0] + lens, device=seqs[0].device).cumsum(0).to(torch.int32)
        from latents.models import main_pass  # the same main-network pass as forward_lm (hidden-state feedback included)
        return main_pass(self.m, torch.cat(seqs, 0), cu, max(lens), valid=valid), cu

    def thought_of(self, z_last):
        """The next input made from a main-network output (D1)."""
        return z_last if self.feed == "full" else self.pad(z_last[:, : self.D0])

    def byte_seqs(self, B, L, starts, device):
        """Sequence offsets for the byte-level encoder and decoder: one sequence per row, or (fresh) two for a row
        with thoughts, the second starting at its <start> byte. Returns (cu, longest sequence)."""
        pts = {b * L for b in range(B + 1)}
        if self.fresh:
            pts |= {b * L + s for b, s in enumerate(starts) if s > 0}
        pts = sorted(pts)
        return torch.tensor(pts, device=device, dtype=torch.int32), max(j - i for i, j in zip(pts[:-1], pts[1:]))

    def encode(self, x, force, cu_b=None, max_b=None):
        """Encoder + router with boundaries forced at the flat byte indices `force`. cu_b / max_b: the byte-level
        sequences the encoder runs over (default: one per row)."""
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        if cu_b is None:
            cu_b, max_b = cu, L
        h = self.bb.encoder(self.m.embeddings(x).flatten(0, 1), cu_seqlens=cu_b, max_seqlen=max_b)
        if self.mc.get("pool_k"):
            bp = fixed_routing(B, L, self.mc["pool_k"], x.device)
        else:
            bp = self.bb.routing_module(h, cu_seqlens=cu)
        if force.numel():
            mask, prob, sel = bp.boundary_mask.clone(), bp.boundary_prob.clone(), bp.selected_probs.clone()
            mask[force] = True
            prob[force] = torch.tensor([0.0, 1.0], device=x.device, dtype=prob.dtype)
            sel[force] = 1.0
            bp = RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)
        return h, cu, bp

    # ------------------------------------------------------------------ grounding: the reasoning's own chunk latents
    @torch.no_grad()
    def reasoning_chunks(self, probs, stages):
        """Per problem: the encoder's chunk vectors (n, D0) of the chunks that begin inside the steps deleted at its
        stage, from the model's own chunking of the full written example (no markers, no thoughts)."""
        seqs, spans_b = [], []
        for p, st in zip(probs, stages):
            q = bytes([BOS]) + p["q"].encode()
            spans = step_spans(p["cot"])
            k = min(st, len(spans))
            cut = 0 if k == 0 else (spans[k][0] if k < len(spans) else len(p["cot"]))
            seqs.append(q + (p["cot"] + p["answer"]).encode())
            spans_b.append((len(q), len(q) + cut))
        x = self.batch(seqs)
        B, L = x.shape
        h, cu, bp = self.encode(x, torch.zeros(0, dtype=torch.long, device=x.device))
        hc, cu1, _, _ = self.bb.chunk_layer(h, bp.boundary_mask, cu)
        mv = bp.boundary_mask.view(B, L)
        out = []
        for b, (lo, hi) in enumerate(spans_b):
            first = int(cu1[b]) + int(mv[b, :lo].sum())
            out.append(hc[first: first + int(mv[b, lo:hi].sum())].float())
        return out

    # ------------------------------------------------------------------ thoughts and the forward with them
    def thoughts(self, prefixes, ns):
        """prefixes: per row, the main-network inputs up to and including the <start> chunk. Returns per row the
        (n, D1) thoughts, generated one after another (each depends on the previous ones; gradients flow through)."""
        seqs = list(prefixes)
        outs = [[] for _ in seqs]
        for t in range(max(ns, default=0)):
            act = [b for b in range(len(seqs)) if t < ns[b]]
            z, cu = self.run_main([seqs[b] for b in act])
            th = self.thought_of(z[cu[1:].long() - 1])
            for i, b in enumerate(act):
                outs[b].append(th[i])
                seqs[b] = torch.cat((seqs[b], th[i:i + 1].to(seqs[b].dtype)), 0)
        D1 = prefixes[0].shape[-1]
        return [torch.stack(o) if o else prefixes[0].new_zeros(0, D1) for o in outs]

    def forward(self, byte_seqs, stops, n_thoughts, thoughts=None):
        """byte_seqs: per row the bytes; stops: per row the index of <stop> (-1: no thoughts); n_thoughts per row.
        thoughts: precomputed per-row thoughts (generation reuses them), else made here. Returns (logits (B, L, V),
        the boundary output, real-byte mask (B, L), thoughts)."""
        x = self.batch(byte_seqs)
        B, L = x.shape
        force = torch.tensor([b * L + i for b, s in enumerate(stops) if s > 0 for i in (s - 1, s)], device=x.device,
                             dtype=torch.long)
        cu_b, max_b = self.byte_seqs(B, L, [s - 1 if s > 0 else -1 for s in stops], x.device)
        h, cu, bp = self.encode(x, force, cu_b, max_b)
        residual = self.bb.residual_proj(h.to(self.bb.residual_proj.weight.dtype))
        hc, cu1, _, _ = self.bb.chunk_layer(h, bp.boundary_mask, cu)
        mv = bp.boundary_mask.view(B, L)
        rows = [self.pad(hc[cu1[b]:cu1[b + 1]]) for b in range(B)]
        # chunks up to and including the <start> chunk (= every chunk whose boundary is at or before <start>)
        qn = [int(mv[b, : s].sum()) if s > 0 else 0 for b, s in enumerate(stops)]
        if thoughts is None:
            thoughts = self.thoughts([rows[b][: qn[b]] if n_thoughts[b] else rows[b][:1] for b in range(B)],
                                     [n if s > 0 else 0 for n, s in zip(n_thoughts, stops)])
        nt = [thoughts[b].shape[0] for b in range(B)]
        # read="thought": the <stop> chunk (index qn) is left out of the main network and gets the output at the last thought
        skip = [1 if (self.read == "thought" and nt[b] > 0) else 0 for b in range(B)]
        fed = thoughts if self.thought_byte_grad else [t.detach() for t in thoughts]
        seqs = [torch.cat((rows[b][: qn[b]], fed[b].to(rows[b].dtype), rows[b][qn[b] + skip[b]:]), 0) for b in range(B)]
        # main-network positions that matter per row: its chunks that start inside the real bytes, plus its thoughts
        nreal = [int(mv[b, : len(byte_seqs[b])].sum()) + nt[b] - skip[b] for b in range(B)]
        z, zcu = self.run_main(seqs, valid=nreal)
        idx = []  # per real chunk, its position in the main-network sequence
        for b in range(B):
            o = int(zcu[b])
            idx.append(torch.arange(qn[b]) + o)
            if skip[b]:
                idx.append(torch.tensor([o + qn[b] + nt[b] - 1]))
            idx.append(torch.arange(rows[b].shape[0] - qn[b] - skip[b]) + o + qn[b] + nt[b])
        zc = z[torch.cat(idx).to(z.device)][..., : self.D0]
        out = self.bb.dechunk_layer(zc, bp.boundary_mask, bp.boundary_prob, cu1)
        hd = self.bb.residual_func(out.to(residual.dtype), residual, bp.selected_probs).to(out.dtype)
        hd = self.bb.decoder(hd, cu_seqlens=cu_b, max_seqlen=max_b)
        real = torch.arange(L, device=x.device)[None, :] < torch.tensor([len(s) for s in byte_seqs], device=x.device)[:, None]
        return self.m.lm_head(hd).view(B, L, -1), bp, real, thoughts

    # ------------------------------------------------------------------ training loss
    def loss(self, probs, stages, alpha=None, q_loss=True, ratio="row"):
        """Next-byte loss on what follows the thoughts (stage 0: on what follows the question), plus H-Net's ratio
        loss on the real bytes. stages: per problem.
        q_loss: also the normal next-byte loss on the question's own bytes (not on the markers). Without it nothing
        but the ratio loss acts on the question's cuts, and they are removed within a stage or two (2026-11-20: the
        question became one chunk and its numbers could no longer be read).
        ratio: "row" = H-Net's ratio loss over all real bytes; "parts" = the same loss on the question's bytes and on
        the bytes from <start> on separately (their mean), so that cuts cannot be moved from the question to the part
        after thinking (coco-fresh, 2026-11-20: question cut rate 0.135 -> 0.03 in 1300 steps with "row")."""
        exs = [example(p, s, self.c) for p, s in zip(probs, stages)]
        byte_seqs = [e[0] for e in exs]
        targets = self.reasoning_chunks(probs, stages) if self.ground > 0 else None
        if targets is not None:  # one thought per chunk of the deleted reasoning
            exs = [(e[0], e[1], e[2], t.shape[0] if e[1] > 0 else 0) for e, t in zip(exs, targets)]
        logits, bp, real, th = self.forward(byte_seqs, [e[1] for e in exs], [e[3] for e in exs])
        B, L, V = logits.shape
        x = self.batch(byte_seqs)
        pos = torch.arange(L - 1, device=x.device)[None, :]
        first = torch.tensor([e[2] for e in exs], device=x.device)[:, None]
        qlen = torch.tensor([e[1] - 1 if e[1] > 0 else e[2] for e in exs], device=x.device)[:, None]  # question bytes
        lens = torch.tensor([len(s) for s in byte_seqs], device=x.device)[:, None]
        on = (pos + 1 >= first) & (pos + 1 < lens)  # predict bytes first..end-1 from positions first-1..end-2
        on_q = pos + 1 < qlen
        ce = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, L - 1)
        l_ce = (ce * on).sum() / on.sum().clamp(min=1)
        l_q = (ce * on_q).sum() / on_q.sum().clamp(min=1)
        l_byte = (ce * (on | on_q)).sum() / (on | on_q).sum().clamp(min=1) if q_loss else l_ce
        N = float(self.mc.get("N", 6))
        mq = (torch.arange(L, device=x.device)[None, :] < qlen)
        bm, bpr = bp.boundary_mask.view(B, L), bp.boundary_prob[..., -1].view(B, L)

        def ratio_of(mask):
            Fm_, G_ = bm[mask].float().mean(), bpr[mask].float().mean()
            return N / (N - 1) * ((N - 1) * Fm_ * G_ + (1 - Fm_) * (1 - G_))
        Fm = bm[real].float().mean()
        l_ratio = 0.5 * (ratio_of(mq) + ratio_of(real & ~mq)) if ratio == "parts" else ratio_of(real)
        a = float(self.mc.get("alpha", 0.03) if alpha is None else alpha)
        cut_q = bm[mq].float().mean()
        st = {"ce": l_ce.detach(), "ce_question": l_q.detach(), "ratio_loss": l_ratio.detach(),
              "thoughts": sum(e[3] for e in exs) / len(exs), "cut_rate": Fm.detach(), "cut_rate_question": cut_q.detach()}
        total = l_byte + a * l_ratio
        if targets is not None:
            pr = torch.cat([th[b][:, : self.D0].float() for b in range(B) if exs[b][3] > 0] or [logits.new_zeros(0, self.D0)])
            tg = torch.cat([targets[b] for b in range(B) if exs[b][3] > 0] or [logits.new_zeros(0, self.D0)])
            if pr.shape[0]:
                l_g = F.smooth_l1_loss(pr, tg)
                total = total + self.ground * l_g
                var = (tg - tg.mean(0)).pow(2).sum(-1).mean().clamp(min=1e-8)
                st["ground_loss"] = l_g.detach()
                st["ground_explained"] = (1 - (pr - tg).pow(2).sum(-1).mean() / var).detach()
        return total, st

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def answer(self, probs, stage=None, n_thoughts=None, max_new=160):
        """Greedy generation after the question. stage None = the final stage: every step is a thought and the
        model writes only the answer; stage s: s steps are thoughts and the model writes the rest of the reasoning.
        n_thoughts: per problem override (default c x replaced steps, the count read off the question).
        Returns per problem (generated text, number of thoughts)."""
        if self.ground > 0 and n_thoughts is None:  # as many thoughts as the true reasoning has chunks (oracle count)
            st_all = [n_written_steps(p["q"]) if stage is None else stage for p in probs]
            n_thoughts = [t.shape[0] for t in self.reasoning_chunks(probs, st_all)]
        ctx, stops, ns = [], [], []
        for i, p in enumerate(probs):
            k = n_written_steps(p["q"]) if stage is None else min(stage, n_written_steps(p["q"]))
            q = bytes([BOS]) + p["q"].encode()
            n = (n_thoughts[i] if n_thoughts is not None else self.c * k)
            if k == 0 and n == 0:
                ctx.append(bytearray(q))
                stops.append(-1)
            else:
                ctx.append(bytearray(q + bytes([START, STOP])))
                stops.append(len(q) + 1)
            ns.append(n)
        starts = [len(c) for c in ctx]
        thoughts = None
        done = [False] * len(ctx)
        for _ in range(max_new):
            logits, _, _, thoughts = self.forward([bytes(c) for c in ctx], stops, ns, thoughts=thoughts)
            for b in range(len(ctx)):
                if done[b]:
                    continue
                nxt = int(logits[b, len(ctx[b]) - 1].argmax())
                ctx[b].append(nxt)
                gen = bytes(ctx[b][starts[b]:])
                if b"####" in gen and gen.endswith(b"\n") or len(gen) >= max_new:
                    done[b] = True
            if all(done):
                break
        return [(bytes(c[s:]).decode("utf-8", "replace"), n) for c, s, n in zip(ctx, starts, ns)]


def score(probs, outs):
    """Accuracy overall and by number of computation steps; an answer is the number after the last '####'."""
    by = {}
    right = 0
    for p, (txt, _) in zip(probs, outs):
        ans = txt.split("####")[-1].strip() if "####" in txt else ""
        ok = ans == str(p["value"])
        right += ok
        by.setdefault(p["steps"], []).append(ok)
    return {"acc": right / max(len(probs), 1),
            "by_steps": {k: sum(v) / len(v) for k, v in sorted(by.items())},
            "mean_thoughts": sum(n for _, n in outs) / max(len(outs), 1),
            "mean_bytes_written": sum(len(t.encode()) for t, _ in outs) / max(len(outs), 1)}


def carried(probs, outs, stage=None):
    """The first number the model has to know from its thoughts: the value of the last replaced step, which opens the
    first step it writes ("c=42*3=26." -> 42), or the answer when every step is a thought. Share exactly right, and
    with the units / tens digit right (tens of a one-digit number = 0)."""
    import re
    ex = un = te = 0
    for p, (txt, _) in zip(probs, outs):
        spans = step_spans(p["cot"])
        k = len(spans) if stage is None else min(stage, len(spans))
        if k == 0:
            continue
        true = spans[k - 1][3]
        m = re.match(r"[a-z]=(\d+)", txt) if k < len(spans) else re.match(r"#### (\d+)", txt)
        got = int(m.group(1)) if m else -1
        ex += got == true
        un += got >= 0 and got % 10 == true % 10
        te += got >= 0 and got // 10 == true // 10
    n = max(len(probs), 1)
    return {"exact": ex / n, "units": un / n, "tens": te / n}
