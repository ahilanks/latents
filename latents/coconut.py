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
    def __init__(self, model, mc, c=1, feed="full"):
        assert not mc.get("nest"), "1-level H-Nets only"
        assert feed in ("full", "input")
        self.m, self.mc, self.c, self.feed = model, mc, int(c), feed
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

    def run_main(self, seqs):
        lens = [s.shape[0] for s in seqs]
        cu = torch.tensor([0] + lens, device=seqs[0].device).cumsum(0).to(torch.int32)
        return self.iso(torch.cat(seqs, 0), cu_seqlens=cu, max_seqlen=max(lens)), cu

    def thought_of(self, z_last):
        """The next input made from a main-network output (D1)."""
        return z_last if self.feed == "full" else self.pad(z_last[:, : self.D0])

    def encode(self, x, force):
        """Encoder + router with boundaries forced at the flat byte indices `force`."""
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        h = self.bb.encoder(self.m.embeddings(x).flatten(0, 1), cu_seqlens=cu, max_seqlen=L)
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
        h, cu, bp = self.encode(x, force)
        residual = self.bb.residual_proj(h.to(self.bb.residual_proj.weight.dtype))
        hc, cu1, _, _ = self.bb.chunk_layer(h, bp.boundary_mask, cu)
        mv = bp.boundary_mask.view(B, L)
        rows = [self.pad(hc[cu1[b]:cu1[b + 1]]) for b in range(B)]
        # chunks up to and including the <start> chunk (= every chunk whose boundary is at or before <start>)
        qn = [int(mv[b, : s].sum()) if s > 0 else 0 for b, s in enumerate(stops)]
        if thoughts is None:
            thoughts = self.thoughts([rows[b][: qn[b]] if n_thoughts[b] else rows[b][:1] for b in range(B)],
                                     [n if s > 0 else 0 for n, s in zip(n_thoughts, stops)])
        seqs = [torch.cat((rows[b][: qn[b]], thoughts[b].to(rows[b].dtype), rows[b][qn[b]:]), 0) for b in range(B)]
        z, zcu = self.run_main(seqs)
        keep = torch.cat([torch.cat((torch.ones(qn[b], dtype=torch.bool), torch.zeros(thoughts[b].shape[0], dtype=torch.bool),
                                     torch.ones(rows[b].shape[0] - qn[b], dtype=torch.bool))) for b in range(B)]).to(z.device)
        zc = z[keep][..., : self.D0]
        out = self.bb.dechunk_layer(zc, bp.boundary_mask, bp.boundary_prob, cu1)
        hd = self.bb.residual_func(out.to(residual.dtype), residual, bp.selected_probs).to(out.dtype)
        hd = self.bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
        real = torch.arange(L, device=x.device)[None, :] < torch.tensor([len(s) for s in byte_seqs], device=x.device)[:, None]
        return self.m.lm_head(hd).view(B, L, -1), bp, real, thoughts

    # ------------------------------------------------------------------ training loss
    def loss(self, probs, stages, alpha=None):
        """Next-byte loss on what follows the thoughts (stage 0: on what follows the question), plus H-Net's ratio
        loss on the real bytes. stages: per problem."""
        exs = [example(p, s, self.c) for p, s in zip(probs, stages)]
        byte_seqs = [e[0] for e in exs]
        logits, bp, real, _ = self.forward(byte_seqs, [e[1] for e in exs], [e[3] for e in exs])
        B, L, V = logits.shape
        x = self.batch(byte_seqs)
        pos = torch.arange(L - 1, device=x.device)[None, :]
        first = torch.tensor([e[2] for e in exs], device=x.device)[:, None]
        lens = torch.tensor([len(s) for s in byte_seqs], device=x.device)[:, None]
        on = (pos + 1 >= first) & (pos + 1 < lens)  # predict bytes first..end-1 from positions first-1..end-2
        ce = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(B, L - 1)
        l_ce = (ce * on).sum() / on.sum().clamp(min=1)
        N = float(self.mc.get("N", 6))
        rm = real.flatten()
        Fm = bp.boundary_mask[rm].float().mean()
        G = bp.boundary_prob[..., -1][rm].float().mean()
        l_ratio = N / (N - 1) * ((N - 1) * Fm * G + (1 - Fm) * (1 - G))
        a = float(self.mc.get("alpha", 0.03) if alpha is None else alpha)
        return l_ce + a * l_ratio, {"ce": l_ce.detach(), "ratio_loss": l_ratio.detach(),
                                    "thoughts": sum(e[3] for e in exs) / len(exs), "cut_rate": Fm.detach()}

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def answer(self, probs, stage=None, n_thoughts=None, max_new=160):
        """Greedy generation after the question. stage None = the final stage: every step is a thought and the
        model writes only the answer; stage s: s steps are thoughts and the model writes the rest of the reasoning.
        n_thoughts: per problem override (default c x replaced steps, the count read off the question).
        Returns per problem (generated text, number of thoughts)."""
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
