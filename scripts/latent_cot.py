"""Latent chain of thought (ladder phase 6): does rolling the predictor forward in latent space, without decoding
any bytes, carry the reasoning?

For each test problem (latents/tasks.py) and each fine-tuned model, three ways to answer:
  direct     context "Q ... #### " -> greedy answer bytes (no chain of thought)
  byte CoT   context "Q" -> greedy generation of the chain of thought and the answer, byte by byte
  latent CoT the main network gets k extra inputs after the question's last chunk: its own predicted next-chunk
             latents ĉ fed back (k sequential main-net steps, no encoder/decoder work), then the answer is decoded
             from context "Q ... #### " with those k thoughts in the main network's sequence.
             k in --k (e.g. 0,2,4,8,16,oracle; oracle = number of chunks the model's router puts on the true CoT).
A boundary is forced at the start of "#### " so the answer bytes read main-net outputs that come after the thoughts.
Only 1-level models (no model.nest).

usage: python scripts/latent_cot.py runs/<run> [runs/<run> ...] [--k 0,2,4,8,16,oracle] [--n 1000] [--n_cot 300]
Writes runs/<run>/latent_cot.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from hnet.modules.dc import RoutingModuleOutput  # noqa: E402
from latents.models import build_model, fixed_routing, model_forward  # noqa: E402
from latents import latent as latent_lib  # noqa: E402
from latents import flops as flops_lib  # noqa: E402

BOS = 254


class Model:
    def __init__(self, run_dir, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(Path(run_dir) / "final.pt", map_location=device, weights_only=False)
        self.cfg, self.mc, self.device = ck["cfg"], ck["cfg"]["model"], device
        assert not self.mc.get("nest"), "latent CoT eval supports 1-level models only"
        self.m = build_model(self.mc, device=device)
        self.m.load_state_dict(ck["model"])
        self.m.eval()
        self.lat = None  # models without a latent loss can still answer directly and with byte CoT
        if self.cfg.get("latent"):
            self.lat = latent_lib.build(self.cfg["latent"], self.m, self.mc)
            missing, unexpected = self.lat.load_state_dict(ck["latent"], strict=False)
            assert not unexpected and set(missing) <= {"gamma_est", "rf_baseline"}, (missing, unexpected)
        self.ac = dict(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                       enabled=device.startswith("cuda"))

    def tensor(self, b):
        return torch.tensor([list(b)], device=self.device)

    @torch.no_grad()
    def route(self, x, h, cu):
        bb, mc = self.m.backbone, self.mc
        if mc.get("pool_k"):
            return fixed_routing(1, x.shape[1], mc["pool_k"], x.device)
        if getattr(self.m, "surprise", None) is not None:
            tgt = self.lat.ema_states(self.m, x) if (self.lat is not None and self.lat.target == "ema") else None
            return self.m.surprise(h, cu, tgt)
        return bb.routing_module(h, cu_seqlens=cu)

    @torch.no_grad()
    def chunks(self, x, force_at=None):
        """Encoder + router (+ a forced boundary at byte force_at). Returns (h, residual, bp, hc, mask idx)."""
        bb = self.m.backbone
        L = x.shape[1]
        cu = torch.tensor([0, L], device=x.device, dtype=torch.int32)
        h = bb.encoder(self.m.embeddings(x).flatten(0, 1), cu_seqlens=cu, max_seqlen=L)
        bp = self.route(x, h, cu)
        if force_at is not None and force_at < L:
            mask = bp.boundary_mask.clone()
            prob = bp.boundary_prob.clone()
            sel = bp.selected_probs.clone()
            mask[force_at] = True
            prob[force_at] = torch.tensor([0.0, 1.0], device=x.device, dtype=prob.dtype)
            sel[force_at] = 1.0
            bp = RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)
        return h, cu, bp

    def pad(self, v):
        inner = self.m.backbone.main_network
        return v if inner.pad_dimension is None else torch.cat((v, inner.pad_dimension.expand(v.shape[0], -1).to(v.dtype)), -1)

    def main(self, seq):
        iso = self.m.backbone.main_network.main_network
        n = seq.shape[0]
        return iso(seq, cu_seqlens=torch.tensor([0, n], device=seq.device, dtype=torch.int32), max_seqlen=n)

    @torch.no_grad()
    def thoughts(self, q_bytes, k):
        """k latent steps after the question: ĉ fed back as main-net inputs."""
        x = self.tensor(q_bytes)
        with torch.autocast(**self.ac):
            h, cu, bp = self.chunks(x)
            hc = h[bp.boundary_mask]
            seq = self.pad(hc)
            out = []
            for _ in range(k):
                z = self.main(seq)
                c = self.lat.head(z[-1:]).to(hc.dtype)
                out.append(c)
                seq = torch.cat((seq, self.pad(c)), 0)
        return torch.cat(out, 0) if out else hc[:0]

    @torch.no_grad()
    def logits_with_thoughts(self, ctx, q_len, th):
        bb = self.m.backbone
        x = self.tensor(ctx)
        with torch.autocast(**self.ac):
            h, cu, bp = self.chunks(x, force_at=q_len)
            residual = bb.residual_proj(h.to(bb.residual_proj.weight.dtype))
            hc, cu1, max1, _ = bb.chunk_layer(h, bp.boundary_mask, cu)
            q = int(bp.boundary_mask[:q_len].sum())
            seq = torch.cat((self.pad(hc[:q]), self.pad(th.to(hc.dtype)), self.pad(hc[q:])), 0)
            z = self.main(seq)
            z = torch.cat((z[:q], z[q + th.shape[0]:]), 0)
            src = self.lat.head(z).to(hc.dtype) if (self.lat is not None and self.lat.decode_from_pred) else z[..., : hc.shape[-1]]
            out = bb.dechunk_layer(src, bp.boundary_mask, bp.boundary_prob, cu1)
            hd = bb.residual_func(out.to(residual.dtype), residual, bp.selected_probs).to(out.dtype)
            if getattr(self.m, "dec_emb", None) is not None:
                hd = hd + self.m.dec_emb(x).flatten(0, 1).to(hd.dtype)
            hd = bb.decoder(hd, cu_seqlens=cu, max_seqlen=x.shape[1])
            return self.m.lm_head(hd)

    @torch.no_grad()
    def answer_latent(self, q_bytes, th, max_len=4):
        ctx = bytearray(q_bytes + b"#### ")
        q_len = len(q_bytes)
        for _ in range(max_len):
            nxt = int(self.logits_with_thoughts(bytes(ctx), q_len, th)[-1].argmax())
            if nxt == 10:
                break
            ctx.append(nxt)
        return bytes(ctx[q_len + 5:]).decode("utf-8", "replace").strip()

    @torch.no_grad()
    def generate(self, prompts, max_new=400, stop=b"\n####"):
        """Batched greedy byte generation with the normal forward (no thoughts). Returns generated bytes."""
        outs = [bytearray() for _ in prompts]
        done = [False] * len(prompts)
        for _ in range(max_new):
            seqs = [p + bytes(o) for p, o in zip(prompts, outs)]
            n = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), (n + 63) // 64 * 64, dtype=torch.long, device=self.device)
            for i, s in enumerate(seqs):
                x[i, :len(s)] = torch.tensor(list(s), device=self.device)
            with torch.autocast(**self.ac):
                lg = model_forward(self.m, x, self.mc, self.lat).logits
            for i, s in enumerate(seqs):
                if done[i]:
                    continue
                b = int(lg[i, len(s) - 1].argmax())
                outs[i].append(b)
                txt = bytes(outs[i])
                if b"####" in txt and txt.endswith(b"\n"):
                    done[i] = True
            if all(done):
                break
        return [bytes(o) for o in outs]

    @torch.no_grad()
    def cot_chunks(self, q_bytes, cot_bytes):
        x = self.tensor(q_bytes + cot_bytes)
        with torch.autocast(**self.ac):
            h, cu, bp = self.chunks(x)
        return int(bp.boundary_mask[len(q_bytes):].sum())

    def costs(self):
        mc = self.mc
        enc, (main,), dec = mc["arch_layout"]
        d0, d1 = mc["d_model"]
        di0, di1 = mc["d_intermediate"]
        ctx0 = flops_lib.causal_mean_ctx(mc["seq_len"], mc["window_size"][0])
        outer = flops_lib.layout_fwd(enc, d0, ctx0, di0) + flops_lib.layout_fwd(dec, d0, ctx0, di0) + 2 * d0 * 256
        inner = flops_lib.layout_fwd(main, d1, 256, di1)
        return outer, inner


def evaluate(run, ks, n, n_cot):
    M = Model(run)
    probs = [json.loads(l) for l in open(ROOT / M.cfg.get("data_dir", "data/task_arith") / "test.jsonl")][:n]
    outer, inner = M.costs()
    res = {"model": Path(run).name, "per_byte_mflop": outer / 1e6, "per_chunk_mflop": inner / 1e6, "modes": {}}
    by = lambda rows: {s: sum(r for st, r in rows if st == s) / max(1, sum(1 for st, _ in rows if st == s))  # noqa: E731
                       for s in sorted({st for st, _ in rows})}
    # latent CoT for every k (k=0 is the direct answer); a model without a latent head only answers directly
    for k in (ks if M.lat is not None else ["0"]):
        rows, ksz = [], []
        for p in probs:
            qb = bytes([BOS]) + p["q"].encode()
            kk = M.cot_chunks(qb, p["cot"].encode()) if k == "oracle" else int(k)
            ksz.append(kk)
            ans = M.answer_latent(qb, M.thoughts(qb, kk))
            rows.append((p["steps"], ans == str(p["value"])))
        acc = sum(r for _, r in rows) / len(rows)
        name = "direct" if k == "0" else f"latent_k{k}"
        res["modes"][name] = dict(acc=acc, acc_by_steps=by(rows), mean_k=sum(ksz) / len(ksz),
                                  sequential_main_steps=sum(ksz) / len(ksz), mflop=sum(ksz) / len(ksz) * inner / 1e6)
        print(name, json.dumps(res["modes"][name]), flush=True)
    # byte CoT
    sub = probs[:n_cot]
    rows, gen_b, gen_c = [], [], []
    for i in range(0, len(sub), 64):
        batch = sub[i:i + 64]
        prompts = [bytes([BOS]) + p["q"].encode() for p in batch]
        outs = M.generate(prompts)
        for p, pr, o in zip(batch, prompts, outs):
            txt = o.decode("utf-8", "replace")
            ans = txt.split("####")[-1].strip() if "####" in txt else ""
            rows.append((p["steps"], ans == str(p["value"])))
            gen_b.append(len(o))
            gen_c.append(M.cot_chunks(pr, o))
    nb, nc = sum(gen_b) / len(gen_b), sum(gen_c) / len(gen_c)
    res["modes"]["byte_cot"] = dict(acc=sum(r for _, r in rows) / len(rows), acc_by_steps=by(rows), n=len(rows),
                                    gen_bytes=nb, sequential_decoder_steps=nb, sequential_main_steps=nc,
                                    mflop=(nb * outer + nc * inner) / 1e6)
    print("byte_cot", json.dumps(res["modes"]["byte_cot"]), flush=True)
    json.dump(res, open(Path(run) / "latent_cot.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", default="0,2,4,8,16,oracle")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_cot", type=int, default=300)
    a = ap.parse_args()
    for r in a.runs:
        evaluate(r, a.k.split(","), a.n, a.n_cot)


if __name__ == "__main__":
    main()
