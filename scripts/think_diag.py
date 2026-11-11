"""Where does teacher-forced latent thinking lose the answer? (diagnostic for runs trained by scripts/think_train.py)

Every test problem gets its true reasoning latents as thoughts (the "teacher" mode of scripts/latent_cot.py). The
answer "#### 94\n" is then scored teacher-forced (each byte given the true earlier bytes) with the decoder reading,
for the chunks from the question's last one on:
  pred     the trunk's predicted next-chunk latents (what the model really does)
  oracle   the true next-chunk latents (EMA target encoder): can the decoder read the digits off a perfect latent?
  zero     zeros: how much the decoder gets from its own byte context alone
High oracle + low pred = the trunk mis-predicts the answer latent; low oracle = the readout itself is the bottleneck.
Also reports per-byte accuracy (tens digit, units digit, newline) and whether the predicted answer latents point to
the right answer (nearest true latent among the other problems' answers, same bytes-to-boundary layout).

usage: python scripts/think_diag.py runs/<run> [--n 2000] [--batch 100]
Writes runs/<run>/think_diag.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hnet.modules.dc import RoutingModuleOutput  # noqa: E402

from latents.models import model_forward  # noqa: E402
from latents.think import BOS, Thinker  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from think_train import load  # noqa: E402


@torch.no_grad()
def teacher(th, probs):
    """True reasoning-chunk states (encoder chunks of the full written example), as in loss_pure."""
    full = [bytes([BOS]) + (p["q"] + p["cot"] + p["answer"]).encode() for p in probs]
    x = th.batch(full)
    B, L = x.shape
    out = model_forward(th.m, x, th.mc, th.lat)
    mv = out.bpred[0].boundary_mask.view(B, L)
    off = torch.cat((torch.zeros(1, device=x.device, dtype=torch.long), mv.sum(1).cumsum(0)))
    res = []
    for b, p in enumerate(probs):
        q = 1 + len(p["q"].encode())
        r = q + len(p["cot"].encode())
        qn, k = int(mv[b, :q].sum()), int(mv[b, q:r].sum())
        res.append(out.chunks[off[b] + qn: off[b] + qn + k])
    return res


@torch.no_grad()
def answer_pass(th, ctx_bytes, q_lens, thoughts, mode):
    """Thinker.answer_logits with the decoder's latent source replaced from the question's last chunk on."""
    bb, lat = th.bb, th.lat
    x = th.batch(ctx_bytes)
    B, L = x.shape
    h, cu, bp = th.encode(x)
    idx = torch.tensor([b * L + q for b, q in enumerate(q_lens)], device=x.device)
    mask, prob, sel = bp.boundary_mask.clone(), bp.boundary_prob.clone(), bp.selected_probs.clone()
    mask[idx] = True
    prob[idx] = torch.tensor([0.0, 1.0], device=x.device, dtype=prob.dtype)
    sel[idx] = 1.0
    residual = bb.residual_proj(h.to(bb.residual_proj.weight.dtype))
    hc, cu1, _, _ = bb.chunk_layer(h, mask, cu)
    mv = mask.view(B, L)
    qn = [int(mv[b, :q_lens[b]].sum()) for b in range(B)]
    rows = [hc[cu1[b]:cu1[b + 1]] for b in range(B)]
    seqs = [torch.cat((th.pad(rows[b][:qn[b]]), th.pad(thoughts[b].to(hc.dtype)), th.pad(rows[b][qn[b]:])), 0)
            for b in range(B)]
    z, zcu = th.run_main(seqs)
    zr = [z[zcu[b]:zcu[b + 1]] for b in range(B)]
    zc = torch.cat([torch.cat((zr[b][:qn[b]], zr[b][qn[b] + thoughts[b].shape[0]:]), 0) for b in range(B)], 0)
    src = lat.head(zc).to(hc.dtype)
    tgt = lat.ema_states(th.m, x)[mask].float()
    if getattr(lat, "target_norm", None) == "ln":
        tgt = F.layer_norm(tgt, tgt.shape[-1:])
    tgt = tgt.to(hc.dtype)
    # per example: the chunks the scored decoder positions (answer bytes after "#### ", predicted from the byte
    # before) sit in; chunk row j reads the latent of chunk j+1 (what the head predicts)
    cum = mv.long().cumsum(1) - 1
    pairs, repl = [], []
    for b in range(B):
        c_len = len(ctx_bytes[b])
        lo, hi = int(cum[b, q_lens[b] + 4]), int(cum[b, c_len - 2])
        pairs.append((b, torch.arange(cu1[b] + lo, cu1[b] + hi + 1, device=x.device)))
        repl.append(torch.arange(cu1[b] + qn[b] - 1, cu1[b] + hi + 1, device=x.device))
    if mode != "pred":
        src = src.clone()
        for j in repl:
            src[j] = tgt[j + 1] if mode == "oracle" else 0
    out = bb.dechunk_layer(src, mask, prob, cu1)
    hd = bb.residual_func(out.to(residual.dtype), residual, sel).to(out.dtype)
    if getattr(th.m, "dec_emb", None) is not None:
        hd = hd + th.m.dec_emb(x).flatten(0, 1).to(hd.dtype)
    hd = bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
    lg = th.m.lm_head(hd).view(B, L, -1)
    pred_ans = [lat.head(zc[j]).float() for _, j in pairs] if mode == "pred" else None
    true_ans = [tgt[j + 1].float() for _, j in pairs] if mode == "pred" else None
    return lg, pred_ans, true_ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    cfg, model, lat = load(run, device)
    model.eval()
    th = Thinker(model, lat, cfg["model"])
    probs = [json.loads(l) for l in open(ROOT / cfg.get("data_dir", "data/task_arith") / "test.jsonl")][:a.n]
    ac = dict(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda")
    stats = {m: dict(exact=0, tens=[0, 0], units=[0, 0], nl=[0, 0], by_steps={}) for m in ("pred", "oracle", "zero")}
    lat_rows = []  # (layout key, value, pred latents, true latents)
    for i in range(0, len(probs), a.batch):
        bp_ = probs[i:i + a.batch]
        with torch.autocast(**ac):
            teach = teacher(th, bp_)
            qb = [bytes([BOS]) + p["q"].encode() for p in bp_]
            ctx = [q + p["answer"].encode() for q, p in zip(qb, bp_)]
            for mode in stats:
                lg, pa, ta = answer_pass(th, ctx, [len(q) for q in qb], teach, mode)
                for b, (q, c, p) in enumerate(zip(qb, ctx, bp_)):
                    a0 = len(q) + 5
                    pr = lg[b, a0 - 1: len(c) - 1].argmax(-1).tolist()
                    tr = list(c[a0:])
                    ok = [int(u == v) for u, v in zip(pr, tr)]
                    s = stats[mode]
                    s["exact"] += int(all(ok))
                    d = len(tr) - 1  # digits
                    if d == 2:
                        s["tens"][0] += ok[0]; s["tens"][1] += 1  # noqa: E702
                    s["units"][0] += ok[d - 1]; s["units"][1] += 1  # noqa: E702
                    s["nl"][0] += ok[d]; s["nl"][1] += 1  # noqa: E702
                    bs = s["by_steps"].setdefault(str(p["steps"]), [0, 0])
                    bs[0] += int(all(ok)); bs[1] += 1  # noqa: E702
                    if mode == "pred":
                        lat_rows.append(((len(q) % th.mc["pool_k"], d), p["value"], pa[b].cpu(), ta[b].cpu()))
        print(f"{i + len(bp_)}/{len(probs)}", {m: round(s["exact"] / (i + len(bp_)), 3) for m, s in stats.items()}, flush=True)
    N = len(probs)
    res = {"run": run.name, "n": N, "modes": {}}
    for m, s in stats.items():
        res["modes"][m] = dict(exact=s["exact"] / N, tens=s["tens"][0] / max(1, s["tens"][1]),
                               units=s["units"][0] / s["units"][1], newline=s["nl"][0] / s["nl"][1],
                               by_steps={k: v[0] / v[1] for k, v in sorted(s["by_steps"].items())})
    # does the predicted answer latent point to the right answer? nearest true latent (L1) among problems with the same
    # chunk layout (question length mod pool_k, digit count), using the answer chunks' latents concatenated
    groups = {}
    for key, v, pa, ta in lat_rows:
        groups.setdefault(key, []).append((v, pa.flatten(), ta.flatten()))
    hit = tot = 0
    l1_true = l1_other = 0.0
    for key, g in groups.items():
        # one true latent per distinct value
        cand = {}
        for v, _, t in g:
            cand.setdefault(v, t)
        if len(cand) < 2:
            continue
        vals = list(cand)
        T = torch.stack([cand[v] for v in vals])
        for v, p, _ in g:
            d = (T - p).abs().mean(-1)
            hit += int(vals[int(d.argmin())] == v)
            tot += 1
            l1_true += float(d[vals.index(v)])
            l1_other += float(d.sum() - d[vals.index(v)]) / (len(vals) - 1)
    res["latent_nn"] = dict(acc=hit / max(1, tot), n=tot, mean_candidates=sum(len({v for v, _, _ in g}) for g in groups.values()) / len(groups),
                            l1_to_true=l1_true / max(1, tot), l1_to_other=l1_other / max(1, tot))
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(run / "think_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
