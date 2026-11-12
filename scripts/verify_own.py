"""Independent check of whole-chain latent reasoning (learned-cut models, latents/think_lc.py).
usage: python scripts/verify_own.py runs/<thought run> [task_arith|task_arith_sp|task_arith_long]
 Nothing here ever encodes the written reasoning or the answer:
the question is encoded ALONE, the number of thoughts comes from the question (written steps + 1), the model rolls out
its own thoughts and the answer is decoded from latents. Also: thought count off by one, and no thoughts at all."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import latents, torch
import think_withhold as W
from think_train import load
from latents.think import BOS
from latents.think_lc import make_thinker
run, data = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "task_arith")
cfg, model, lat = load(ROOT / run, "cuda"); model.eval()
th = make_thinker(model, lat, cfg["model"])
probs = [json.loads(l) for l in open(ROOT / "data" / data / "test.jsonl")]
res = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(probs), 250):
        P = probs[i:i + 250]
        qb = [bytes([BOS]) + p["q"].encode() for p in P]          # the question only
        x = th.batch(qb); B, L = x.shape
        idx = th.flat(B, L, [[len(q) - 1] for q in qb], x.device)  # one forced cut: the question's last byte
        h, cu, bp, hc, cu1 = th.enc_chunks(x, idx)
        mv = bp.boundary_mask.view(B, L)
        qin = []
        for b in range(B):
            n = int(mv[b, :len(qb[b])].sum()); qin.append(hc[cu1[b]: cu1[b] + n])
        nsteps = [p["q"].count(";") for p in P]                    # written steps + 1, read off the question
        for name, ks in (("thoughts = steps + 1 (from the question)", nsteps), ("one thought fewer", [k - 1 for k in nsteps]),
                         ("one thought more", [k + 1 for k in nsteps]), ("no thoughts", [0] * B)):
            seqs = [th.pad(q) for q in qin]
            own = [[] for _ in range(B)]
            for t in range(max(ks)):
                act = [b for b in range(B) if t < ks[b]]
                z, zcu = th.run_main([seqs[b] for b in act])
                c = th.th_head(z[zcu[1:].long() - 1]).to(seqs[0].dtype)
                for j, b in enumerate(act):
                    own[b].append(c[j]); seqs[b] = torch.cat((seqs[b], th.pad(c[j:j + 1])), 0)
            D = hc.shape[-1]
            ths = [torch.stack(o) if o else hc.new_zeros(0, D) for o in own]
            ans = W.answers(th, qb, ths)
            r = res.setdefault(name, {})
            for p, a in zip(P, ans):
                c_ = r.setdefault(p["steps"], [0, 0]); c_[0] += a == str(p["value"]); c_[1] += 1
for name, r in res.items():
    tot = sum(a for a, n in r.values()) / sum(n for a, n in r.values())
    print(f"{name:42s} {100 * tot:5.1f}%   by steps: " + " ".join(f"{s}:{100 * a / n:.0f}" for s, (a, n) in sorted(r.items())))
