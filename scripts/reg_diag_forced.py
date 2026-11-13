"""Anti-collapse term on one-problem rows with the forced pieces (row start, ends of question / reasoning / answer)
left out or kept. usage: python scripts/reg_diag_forced.py [runs/<run>]"""
import sys, random, torch, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import latents
from latents import tasks, train as T
from latents.think_lc import make_thinker
from latents.latent import row_ids
from think_train import load
import reg_diag as R
cfg, model, lat = load(str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/R4n6-3B-s0")), T.DEV); mc = cfg["model"]; model.train()
th = make_thinker(model, lat, mc)
rng = random.Random(7); probs = [tasks.problem(rng, rng.randint(2, 6)) for _ in range(512)]
ac = dict(device_type="cuda", dtype=torch.bfloat16)
acc = {}
for i in range(0, 512, 64):
    with torch.autocast(**ac):
        f = th.full(probs[i:i+64])
    bm = f.out.bpred[0].boundary_mask  # flat B*L
    pos = torch.nonzero(bm).squeeze(1)
    b, t = pos // f.L, pos % f.L
    real = f.real.flatten()[bm]
    q = torch.tensor(f.q, device=t.device)[b]; r = torch.tensor(f.r, device=t.device)[b]; ln = f.lens[b]
    first = t == 0
    forced = (t == q - 1) | (t == r - 1) | (t == ln - 1) | (t == q) | (t == r) | (t == ln)
    sets = {"all real": real, "no row-start": real & ~first, "no forced": real & ~forced, "no row-start, no forced": real & ~first & ~forced,
            "only free cuts in reasoning": real & ~first & ~forced & (t >= q) & (t < r)}
    for k, cv in sets.items():
        res = R.measure(lat, model, f.out, f.x, chunk_valid=cv)
        res["frac_kept"] = float(cv.sum() / real.sum())
        acc.setdefault(k, []).append(res)
print("| chunks used | share kept | input cov part | local cov part | input var part | grad to router | grad to encoder | noise level 256/N |")
print("|---|---|---|---|---|---|---|---|")
for k, v in acc.items():
    m = {kk: np.mean([d[kk] for d in v]) for kk in v[0]}
    print(f"| {k} | {m['frac_kept']:.2f} | {m['in_cov_part']:.3g} | {m.get('local_cov_part', float('nan')):.3g} | {m['in_var_part']:.3g} | {m['reg_grad_router']:.3g} | {m['reg_grad_encoder']:.3g} | {256/m['in_n_chunks']:.2g} |")
