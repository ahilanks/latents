"""How much of the answer latents' variance is the answer itself? (latent geometry vs the prediction loss)

At each scored answer position (scripts/think_probe.py setup) take the true next-chunk latent (EMA target, LN) and the
trunk's prediction. Within each layout group (question length mod pool_k, position in the answer):
  content share   variance of the per-value mean target / total target variance: how much of the L1/L2 loss the
                  answer value can account for (the rest is context, format, position)
  pred content    the same for the predictions (did the predictor learn the value-dependent part?)
  err split       mean |pred - true| along the content directions (value means) vs. the rest
usage: python scripts/think_geometry.py runs/<run> [--n 6000]
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents import tasks  # noqa: E402
from latents.think import BOS, Thinker  # noqa: E402
from think_probe import features  # noqa: E402
from think_train import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    run = ROOT / a.run
    cfg, model, lat = load(run, "cuda")
    model.eval()
    th = Thinker(model, lat, cfg["model"])
    rng = random.Random(4242)
    probs = [tasks.sample(rng, (2, 8)) for _ in range(a.n)]
    groups = {}
    for i in range(0, len(probs), 100):
        bp = probs[i:i + 100]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fs = features(th, bp)
        for p, ex in zip(bp, fs):
            ql = 1 + len(p["q"].encode())
            for pos, (tr, pr, off, cur, nxt, zz) in enumerate(ex):
                groups.setdefault((ql % th.mc["pool_k"], len(ex), pos), []).append((p["value"], tr.float(), pr.float()))
    tot = {"share_true": 0.0, "share_pred": 0.0, "err_content": 0.0, "err_rest": 0.0, "var_true": 0.0, "var_pred": 0.0, "w": 0}
    for key, g in groups.items():
        if len(g) < 200:
            continue
        v = torch.tensor([x[0] for x in g], device="cuda")
        T = torch.stack([x[1] for x in g])
        P = torch.stack([x[2] for x in g])
        def share(X):
            Xc = X - X.mean(0)
            means = torch.zeros(100, X.shape[1], device=X.device).index_add_(0, v, Xc)
            cnt = torch.bincount(v, minlength=100).clamp(min=1)[:, None].float()
            between = ((means / cnt)[v]).pow(2).sum(1).mean()
            return float(between / Xc.pow(2).sum(1).mean()), float(Xc.pow(2).sum(1).mean())
        st, vt = share(T)
        sp, vp = share(P)
        # content subspace: span of the per-value mean targets (top PCs of the value means)
        Tc = T - T.mean(0)
        means = torch.zeros(100, T.shape[1], device=T.device).index_add_(0, v, Tc) / torch.bincount(v, minlength=100).clamp(min=1)[:, None]
        U, S, Vh = torch.linalg.svd(means, full_matrices=False)
        k = int((S.pow(2).cumsum(0) / S.pow(2).sum() < 0.95).sum()) + 1
        B = Vh[:k]
        E = P - T
        e_c = (E @ B.T).pow(2).sum(1).mean()
        e_r = (E - (E @ B.T) @ B).pow(2).sum(1).mean()
        w = len(g)
        for kk, val in (("share_true", st), ("share_pred", sp), ("err_content", float(e_c)), ("err_rest", float(e_r)),
                        ("var_true", vt), ("var_pred", vp)):
            tot[kk] += w * val
        tot["w"] += w
        tot.setdefault("content_dims", []).append(k)
    W = tot.pop("w")
    dims = tot.pop("content_dims")
    res = {k: v / W for k, v in tot.items()} | {"content_dims_mean": sum(dims) / len(dims), "groups": len(dims)}
    res["err_content_frac"] = res["err_content"] / (res["err_content"] + res["err_rest"])
    print(json.dumps(res, indent=1))
    json.dump(res, open(run / "think_geometry.json", "w"), indent=1)


if __name__ == "__main__":
    main()
