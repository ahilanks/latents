"""What happens to the cuts on the reasoning text during end-to-end reasoning training?
For a list of checkpoints (first = reference, usually the pretrained model): thoughts per problem, the share of cuts that
sit on the first letter of a written step, agreement with the reference cuts, how many cuts moved since the previous
checkpoint, and two example segmentations. Forced boundaries (ends of the parts) are left out of the statistics.

usage: python scripts/cut_drift.py runs/<reference> runs/<ckpt> [runs/<ckpt> ...] [--data task_arith] [--n 500]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa
import torch
from think_train import load
from latents.think_lc import make_thinker, step_spans

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--data", default="task_arith"); ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
probs = [json.loads(l) for l in open(ROOT / "data" / a.data / "test.jsonl")][:a.n]
masks, shown = {}, {}
for run in a.runs:
    cfg, model, lat = load(ROOT / run, "cuda"); model.eval()
    lat.cfg.pop("cut_from", None)  # always the model's OWN router here
    th = make_thinker(model, lat, cfg["model"])
    ms = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(probs), 250):
            f = th.full(probs[i:i + 250])
            for b, p in enumerate(probs[i:i + 250]):
                m = f.mv[b, f.q[b]: f.r[b]].clone().cpu()
                m[-1] = False  # the forced cut at the reasoning's last byte
                ms.append(m)
    masks[run] = ms
    del model, lat, th
ref = masks[a.runs[0]]
prev = None
print(f"{'checkpoint':38s} {'thoughts':>8s} {'at step start':>13s} {'same as ref':>11s} {'moved vs prev':>13s}")
for run in a.runs:
    ms = masks[run]
    n_cut = sum(int(m.sum()) for m in ms)
    at_start = 0
    for m, p in zip(ms, probs):
        starts = {s[0] for s in step_spans(p["cot"])}
        at_start += sum(1 for t in torch.nonzero(m).squeeze(1).tolist() if t in starts)
    tp = sum(int((m & r).sum()) for m, r in zip(ms, ref)); n_ref = sum(int(r.sum()) for r in ref)
    f1 = 2 * tp / max(n_cut + n_ref, 1)
    mv = "" if prev is None else f"{sum(int((m ^ q).sum()) for m, q in zip(ms, prev)) / max(sum(int(q.sum()) for q in prev), 1):.2f}"
    print(f"{Path(run).name:38s} {n_cut / len(ms) + 1:8.1f} {at_start / max(n_cut, 1):13.2f} {f1:11.2f} {mv:>13s}")
    prev = ms
for j in (1, 4):
    print()
    for run in a.runs:
        cot, m = probs[j]["cot"], masks[run][j]
        out, last = [], 0
        for t in torch.nonzero(m).squeeze(1).tolist():
            out.append(cot[last:t + 1]); last = t + 1
        out.append(cot[last:])
        print(f"{Path(run).name:38s} " + "|".join(x.replace(chr(10), "\\n") for x in out))
