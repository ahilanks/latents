"""Coconut curriculum on a pretrained byte H-Net (latents/coconut.py): stage s replaces the first s written reasoning
steps with c*s thoughts (the main network's output fed back), trained by the next-byte loss on what follows.

usage: python scripts/coconut_train.py --init runs/C-B1-plain-cot --name=coco-c1 [--c 1] [--feed full|input]
           [--stages 7] [--stage_steps 500] [--batch 128] [--lr 1e-4] [--p_mix 0.0] [--eval_n 200]
Starts at stage 1 (the init model is the written-reasoning model, Coconut's stage 0). The optimizer is reset at every
stage switch (as in Coconut). At the end of every stage: a quick eval on --eval_n test problems, with that stage's
thoughts (the rest written) and with every step a thought. Writes runs/<name>/{final.pt, metrics.jsonl, summary.json}
and runs/<name>-stage<s>/final.pt.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents import tasks  # noqa: E402
from latents.coconut import Coconut, score  # noqa: E402
from latents.models import param_groups  # noqa: E402
from show_cuts import load_any  # noqa: E402


def quick_eval(co, probs, stage, ac, bs=128):
    co.m.eval()
    res = {}
    for name, st in (("this_stage", stage), ("all_thoughts", None)):
        outs = []
        with torch.autocast(**ac):
            for i in range(0, len(probs), bs):
                outs += co.answer(probs[i:i + bs], stage=st)
        res[name] = score(probs, outs)
    co.m.train()
    return res


def save(run_dir, model, cfg, extra):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": dict(cfg, coconut=extra)}, run_dir / "final.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c", type=int, default=1, help="thoughts per replaced written step")
    ap.add_argument("--feed", choices=["full", "input"], default="full")
    ap.add_argument("--stages", type=int, default=7, help="last stage (7 = every step of the longest training problem)")
    ap.add_argument("--stage_steps", type=int, default=500)
    ap.add_argument("--last_stage_extra", type=int, default=0, help="extra steps at the last stage")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--p_mix", type=float, default=0.0, help="per problem, a random earlier stage with this probability")
    ap.add_argument("--train_steps", default="2,6")
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--data_dir", default="data/task_arith")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg, model, _ = load_any(a.init, dev)
    model.train()
    mc = cfg["model"]
    co = Coconut(model, mc, c=a.c, feed=a.feed)
    groups, _ = param_groups(model, mc, a.wd)
    rng = random.Random(1000 + a.seed)
    mix = random.Random(2000 + a.seed)
    tr_steps = tuple(int(v) for v in a.train_steps.split(","))
    test = [json.loads(l) for l in open(ROOT / a.data_dir / "test.jsonl")][: a.eval_n]
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mlog = open(run_dir / "metrics.jsonl", "w")
    extra = dict(c=a.c, feed=a.feed, stages=a.stages, init=a.init)
    t0, step = time.time(), 0
    summary = {}
    for stage in range(1, a.stages + 1):
        opt = torch.optim.AdamW([{"params": g["params"], "weight_decay": g["weight_decay"], "lr": a.lr * g["lr_mult"]}
                                 for g in groups], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)  # reset at every stage
        base = [g["lr"] for g in opt.param_groups]
        params = [p for g in opt.param_groups for p in g["params"]]
        n_steps = a.stage_steps + (a.last_stage_extra if stage == a.stages else 0)
        for s in range(n_steps):
            w = max(1, int(0.05 * n_steps))
            f = (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, n_steps - w)))
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * f
            probs = [tasks.sample(rng, steps=tr_steps) for _ in range(a.batch)]
            stages = [mix.randint(0, stage) if mix.random() < a.p_mix else stage for _ in probs]
            with torch.autocast(**ac):
                loss, st = co.loss(probs, stages)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            if s % a.log_every == 0 or s == n_steps - 1:
                rec = {"step": step, "stage": stage, "loss": float(loss), "grad_norm": float(gn),
                       **{k: (float(v) if torch.is_tensor(v) else v) for k, v in st.items()},
                       "min": (time.time() - t0) / 60}
                mlog.write(json.dumps(rec) + "\n")
                mlog.flush()
                print(json.dumps(rec), flush=True)
            step += 1
        ev = quick_eval(co, test, stage, ac)
        rec = {"step": step, "stage": stage, "eval": ev, "min": (time.time() - t0) / 60}
        mlog.write(json.dumps(rec) + "\n")
        mlog.flush()
        print(f"[stage {stage}] this stage {ev['this_stage']['acc']:.3f}  all thoughts {ev['all_thoughts']['acc']:.3f}  "
              f"{ev['all_thoughts']['by_steps']}", flush=True)
        summary[f"stage{stage}"] = ev
        save(ROOT / "runs" / f"{a.name}-stage{stage}", model, cfg, dict(extra, stage=stage))
    save(run_dir, model, cfg, dict(extra, stage=a.stages))
    json.dump({"name": a.name, **extra, "stages_eval": summary, "min": (time.time() - t0) / 60},
              open(run_dir / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
