"""Fine-tune a 1-level latent H-Net to reason in latents (latents/think.py): the written reasoning is replaced by
thoughts that are never decoded; only the answer is.

usage: python scripts/think_train.py --init runs/<pretrained run> --name <run> [--steps 1500] [--batch 64]
           [--lr 5e-4] [--beta 1.0] [--warmup 0.05] [--seed 0]
           [--mode pure|e2e] [--p_true_frac 0.6 | --teacher_only] [--gamma 1.0] [--w_lm 1.0]
mode pure (default; latents/think.py loss_pure): the decoder is trained on stop-gradient latents and never shapes
the thoughts; thought inputs start as the true reasoning latents (p_true = 1) and anneal to the model's own over
the first p_true_frac of training (--teacher_only keeps them true: the "read" step alone). Every step also trains
on the full written examples (w_lm). mode e2e: the earlier loss (answer CE back-propagated through the thoughts).
Writes runs/<name>/{final.pt, summary.json, metrics.jsonl} in the latents.train format, with data_dir set to the
arithmetic task, so `python scripts/latent_cot.py runs/<name>` evaluates it.
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
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents import latent as latent_lib  # noqa: E402
from latents import tasks  # noqa: E402
from latents.models import build_model, param_groups  # noqa: E402
from latents.think import Thinker  # noqa: E402


def load(run, device):
    ck = torch.load(Path(run) / "final.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], device=device)
    model.load_state_dict(ck["model"])
    lat = latent_lib.build(cfg["latent"], model, cfg["model"])
    missing, unexpected = lat.load_state_dict(ck["latent"], strict=False)
    assert not unexpected and set(missing) <= {"gamma_est", "rf_baseline"}, (missing, unexpected)
    return cfg, model, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--data_dir", default="data/task_arith")
    ap.add_argument("--mode", choices=["pure", "e2e"], default="pure")
    ap.add_argument("--p_true_frac", type=float, default=0.6)
    ap.add_argument("--teacher_only", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--w_lm", type=float, default=1.0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg, model, lat = load(ROOT / a.init if not Path(a.init).is_absolute() else a.init, device)
    mc = cfg["model"]
    groups, _ = param_groups(model, mc, a.wd)
    opt_groups = [{k: g[k] for k in ("params", "weight_decay")} | {"lr": a.lr * g["lr_mult"]} for g in groups]
    lat_params = [p for p in lat.parameters() if p.requires_grad]
    if lat_params:
        opt_groups.append({"params": lat_params, "weight_decay": a.wd, "lr": a.lr * lat.lr_mult})
    base_lrs = [g["lr"] for g in opt_groups]
    opt = torch.optim.AdamW(opt_groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    params = [p for g in opt_groups for p in g["params"]]
    th = Thinker(model, lat, mc)
    rng = random.Random(1000 + a.seed)  # training problems; the test set uses seed 1 in latents/tasks.py
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mlog = open(run_dir / "metrics.jsonl", "w")
    ac = dict(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda")
    t0, n_seen = time.time(), 0
    model.train()
    for step in range(a.steps):
        w = max(1, int(a.warmup * a.steps))
        f = (step + 1) / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, a.steps - w)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        probs = [tasks.sample(rng) for _ in range(a.batch)]
        n_seen += len(probs)
        if a.teacher_only:
            p_true = 1.0
        else:
            p_true = max(0.0, 1.0 - step / max(1.0, a.p_true_frac * a.steps)) if a.p_true_frac > 0 else 0.0
        with torch.autocast(**ac):
            if a.mode == "pure":
                loss, st = th.loss_pure(probs, p_true=p_true, beta=a.beta, gamma=a.gamma, w_lm=a.w_lm,
                                        step=step, total_steps=a.steps)
            else:
                loss, st = th.loss(probs, beta=a.beta)
        if not torch.isfinite(loss):
            print("non-finite loss; abort", flush=True)
            sys.exit(4)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lat.after_step(model, step, a.steps)
        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss), "grad_norm": float(gn), "lr_scale": f, "problems": n_seen,
                   "sec": round(time.time() - t0, 1)}
            rec.update({k: (float(v) if torch.is_tensor(v) else v) for k, v in st.items() if not k.startswith("_")})
            mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
            print(json.dumps(rec), flush=True)
    model.eval()
    cfg = dict(cfg, name=a.name, data_dir=a.data_dir, think=dict(vars(a)))
    torch.save({"model": model.state_dict(), "cfg": cfg, "latent": lat.state_dict()}, run_dir / "final.pt")
    json.dump({"name": a.name, "init": a.init, "steps": a.steps, "problems_seen": n_seen, "final_answer_ce": rec["answer_ce"],
               "final_thought_l1": rec["thought_l1"], "wall_min": (time.time() - t0) / 60}, open(run_dir / "summary.json", "w"), indent=1)
    print("saved", run_dir, flush=True)


if __name__ == "__main__":
    main()
