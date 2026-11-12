"""Fine-tune a 1-level latent H-Net to reason in latents (latents/think.py): the written reasoning is replaced by
thoughts that are never decoded; only the answer is.

usage: python scripts/think_train.py --init runs/<pretrained run> --name <run> [--steps 1500] [--batch 64]
           [--lr 5e-4] [--beta 1.0] [--warmup 0.05] [--seed 0]
           [--mode pure|e2e] [--p_true_frac 0.6 | --teacher_only] [--gamma 1.0] [--w_lm 1.0]
mode pure (default; latents/think.py loss_pure): the decoder is trained on stop-gradient latents and never shapes
the thoughts; thought inputs start as the true reasoning latents (p_true = 1) and anneal to the model's own over
the first p_true_frac of training (--teacher_only keeps them true: the "read" step alone). Every step also trains
on the full written examples (w_lm). mode e2e: the earlier loss (answer CE back-propagated through the thoughts).
mode coconut (latents/think.py loss_coconut): Coconut's curriculum in latent space. Stage s replaces the first s written
steps with own thoughts trained only by the latent loss on what follows; stage = step // stage_steps (up to max_stage),
a random stage per problem with probability p_mix, optimizer state reset at each switch.
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
from latents.think_lc import make_thinker  # noqa: E402


def load(run, device, lat_over=None):
    ck = torch.load(Path(run) / "final.pt", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    if lat_over:
        cfg["latent"] = dict(cfg["latent"], **lat_over)
    model = build_model(cfg["model"], device=device)
    model.load_state_dict(ck["model"])
    lat = latent_lib.build(cfg["latent"], model, cfg["model"])
    missing, unexpected = lat.load_state_dict(ck["latent"], strict=False)
    assert not unexpected and all(k in ("gamma_est", "rf_baseline") or k.startswith("in_head.") for k in missing), \
        (missing, unexpected)  # in_head: a second head added at the thought stage starts untrained
    return cfg, model, lat


def parse_over(items):
    """--lat key=value overrides of the latent config (yaml-typed values), e.g. --lat reg=sigreg --lat dist=l2."""
    import yaml
    return {k: yaml.safe_load(v) for k, v in (it.split("=", 1) for it in items or [])}


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
    ap.add_argument("--train_steps", default="2,6", help="min,max reasoning steps of the training problems")
    ap.add_argument("--spaced", action="store_true",
                    help="spaced task format (a space between every piece); pass --data_dir data/task_arith_sp for the evals")
    ap.add_argument("--mode", choices=["pure", "e2e", "coconut"], default="pure")
    ap.add_argument("--max_stage", type=int, default=7, help="coconut: the last stage replaces this many written steps")
    ap.add_argument("--stage_steps", type=int, default=400, help="coconut: optimizer steps per stage (stage 0 first)")
    ap.add_argument("--p_mix", type=float, default=0.3, help="coconut: per problem, a random stage with this probability")
    ap.add_argument("--c", type=int, default=0, help="coconut: thoughts per replaced step (0 = the replaced text's chunk count)")
    ap.add_argument("--beta_rest", type=float, default=1.0, help="coconut: weight of the latent loss after the thoughts")
    ap.add_argument("--no_reset_opt", action="store_true", help="coconut: keep the optimizer state across stage switches")
    ap.add_argument("--beta_anchor", type=float, default=0.0,
                    help="coconut: replaced thoughts also get their latent targets, weight fading linearly to 0 over each stage")
    ap.add_argument("--anchor_fade", choices=["stage", "none", "run"], default="stage",
                    help="coconut anchors: fade 1 -> 0 over each stage, keep constant, or fade linearly over the whole run")
    ap.add_argument("--bptt", choices=["full", "1"], default="full",
                    help="coconut: gradient through the whole thought chain, or truncated to one hop (fed-back thoughts detached)")
    ap.add_argument("--p_true_frac", type=float, default=0.6)
    ap.add_argument("--teacher_only", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--w_lm", type=float, default=1.0)
    ap.add_argument("--dec_true", type=float, default=0.0, help="weight of the decoder's CE when reading the true next-chunk latents")
    ap.add_argument("--thought_grad", action="store_true",
                    help="thought/answer latent losses also train the encoder through the teacher thought inputs")
    ap.add_argument("--lat", action="append", default=[], help="latent-config override key=value (repeatable)")
    ap.add_argument("--save_every", type=int, default=0, help="also save runs/<name>-s<step>/final.pt every N steps")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg, model, lat = load(ROOT / a.init if not Path(a.init).is_absolute() else a.init, device, parse_over(a.lat))
    mc = cfg["model"]
    groups, _ = param_groups(model, mc, a.wd)
    opt_groups = [{k: g[k] for k in ("params", "weight_decay")} | {"lr": a.lr * g["lr_mult"]} for g in groups]
    lat_params = [p for p in lat.parameters() if p.requires_grad]
    if lat_params:
        opt_groups.append({"params": lat_params, "weight_decay": a.wd, "lr": a.lr * lat.lr_mult})
    base_lrs = [g["lr"] for g in opt_groups]
    opt = torch.optim.AdamW(opt_groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    params = [p for g in opt_groups for p in g["params"]]
    th = make_thinker(model, lat, mc)  # learned-cut models (chunk_pool = mean) get latents/think_lc.py
    train_steps = tuple(int(v) for v in a.train_steps.split(","))
    rng = random.Random(1000 + a.seed)  # training problems; the test set uses seed 1 in latents/tasks.py
    mix_rng = random.Random(2000 + a.seed)  # coconut stage mixing (separate, so the problems match the other modes)
    cur_stage = 0
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
        probs = [tasks.sample(rng, steps=train_steps, spaced=a.spaced) for _ in range(a.batch)]
        n_seen += len(probs)
        if a.teacher_only:
            p_true = 1.0
        else:
            p_true = max(0.0, 1.0 - step / max(1.0, a.p_true_frac * a.steps)) if a.p_true_frac > 0 else 0.0
        if a.mode == "coconut":
            stage = min(a.max_stage, step // a.stage_steps)
            if stage != cur_stage:
                cur_stage = stage
                if not a.no_reset_opt:
                    opt.state.clear()  # Coconut resets the optimizer state when the stage switches
                print(f"stage {stage} from step {step}", flush=True)
            stages = [mix_rng.randint(0, a.max_stage) if mix_rng.random() < a.p_mix else stage for _ in probs]
        with torch.autocast(**ac):
            if a.mode == "coconut":
                phase = step - cur_stage * a.stage_steps  # steps into the current stage (the last stage runs on)
                fade = {"stage": max(0.0, 1.0 - phase / a.stage_steps), "none": 1.0,
                        "run": max(0.0, 1.0 - step / a.steps)}[a.anchor_fade]
                anchor_w = a.beta_anchor * fade
                loss, st = th.loss_coconut(probs, stages, c=a.c, w_lm=a.w_lm, step=step, total_steps=a.steps,
                                           dec_true=a.dec_true, beta_rest=a.beta_rest, anchor_w=anchor_w,
                                           bptt=a.bptt)
                st["stage"] = cur_stage
            elif a.mode == "pure":
                loss, st = th.loss_pure(probs, p_true=p_true, beta=a.beta, gamma=a.gamma, w_lm=a.w_lm,
                                        step=step, total_steps=a.steps, dec_true=a.dec_true,
                                        thought_grad=a.thought_grad)
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
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < a.steps:
            model.eval()
            d = ROOT / "runs" / f"{a.name}-s{step + 1}"
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": dict(cfg, name=d.name, data_dir=a.data_dir, think=dict(vars(a))),
                        "latent": lat.state_dict()}, d / "final.pt")
            model.train()
        if step % a.log_every == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": float(loss.detach()), "grad_norm": float(gn), "lr_scale": f, "problems": n_seen,
                   "sec": round(time.time() - t0, 1)}
            rec.update({k: (float(v) if torch.is_tensor(v) else v) for k, v in st.items() if not k.startswith("_")})
            mlog.write(json.dumps(rec) + "\n")
            mlog.flush()
            print(json.dumps(rec), flush=True)
    model.eval()
    cfg = dict(cfg, name=a.name, data_dir=a.data_dir, think=dict(vars(a)))
    torch.save({"model": model.state_dict(), "cfg": cfg, "latent": lat.state_dict()}, run_dir / "final.pt")
    json.dump({"name": a.name, "init": a.init, "steps": a.steps, "problems_seen": n_seen, "final_answer_ce": rec["answer_ce"],
               "final_thought_l1": rec.get("thought_l1"), "final_rest_l1": rec.get("rest_l1"), "wall_min": (time.time() - t0) / 60}, open(run_dir / "summary.json", "w"), indent=1)
    print("saved", run_dir, flush=True)


if __name__ == "__main__":
    main()
