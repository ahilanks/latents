"""Coconut curriculum on a pretrained byte H-Net (latents/coconut.py): stage s replaces the first s written reasoning
steps with c*s thoughts (the main network's output fed back), trained by the next-byte loss on what follows.

usage: python scripts/coconut_train.py --init runs/C-B1-plain-cot --name=coco-c1 [--c 1] [--feed full|input]
           [--stages 7] [--stage_steps 500] [--batch 128] [--lr 1e-4] [--p_mix 0.0] [--eval_n 200]
           [--first_stage 1] [--eval_every 0] [--advance_at 0] [--q_loss 1] [--wandb 1]
Starts at stage --first_stage (the init model is the written-reasoning model, Coconut's stage 0). The optimizer is
reset at every stage switch (as in Coconut). At the end of every stage (and every --eval_every steps inside one): a
quick eval on --eval_n test problems, with that stage's thoughts (the rest written) and with every step a thought;
"carried" = is the first number the model needs from its thoughts right (exact / units digit / tens digit).
--q_loss 1 (default) keeps the normal next-byte loss on the question; 0 = loss only after the thoughts (Coconut as
published; on the H-Net the question's cuts then disappear). Writes runs/<name>/{final.pt, metrics.jsonl,
summary.json} and runs/<name>-stage<s>/final.pt; logs to W&B unless --wandb 0.
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
from latents.coconut import Coconut, carried, score  # noqa: E402
from latents.models import param_groups  # noqa: E402
from show_cuts import load_any  # noqa: E402


def quick_eval(co, probs, stage, ac, bs=128):
    co.m.eval()
    res = {}
    for name, st in (("this_stage", stage), ("all_thoughts", None)):
        if stage == 0 and name == "all_thoughts":  # written-reasoning training: there are no thoughts to evaluate
            res[name] = res["this_stage"]
            continue
        outs = []
        with torch.autocast(**ac):
            for i in range(0, len(probs), bs):
                outs += co.answer(probs[i:i + bs], stage=st, max_new=400 if st == 0 else 160)
        res[name] = dict(score(probs, outs), carried=carried(probs, outs, st))
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
    ap.add_argument("--read", choices=["stop", "thought"], default="stop",
                    help="where the first bytes after thinking are decoded from: the <stop> chunk's own main-network "
                         "position, or the output at the last thought (latents/coconut.py)")
    ap.add_argument("--fresh", type=int, default=0,
                    help="1: the byte-level encoder and decoder restart at <start> (only the main network connects the "
                         "question to what is written after thinking)")
    ap.add_argument("--ratio", choices=["row", "parts"], default="row",
                    help="cut-count loss over the whole row, or on the question and on the part from <start> on separately")
    ap.add_argument("--feedback", choices=["none", "loop"], default="none",
                    help="loop: add the hidden-state loop to the loaded model (the main network's last-layer output at "
                         "one chunk is added to its input at the next; starts at zero). With --first_stage 0 --stages 0 "
                         "this is plain written-reasoning training on one-problem rows, with or without the loop.")
    ap.add_argument("--ground", type=float, default=0.0,
                    help="weight of the grounding loss: every thought is trained toward the latent of the reasoning chunk "
                         "it replaces (the model's own chunking of the written reasoning gives the thought count and targets)")
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
    ap.add_argument("--first_stage", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=0, help="also a quick eval every this many steps inside a stage")
    ap.add_argument("--q_loss", type=int, default=1, help="1: next-byte loss on the question too; 0: only after the thoughts")
    ap.add_argument("--wandb", type=int, default=1)
    ap.add_argument("--advance_at", type=float, default=0.0,
                    help="leave a stage early once the carried value is exactly right on this share of the quick-eval "
                         "problems (checked at every --eval_every eval); 0 = always run the full stage")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg, model, _ = load_any(a.init, dev)
    model.train()
    mc = cfg["model"]
    if a.feedback == "loop" and not mc.get("feedback"):
        from latents.models import attach_feedback
        mc["feedback"] = {"mode": "loop"}
        attach_feedback(model, mc, dev)
    co = Coconut(model, mc, c=a.c, feed=a.feed, read=a.read, fresh=a.fresh, ground=a.ground)
    groups, _ = param_groups(model, mc, a.wd)
    rng = random.Random(1000 + a.seed)
    mix = random.Random(2000 + a.seed)
    tr_steps = tuple(int(v) for v in a.train_steps.split(","))
    test = [json.loads(l) for l in open(ROOT / a.data_dir / "test.jsonl")][: a.eval_n]
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mlog = open(run_dir / "metrics.jsonl", "w")
    extra = dict(c=a.c, feed=a.feed, read=a.read, fresh=a.fresh, ratio=a.ratio, stages=a.stages, init=a.init, q_loss=a.q_loss,
                 feedback=a.feedback, ground=a.ground)
    wb = None
    if a.wandb:
        try:  # never lose a run to a W&B problem
            import wandb
            from latents.train import WANDB_ENTITY, WANDB_PROJECT
            wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=a.name, group="coconut", config=vars(a),
                            dir=str(ROOT / "runs"))
        except Exception as e:
            print("W&B off:", repr(e)[:200], flush=True)

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}/"))
            elif isinstance(v, (int, float)):
                out[f"{pre}{k}"] = v
        return out

    def log(rec):
        mlog.write(json.dumps(rec) + "\n")
        mlog.flush()
        if wb is not None:
            wb.log(flat({k: v for k, v in rec.items() if k != "step"}), step=rec["step"])

    def run_eval(stage, step):
        ev = quick_eval(co, test, stage, ac)
        log({"step": step, "stage": stage, "eval": ev, "min": (time.time() - t0) / 60})
        c1, c2 = ev["this_stage"]["carried"], ev["all_thoughts"]["carried"]
        print(f"[stage {stage} step {step}] this stage {ev['this_stage']['acc']:.3f} (carried exact {c1['exact']:.3f} units "
              f"{c1['units']:.3f} tens {c1['tens']:.3f})  all thoughts {ev['all_thoughts']['acc']:.3f} (exact {c2['exact']:.3f})  "
              f"{ev['all_thoughts']['by_steps']}", flush=True)
        return ev

    t0, step = time.time(), 0
    summary = {}
    for stage in range(a.first_stage, a.stages + 1):
        opt = torch.optim.AdamW([{"params": g["params"], "weight_decay": g["weight_decay"], "lr": a.lr * g["lr_mult"]}
                                 for g in groups], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)  # reset at every stage
        base = [g["lr"] for g in opt.param_groups]
        params = [p for g in opt.param_groups for p in g["params"]]
        n_steps = a.stage_steps + (a.last_stage_extra if stage == a.stages else 0)
        done_early = None
        for s in range(n_steps):
            w = max(1, int(0.05 * n_steps))
            f = (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, n_steps - w)))
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * f
            probs = [tasks.sample(rng, steps=tr_steps) for _ in range(a.batch)]
            stages = [mix.randint(0, stage) if mix.random() < a.p_mix else stage for _ in probs]
            with torch.autocast(**ac):
                loss, st = co.loss(probs, stages, q_loss=bool(a.q_loss), ratio=a.ratio)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            if s % a.log_every == 0 or s == n_steps - 1:
                rec = {"step": step, "stage": stage, "loss": float(loss), "grad_norm": float(gn),
                       **{k: (float(v) if torch.is_tensor(v) else v) for k, v in st.items()},
                       "min": (time.time() - t0) / 60}
                log(rec)
                print(json.dumps(rec), flush=True)
            step += 1
            if a.eval_every and (s + 1) % a.eval_every == 0 and s + 1 < n_steps:
                last_ev = run_eval(stage, step)
                if a.advance_at and last_ev["this_stage"]["carried"]["exact"] >= a.advance_at:
                    print(f"[stage {stage}] learned at step {step} ({s + 1} steps in this stage)", flush=True)
                    done_early = last_ev
                    break
        summary[f"stage{stage}"] = done_early or run_eval(stage, step)
        summary[f"stage{stage}"]["steps_used"] = s + 1
        save(ROOT / "runs" / f"{a.name}-stage{stage}", model, cfg, dict(extra, stage=stage))
    save(run_dir, model, cfg, dict(extra, stage=a.stages))
    json.dump({"name": a.name, **extra, "stages_eval": summary, "min": (time.time() - t0) / 60},
              open(run_dir / "summary.json", "w"), indent=1)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
