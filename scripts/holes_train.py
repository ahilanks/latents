"""Reasoning in the holes (NEXT_STEPS.md, plan of 2026-11-23, "Reasoning"): fine-tune a model pretrained with the loop
and holes so that its thoughts are the same holes. Stage s replaces the first s written steps of every training
problem with --per_step marker bytes each (the count read off the question); the model is trained with the next-byte
loss on everything but the markers (question, the rest of the reasoning, the answer), the loop in parallel rounds as in
pretraining (15% plain updates). One problem per row. The optimizer is reset at every stage (as in Coconut).

  --stages 1,2,3   the curriculum (stage s: s steps hidden; at stage 1 only the restated first value)
  --stages all     every step hidden from the start: question, markers, answer
  --marks 0        the no-thoughts control: the hidden steps are deleted, nothing is left in their place
                   (with --stages all: answering directly)
Quick evals every --eval_every steps on --eval_n test problems with the exact loop: answers right with this stage's
steps hidden (the rest written) and with every step hidden. A stage ends early when answers are right on --advance_at
of the problems or have not improved by a point in --plateau evals. Health at every eval (latents/health.py): the web
panel on --web_windows web windows; on the reasoning text the cut rate on the question and on the rest, the share of
the starting model's cuts still there, cuts moved since the last eval (forced cuts at markers left out).

usage: python scripts/holes_train.py --init runs/<written-reasoning run> --name=<run> [--task arith|logic]
           [--stages 1,2,3] [--stage_steps 3000] [--marks 1] [--per_step 3] [--batch 128] [--lr 1e-4]
Writes runs/<name>/{final.pt, metrics.jsonl, summary.json}, runs/<name>-stage<s>/final.pt.
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents import holes as H  # noqa: E402
from latents import health as health_lib  # noqa: E402
from latents.models import feedback_mode, model_forward, param_groups, ratio_loss  # noqa: E402
from show_cuts import load_any  # noqa: E402

TASKS = {"arith": ("latents.tasks", "data/task_arith"), "logic": ("latents.tasks_logic", "data/task_logic")}


def task_module(name):
    import importlib
    return importlib.import_module(TASKS[name][0])


def batch_of(probs, k, per_step, marks, mark, dev):
    """Padded rows (one problem each), real lengths, and the loss mask over targets (real bytes, not markers)."""
    seqs = [H.example(p, (H.count_steps(p["q"]) if k is None else k), per_step, marks, mark)[0] for p in probs]
    x = H.pad_batch(seqs, dev)
    lens = torch.tensor([len(s) for s in seqs], device=dev)
    pos = torch.arange(x.shape[1] - 1, device=dev)[None]
    keep = (pos + 1 < lens[:, None]) & (x[:, 1:] != mark)
    return x, lens, keep


def loss_on(model, mc, probs, k, a, mark, dev):
    x, lens, keep = batch_of(probs, k, a.per_step, bool(a.marks), mark, dev)
    out = model_forward(model, x, mc)
    ce = F.cross_entropy(out.logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(keep.shape)
    l_ce = (ce * keep).sum() / keep.sum()
    real = (torch.arange(x.shape[1], device=dev)[None] < lens[:, None]).flatten()
    forced = out.extras.get("forced")
    free = real if forced is None else real & ~forced
    l_ratio = ratio_loss(out.bpred[0], mc.get("N", 6), keep=free)
    bm = out.bpred[0].boundary_mask
    return l_ce + float(mc.get("alpha", 0.03)) * l_ratio, {"ce": float(l_ce.detach()), "ratio_loss": float(l_ratio.detach()),
                                                          "cut_rate": float(bm[free].float().mean())}


@torch.no_grad()
def cut_masks(model, mc, probs, k, a, mark, dev):
    """Per eval problem: boundary mask on its real bytes (forced cuts removed), and the question mask."""
    x, lens, _ = batch_of(probs, k, a.per_step, bool(a.marks), mark, dev)
    out = model_forward(model, x, mc)
    bm = out.bpred[0].boundary_mask.view(x.shape)
    if "forced" in out.extras:
        bm = bm & ~out.extras["forced"].view(x.shape)
    real = torch.arange(x.shape[1], device=dev)[None] < lens[:, None]
    return bm & real, health_lib.question_mask(x) & real, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", choices=list(TASKS), default="arith")
    ap.add_argument("--stages", default="1,2,3", help="comma list of stages, or 'all' (every step hidden at once)")
    ap.add_argument("--stage_steps", type=int, default=3000)
    ap.add_argument("--marks", type=int, default=1, help="0: hidden steps deleted with nothing in their place")
    ap.add_argument("--per_step", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=0, help="parallel rounds in training (0: as pretrained)")
    ap.add_argument("--plain_frac", type=float, default=-1, help="share of plain updates (-1: as pretrained)")
    ap.add_argument("--train_steps", default="2,6")
    ap.add_argument("--data_dir", default=None, help="test set (default: the task's)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--advance_at", type=float, default=0.98)
    ap.add_argument("--plateau", type=int, default=3)
    ap.add_argument("--web_windows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--wandb", type=int, default=1)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    cfg, model, _ = load_any(a.init, dev)
    model.train()
    mc = cfg["model"]
    if not mc.get("holes"):  # a model pretrained without holes: markers are then plain extra slots
        mc["holes"] = {"mark": H.MARK}
    mark = H.hole_mark_of(mc)
    fbc = mc.get("feedback") or {}
    rounds = a.rounds or (int(fbc.get("passes", 1)) if fbc else 1)
    plain_frac = float(fbc.get("plain_frac", 0) or 0) if a.plain_frac < 0 else a.plain_frac
    if fbc:
        mc["feedback"] = dict(fbc, passes=rounds)
        model._feedback_cfg = dict(model._feedback_cfg, passes=rounds, mode="passes")
    T = task_module(a.task)
    data_dir = a.data_dir or TASKS[a.task][1]
    test = [json.loads(l) for l in open(ROOT / data_dir / "test.jsonl")][: a.eval_n]
    stages = [None] if a.stages == "all" else [int(s) for s in a.stages.split(",")]
    tr_steps = tuple(int(v) for v in a.train_steps.split(","))
    rng, fb_rng = random.Random(1000 + a.seed), random.Random(77 + a.seed)
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    groups, _ = param_groups(model, mc, a.wd)
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    mlog = open(run_dir / "metrics.jsonl", "w")
    extra = dict(task=a.task, stages=a.stages, marks=a.marks, per_step=a.per_step, init=a.init, rounds=rounds,
                 plain_frac=plain_frac, stage_steps=a.stage_steps)

    from latents import train as TR
    web = None
    if (ROOT / "data/fwe/val.u8").exists() and a.web_windows:
        web = TR.Data("bytes", "data/fwe")
    wtc = dict(seq_len=8192, micro_batch=8)
    wb = None
    if a.wandb:
        try:
            import wandb
            wb = wandb.init(entity=TR.WANDB_ENTITY, project=TR.WANDB_PROJECT, name=a.name, group="holes",
                            config=dict(vars(a), **extra), dir=str(ROOT / "runs"))
        except Exception as e:
            print("W&B off:", repr(e)[:200], flush=True)

    def log(rec):
        mlog.write(json.dumps(rec) + "\n")
        mlog.flush()
        if wb is not None:
            flat = {}
            for k, v in rec.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            flat[f"{k}/{k2}"] = v2
                elif isinstance(v, (int, float)) and k != "step":
                    flat[k] = v
            wb.log(flat, step=rec["step"])

    # the starting model's cuts on the eval problems, in every stage's format
    model.eval()
    with torch.autocast(**ac):
        start_cuts = {s: cut_masks(model, mc, test, s, a, mark, dev)[0] for s in stages}
    model.train()
    prev_cuts = dict(start_cuts)

    def run_eval(stage, step):
        model.eval()
        res = {}
        with torch.autocast(**ac):
            st_k = stage
            res["this_stage"] = H.evaluate_reasoning(model, mc, test, st_k, a.per_step, bool(a.marks))
            if stage is not None:
                res["all_hidden"] = H.evaluate_reasoning(model, mc, test, None, a.per_step, bool(a.marks))
            bm, q, real = cut_masks(model, mc, test, stage, a, mark, dev)
            s0, sp = start_cuts[stage], prev_cuts[stage]
            hl = {"cut_rate_question": float((bm & q).sum() / q.sum().clamp(min=1)),
                  "cut_rate_rest": float((bm & real & ~q).sum() / (real & ~q).sum().clamp(min=1)),
                  "cuts_kept_from_start": float((bm & s0).sum() / s0.sum().clamp(min=1)),
                  "cuts_moved": float((bm ^ sp).sum() / sp.sum().clamp(min=1))}
            prev_cuts[stage] = bm
            if web is not None:
                hl.update(TR.evaluate(model, web, {**mc}, wtc, max_windows=a.web_windows, prefix="web"))
        model.train()
        rec = {"step": step, "stage": stage if stage is not None else "all", "eval": res, "health": hl,
               "min": (time.time() - t0) / 60}
        log(rec)
        msg = f"this stage {res['this_stage']['acc']:.3f} {res['this_stage']['by_steps']}"
        if "all_hidden" in res:
            msg += f"  all hidden {res['all_hidden']['acc']:.3f}"
        print(f"[stage {rec['stage']} step {step}] {msg}  q-cut {hl['cut_rate_question']:.3f} rest-cut "
              f"{hl['cut_rate_rest']:.3f} kept {hl['cuts_kept_from_start']:.3f} web bpb {hl.get('web/bpb', float('nan')):.4f}",
              flush=True)
        return res

    t0, step = time.time(), 0
    summary = {}
    for stage in stages:
        opt = torch.optim.AdamW([{"params": g["params"], "weight_decay": g["weight_decay"], "lr": a.lr * g["lr_mult"]}
                                 for g in groups], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)  # reset at every stage
        base = [g["lr"] for g in opt.param_groups]
        params = [p for g in opt.param_groups for p in g["params"]]
        n = a.stage_steps
        best, since, done, ev = -1.0, 0, None, None
        gmax = 0.0
        for s in range(n):
            w = max(1, int(0.05 * n))
            f = (s + 1) / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / max(1, n - w)))
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * f
            probs = [T.sample(rng, steps=tr_steps) for _ in range(a.batch)]
            plain_now = plain_frac > 0 and fb_rng.random() < plain_frac
            with torch.autocast(**ac), feedback_mode(model, rounds=1 if plain_now else None):
                loss, st = loss_on(model, mc, probs, stage, a, mark, dev)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            gmax = max(gmax, float(gn))
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            if s % a.log_every == 0 or s == n - 1:
                rec = {"step": step, "stage": stage if stage is not None else "all", "loss": float(loss),
                       "grad_norm": float(gn), "grad_norm_max": gmax, **st, "min": (time.time() - t0) / 60}
                gmax = 0.0
                log(rec)
                if s % (a.log_every * 8) == 0:
                    print(json.dumps(rec), flush=True)
            step += 1
            if a.eval_every and (s + 1) % a.eval_every == 0 and s + 1 < n:
                ev = run_eval(stage, step)
                acc = ev["this_stage"]["acc"]
                if acc >= a.advance_at:
                    done = f"learned ({acc:.3f})"
                elif acc > best + 0.01:
                    best, since = acc, 0
                else:
                    since += 1
                    if since >= a.plateau:
                        done = f"plateau ({acc:.3f}, best {best:.3f})"
                if done:
                    print(f"[stage {stage}] ends at step {step} ({s + 1} steps): {done}", flush=True)
                    break
        if not done:
            ev = run_eval(stage, step)
        key = f"stage{stage}" if stage is not None else "all"
        summary[key] = dict(ev, steps_used=s + 1, ended=done or "full stage")
        sdir = ROOT / "runs" / f"{a.name}-{key}"
        sdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": dict(cfg, holes_train=dict(extra, stage=stage))}, sdir / "final.pt")
    torch.save({"model": model.state_dict(), "cfg": dict(cfg, holes_train=extra)}, run_dir / "final.pt")
    json.dump({"name": a.name, **extra, "stages_eval": summary, "min": (time.time() - t0) / 60},
              open(run_dir / "summary.json", "w"), indent=1)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
