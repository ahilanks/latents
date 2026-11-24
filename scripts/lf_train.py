"""Latent Feedback's run on our web bytes (diagnosis, 2026-11-24; user: "only do what latent feedback does ... if needed
do their run too"). Their Transformer (latents/lf.py) at their 60M size (512 wide, 8 layers, 4 heads, MLP 1536) with
their recipe, from ifm-ai/LoRM @ 2460ca8 (works/latent_feedback: configs/base.yaml, feedback.yaml, make_config.py,
runs.csv row feedback-60m-u12.5-f1 / transformer-60m-u12.5-f1; lorm/train: algo.py SampledPasses, optim/__init__.py,
scheduler.py, main.py clipping):
  - NorMuon on the block matrices and the feedback W (lr, cautious weight decay, momentum 0.85 -> 0.95 over the
    momentum warmup, beta2); AdamW (0.9, beta2) on the embedding (lr x 0.905, no decay), the head (lr x 0.08) and
    the norm gains (no decay)
  - WSD schedule: linear warmup, constant, linear decay to 0 over the last 20%; gradient norm clipped at 1
  - feedback arm: each update draws 1 pass (15%) or 2 passes (85%) from (seed, step); earlier passes without gradient
    and the fed state detached; evaluation at 4 passes
Differences, forced by our setting: bytes (vocabulary 256) instead of Llama-2 tokens; our web text (data/fwe) instead
of their Dolma 3 mix; 0.6B bytes as in our H-Net runs; one GPU.
Health every --eval_every steps on --eval_windows validation windows: bits per byte at 1, 2 and 4 passes; the top state
before the final norm (size, share that is the same for every position) and after it (variation between positions,
effective rank), the fed-back vector's size relative to the token embedding and how much it varies between positions.

usage: python scripts/lf_train.py --name LF-fb-s0 [--feedback 1] [--bytes 6e8]
Writes runs/<name>/{metrics.jsonl, summary.json, final.pt}.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents.health import participation_ratio  # noqa: E402
from latents.lf import LFTransformer  # noqa: E402
from latents.normuon import NorMuon  # noqa: E402

LN2 = math.log(2)


def wsd(step, total, warmup, decay):
    if step < warmup:
        return (step + 1) / warmup
    if step >= total - decay:
        return max(0.0, (total - step) / decay)
    return 1.0


def build_optimizers(model, a):
    mats = model.hidden_matrices()
    ids = {id(p) for p in mats}
    emb, head = model.embedding.weight, model.head.weight
    rest = [p for p in model.parameters() if id(p) not in ids and p is not emb and p is not head]
    assert all(p.ndim == 1 for p in rest)
    adamw = torch.optim.AdamW([{"params": [emb], "lr": a.lr * 0.9050966799189999, "weight_decay": 0.0},
                               {"params": [head], "lr": a.lr * 0.08, "weight_decay": a.wd},
                               {"params": rest, "lr": a.lr, "weight_decay": 0.0}],
                              lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, weight_decay=a.wd)
    muon = NorMuon(mats, lr=a.lr, weight_decay=a.wd, momentum=0.95, momentum_start=0.85,
                   momentum_warmup_steps=a.momentum_warmup, beta2=a.beta2)
    for o in (adamw, muon):
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return adamw, muon


@torch.no_grad()
def health(model, xv, ac):
    """Validation bits per byte at 1, 2, 4 passes and the state numbers at 4 passes (their evaluation)."""
    res = {}
    for p in (1, 2, 4):
        if model.fusion is None and p > 1:
            continue
        tot, n = 0.0, 0
        for i in range(0, xv.shape[0], 16):
            x = xv[i:i + 16]
            with torch.autocast(**ac):
                lg = model(x[:, :-1], passes=p)
            tot += float(F.cross_entropy(lg.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum"))
            n += x[:, 1:].numel()
        res[f"bpb_p{p}"] = tot / n / LN2
    x = xv[:8]
    with torch.autocast(**ac):
        _, st = model(x[:, :-1], passes=4, return_state=True)
    top = st["top"].float().flatten(0, 1)
    res["top_norm"] = float(top.norm(dim=-1).mean())
    res["top_const"] = float(top.mean(0).pow(2).sum() / top.pow(2).sum(-1).mean().clamp(min=1e-12))
    z = model.norm(st["top"]).float().flatten(0, 1)
    res["out_variation"] = float((z - z.mean(0)).pow(2).sum(-1).mean() / z.pow(2).sum(-1).mean().clamp(min=1e-12))
    res["out_rank"] = participation_ratio(z[:20000])
    res["emb_norm"] = float(st["emb"].float().norm(dim=-1).mean())
    if model.fusion is not None:
        with torch.autocast(**ac):
            f = model.fed(st["top"]).float()[:, :-1].flatten(0, 1)
        e = st["emb"].float()[:, 1:].flatten(0, 1)
        res["fb_share"] = float(f.norm(dim=-1).mean() / e.norm(dim=-1).mean().clamp(min=1e-12))
        res["fb_variation"] = float((f - f.mean(0)).pow(2).sum(-1).mean() / f.pow(2).sum(-1).mean().clamp(min=1e-12))
        res["fb_weight"] = float(model.fusion.weight.float().norm())
        res["loop_gain_bpb"] = res["bpb_p1"] - res["bpb_p4"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--feedback", type=int, default=1)
    ap.add_argument("--bytes", type=float, default=6e8)
    ap.add_argument("--data", default="data/fwe")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64, help="sequences per update (theirs at 60M, U/N 12.5: 64)")
    ap.add_argument("--micro", type=int, default=32, help="sequences per microbatch (theirs: half the batch, x2)")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1536)
    # feedback-60m-u12.5-f1 (transformer arm: --lr 0.025192911532357438 --wd 0.0156)
    ap.add_argument("--lr", type=float, default=0.03562815716472544)
    ap.add_argument("--wd", type=float, default=0.00987)
    ap.add_argument("--beta2", type=float, default=0.987)
    ap.add_argument("--momentum_warmup", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=458)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--pass_mix", default="0.15,0.85")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_windows", type=int, default=64)
    ap.add_argument("--steps", type=int, default=0, help="override the step count (tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", type=int, default=1)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    train = np.memmap(ROOT / a.data / "train.u8", dtype=np.uint8, mode="r")
    val = np.memmap(ROOT / a.data / "val.u8", dtype=np.uint8, mode="r")
    W = a.seq + 1
    steps = a.steps or int(a.bytes // (a.batch * a.seq))
    n_win = min(len(train) // W, steps * a.batch)
    order = np.random.default_rng(a.seed).permutation(n_win)
    vstarts = np.linspace(0, len(val) - W - 1, a.eval_windows).astype(np.int64)
    xv = torch.from_numpy(np.stack([val[s:s + W] for s in vstarts]).astype(np.int64)).to(dev)
    mix = [float(v) for v in a.pass_mix.split(",")]

    model = LFTransformer(a.dim, a.layers, a.heads, a.hidden, feedback=bool(a.feedback)).to(dev)
    adamw, muon = build_optimizers(model, a)
    decay = int(round(a.decay_frac * steps))
    run_dir = ROOT / "runs" / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a), steps_total=steps, decay_steps=decay, n_params=sum(p.numel() for p in model.parameters()),
               recipe="Latent Feedback (ifm-ai/LoRM @ 2460ca8), feedback-60m-u12.5-f1 settings, bytes")
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=1)
    mlog = open(run_dir / "metrics.jsonl", "w")
    use_wandb = bool(a.wandb)
    if use_wandb:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            import wandb
            from latents.train import WANDB_ENTITY, WANDB_PROJECT
            wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=a.name, group="lf", config=cfg,
                       dir=str(ROOT / "runs"))
        except Exception as e:
            print("W&B off:", repr(e)[:200], flush=True)
            use_wandb = False

    def log(d, step):
        d = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        mlog.write(json.dumps({"step": step, **d}) + "\n")
        mlog.flush()
        if use_wandb:
            wandb.log(d, step=step)

    print(json.dumps(cfg), flush=True)
    t0 = time.time()
    ev = {}
    for step in range(steps):
        s = wsd(step, steps, a.warmup, decay)
        for o in (adamw, muon):
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * s
        passes = (int(np.random.default_rng([a.seed, step]).choice(len(mix), p=mix)) + 1) if model.fusion is not None else 1
        idx = order[(step * a.batch) % n_win:(step * a.batch) % n_win + a.batch]
        xb = torch.from_numpy(np.stack([train[i * W:(i + 1) * W] for i in idx]).astype(np.int64)).to(dev)
        tot = 0.0
        for j in range(0, a.batch, a.micro):
            x = xb[j:j + a.micro]
            with torch.autocast(**ac):
                lg = model(x[:, :-1], passes=passes)
            loss = F.cross_entropy(lg.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum") / (a.batch * a.seq)
            loss.backward()
            tot += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        adamw.step()
        muon.step()
        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == steps - 1:
            log({"train/bpb": tot / LN2, "train/grad_norm": float(gn), "train/passes": passes, "lr_scale": s,
                 "elapsed_min": (time.time() - t0) / 60}, step)
        if step % a.eval_every == 0 or step == steps - 1:
            model.eval()
            ev = {f"val/{k}": v for k, v in health(model, xv, ac).items()}
            model.train()
            log(ev, step)
            print(f"step {step}/{steps} " + " ".join(f"{k[4:]} {v:.4g}" for k, v in ev.items()), flush=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, run_dir / "final.pt")
    json.dump({"steps": steps, "minutes": (time.time() - t0) / 60, **ev}, open(run_dir / "summary.json", "w"), indent=1)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
