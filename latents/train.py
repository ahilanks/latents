"""Trainer for BPE / byte-isotropic / H-Net (+ latent objectives, nested levels). 1 GPU, multi-GPU or CPU.

usage: python -m latents.train configs/<cfg>.yaml [dotted.key=value ...]
       torchrun --standalone --nproc_per_node=N -m latents.train configs/<cfg>.yaml ...   (data parallel)
Logs: W&B (project hnet-latents), local runs/<name>/{config.yaml,metrics.jsonl,summary.json,chunks.txt},
final checkpoint + logs uploaded to HF repo a5ilank/hnet-latents under runs/<name>/.

Early kill (train.kill_ref=<run> [train.kill_margin=0.03 train.kill_after=0.8]): at each eval after
kill_after of training, if val BPB is worse than the reference run's val BPB at the same step by more than
kill_margin, stop, write summary.json with killed=true and exit with code 3.
Manual kill: create runs/<name>/KILL (its text = the reason); the run stops the same way at its next eval, so the
ladder gate sees a killed run instead of waiting forever for a summary.
"""
import copy
import json
import math
import os
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Early kill only late in training: on H100 the val BPB at ~45% swings by 0.05-0.07 between seeds and between
# identical runs (R0: seed order flips between step 1000 and the end), so a 0.03 margin at 33% killed sound runs.
KILL_AFTER = 0.8
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

from latents import flops as flops_lib  # noqa: E402
from latents.models import build_model, model_forward, param_groups, ratio_loss  # noqa: E402
from latents import latent as latent_lib  # noqa: E402

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ahilan-uc-berkeley-electrical-engineering-computer-sciences")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "hnet-latents")
HF_REPO = os.environ.get("HF_REPO", "a5ilank/hnet-latents")
PEAK_BF16 = {"H100": 989e12, "A100": 312e12}  # dense bf16 peak per GPU, for the logged MFU
LN2 = math.log(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WS_BYTES = (32, 9, 10, 13)


# ----------------------------------------------------------------------------- distributed
def peak_flops():
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    return next((v for k, v in PEAK_BF16.items() if k in name), PEAK_BF16["A100"])


def dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def setup_dist():
    global DEV
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local)
            DEV = f"cuda:{local}"
    return dist_info()


def all_reduce_(t, op="sum"):
    if dist_info()[1] > 1:
        dist.all_reduce(t)  # SUM (AVG is NCCL-only)
        if op != "sum":
            t /= dist.get_world_size()
    return t


def sync_grads(params):
    """Average gradients across ranks (one flat all-reduce; missing grads count as zeros)."""
    if dist_info()[1] == 1:
        return
    gs = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        gs.append(p.grad)
    flat = torch._utils._flatten_dense_tensors(gs)
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    for g, f in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
        g.copy_(f)


def broadcast_state(*modules):
    if dist_info()[1] == 1:
        return
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            dist.broadcast(t.data, 0)


# ----------------------------------------------------------------------------- config
def load_yaml(path):
    cfg = yaml.safe_load(open(path))
    if "base" in cfg:  # recursive inheritance: configs/<base>
        cfg = deep_merge(load_yaml(ROOT / "configs" / cfg.pop("base")), cfg)
    return cfg


def load_cfg(argv):
    cfg = load_yaml(argv[0])
    for kv in argv[1:]:
        k, v = kv.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = _num(yaml.safe_load(v))
    return _num_tree(cfg)


def _num(v):
    if isinstance(v, str):  # YAML 1.1 reads "6e8"/"1e-3" as strings
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _num_tree(d):
    if isinstance(d, dict):
        return {k: (_num_tree(v) if k not in ("name", "group", "notes", "kill_ref", "init_from") else v) for k, v in d.items()}
    if isinstance(d, list):
        return [_num_tree(v) for v in d]
    return _num(d)


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def int_keys(mc):
    for k in ("N", "pool_k"):
        if k in mc and isinstance(mc[k], float) and mc[k].is_integer():
            mc[k] = int(mc[k])
    return mc


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, tokenizer, data_dir):
        dd = ROOT / data_dir
        self.bpe = tokenizer == "gpt2"
        suf = "gpt2.u16" if self.bpe else "u8"
        dt = np.uint16 if self.bpe else np.uint8
        self.train = np.memmap(dd / f"train.{suf}", dtype=dt, mode="r")
        self.val = np.fromfile(dd / f"val.{suf}", dtype=dt)
        if self.bpe:
            self.tok_bytes = torch.from_numpy(np.fromfile(dd / "gpt2_token_bytes.u16", dtype=np.uint16).astype(np.int64))
            tr_bytes = os.path.getsize(dd / "train.u8")
            self.bytes_per_token = tr_bytes / len(self.train)
        else:
            self.tok_bytes = None
            self.bytes_per_token = 1.0

    def batches(self, B, L, seed):
        """Uniformly random windows of L+1 tokens; background-thread prefetch."""
        q = queue.Queue(maxsize=8)
        rng = np.random.default_rng(seed)
        ar = np.arange(L + 1)
        cuda = DEV.startswith("cuda")

        def worker():
            while True:
                idx = rng.integers(0, len(self.train) - L - 1, size=B)
                w = torch.from_numpy(self.train[idx[:, None] + ar].astype(np.int64))
                q.put(w.pin_memory() if cuda else w)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            w = q.get().to(DEV, non_blocking=True)
            yield w[:, :-1], w[:, 1:]

    def target_bytes(self, y):
        if self.tok_bytes is None:
            return torch.ones_like(y)
        return self.tok_bytes.to(y.device)[y]


# ----------------------------------------------------------------------------- schedule
def wsd(step, total, warmup_frac, decay_frac):
    w = max(1, int(total * warmup_frac))
    d0 = int(total * (1 - decay_frac))
    if step < w:
        return (step + 1) / w
    if step < d0:
        return 1.0
    return max(0.0, 1 - math.sqrt((step - d0) / max(1, total - d0)))  # 1-sqrt cooldown


def latent_scale(lat, step, total, decay_frac):
    """latent.beta_schedule: const (1) | cooldown (decays like the LR during the cooldown, to 0)."""
    if lat is None or lat.cfg.get("beta_schedule", "const") == "const":
        return 1.0
    d0 = int(total * (1 - decay_frac))
    return 1.0 if step < d0 else max(0.0, 1 - math.sqrt((step - d0) / max(1, total - d0)))


def autocast():
    return torch.autocast("cuda" if DEV.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                          enabled=DEV.startswith("cuda"))


def routing_losses(model, out, mc):
    """Ratio losses for learned routers (level 1 unless pooled / surprise) and nested levels."""
    loss, st = 0.0, {}
    if out.bpred and not mc.get("pool_k") and mc.get("router", "learned") != "surprise":
        rl = ratio_loss(out.bpred[0], mc.get("N", 6))
        loss = loss + mc.get("alpha", 0.03) * rl
        st["ratio_loss"] = rl.detach()
    nest = getattr(model, "nest", None)
    if nest is not None and out.extras.get("levels") is not None:
        l2, st2 = nest.router_losses(out.extras["levels"])
        loss = loss + l2
        st.update(st2)
    return loss, st


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, data, mc, tc, lat=None, max_windows=None):
    rank, world = dist_info()
    model.eval()
    L = tc["seq_len"]
    v = torch.from_numpy(data.val.astype(np.int64))
    n = (len(v) - 1) // L
    if max_windows:
        n = min(n, max_windows)
    B = tc["micro_batch"]
    mine = list(range(n))[rank::world]
    acc = torch.zeros(8, dtype=torch.float64, device=DEV)  # ce, bytes, nb, nc, nprec, nsp, nrec, nbatches
    lat_acc = {}
    ws = torch.tensor(WS_BYTES, device=DEV)
    for i in range(0, len(mine), B):
        idx = mine[i:i + B]
        w = torch.stack([v[j * L: j * L + L + 1] for j in idx]).to(DEV)
        x, y = w[:, :-1], w[:, 1:]
        with autocast():
            out = model_forward(model, x, mc, lat)
            if lat is not None:
                for k, val in lat.eval_metrics(model, out, x).items():
                    lat_acc[k] = lat_acc.get(k, 0.0) + val
        ce = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="sum")
        acc[0] += ce.double()
        acc[1] += data.target_bytes(y).sum().double()
        acc[7] += 1
        if out.bpred:
            b = out.bpred[0].boundary_mask.view(x.shape)
            acc[2] += b.sum()
            acc[3] += b.numel()
            isws = torch.isin(x, ws)
            edge = isws | F.pad(isws[:, :-1], (1, 0))
            acc[4] += (b & edge).sum()
            acc[5] += isws.sum()
            acc[6] += (isws & (b | F.pad(b[:, 1:], (0, 1)))).sum()
    all_reduce_(acc)
    keys = sorted(lat_acc)
    if world > 1:
        keys_all = [None] * world
        dist.all_gather_object(keys_all, keys)
        keys = sorted(set(k for ks in keys_all for k in ks))
    lt = torch.tensor([lat_acc.get(k, 0.0) for k in keys] + [len(lat_acc) > 0], dtype=torch.float64, device=DEV)
    all_reduce_(lt)
    model.train()
    res = {"val/bpb": (acc[0] / LN2 / acc[1]).item()}
    if acc[3] > 0:
        res["val/compression_L1/L0"] = (acc[2] / acc[3]).item()
        res["val/edge_prec"] = (acc[4] / acc[2].clamp(min=1)).item()
        res["val/space_recall"] = (acc[6] / acc[5].clamp(min=1)).item()
    for k, val in zip(keys, lt[:-1].tolist()):
        res[f"val/{k}"] = val / acc[7].item()
    return res


@torch.no_grad()
def chunk_sample(model, data, mc, lat=None, n_bytes=1200):
    """Render boundaries of the first val document(s) as text with '|' separators."""
    if data.bpe or len(mc["d_model"]) == 1:
        return None
    model.eval()
    x = torch.from_numpy(data.val[: n_bytes].astype(np.int64))[None].to(DEV)
    with autocast():
        out = model_forward(model, x, mc, lat)
    model.train()
    mask = out.bpred[0].boundary_mask.cpu().numpy()
    bs = data.val[:n_bytes].tobytes()
    pieces, cur = [], bytearray()
    for b, m in zip(bs, mask):
        if m and cur:
            pieces.append(bytes(cur).decode("utf-8", "replace"))
            cur = bytearray()
        cur.append(b)
    pieces.append(bytes(cur).decode("utf-8", "replace"))
    return "|".join(p.replace("�", "?") for p in pieces).replace(chr(254), "<s>")


def ref_bpb_at(ref, step):
    p = ROOT / "runs" / ref / "metrics.jsonl"
    if not p.exists():
        return None
    for line in open(p):
        r = json.loads(line)
        if r.get("step") == step and "val/bpb" in r:
            return r["val/bpb"]
    return None


# ----------------------------------------------------------------------------- build
def prepare(cfg, data):
    mc, tc = cfg["model"], cfg["train"]
    int_keys(mc)
    mc["vocab_size_padded"] = 50304 if data.bpe else 256
    mc["vocab_size"] = 50257 if data.bpe else 256
    mc["bytes_per_token"] = data.bytes_per_token
    mc["seq_len"] = tc["seq_len"]  # in tokens of the model's own vocab
    if cfg.get("latent"):
        mc["_latent_head"] = cfg["latent"].get("head", "linear")
    return mc, tc


def load_init(model, lat, path):
    """train.init_from: start from a checkpoint (model + latent state); missing/new keys are reported."""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    r = model.load_state_dict(ck["model"], strict=False)
    print("init_from", path, "missing:", r.missing_keys, "unexpected:", r.unexpected_keys, flush=True)
    if lat is not None and "latent" in ck:
        print("latent:", lat.load_state_dict(ck["latent"], strict=False), flush=True)


# ----------------------------------------------------------------------------- main
def main(argv):
    rank, world = setup_dist()
    cfg = load_cfg(argv)
    name, tc = cfg["name"], cfg["train"]
    torch.manual_seed(tc.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = Data(cfg["model"]["tokenizer"], cfg.get("data_dir", "data/fwe"))
    mc, tc = prepare(cfg, data)
    B, L = tc["batch_size"], tc["seq_len"]
    bytes_per_step = B * L * data.bytes_per_token
    total_steps = tc.get("steps") or int(tc["total_bytes"] / bytes_per_step)
    is_hnet = len(mc["d_model"]) > 1

    model = build_model(mc, device=DEV)
    groups, mults = param_groups(model, mc, tc.get("wd", 0.1))
    lat = latent_lib.build(cfg.get("latent"), model, mc) if cfg.get("latent") else None
    if tc.get("init_from"):
        load_init(model, lat, tc["init_from"])
    opt_groups = [{k: g[k] for k in ("params", "weight_decay")} | {"lr": tc["lr"] * g["lr_mult"]} for g in groups]
    if lat is not None:
        lat_params = [p for p in lat.parameters() if p.requires_grad]
        if lat_params:
            opt_groups.append({"params": lat_params, "weight_decay": tc.get("wd", 0.1), "lr": tc["lr"] * lat.lr_mult})
            groups.append({"lr_mult": lat.lr_mult})
    opt = torch.optim.AdamW(opt_groups, lr=tc["lr"], betas=(0.9, 0.95), eps=1e-8, fused=DEV.startswith("cuda"))
    broadcast_state(model, *( [lat] if lat is not None else []))
    train_params = [p for g in opt.param_groups for p in g["params"]]
    n_params = sum(p.numel() for p in model.parameters())

    est_fwd = flops_lib.model_fwd_per_byte(mc)
    run_dir = ROOT / "runs" / name
    cfg["derived"] = dict(
        n_params=n_params, total_steps=total_steps, bytes_per_step=bytes_per_step, lr_multipliers=mults,
        est_train_flops_per_byte=3 * est_fwd, est_total_flops=3 * est_fwd * bytes_per_step * total_steps, world=world,
    )
    use_wandb = cfg.get("wandb", True) and rank == 0
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "KILL").unlink(missing_ok=True)  # a leftover from an earlier run of this name must not kill this one
        yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
        print(json.dumps(cfg["derived"], indent=1), flush=True)
        mlog = open(run_dir / "metrics.jsonl", "w")
    if use_wandb:
        import wandb
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, group=cfg.get("group"),
                   tags=cfg.get("tags", []), notes=cfg.get("notes"), config=cfg, dir=str(ROOT / "runs"))

    def log(d, step):
        if rank != 0:
            return
        d = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        mlog.write(json.dumps({"step": step, **d}) + "\n")
        mlog.flush()
        if use_wandb:
            wandb.log(d, step=step)

    mb = tc.get("micro_batch", B)
    assert B % (mb * world) == 0, f"batch {B} must split into micro-batches of {mb} on {world} ranks"
    accum = B // (mb * world)
    it = data.batches(B // world, L, seed=tc.get("seed", 0) + 1000 * rank)
    t0 = time.time()
    tlast = t0
    cum_flops = 0.0
    cum_bytes = 0.0
    killed = None
    ev = {}
    for step in range(total_steps):
        lr_scale = wsd(step, total_steps, tc.get("warmup_frac", 0.1), tc.get("decay_frac", 0.2))
        lscale = latent_scale(lat, step, total_steps, tc.get("decay_frac", 0.2))
        for pg, g in zip(opt.param_groups, groups):
            pg["lr"] = tc["lr"] * g["lr_mult"] * lr_scale
        x_all, y_all = next(it)
        stats = {}
        f_step = 0.0
        for a in range(accum):
            x, y = x_all[a * mb:(a + 1) * mb], y_all[a * mb:(a + 1) * mb]
            with autocast():
                out = model_forward(model, x, mc, lat, step_frac=step / total_steps)
                ce_tok = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="none")
                ce = ce_tok.mean()
                loss = ce
                if is_hnet:
                    l_rt, st_rt = routing_losses(model, out, mc)
                    loss = loss + l_rt
                    for k, v in st_rt.items():
                        stats[k] = stats.get(k, 0) + v / accum
                if lat is not None:
                    l_lat, lat_stats = lat.loss(model, out, x, step=step, total_steps=total_steps, scale=lscale)
                    loss = loss + l_lat
                    for k, v in lat_stats.items():
                        stats[k] = stats.get(k, 0) + v.detach() / accum
            (loss / accum).backward()
            f_step += flops_lib.step_train_flops(mc, out, lat, getattr(model, "nest", None)) if is_hnet else \
                3 * flops_lib.model_fwd_per_byte(mc) * x.numel() * data.bytes_per_token
            nbytes = data.target_bytes(y).sum()
            stats["ce"] = stats.get("ce", 0) + ce.detach() / accum
            stats["bpb"] = stats.get("bpb", 0) + (ce_tok.detach().sum() / LN2 / nbytes) / accum
            if out.bpred:
                stats["compression_L1/L0"] = stats.get("compression_L1/L0", 0) + out.bpred[0].boundary_mask.float().mean().detach() / accum
                stats["boundary_prob_mean"] = stats.get("boundary_prob_mean", 0) + out.bpred[0].boundary_prob[..., -1].float().mean().detach() / accum
        sync_grads(train_params)
        gn = torch.nn.utils.clip_grad_norm_(train_params, tc.get("clip", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if lat is not None:
            lat.after_step(model, step, total_steps)
        cum_flops += f_step * world
        cum_bytes += bytes_per_step

        if step % tc.get("log_every", 50) == 0 or step == total_steps - 1:
            if DEV.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.time()
            dt = (now - tlast) / (tc.get("log_every", 50) if step else 1)
            tlast = now
            d = {f"train/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in stats.items()}
            d.update({"train/grad_norm": gn.item(), "lr": tc["lr"] * lr_scale, "lat_scale": lscale, "flops": cum_flops,
                      "bytes": cum_bytes, "perf/sec_per_step": dt, "perf/bytes_per_sec": bytes_per_step / dt,
                      "perf/mfu": f_step / dt / peak_flops()})
            if DEV.startswith("cuda"):
                d["perf/max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log(d, step)
            if rank == 0 and step % (tc.get("log_every", 50) * 10) == 0:
                print(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in d.items() if isinstance(v, float)), flush=True)
            bad = torch.tensor(float(not math.isfinite(d["train/ce"])), device=DEV)
            if all_reduce_(bad) > 0:
                print("non-finite loss; abort", flush=True)
                killed = "non-finite loss"
                break
        if (step > 0 and step % tc.get("eval_every", 1000) == 0) or step == total_steps - 1:
            ev = evaluate(model, data, mc, tc, lat=lat, max_windows=tc.get("eval_windows"))
            ev["flops"] = cum_flops
            log(ev, step)
            if rank == 0:
                print(f"[eval] step {step} " + " ".join(f"{k}={v:.5g}" for k, v in ev.items()), flush=True)
            if step < total_steps - 1:
                # early kill: worse than the reference run at the same step, or a runs/<name>/KILL file (a person or
                # monitor judged the run hopeless; its text is the reason). Either way: summary with killed, exit 3.
                flag = torch.zeros(1, device=DEV)
                if rank == 0:
                    kf = run_dir / "KILL"
                    ref = ref_bpb_at(tc["kill_ref"], step) if (tc.get("kill_ref") and
                                                              step >= tc.get("kill_after", KILL_AFTER) * total_steps) else None
                    if kf.exists():
                        flag += 1
                        killed = "manual: " + (kf.read_text().strip() or "KILL file")
                    elif ref is not None and ev["val/bpb"] - ref > tc.get("kill_margin", 0.03):
                        flag += 1
                        killed = f"val bpb {ev['val/bpb']:.4f} vs {tc['kill_ref']} {ref:.4f} at step {step}"
                all_reduce_(flag)
                if flag.item() > 0:
                    killed = killed or "early kill (rank 0)"
                    if rank == 0:
                        print("[kill]", killed, flush=True)
                    break

    # ---- final: full val, chunk sample, summary, checkpoint
    if killed:
        final = ev if ev else {}
    else:
        final = evaluate(model, data, mc, tc, lat=lat, max_windows=tc.get("final_eval_windows"))
    if rank == 0:
        sample = None if killed else chunk_sample(model, data, mc, lat)
        summary = {"name": name, "group": cfg.get("group"), **{k: v for k, v in final.items()},
                   "train_flops": cum_flops, "train_bytes": cum_bytes, "n_params": n_params,
                   "wall_min": (time.time() - t0) / 60, "steps": total_steps, "world": world}
        if killed:
            summary["killed"] = killed
        print("[final]", json.dumps(summary), flush=True)
        json.dump(summary, open(run_dir / "summary.json", "w"), indent=1)
        if sample:
            open(run_dir / "chunks.txt", "w").write(sample)
        if not killed:
            torch.save({"model": model.state_dict(), "cfg": cfg, **({"latent": lat.state_dict()} if lat is not None else {})},
                       run_dir / "final.pt")
        if use_wandb:
            for k, v in summary.items():
                wandb.run.summary[f"final/{k}"] = v
            if sample:
                wandb.log({"chunks": wandb.Html("<pre style='white-space:pre-wrap'>" + sample.replace("<", "&lt;") + "</pre>")})
            wandb.finish()
        mlog.close()
        if cfg.get("upload_hf", True):
            try:
                from huggingface_hub import HfApi
                HfApi().upload_folder(repo_id=HF_REPO, folder_path=str(run_dir), path_in_repo=f"runs/{name}",
                                      commit_message=f"run {name}: val bpb {final.get('val/bpb', float('nan')):.4f}")
                print("uploaded to HF", flush=True)
            except Exception as e:  # never lose a run because of an upload failure
                print("HF upload failed:", e, flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if killed:
        sys.exit(4 if killed.startswith("non-finite") else 3)


if __name__ == "__main__":
    main(sys.argv[1:])
