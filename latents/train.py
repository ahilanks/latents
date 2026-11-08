"""Single-GPU trainer for BPE / byte-isotropic / H-Net (+ latent objectives).

usage: python -m latents.train configs/<cfg>.yaml [dotted.key=value ...]
Logs: W&B (project hnet-latents), local runs/<name>/{config.yaml,metrics.jsonl,summary.json,chunks.txt},
final checkpoint + logs uploaded to HF repo a5ilank/hnet-latents under runs/<name>/.
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
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from latents import flops as flops_lib  # noqa: E402
from latents.models import build_model, forward_lm, param_groups, ratio_loss  # noqa: E402
from latents import latent as latent_lib  # noqa: E402

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "ahilan-uc-berkeley-electrical-engineering-computer-sciences")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "hnet-latents")
HF_REPO = os.environ.get("HF_REPO", "a5ilank/hnet-latents")
A100_BF16 = 312e12
LN2 = math.log(2)


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
        return {k: (_num_tree(v) if k not in ("name", "group", "notes") else v) for k, v in d.items()}
    if isinstance(d, list):
        return [_num_tree(v) for v in d]
    return _num(d)


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


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

        def worker():
            while True:
                idx = rng.integers(0, len(self.train) - L - 1, size=B)
                w = torch.from_numpy(self.train[idx[:, None] + ar].astype(np.int64)).pin_memory()
                q.put(w)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            w = q.get().cuda(non_blocking=True)
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


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, data, mc, tc, lat=None, max_windows=None):
    model.eval()
    L = tc["seq_len"]
    v = torch.from_numpy(data.val.astype(np.int64))
    n = (len(v) - 1) // L
    if max_windows:
        n = min(n, max_windows)
    B = tc["micro_batch"]
    ce_sum = byte_sum = 0.0
    nb = nc = 0
    lat_acc = {}
    for i in range(0, n, B):
        idx = torch.arange(i, min(n, i + B))
        w = torch.stack([v[j * L : j * L + L + 1] for j in idx]).cuda()
        x, y = w[:, :-1], w[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward_lm(model, x, pool_k=mc.get("pool_k", 0), detach_main=bool(lat and lat.detach_main), zero_main=bool(mc.get("zero_main", False)), main_out_fn=(lat.head if (lat and lat.decode_from_pred) else None))
            if lat is not None:
                for k, val in lat.eval_metrics(model, out, x).items():
                    lat_acc[k] = lat_acc.get(k, 0.0) + val
        ce = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="sum")
        ce_sum += ce.item()
        byte_sum += data.target_bytes(y).sum().item()
        if out.bpred:
            nb += out.bpred[0].boundary_mask.sum().item()
            nc += out.bpred[0].boundary_mask.numel()
    model.train()
    res = {"val/bpb": ce_sum / LN2 / byte_sum}
    if nc:
        res["val/compression_L1/L0"] = nb / nc
    nbatches = math.ceil(n / B)
    for k, val in lat_acc.items():
        res[f"val/{k}"] = val / nbatches
    return res


@torch.no_grad()
def chunk_sample(model, data, mc, n_bytes=1200):
    """Render boundaries of the first val document(s) as text with '|' separators."""
    if data.bpe or len(mc["d_model"]) == 1:
        return None
    model.eval()
    x = torch.from_numpy(data.val[: n_bytes].astype(np.int64))[None].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = forward_lm(model, x, pool_k=mc.get("pool_k", 0))
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


# ----------------------------------------------------------------------------- main
def main(argv):
    cfg = load_cfg(argv)
    name, mc, tc = cfg["name"], cfg["model"], cfg["train"]
    torch.manual_seed(tc.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = Data(mc["tokenizer"], cfg.get("data_dir", "data/fwe"))
    mc["vocab_size_padded"] = 50304 if data.bpe else 256
    mc["vocab_size"] = 50257 if data.bpe else 256
    mc["bytes_per_token"] = data.bytes_per_token
    mc["seq_len"] = tc["seq_len"] if not data.bpe else tc["seq_len"]  # in tokens of the model's own vocab
    B, L = tc["batch_size"], tc["seq_len"]
    bytes_per_step = B * L * data.bytes_per_token
    total_steps = tc.get("steps") or int(tc["total_bytes"] / bytes_per_step)
    is_hnet = len(mc["d_model"]) > 1
    pool_k = mc.get("pool_k", 0)

    model = build_model(mc)
    n_params = sum(p.numel() for p in model.parameters())
    groups, mults = param_groups(model, mc, tc.get("wd", 0.1))
    opt = torch.optim.AdamW(
        [{k: g[k] for k in ("params", "weight_decay")} | {"lr": tc["lr"] * g["lr_mult"]} for g in groups],
        lr=tc["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    lat = latent_lib.build(cfg.get("latent"), model, mc) if cfg.get("latent") else None
    if lat is not None:
        lat_params = [p for p in lat.parameters() if p.requires_grad]
        if lat_params:
            opt.add_param_group({"params": lat_params, "weight_decay": tc.get("wd", 0.1), "lr": tc["lr"] * lat.lr_mult})
            groups.append({"lr_mult": lat.lr_mult})

    est_fwd = flops_lib.model_fwd_per_byte(mc)
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["derived"] = dict(
        n_params=n_params, total_steps=total_steps, bytes_per_step=bytes_per_step, lr_multipliers=mults,
        est_train_flops_per_byte=3 * est_fwd, est_total_flops=3 * est_fwd * bytes_per_step * total_steps,
    )
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    print(json.dumps(cfg["derived"], indent=1), flush=True)

    use_wandb = cfg.get("wandb", True)
    if use_wandb:
        import wandb
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, group=cfg.get("group"),
                   tags=cfg.get("tags", []), notes=cfg.get("notes"), config=cfg, dir=str(ROOT / "runs"))
    mlog = open(run_dir / "metrics.jsonl", "w")

    def log(d, step):
        d = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        mlog.write(json.dumps({"step": step, **d}) + "\n")
        mlog.flush()
        if use_wandb:
            wandb.log(d, step=step)

    mb = tc.get("micro_batch", B)
    assert B % mb == 0
    accum = B // mb
    it = data.batches(B, L, seed=tc.get("seed", 0))
    t0 = time.time()
    tlast = t0
    cum_flops = 0.0
    cum_bytes = 0.0
    acc = {}
    best = {}
    for step in range(total_steps):
        lr_scale = wsd(step, total_steps, tc.get("warmup_frac", 0.1), tc.get("decay_frac", 0.2))
        for pg, g in zip(opt.param_groups, groups):
            pg["lr"] = tc["lr"] * g["lr_mult"] * lr_scale
        x_all, y_all = next(it)
        stats = {}
        for a in range(accum):
            x, y = x_all[a * mb:(a + 1) * mb], y_all[a * mb:(a + 1) * mb]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = forward_lm(model, x, pool_k=pool_k, detach_main=bool(lat and lat.detach_main), zero_main=bool(mc.get("zero_main", False)), main_out_fn=(lat.head if (lat and lat.decode_from_pred) else None))
                ce_tok = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="none")
                ce = ce_tok.mean()
                loss = ce
                if is_hnet and not pool_k:
                    rl = ratio_loss(out.bpred[0], mc.get("N", 6))
                    loss = loss + mc.get("alpha", 0.03) * rl
                    stats["ratio_loss"] = stats.get("ratio_loss", 0) + rl.detach() / accum
                if lat is not None:
                    l_lat, lat_stats = lat.loss(model, out, x, step=step, total_steps=total_steps)
                    loss = loss + l_lat
                    for k, v in lat_stats.items():
                        stats[k] = stats.get(k, 0) + v.detach() / accum
            (loss / accum).backward()
            nbytes = data.target_bytes(y).sum()
            stats["ce"] = stats.get("ce", 0) + ce.detach() / accum
            stats["bpb"] = stats.get("bpb", 0) + (ce_tok.detach().sum() / LN2 / nbytes) / accum
            if out.bpred:
                stats["compression_L1/L0"] = stats.get("compression_L1/L0", 0) + out.bpred[0].boundary_mask.float().mean().detach() / accum
                stats["boundary_prob_mean"] = stats.get("boundary_prob_mean", 0) + out.bpred[0].boundary_prob[..., -1].float().mean().detach() / accum
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], tc.get("clip", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if lat is not None:
            lat.after_step(model, step, total_steps)

        comp = stats.get("compression_L1/L0")
        comp = comp.item() if comp is not None and (step % tc.get("log_every", 50) == 0) else (acc.get("last_comp") or 1.0 / mc.get("N", 6))
        acc["last_comp"] = comp
        f_step = 3 * flops_lib.model_fwd_per_byte(mc, compression=comp if is_hnet else None) * bytes_per_step
        cum_flops += f_step
        cum_bytes += bytes_per_step

        if step % tc.get("log_every", 50) == 0 or step == total_steps - 1:
            torch.cuda.synchronize()
            now = time.time()
            dt = (now - tlast) / (tc.get("log_every", 50) if step else 1)
            tlast = now
            d = {f"train/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in stats.items()}
            d.update({"train/grad_norm": gn.item(), "lr": tc["lr"] * lr_scale, "flops": cum_flops, "bytes": cum_bytes,
                      "perf/sec_per_step": dt, "perf/bytes_per_sec": bytes_per_step / dt, "perf/mfu": f_step / dt / A100_BF16,
                      "perf/max_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
            log(d, step)
            if step % (tc.get("log_every", 50) * 10) == 0:
                print(f"step {step}/{total_steps} " + " ".join(f"{k}={v:.4g}" for k, v in d.items() if isinstance(v, float)), flush=True)
            if not math.isfinite(d["train/ce"]):
                print("non-finite loss; abort", flush=True)
                break
        if (step > 0 and step % tc.get("eval_every", 1000) == 0) or step == total_steps - 1:
            ev = evaluate(model, data, mc, tc, lat=lat, max_windows=tc.get("eval_windows"))
            ev["flops"] = cum_flops
            log(ev, step)
            print(f"[eval] step {step} " + " ".join(f"{k}={v:.5g}" for k, v in ev.items()), flush=True)
            best = ev

    # ---- final: full val, chunk sample, summary, checkpoint
    final = evaluate(model, data, mc, tc, lat=lat)
    sample = chunk_sample(model, data, mc)
    summary = {"name": name, "group": cfg.get("group"), **{k: v for k, v in final.items()},
               "train_flops": cum_flops, "train_bytes": cum_bytes, "n_params": n_params,
               "wall_min": (time.time() - t0) / 60, "steps": total_steps}
    print("[final]", json.dumps(summary), flush=True)
    json.dump(summary, open(run_dir / "summary.json", "w"), indent=1)
    if sample:
        open(run_dir / "chunks.txt", "w").write(sample)
    torch.save({"model": model.state_dict(), "cfg": cfg, **({"latent": lat.state_dict()} if lat is not None else {})}, run_dir / "final.pt")
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
                                  commit_message=f"run {name}: val bpb {final['val/bpb']:.4f}")
            print("uploaded to HF", flush=True)
        except Exception as e:  # never lose a run because of an upload failure
            print("HF upload failed:", e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
