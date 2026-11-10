"""GPU preflight: catch crashes, OOMs and wrong gradient wiring in minutes, before any real run.

usage: python scripts/preflight.py [configs/ladder/r6_detach_trunk.yaml ...]   (default: every ladder config)
       python scripts/run_jobs.py scripts/ladder/preflight.txt                  (configs spread over the GPUs)

Per config:
  1. one full-size micro-batch forward/backward (real seq_len and micro_batch -> memory), all losses finite
  2. gradient wiring: e.g. with detach_trunk the byte loss must not reach the encoder, with detach_main not the
     main net; frozen routers get no gradient; a passenger level gets nothing from level 1; and so on
  3. a 4-step run of latents.train with 2 evals, summary + checkpoint, and a strict checkpoint reload
Phase-winner configs (p2_best / p3_best / p4_best) that do not exist yet are stubbed with a representative rung.
Prints OK / FAIL per config and exits 1 if anything failed.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents import train as T  # noqa: E402
from latents.models import build_model, model_forward  # noqa: E402
from latents import latent as latent_lib  # noqa: E402

STUBS = {"p2_best": "ladder/r8_ste_router.yaml", "p3_best": "ladder/r11a_rollout4.yaml", "p4_best": "ladder/r15_level2_rollout.yaml"}


def ensure_stubs():
    made = []
    for name, base in STUBS.items():
        p = ROOT / "configs" / "ladder" / f"{name}.yaml"
        if not p.exists():
            p.write_text(f"# preflight stub (scripts/preflight.py), replaced by scripts/ladder.py select\nbase: {base}\n")
            made.append(p)
    return made


def grads_of(mod):
    if mod is None:
        return 0.0
    return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)


def wiring(cfg_path):
    cfg = T.load_cfg([str(cfg_path)])
    data = T.Data(cfg["model"]["tokenizer"], cfg.get("data_dir", "data/fwe"))
    mc, tc = T.prepare(cfg, data)
    model = build_model(mc, device=T.DEV)
    T.param_groups(model, mc, 0.1)
    lat = latent_lib.build(cfg.get("latent"), model, mc) if cfg.get("latent") else None
    mb, L = tc.get("micro_batch", tc["batch_size"]), tc["seq_len"]
    it = data.batches(mb, L, seed=0)
    x, y = next(it)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    msgs, ok = [], True

    def fwd():
        with T.autocast():
            return model_forward(model, x, mc, lat, step_frac=0.5)

    def zero():
        model.zero_grad(set_to_none=True)
        if lat is not None:
            lat.zero_grad(set_to_none=True)

    # (a) everything: finite + memory
    out = fwd()
    with T.autocast():
        ce = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten())
        l_rt, _ = T.routing_losses(model, out, mc)
        l_lat, st = lat.loss(model, out, x, step=10, total_steps=100) if lat is not None else (0.0, {})
        loss = ce + l_rt + l_lat
    if not torch.isfinite(torch.as_tensor(loss)).all():
        ok = False
        msgs.append(f"non-finite loss {float(loss)}")
    loss.backward()
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    msgs.append(f"mem {mem:.1f}GB, fwd+bwd {time.time() - t0:.1f}s, ce {float(ce):.3f}")
    if mem > 75:
        ok = False
        msgs.append("memory too close to 80GB: lower train.micro_batch")
    bb = model.backbone
    lc = (cfg.get("latent") or {})
    # (b) byte loss alone. residual_proj starts at zero (official H-Net init), which closes the residual path into
    # the encoder at init: with fixed chunks (R4/R5) the encoder would get no byte gradient yet, and the
    # detach_trunk check (R6) would pass even without the detach. Open it, as after the first optimizer step.
    rp = bb.residual_proj.weight
    rp0 = rp.detach().clone()
    if rp0.abs().sum() == 0:
        with torch.no_grad():
            rp.normal_(std=0.02)
    zero()
    out = fwd()
    with T.autocast():
        ce = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten())
    ce.backward()
    g_enc, g_main = grads_of(bb.encoder), grads_of(bb.main_network)
    g_head = grads_of(lat.head) if lat is not None else 0.0
    g_dec = grads_of(bb.decoder)
    expect = []
    if lc.get("detach_trunk") or mc.get("floor"):
        expect += [("encoder", g_enc, 0), ("main net", g_main, 0), ("latent head", g_head, 0)]
    elif lc.get("detach_main"):
        expect += [("main net", g_main, 0), ("latent head", g_head, 0), ("encoder", g_enc, 1)]
    else:
        expect += [("encoder", g_enc, 1)]
    expect.append(("decoder", g_dec, 1))
    if getattr(model, "nest", None) is not None and lc.get("detach_trunk"):
        expect.append(("towers", grads_of(model.nest), 0))
    for name, g, want in expect:
        if (g > 0) != bool(want):
            ok = False
            msgs.append(f"byte-loss grad into {name} = {g:.3g}, expected {'>0' if want else '0'}")
    with torch.no_grad():
        rp.copy_(rp0)
    # (c) latent loss alone
    if lat is not None:
        zero()
        out = fwd()
        with T.autocast():
            l_lat, st = lat.loss(model, out, x, step=10, total_steps=100)
        l_lat.backward()
        rm = bb.routing_module
        g_router = sum(float(p.grad.abs().sum()) for p in (rm.q_proj_layer.weight, rm.k_proj_layer.weight) if p.grad is not None)
        if lc.get("router_grad") == "ste" and not mc.get("pool_k") and g_router == 0:
            ok = False
            msgs.append("STE router got no gradient from the latent loss")
        if grads_of(lat.head) == 0:
            ok = False
            msgs.append("latent head got no gradient from the latent loss")
        nest = getattr(model, "nest", None)
        if nest is not None:
            nc = mc["nest"]
            levels = out.extras["levels"]["levels"]
            msgs.append(f"levels built: {sorted(r['level'] for r in levels)}; lens " +
                        ", ".join(f"L{r['level']}={r['max']}" for r in sorted(levels, key=lambda r: r['level'])))
            if not nc.get("topdown", True):
                # passenger: the level-1 latent loss must not reach the towers
                zero()
                out = fwd()
                with T.autocast():
                    pa = lat.head(out.z_full)
                    j = latent_lib.next_pairs(out.cu1, pa.shape[0])
                    tg = lat.targets(model, out, x).float()
                    (pa[j].float() - tg[j + 1]).abs().mean().backward()
                if grads_of(nest) > 0:
                    ok = False
                    msgs.append("passenger level received gradient from level 1")
        for k, v in st.items():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                ok = False
                msgs.append(f"{k} not finite")
    del model, lat
    torch.cuda.empty_cache()
    return ok, "; ".join(msgs)


def short_run(cfg_path):
    name = "preflight-" + cfg_path.stem
    cmd = [sys.executable, "-m", "latents.train", str(cfg_path), f"name={name}", "train.steps=4", "train.eval_every=2",
           "train.eval_windows=2", "train.final_eval_windows=4", "wandb=false", "upload_hf=false"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    run = ROOT / "runs" / name
    if r.returncode != 0 or not (run / "summary.json").exists():
        return False, "train failed: " + " | ".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    sm = json.load(open(run / "summary.json"))
    ck = torch.load(run / "final.pt", map_location="cuda", weights_only=False)
    m = build_model(ck["cfg"]["model"], device="cuda")
    m.load_state_dict(ck["model"], strict=True)
    if ck["cfg"].get("latent"):
        lat = latent_lib.build(ck["cfg"]["latent"], m, ck["cfg"]["model"])
        lat.load_state_dict(ck["latent"], strict=True)
    return True, f"4-step run ok (val bpb {sm.get('val/bpb', float('nan')):.3f}, flops {sm['train_flops']:.2e}), checkpoint reloads"


def remove_stubs():
    for name in STUBS:
        p = ROOT / "configs" / "ladder" / f"{name}.yaml"
        if p.exists() and p.read_text().startswith("# preflight stub"):
            p.unlink()


def main():
    args = sys.argv[1:]
    if "--stubs-only" in args:
        ensure_stubs()
        return
    if "--remove-stubs" in args:
        remove_stubs()
        return
    keep = "--keep-stubs" in args
    args = [a for a in args if not a.startswith("--") and "=" not in a]
    stubs = [] if keep else ensure_stubs()
    paths = [Path(p) for p in args] or sorted(p for p in (ROOT / "configs" / "ladder").glob("*.yaml"))
    paths = [p if p.is_absolute() else ROOT / p for p in paths]
    bad = 0
    try:
        for p in paths:
            try:
                ok1, m1 = wiring(p)
                ok2, m2 = short_run(p) if ok1 else (False, "skipped")
            except Exception as e:  # report and continue with the next config
                ok1, m1, ok2, m2 = False, f"{type(e).__name__}: {e}", False, ""
            ok = ok1 and ok2
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {p.relative_to(ROOT)}: {m1} | {m2}", flush=True)
    finally:
        for s in stubs:
            s.unlink()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
