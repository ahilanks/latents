"""Next round (math in pretraining, rollout in pretraining, the anti-collapse diagnostic), end to end on a tiny model."""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import tests.cpu_stubs as S
from latents import prepare_math, tasks
from latents import train as T
from latents.latent import LatentObjective
from latents.models import build_model, model_forward

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
WEB, MATH = "data/_cpu_web", "data/_cpu_math"
LAT = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp", dist="l2", decode_from_pred=True, detach_main=True,
           detach_trunk=True, target_pool=True, reg="vicreg", reg_scope="within", reg_gamma=1.0, beta_cov=1.0,
           target_local=True, target_local_cap=64, loss_weight="len")
ROLL = dict(in_head=True, in_loss=True, beta_r=1.0, rollout_feed="in_head", rollout_predictor_only=False,
            rollout_schedule=[[0.0, 2], [0.5, 3]])


@pytest.fixture(scope="module")
def math_dir():
    rng = random.Random(0)
    web = ROOT / WEB
    web.mkdir(parents=True, exist_ok=True)
    words = "the cat sat on a mat while dogs ran far away from home and back".split()
    docs = [bytes([254]) + " ".join(rng.choice(words) for _ in range(rng.randint(20, 80))).encode() for _ in range(3000)]
    np.frombuffer(b"".join(docs[:2500]), dtype=np.uint8).tofile(web / "train.u8")
    np.frombuffer(b"".join(docs[2500:]), dtype=np.uint8).tofile(web / "val.u8")
    mj = web / "math.jsonl"
    with open(mj, "w") as f:
        for i in range(2000):
            f.write(json.dumps({"text": f"Let x = {i}. Then x^2 = {i * i} and x + 7 = {i + 7}. " * 3}) + "\n")
    meta = prepare_math.write(ROOT / MATH, total=4e5, frac=(0.75, 0.20, 0.05), web=web, math_jsonl=mj,
                              val_bytes=20_000, arith_val_bytes=20_000, block=4096)
    yield meta
    shutil.rmtree(ROOT / WEB, ignore_errors=True)
    shutil.rmtree(ROOT / MATH, ignore_errors=True)


def test_math_mix_hits_the_shares_and_keeps_documents_whole(math_dir):
    d = ROOT / MATH
    for f in ("train.u8", "val.u8", "val_math.u8", "val_arith.u8", "meta.json"):
        assert (d / f).exists(), f
    fr = math_dir["frac"]
    assert abs(fr["web"] - 0.75) < 0.05 and abs(fr["math"] - 0.20) < 0.05 and abs(fr["arith"] - 0.05) < 0.03
    tr = np.fromfile(d / "train.u8", dtype=np.uint8)
    assert tr[0] == 254  # every block starts at a document
    vm = (d / "val_math.u8").read_bytes()
    assert b"x^2" in vm and vm[:1] == bytes([254])
    assert b"####" in (d / "val_arith.u8").read_bytes()
    # held-out math never appears in training: its first document is not in train.u8
    first = vm.split(bytes([254]))[1]
    assert first not in tr.tobytes()


def write_cfg(tmp, name, latent, **over):
    mc = S.tiny_mc(arch_layout=["m1", ["T2"], "m1"], dec_byte_emb=True, pool_k=0, N=4, chunk_pool="mean", alpha=1.0)
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    tc = dict(seq_len=256, batch_size=4, micro_batch=2, steps=6, total_bytes=1e4, lr=1e-3, wd=0.1, eval_every=3,
              eval_windows=2, final_eval_windows=2, extra_val_windows=2, final_extra_val_windows=2, log_every=1, seed=0,
              warmup_frac=0.2, decay_frac=0.2, extra_val={"math": "val_math.u8", "arith": "val_arith.u8"})
    tc.update(over.get("train", {}))
    cfg = dict(name=name, group="cpu-test", data_dir=MATH, wandb=False, upload_hf=False, model=mc, train=tc, latent=latent)
    p = Path(tmp) / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_pretraining_with_math_rollout_and_second_head(tmp_path, math_dir):
    name = "cpu-next-roll"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    T.main([str(write_cfg(tmp_path, name, dict(LAT, **ROLL)))])
    run = ROOT / "runs" / name
    sm = json.load(open(run / "summary.json"))
    for k in ("val/bpb", "val_math/bpb", "val_arith/bpb", "val_math/compression_L1/L0", "val/lat/in_l",
              "val/lat/r2_l1", "val/lat/r3_l1", "val/lat/copy3_l1"):
        assert k in sm and sm[k] == sm[k], k
    assert sm["val/lat/rollout_k"] == 3.0  # the schedule's last horizon at the end
    ks = [r["train/lat/rollout_k"] for r in map(json.loads, open(run / "metrics.jsonl")) if "train/lat/rollout_k" in r]
    assert ks[0] == 2.0 and ks[-1] == 3.0

    # the scripts that read these checkpoints
    r = subprocess.run([sys.executable, "-c", "import tests.cpu_stubs as S; S.install(); import runpy, sys; "
                        "sys.argv=['eval_extra_val.py', 'runs/cpu-next-roll', '--data', '" + MATH + "', '--windows', '2'];"
                        "runpy.run_path('scripts/eval_extra_val.py', run_name='__main__')"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = json.load(open(run / "extra_val.json"))
    assert "val_math/bpb" in ev and "val_arith/edge_prec" in ev
    r = subprocess.run([sys.executable, "-c", "import tests.cpu_stubs as S; S.install(); "
                        "from torch.nn.attention import SDPBackend; import latents.rollout as R; "
                        "R.sdpa_kernel_backend = SDPBackend.MATH; import runpy, sys; "
                        "sys.argv=['reg_diag.py', 'runs/cpu-next-roll', '--n', '16', '--windows', '4', '--batch', '8', "
                        "'--data', '" + MATH + "', '--seq_len', '256'];"
                        "runpy.run_path('scripts/reg_diag.py', run_name='__main__')"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    diag = json.load(open(run / "reg_diag.json"))
    assert set(diag) == {"web", "arith_rows", "arith_batch", "arith_packed"}
    for row in diag.values():
        assert abs(row["in_total"] - row["in_var_part"] - row["in_cov_part"]) < 1e-4
        assert row["reg_grad_encoder"] >= 0 and row["latent_grad_router"] >= 0
    shutil.rmtree(run, ignore_errors=True)


def test_rollout_gradients_flow_end_to_end():
    torch.manual_seed(0)
    mc = S.tiny_mc(seq_len=256, pool_k=0, N=4, dec_byte_emb=True, chunk_pool="mean", alpha=1.0, arch_layout=["m1", ["T2"], "m1"])
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    lat = LatentObjective(dict(LAT, **ROLL), m, mc)
    x = torch.randint(32, 127, (2, 256))
    out = model_forward(m, x, mc, lat)
    lat.rollout.k = 3
    pa, tgt, it = lat.head(out.z_full), lat.targets(m, out, x).float(), lat.input_targets(m, out, x).float()
    lr_, _ = lat.rollout.losses(lat.head, out.chunks, out.cu1, pa, tgt, feed_all=lat.in_head(out.z_full),
                                feed_head=lat.in_head, in_tgt=it, dist=lat.dist_fn)
    assert set(lr_) == {2, 3} and set(lat.rollout.last_in) == {2, 3}
    loss = sum(lr_.values()) + sum(lat.rollout.last_in.values())
    for prm in (m.backbone.routing_module.q_proj_layer.weight, next(m.backbone.encoder.parameters()),
                next(lat.in_head.parameters())):
        g = torch.autograd.grad(loss, prm, retain_graph=True, allow_unused=True)[0]
        assert g is not None and g.abs().sum() > 0


def test_second_head_counted_once_in_the_reasoning_stage():
    """A model pretrained with the second head's loss: the reasoning stage reuses that term instead of adding it again."""
    from latents.think_lc import make_thinker
    torch.manual_seed(0)
    mc = S.tiny_mc(seq_len=256, pool_k=0, N=4, dec_byte_emb=True, chunk_pool="mean", alpha=1.0)
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    rng = random.Random(0)
    probs = [tasks.problem(rng, 3) for _ in range(3)]
    vals = {}
    for in_loss in (False, True):
        torch.manual_seed(1)  # the same head weights in both cases
        lat = LatentObjective(dict(LAT, in_head=True, in_loss=in_loss), m, mc)
        th = make_thinker(m, lat, mc)
        f = th.full(probs)
        _, l_lat, _, _, _, _ = th.base_terms(f, 5, 10)
        vals[in_loss] = (float(l_lat), float(f.l_in))
    assert abs(vals[False][0] - vals[True][0]) < 1e-4 and abs(vals[False][1] - vals[True][1]) < 1e-4
