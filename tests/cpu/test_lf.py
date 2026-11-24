"""Latent Feedback's pieces (2026-11-24): NorMuon (latents/normuon.py), their Transformer with the loop (latents/lf.py),
their run on bytes (scripts/lf_train.py), and the H-Net loop with its W trained by NorMuon (model.feedback.optimizer)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import tests.cpu_stubs as S
from latents import train as T
from latents.lf import LFTransformer
from latents.models import build_model, param_groups
from latents.normuon import NorMuon

ROOT = Path(__file__).resolve().parents[2]


def test_normuon_lowers_a_loss_skips_matrices_without_gradient_and_keeps_its_update_size():
    torch.manual_seed(0)
    A, X = torch.randn(16, 8), torch.randn(64, 8)
    Y = X @ A.T
    w = torch.nn.Parameter(torch.zeros(16, 8))
    idle = torch.nn.Parameter(torch.ones(4, 4))
    opt = NorMuon([w, idle], lr=0.05, weight_decay=0.01, momentum_start=0.85, momentum_warmup_steps=10, beta2=0.95)
    first = None
    for _ in range(60):
        loss = (X @ w.T - Y).pow(2).mean()
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.2 * first
    assert torch.equal(idle.data, torch.ones(4, 4)) and not opt.state[idle]


def tiny(feedback=True):
    torch.manual_seed(0)
    return LFTransformer(dim=32, layers=2, heads=2, hidden=48, feedback=feedback)


def test_lf_model_is_their_form_causal_and_trains_w_only_through_the_last_pass():
    m = tiny()
    assert m.fusion.bias is None and abs(float(m.fusion.weight.std()) - 0.02) < 0.01
    assert all(float(p.data.mean()) == 1.0 for p in m.parameters() if p.ndim == 1)
    x = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        p1 = m(x, passes=1)
        e = m.embedding(x)
        from latents.lf import rotary
        cos, sin = rotary(40, 16, m.theta, x.device)
        top = m.stack(e, cos, sin)
        assert torch.allclose(p1, m.head(m.norm(top)), atol=1e-6)
        top2 = m.stack(m.mix(e, top), cos, sin)  # u_t = e_t + W rms(h_{t-1}); the first position keeps e_0
        assert torch.allclose(m(x, passes=2), m.head(m.norm(top2)), atol=1e-5)
        y = x.clone()
        y[:, 30:] = (y[:, 30:] + 1) % 256
        assert torch.allclose(m(x, passes=4)[:, :30], m(y, passes=4)[:, :30], atol=1e-5)
    # with 2 passes the first runs without gradient: the loss reaches W only through the last pass's input
    m.zero_grad()
    m(x, passes=2).float().pow(2).mean().backward()
    assert float(m.fusion.weight.grad.abs().sum()) > 0
    m.zero_grad()
    m(x, passes=1).float().pow(2).mean().backward()
    assert m.fusion.weight.grad is None
    assert len(m.hidden_matrices()) == 2 * 7 + 1 and len(tiny(False).hidden_matrices()) == 2 * 7


def test_lf_run_script_trains_and_logs_health(tmp_path):
    data = Path(tmp_path) / "d"
    data.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(32, 127, 200_000, dtype=np.uint8).tofile(data / "train.u8")
    rng.integers(32, 127, 20_000, dtype=np.uint8).tofile(data / "val.u8")
    for fb, name in ((1, "cpu-lf-fb"), (0, "cpu-lf-plain")):
        try:
            r = subprocess.run([sys.executable, "scripts/lf_train.py", "--name", name, "--feedback", str(fb),
                                "--data", str(data), "--seq", "64", "--batch", "4", "--micro", "2", "--dim", "32",
                                "--layers", "2", "--heads", "2", "--hidden", "48", "--steps", "6", "--warmup", "2",
                                "--eval_every", "3", "--eval_windows", "4", "--momentum_warmup", "3", "--wandb", "0"],
                               cwd=ROOT, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            rows = [json.loads(l) for l in open(ROOT / "runs" / name / "metrics.jsonl")]
            ev = [r for r in rows if "val/bpb_p1" in r][-1]
            keys = ["val/top_const", "val/out_variation", "val/out_rank"]
            keys += ["val/fb_share", "val/bpb_p4", "val/loop_gain_bpb"] if fb else []
            for k in keys:
                assert k in ev and np.isfinite(ev[k]), k
            assert {r["train/passes"] for r in rows if "train/passes" in r} <= ({1, 2} if fb else {1})
        finally:
            shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)


def test_hnet_loop_with_w_on_normuon_trains_w_and_keeps_it_out_of_adamw(tmp_path):
    fb = dict(passes=2, plain_frac=0.15, norm="rms", bias=False, affine=False, init_std=0.02, detach=True,
              optimizer=dict(name="normuon", lr=0.03, wd=0.01, beta2=0.987, momentum_warmup=4))
    mc = S.tiny_mc(feedback=fb, holes=dict(mark=3))
    m = build_model(mc, device="cpu")
    groups, _ = param_groups(m, mc, 0.1)
    assert not any(n.startswith("feedback.") for g in groups for n in g["names"])
    name, data = "cpu-lf-hnet", ROOT / "data/_cpu_lf"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rng.integers(32, 127, 300_000, dtype=np.uint8).tofile(data / "train.u8")
    rng.integers(32, 127, 30_000, dtype=np.uint8).tofile(data / "val.u8")
    for k in ("vocab_size_padded", "vocab_size", "seq_len", "bytes_per_token"):
        mc.pop(k)
    tc = dict(seq_len=128, batch_size=4, micro_batch=2, steps=6, total_bytes=1e4, lr=1e-3, wd=0.1, eval_every=3,
              eval_windows=4, final_eval_windows=4, log_every=1, seed=0, warmup_frac=0.2, decay_frac=0.2,
              holes=dict(rate=0.1, max_run=3), hole_eval_windows=4)
    p = Path(tmp_path) / "c.yaml"
    p.write_text(yaml.safe_dump(dict(name=name, group="cpu-test", data_dir="data/_cpu_lf", wandb=False,
                                     upload_hf=False, model=mc, train=tc)))
    try:
        T.main([str(p)])
        rows = [json.loads(l) for l in open(ROOT / "runs" / name / "metrics.jsonl")]
        w = [r["val/fb_weight"] for r in rows if "val/fb_weight" in r]
        assert len(w) >= 2 and w[-1] != w[0]  # W moves under NorMuon
        ck = torch.load(ROOT / "runs" / name / "final.pt", weights_only=False, map_location="cpu")
        assert ck["model"]["feedback.net.weight"].shape == m.feedback.net.weight.shape
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(data, ignore_errors=True)
