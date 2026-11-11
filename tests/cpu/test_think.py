"""Latent chain-of-thought training (latents/think.py) matches the evaluator (scripts/latent_cot.py) and trains."""
import random
import sys
from pathlib import Path

import torch

import tests.cpu_stubs as S
from latents import tasks
from latents.latent import LatentObjective
from latents.models import build_model
from latents.think import BOS, Thinker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
EMA = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp")


def make(seed=0):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=256)
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    lat = LatentObjective(dict(EMA), m, mc)
    return m, lat, mc


def evaluator(m, lat, mc):
    import latent_cot as LC
    M = object.__new__(LC.Model)
    M.m, M.lat, M.mc, M.cfg, M.device = m, lat, mc, {"model": mc}, "cpu"
    M.ac = dict(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    return M


def test_thinker_matches_evaluator():
    m, lat, mc = make()
    m.eval()
    th, M = Thinker(m, lat, mc), evaluator(m, lat, mc)
    rng = random.Random(0)
    probs = [tasks.problem(rng, 2), tasks.problem(rng, 3)]
    with torch.no_grad():
        qb = [bytes([BOS]) + p["q"].encode() for p in probs]
        ks = [3, 5]
        ours, _ = th.thoughts(qb, ks)
        for b in range(2):
            ref = M.thoughts(qb[b], ks[b])
            assert torch.allclose(ours[b], ref, atol=1e-5), (b, (ours[b] - ref).abs().max())
        ctx = [q + b"#### 42\n" for q in qb]
        lg = th.answer_logits(ctx, [len(q) for q in qb], ours)
        for b in range(2):
            ref = M.logits_with_thoughts(ctx[b], len(qb[b]), ours[b])
            assert torch.allclose(lg[b, :len(ctx[b])], ref, atol=1e-4), (b, (lg[b, :len(ctx[b])] - ref).abs().max())


def test_think_loss_trains_through_the_chain():
    m, lat, mc = make(1)
    m.train()
    th = Thinker(m, lat, mc)
    rng = random.Random(1)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    loss, st = th.loss(probs)
    assert torch.isfinite(loss) and st["mean_k"] > 0
    loss.backward()
    g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)  # noqa: E731
    assert g(lat.head) > 0                           # thoughts are trained
    assert g(m.backbone.main_network) > 0            # through the main network
    assert g(m.backbone.encoder) > 0 and g(m.backbone.decoder) > 0
    # the thought chain is differentiable: a thought's gradient reaches the question's chunk states
    m.zero_grad(set_to_none=True)
    lat.zero_grad(set_to_none=True)
    qb = [bytes([BOS]) + probs[0]["q"].encode()]
    out, _ = th.thoughts(qb, [4])
    out[0][-1].sum().backward()
    assert g(m.backbone.encoder) > 0
