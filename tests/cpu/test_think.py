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


def make_latent_only(seed=0):
    """Like M1: fixed chunks, decoder reads the predicted latent, no byte gradient into the trunk."""
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=256, pool_k=4, dec_byte_emb=True)
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    lat = LatentObjective(dict(EMA, decode_from_pred=True, detach_trunk=True), m, mc)
    return m, lat, mc


def grads(mod):
    return sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)


def test_pure_loss_decoder_never_reaches_the_latents():
    m, lat, mc = make_latent_only()
    m.train()
    th = Thinker(m, lat, mc)
    rng = random.Random(2)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    loss, st = th.loss_pure(probs, p_true=0.5)
    assert torch.isfinite(loss) and st["mean_k"] > 0
    st["_ce_ans"].backward(retain_graph=True)  # the decoder's answer loss alone
    bb = m.backbone
    assert grads(bb.decoder) > 0
    for name, mod in (("encoder", bb.encoder), ("main net", bb.main_network), ("latent head", lat.head),
                      ("byte embedding", m.embeddings)):
        assert grads(mod) == 0, f"decoder loss reached the {name}"
    m.zero_grad(set_to_none=True)
    lat.zero_grad(set_to_none=True)
    loss.backward()  # the latent terms do train the trunk
    assert grads(lat.head) > 0 and grads(bb.main_network) > 0 and grads(bb.encoder) > 0


def test_teacher_forced_thought_inputs():
    m, lat, mc = make_latent_only(1)
    th = Thinker(m, lat, mc)
    rng = random.Random(3)
    qb = [bytes([BOS]) + tasks.sample(rng, (2, 2))["q"].encode()]
    teach = [torch.randn(5, mc["d_model"][0])]
    with torch.no_grad():
        _, ins, _ = th.thoughts(qb, [5], teacher=teach, p_true=1.0, return_inputs=True)
        assert torch.allclose(ins[0], teach[0])
        pred, ins0, _ = th.thoughts(qb, [5], teacher=teach, p_true=0.0, return_inputs=True)
        assert torch.allclose(ins0[0], pred[0])


def test_teacher_forced_parallel_matches_sequential():
    m, lat, mc = make_latent_only(4)
    m.eval()
    th = Thinker(m, lat, mc)
    rng = random.Random(5)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(2)]
    qb = [bytes([BOS]) + p["q"].encode() for p in probs]
    teach = [torch.randn(4, mc["d_model"][0]), torch.randn(6, mc["d_model"][0])]
    with torch.no_grad():
        a, ia, _ = th.thoughts(qb, [4, 6], teacher=teach, p_true=1.0, return_inputs=True, parallel=True)
        b, ib, _ = th.thoughts(qb, [4, 6], teacher=teach, p_true=1.0, return_inputs=True, parallel=False)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5), (x - y).abs().max()
    for x, y in zip(ia, ib):
        assert torch.allclose(x, y)


def test_decoder_on_true_latents_is_decoder_only():
    m, lat, mc = make_latent_only()
    m.train()
    th = Thinker(m, lat, mc)
    rng = random.Random(6)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    l0, _ = th.loss_pure(probs, p_true=1.0)
    rng = random.Random(6)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    l1, st = th.loss_pure(probs, p_true=1.0, dec_true=1.0)
    assert "answer_ce_true" in st and "lm_ce_true" in st
    (l1 - l0).backward()  # the true-latent decoder terms alone (same batch, same randomness)
    bb = m.backbone
    assert grads(bb.decoder) > 0
    for name, mod in (("encoder", bb.encoder), ("main net", bb.main_network), ("latent head", lat.head),
                      ("byte embedding", m.embeddings)):
        assert grads(mod) < 1e-6, f"true-latent decoder loss reached the {name}"
    # the true source is the next chunk's target latent, within each sequence
    src, tgt = torch.zeros(5, 2), torch.arange(10.).view(5, 2)
    out = Thinker._true_src(src, tgt, torch.tensor([0, 3, 5]))
    assert torch.equal(out, torch.tensor([[2., 3.], [4., 5.], [0., 0.], [8., 9.], [0., 0.]]))



def test_l2_distance_and_sigreg_options():
    m, lat, mc = make_latent_only()
    lat.dist = "l2"
    lat.reg = "sigreg"
    m.train()
    th = Thinker(m, lat, mc)
    rng = random.Random(8)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    loss, st = th.loss_pure(probs, p_true=1.0)
    assert torch.isfinite(loss) and "answer_lat_loss" in st and "thought_l1" in st
    loss.backward()
    assert grads(m.backbone.encoder) > 0
    p, t = torch.tensor([1.0, -2.0]), torch.zeros(2)
    assert torch.equal(lat.dist_fn(p, t), torch.tensor([1.0, 4.0]))


def test_vicreg_corr_option():
    m, lat, mc = make_latent_only()
    lat.reg, lat.beta_cov = "vicreg_corr", 0.1
    lat.reg_gamma = 1.0  # fixed floor, no calibration
    m.train()
    th = Thinker(m, lat, mc)
    rng = random.Random(9)
    probs = [tasks.sample(rng, (2, 3)) for _ in range(3)]
    loss, st = th.loss_pure(probs, p_true=1.0)
    assert torch.isfinite(loss) and "lat_reg_cov" in st and float(st["lat_participation"]) > 0
    loss.backward()
    assert grads(m.backbone.encoder) > 0
