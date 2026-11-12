"""Latent thoughts on learned-cut models (latents/think_lc.py): the router-brief variants RT2r / RT4r."""
import random
import sys
from pathlib import Path

import pytest
import torch

import tests.cpu_stubs as S
from latents import tasks
from latents.latent import LatentObjective
from latents.models import build_model
from latents.think_lc import LCThinker, make_thinker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
BASE = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp", dist="l2", decode_from_pred=True, detach_main=True,
            detach_trunk=True, target_pool=True, reg="vicreg", reg_scope="within", reg_gamma=1.0)
# RT2r/RT3r-like, RT4r-like, RT4r + per-byte loss + a second head that predicts the next input (what thoughts are made with)
VARIANTS = [dict(), dict(target_local=True), dict(target_local=True, loss_weight="len", in_head=True)]


def make(extra, seed=0):
    torch.manual_seed(seed)
    mc = S.tiny_mc(seq_len=256, pool_k=0, N=4, dec_byte_emb=True, chunk_pool="mean", alpha=1.0)
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    lat = LatentObjective(dict(BASE, **extra), m, mc)
    return m, lat, mc


def probs(n=3, seed=0):
    rng = random.Random(seed)
    return [tasks.problem(rng, 2 + i % 3) for i in range(n)]


@pytest.mark.parametrize("extra", VARIANTS)
def test_stage_a_loss_wiring(extra):
    m, lat, mc = make(extra)
    th = make_thinker(m, lat, mc)
    assert isinstance(th, LCThinker)
    loss, st = th.loss_pure(probs(), dec_true=1.0, step=5, total_steps=10)
    assert torch.isfinite(loss)
    enc_p = next(m.backbone.encoder.parameters())
    rq = m.backbone.routing_module.q_proj_layer.weight
    # the decoder's answer loss reaches neither the encoder nor the router; the whole loss reaches both
    for prm in (enc_p, rq):
        g = torch.autograd.grad(st["_ce_ans"], prm, retain_graph=True, allow_unused=True)[0]
        assert g is None or g.abs().sum() == 0
        g = torch.autograd.grad(loss, prm, retain_graph=True, allow_unused=True)[0]
        assert g is not None and g.abs().sum() > 0
    assert 0 < float(st["cut_rate"]) < 1 and st["mean_k"] >= 1


@pytest.mark.parametrize("extra", VARIANTS)
def test_padding_never_changes_the_losses(extra):
    vals = []
    for pad in (0, 128):
        m, lat, mc = make(extra)
        m.eval()
        th = make_thinker(m, lat, mc)
        th.pad_extra = pad
        with torch.no_grad():
            l1, s1 = th.loss_pure(probs(), dec_true=1.0, step=5, total_steps=10)
            l2, s2 = th.loss_coconut(probs(), [1, 2, 9], dec_true=1.0, anchor_w=1.0, step=5, total_steps=10)
        vals.append((float(l1), float(l2), float(s1["cut_rate"]), float(s2["mean_thoughts"])))
    assert vals[0] == pytest.approx(vals[1], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("extra", VARIANTS)
def test_practice_stage_counts_and_gradients(extra):
    m, lat, mc = make(extra)
    th = make_thinker(m, lat, mc)
    P = probs()
    f = th.full(P)
    # stage 0: no thoughts; a stage past the last step: one thought per reasoning chunk of the full example
    _, s0 = th.loss_coconut(P, [0, 0, 0], anchor_w=1.0)
    assert s0["mean_thoughts"] == 0 and "anchor_loss" not in s0
    loss, s9 = th.loss_coconut(P, [9, 9, 9], anchor_w=1.0, dec_true=1.0)
    assert s9["mean_thoughts"] == pytest.approx(sum(f.k) / len(P))
    assert torch.isfinite(loss) and torch.isfinite(s9["anchor_loss"])
    g = torch.autograd.grad(loss, next(m.backbone.encoder.parameters()), allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0


@pytest.mark.parametrize("extra", VARIANTS)
def test_eval_path_runs_and_teacher_prefix_is_causal(extra):
    import think_withhold as W
    m, lat, mc = make(extra)
    m.eval()
    th = make_thinker(m, lat, mc)
    P = probs(4, seed=3)
    with torch.no_grad():
        ex = W.prepare(th, P)
        for e in ex:
            assert e["teach"].shape[0] == e["k"] == len(e["bpos"]) and e["qin"].shape[0] >= 1
            assert e["bpos"][-1] == len(e["p"]["cot"]) - 1  # the last reasoning chunk ends at the reasoning's last byte
        own = W.rollout(th, ex, [min(2, e["k"]) for e in ex])
        th_in = [torch.cat((e["teach"][:e["k"] - o.shape[0]], o), 0) for e, o in zip(ex, own)]
        ans = W.answers(th, [e["qb"] for e in ex], th_in)
        assert len(ans) == 4 and all(isinstance(a, str) for a in ans)
        # the first j teacher thoughts do not depend on the reasoning bytes after their last boundary
        P2 = [dict(p) for p in P]
        P2[0] = dict(P2[0], cot=P2[0]["cot"][:-4] + "99.\n")
        ex2 = W.prepare(th, P2)
        j = sum(1 for b in ex[0]["bpos"] if b < len(P[0]["cot"]) - 4)
        j2 = sum(1 for b in ex2[0]["bpos"] if b < len(P[0]["cot"]) - 4)
        assert j == j2 and torch.allclose(ex[0]["teach"][:j], ex2[0]["teach"][:j], atol=1e-5)


def test_second_head_makes_thoughts_in_the_input_space():
    """With in_head, a teacher-forced thought prediction is trained toward the next chunk's input (EMA, pooled), and
    the eval's own thoughts come from that head."""
    import think_withhold as W
    m, lat, mc = make(dict(target_local=True, in_head=True))
    th = make_thinker(m, lat, mc)
    assert th.in_space and th.th_head is lat.in_head
    loss, st = th.loss_pure(probs(), dec_true=1.0, step=5, total_steps=10)
    assert "in_loss" in st and torch.isfinite(st["in_loss"])
    g = torch.autograd.grad(loss, next(lat.in_head.parameters()), allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        ex = W.prepare(th, probs(2, seed=4))
        f = th.full([e["p"] for e in ex])
        tin = lat.input_targets(m, f.out, f.x).float()
        o, qn, k = f.off[0], f.qn[0], f.k[0]
        assert torch.allclose(ex[0]["tgt"], tin[o + qn: o + qn + k], atol=1e-5)
        own = W.rollout(th, ex, [1, 1])
        seq = torch.cat((th.pad(ex[0]["qin"]), th.pad(ex[0]["teach"][:-1])), 0)
        z, _ = th.run_main([seq])
        assert torch.allclose(own[0][0], lat.in_head(z[-1]), atol=1e-5)


def test_windowed_ratio_loss_sees_a_region_without_cuts():
    from hnet.modules.dc import RoutingModuleOutput
    m, lat, mc = make(dict(target_local=True))
    th = make_thinker(m, lat, mc)
    L = 128
    mask = torch.zeros(L, dtype=torch.bool)
    mask[torch.arange(0, 64, 3)] = True            # all the cuts in the first half: the global rate is ~1/6
    p = torch.where(mask, torch.tensor(0.95), torch.tensor(0.02))
    bp = RoutingModuleOutput(boundary_prob=torch.stack((1 - p, p), -1), boundary_mask=mask, selected_probs=p[:, None])
    real = torch.ones(L, dtype=torch.bool)
    g = float(th.ratio(bp, real))
    lat.cfg["ratio_window"] = 32
    w = float(th.ratio(bp, real))
    assert w > g + 0.05  # the empty half costs under the windowed loss
    real[96:] = False    # padding windows are ignored
    assert torch.isfinite(th.ratio(bp, real))


def test_cut_from_keeps_the_pretrained_cuts(tmp_path):
    """latent.cut_from (labelled non-end-to-end comparison): cuts come from a frozen copy of the pretrained run, the
    losses give the live router no gradient, and changing the live router does not move the cuts."""
    import latents.think_lc as TL
    m, lat, mc = make(dict(target_local=True, loss_weight="len", in_head=True))
    run = tmp_path / "runs" / "pre"
    run.mkdir(parents=True)
    torch.save({"model": m.state_dict()}, run / "final.pt")
    lat.cfg["cut_from"] = str(run)
    th = make_thinker(m, lat, mc)
    assert th.cut is not None
    k0 = th.full(probs()).k
    loss, st = th.loss_pure(probs(), dec_true=1.0, step=5, total_steps=10)
    g = torch.autograd.grad(loss, m.backbone.routing_module.q_proj_layer.weight, retain_graph=True, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0
    g = torch.autograd.grad(loss, next(m.backbone.encoder.parameters()), allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0
    with torch.no_grad():
        m.backbone.routing_module.q_proj_layer.weight.mul_(-1.0)  # wreck the live router
        for prm in m.backbone.encoder.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    assert th.full(probs()).k == k0
