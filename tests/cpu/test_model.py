"""Model-level checks: equivalence with the official forward, no-op extras, gradient wiring, causality, rollout."""
import copy

import pytest
import torch
import torch.nn.functional as F

import tests.cpu_stubs as S
from latents.models import build_model, forward_lm, model_forward
from latents.latent import LatentObjective, next_pairs
from latents.rollout import Rollout

torch.set_num_threads(4)


def make(mc_over=None, lcfg=None, seed=0):
    torch.manual_seed(seed)
    mc = S.tiny_mc(**(mc_over or {}))
    if lcfg:
        mc["_latent_head"] = lcfg.get("head", "linear")
    m = build_model(mc, device="cpu")
    lat = LatentObjective(lcfg, m, mc) if lcfg else None
    return m, mc, lat


def data(B=2, L=96, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(32, 127, (B, L), generator=g)


EMA = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp")
NEST2 = dict(levels=2, enc="m1", pred="T2", router="learned", ratio=2, topdown=True, input_grad=True, tie="none")


def test_forward_matches_official():
    m, mc, _ = make()
    x = data()
    with torch.no_grad():
        r = m(x, mask=torch.ones_like(x, dtype=torch.bool))
        out = forward_lm(m, x)
    assert torch.allclose(r.logits.float(), out.logits.float(), atol=1e-4, rtol=1e-4)
    assert r.bpred_output[0].boundary_mask.flatten().equal(out.bpred[0].boundary_mask)


def test_extras_are_noops_at_init():
    base, mc, _ = make()
    x = data()
    ref = forward_lm(base, x).logits
    for over in (dict(dec_byte_emb=True), dict(nest=dict(NEST2))):
        m, mc2, _ = make(over)
        m.load_state_dict(base.state_dict(), strict=False)
        out = model_forward(m, x, mc2)
        assert torch.allclose(out.logits, ref, atol=1e-5), over


def ce_grads(m, mc, lat, x):
    m.zero_grad(set_to_none=True)
    out = model_forward(m, x, mc, lat)
    F.cross_entropy(out.logits[:, :-1].flatten(0, 1), x[:, 1:].flatten()).backward()
    g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)  # noqa: E731
    return g(m.backbone.encoder), g(m.backbone.main_network), g(m.backbone.decoder), g(m.backbone.routing_module)


@pytest.mark.parametrize("over,lcfg,enc0,main0", [
    ({}, dict(EMA), False, False),
    ({"dec_byte_emb": True}, dict(EMA, decode_from_pred=True, detach_main=True), False, True),
    ({"dec_byte_emb": True, "pool_k": 4}, dict(EMA, decode_from_pred=True, detach_main=True, detach_trunk=True), True, True),
    ({"dec_byte_emb": True, "router": "identity", "N": 4}, dict(EMA, decode_from_pred=True, detach_trunk=True), True, True),
    ({"dec_byte_emb": True, "N": 4}, dict(EMA, decode_from_pred=True, detach_trunk=True, router_grad="ste"), True, True),
    ({"dec_byte_emb": True, "floor": True}, None, True, True),
])
def test_byte_loss_gradient_wiring(over, lcfg, enc0, main0):
    m, mc, lat = make(over, lcfg)
    g_enc, g_main, g_dec, g_rt = ce_grads(m, mc, lat, data())
    assert (g_enc == 0) == enc0, g_enc
    assert (g_main == 0) == main0, g_main
    assert g_dec > 0
    if lcfg and lcfg.get("detach_trunk"):
        assert g_rt == 0


def test_ste_router_gets_latent_gradient():
    m, mc, lat = make({"dec_byte_emb": True, "N": 4}, dict(EMA, decode_from_pred=True, detach_trunk=True, router_grad="ste"))
    x = data()
    out = model_forward(m, x, mc, lat)
    loss, _ = lat.loss(m, out, x, step=5, total_steps=10)
    loss.backward()
    rm = m.backbone.routing_module
    assert rm.q_proj_layer.weight.grad is not None and rm.q_proj_layer.weight.grad.abs().sum() > 0
    m2, mc2, lat2 = make({"dec_byte_emb": True, "N": 4, "router": "identity"}, dict(EMA, detach_trunk=True))
    assert not m2.backbone.routing_module.q_proj_layer.weight.requires_grad


def causal_ok(m, mc, lat, L=96, k=50, tol=1e-5):
    m.eval()
    x = data(L=L)
    x2 = x.clone()
    x2[:, k:] = torch.randint(32, 127, x2[:, k:].shape, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        a = model_forward(m, x, mc, lat).logits
        b = model_forward(m, x2, mc, lat).logits
    return (a[:, :k] - b[:, :k]).abs().max().item()


@pytest.mark.parametrize("over,lcfg", [
    ({}, None),
    ({"dec_byte_emb": True, "pool_k": 4}, dict(EMA, decode_from_pred=True, detach_trunk=True)),
    ({"N": 4, "nest": dict(NEST2)}, dict(EMA)),
    ({"N": 4, "nest": dict(NEST2, levels=3, tie="all")}, dict(EMA)),
    ({"N": 4, "nest": dict(NEST2, depth_rule="min_len", max_levels=4, min_top_len=4, compute_penalty=0.03)}, dict(EMA)),
])
def test_causality(over, lcfg):
    m, mc, lat = make(over, lcfg)
    assert causal_ok(m, mc, lat) < 1e-5


def test_surprise_router_causal_and_trains():
    m, mc, lat = make({"router": "surprise", "N": 4, "surprise": {"warmup": 0.0}}, dict(EMA))
    x = data()
    m.train()
    out = model_forward(m, x, mc, lat, step_frac=0.5)
    loss, st = lat.loss(m, out, x, step=5, total_steps=10)
    loss.backward()
    assert float(m.surprise.thr) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.surprise.g.parameters())
    assert causal_ok(m, mc, lat) < 1e-5


def test_rollout_teacher_inputs_reproduce_pass1():
    m, mc, _ = make({"arch_layout": ["m1", ["T3"], "m1"]})
    ro = Rollout.for_main(m, k=4)
    x = data()
    with torch.no_grad():
        out = forward_lm(m, x)
        T1 = out.chunks.shape[0]
        layout = ro.layout(out.cu1, T1)
        B, M, lens, row, pos = layout
        kv = ro._pass1_kv(B, M, row, pos)
        for s in (2, 3, 4):
            z = ro.run_pass(s, out.chunks, layout, kv)
            ok = pos >= s - 1
            err = (z[ok] - out.z_full[ok]).abs().max().item()
            assert err < 1e-4, (s, err)


@pytest.mark.parametrize("lcfg,over", [
    (dict(EMA, beta_r=1.0, rollout_k=3, rollout_predictor_only=True), {"arch_layout": ["m1", ["T2"], "m1"]}),
    (dict(EMA, reg="var", reg_gamma="auto"), {"pool_k": 4}),
    (dict(EMA, reg="vicreg", reg_gamma=0.5), {}),
    (dict(EMA, reg="sigreg"), {}),
    (dict(EMA, contrast="cpc"), {}),
    (dict(EMA, router_grad="reinforce"), {"N": 4}),
    (dict(EMA, target="online"), {}),
    (dict(EMA, beta_r=1.0, rollout_k=2), {"N": 4, "arch_layout": ["m1", ["T2"], "m1"],
                                           "nest": dict(NEST2, rollout_k=2, tie="all", levels=3)}),
])
def test_latent_losses_finite_and_backprop(lcfg, over):
    m, mc, lat = make(over, lcfg)
    x = data()
    m.train()
    for step in (0, 5):
        out = model_forward(m, x, mc, lat, step_frac=step / 10)
        loss, st = lat.loss(m, out, x, step=step, total_steps=10)
        assert torch.isfinite(loss), st
        loss.backward()
        lat.after_step(m, step, 10)
    ev = lat.eval_metrics(m, model_forward(m, x, mc, lat), x)
    assert all(v == v for v in ev.values() if isinstance(v, float)), ev


def test_nested_levels_and_targets():
    m, mc, lat = make({"N": 4, "nest": dict(NEST2, levels=3)}, dict(EMA))
    x = data(L=96)
    out = model_forward(m, x, mc, lat)
    levels = sorted(r["level"] for r in out.extras["levels"]["levels"])
    assert levels == [2, 3]
    loss, st = lat.loss(m, out, x)
    assert "L2/tf_l1" in st and "L3/tf_l1" in st
    # passenger: level-1 latent loss sends nothing into the towers
    m2, mc2, lat2 = make({"N": 4, "nest": dict(NEST2, topdown=False, input_grad=False)}, dict(EMA))
    out = model_forward(m2, x, mc2, lat2)
    pa = lat2.head(out.z_full)
    j = next_pairs(out.cu1, pa.shape[0])
    (pa[j] - lat2.targets(m2, out, x)[j + 1]).abs().mean().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m2.nest.parameters())


def test_depth_rule_uses_expected_lengths():
    m, mc, _ = make({"N": 4, "seq_len": 8192, "nest": dict(NEST2, depth_rule="min_len", max_levels=6, min_top_len=128, ratio=3)})
    assert m.nest.top == 4  # 8192/4=2048 -> 683 -> 228 -> 76 (stop)
