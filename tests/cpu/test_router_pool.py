"""Router brief 1A / rule 1: segment pooling (latents.models.segment_pool)."""
import pytest
import torch

from latents.models import segment_pool


def _cu(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)


def test_segments_end_at_boundaries():
    # one row, boundaries at 0, 3, 5 -> S = {0}, {1, 2, 3}, {4, 5}; byte 6 (after the last boundary) is dropped
    x = torch.arange(7, dtype=torch.float32)[:, None]
    mask = torch.tensor([1, 0, 0, 1, 0, 1, 0], dtype=torch.bool)
    p = torch.tensor([1.0, 0.2, 0.3, 0.9, 0.1, 0.6, 0.4])
    z = segment_pool(x, p, mask, _cu([7]))
    exp = torch.tensor([0.0, (0.2 * 1 + 0.3 * 2 + 0.9 * 3) / 1.4, (0.1 * 4 + 0.6 * 5) / 0.7])
    assert torch.allclose(z[:, 0], exp, atol=1e-6)


def test_rows_do_not_leak_and_tail_dropped():
    # two rows; row 0's tail bytes (after its last boundary) must not join row 1's first segment
    x = torch.randn(10, 4)
    mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.bool)  # rows [0:5], [5:10]
    p = torch.rand(10) * 0.4
    p[mask] = 0.8
    cu = _cu([5, 5])
    z = segment_pool(x, p, mask, cu)
    assert z.shape[0] == int(mask.sum())
    x2 = x.clone()
    x2[3:5] += 100.0  # row 0's tail
    assert torch.allclose(segment_pool(x2, p, mask, cu), z)


def test_causal():
    # z_j only depends on bytes <= t_j
    torch.manual_seed(0)
    x = torch.randn(12, 3)
    mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.bool)
    p = torch.rand(12)
    cu = _cu([12])
    z = segment_pool(x, p, mask, cu)
    bpos = torch.nonzero(mask).squeeze(1)
    for t in range(12):
        x2 = x.clone()
        x2[t] += 5.0
        z2 = segment_pool(x2, p, mask, cu)
        unchanged = bpos < t
        assert torch.allclose(z2[unchanged], z[unchanged])


def test_hard_probs_equal_selection():
    # p = 0/1 (fixed routing): pooling reduces to hard selection z_j = x_{t_j}
    x = torch.randn(16, 5)
    mask = torch.arange(16) % 4 == 0
    z = segment_pool(x, mask.float(), mask, _cu([16]))
    assert torch.allclose(z, x[mask])


def test_router_gets_gradient():
    x = torch.randn(8, 3)
    mask = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.bool)
    p = torch.full((8,), 0.3, requires_grad=True)
    segment_pool(x, p, mask, _cu([8])).sum().backward()
    assert p.grad[1:6].abs().sum() > 0  # bytes that did not fire still get gradient (gap 1)
    assert p.grad[7] == 0  # after the last boundary: dropped


# ---------------------------------------------------------------- router brief 3B: chunk-local targets
def _router_model(lcfg_extra=None, over=None):
    import tests.cpu_stubs as S
    from latents.latent import LatentObjective
    from latents.models import build_model
    torch.manual_seed(0)
    mc = S.tiny_mc(**dict(dict(pool_k=0, N=4, dec_byte_emb=True), **(over or {})))
    lcfg = dict(target="ema", ema_tau=0.99, ema_tau_end=0.99, head="mlp", dist="l2", decode_from_pred=True,
                detach_main=True, detach_trunk=True, target_pool=True)
    lcfg.update(lcfg_extra or {})
    mc["_latent_head"] = "mlp"
    m = build_model(mc, device="cpu")
    return m, mc, LatentObjective(lcfg, m, mc)


def test_local_states_only_see_their_segment():
    from latents.latent import LatentObjective
    m, mc, lat = _router_model()
    g = torch.Generator().manual_seed(3)
    x = torch.randint(32, 127, (2, 48), generator=g)
    mask = torch.zeros(96, dtype=torch.bool)
    mask[[0, 5, 9, 20, 48, 50, 61, 90]] = True  # rows [0:48], [48:96]
    with torch.no_grad():
        y = LatentObjective.local_states(lat.tgt_emb, lat.tgt_enc, x, mask, cap=0)
        # the state at boundary 20 (segment = bytes 10..20) ignores every byte outside its segment
        x2 = x.clone().flatten()
        outside = torch.ones(96, dtype=torch.bool)
        outside[10:21] = False
        x2[outside] = torch.randint(32, 127, (int(outside.sum()),), generator=g)
        y2 = LatentObjective.local_states(lat.tgt_emb, lat.tgt_enc, x2.view(2, 48), mask, cap=0)
        assert torch.allclose(y[20], y2[20], atol=1e-5)
        # ... and depends on the bytes inside it
        x3 = x.clone().flatten()
        x3[12] = (x3[12] - 32 + 1) % 95 + 32
        y3 = LatentObjective.local_states(lat.tgt_emb, lat.tgt_enc, x3.view(2, 48), mask, cap=0)
        assert not torch.allclose(y[20], y3[20], atol=1e-5)
        # cap = 4: only the last 4 bytes (17..20) of the segment count
        yc = LatentObjective.local_states(lat.tgt_emb, lat.tgt_enc, x, mask, cap=4)
        yc3 = LatentObjective.local_states(lat.tgt_emb, lat.tgt_enc, x3.view(2, 48), mask, cap=4)
        assert torch.allclose(yc[20], yc3[20], atol=1e-5)


@pytest.mark.parametrize("over,lextra", [
    (dict(), dict()),                                                                   # rt0: baseline
    (dict(chunk_pool="mean"), dict()),                                                  # rt1: 1A
    (dict(chunk_pool="mean"), dict(reg="vicreg", reg_scope="batch", reg_gamma=1.0)),    # rt2: 2B
    (dict(chunk_pool="mean"), dict(reg="vicreg", reg_scope="within", reg_gamma=1.0)),   # rt3: 3A
    (dict(chunk_pool="mean"), dict(reg="vicreg", reg_scope="within", reg_gamma=1.0, target_local=True)),  # rt4: 3B
])
def test_router_rungs_wiring(over, lextra):
    """Latent loss: finite, reaches the encoder; the router gets latent gradient only with pooling (gap 1). The byte
    loss reaches neither the encoder nor the router (detach_trunk). Targets carry no gradient."""
    import torch.nn.functional as F
    from latents.models import model_forward
    m, mc, lat = _router_model(lextra, over)
    g = torch.Generator().manual_seed(5)
    x = torch.randint(32, 127, (2, 96), generator=g)
    out = model_forward(m, x, mc, lat)
    assert not lat.targets(m, out, x).requires_grad
    l_lat, st = lat.loss(m, out, x, step=10, total_steps=20)
    assert torch.isfinite(l_lat)
    rm = m.backbone.routing_module
    gq = torch.autograd.grad(l_lat, rm.q_proj_layer.weight, retain_graph=True, allow_unused=True)[0]
    ge = torch.autograd.grad(l_lat, next(m.backbone.encoder.parameters()), retain_graph=True, allow_unused=True)[0]
    assert ge is not None and ge.abs().sum() > 0
    pooled = over.get("chunk_pool") == "mean"
    assert (gq is not None and gq.abs().sum() > 0) == pooled
    ce = F.cross_entropy(out.logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
    for prm in (rm.q_proj_layer.weight, next(m.backbone.encoder.parameters())):
        gb = torch.autograd.grad(ce, prm, retain_graph=True, allow_unused=True)[0]
        assert gb is None or gb.abs().sum() == 0
    if lextra.get("target_local"):
        assert "lat/reg_local" in st


def test_reg_detach_p_keeps_regularizer_off_the_router():
    from latents.models import model_forward
    g = torch.Generator().manual_seed(5)
    x = torch.randint(32, 127, (2, 96), generator=g)
    grads = {}
    for flag in (False, True):
        m, mc, lat = _router_model(dict(reg="vicreg", reg_scope="within", reg_gamma=1.0, reg_detach_p=flag,
                                        beta_tf=0.0), dict(chunk_pool="mean"))
        out = model_forward(m, x, mc, lat)
        l, _ = lat.loss(m, out, x, step=10, total_steps=20)  # beta_tf = 0: only the regularizer
        gq = torch.autograd.grad(l, m.backbone.routing_module.q_proj_layer.weight, retain_graph=True, allow_unused=True)[0]
        ge = torch.autograd.grad(l, next(m.backbone.encoder.parameters()), allow_unused=True)[0]
        grads[flag] = (0.0 if gq is None else float(gq.abs().sum()), float(ge.abs().sum()))
    assert grads[False][0] > 0 and grads[True][0] == 0  # router: only without the detach
    assert grads[True][1] > 0  # the encoder is still regularized


def test_length_weighted_loss():
    """loss_weight=len: each target chunk is weighted by the bytes in its segment; with equal-length segments it
    equals the per-chunk mean."""
    from latents.models import model_forward
    g = torch.Generator().manual_seed(5)
    x = torch.randint(32, 127, (2, 96), generator=g)
    vals = {}
    for lw in ("chunk", "len"):
        m, mc, lat = _router_model(dict(loss_weight=lw), dict(chunk_pool="mean"))
        out = model_forward(m, x, mc, lat)
        vals[lw] = float(lat.loss(m, out, x, step=10, total_steps=20)[0])
    assert vals["chunk"] != pytest.approx(vals["len"], rel=1e-6)  # learned cuts: unequal segments
    for lw in ("chunk", "len"):
        m, mc, lat = _router_model(dict(loss_weight=lw, target_pool=False), dict(pool_k=4))
        out = model_forward(m, x, mc, lat)
        vals[lw] = float(lat.loss(m, out, x, step=10, total_steps=20)[0])
    assert vals["chunk"] == pytest.approx(vals["len"], rel=1e-5)  # fixed 4-byte chunks: identical
