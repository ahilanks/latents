"""Model wrappers around the official H-Net code (goombalab/hnet, pinned in setup.sh).

All models are `HNetForCausalLM` instances (an isotropic model is a 0-stage H-Net with layout ["T8"]).
`forward_lm` re-implements HNet.forward for 0/1-stage models so intermediate quantities (chunk
representations, main-network outputs) are exposed for the latent/JEPA objectives; it is checked
against the official forward in tests/test_equiv.py.

Project modules that live on the model (so they are saved with it and seen by the optimizer):
  m.dec_emb   decoder's own byte embedding (model.dec_byte_emb, ladder R1)
  m.surprise  prediction-error router (model.router=surprise, ladder R9)
  m.nest      towers for levels >= 2 (latents/nested.py, ladder phases 4-5)
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.hnet import ste_func
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.modules.dc import RoutingModuleOutput


def build_model(mc: dict, device="cuda") -> HNetForCausalLM:
    cfg = HNetConfig(
        arch_layout=mc["arch_layout"],
        d_model=mc["d_model"],
        d_intermediate=mc["d_intermediate"],
        vocab_size=mc["vocab_size_padded"],
        ssm_cfg=SSMConfig(**mc.get("ssm_cfg", {})),
        attn_cfg=AttnConfig(
            num_heads=mc["num_heads"], rotary_emb_dim=mc["rotary_emb_dim"], window_size=mc["window_size"]
        ),
        tie_embeddings=False,
    )
    with torch.device(device):
        m = HNetForCausalLM(cfg, device=device, dtype=torch.float32)
    m.init_weights(initializer_range=0.02)
    attach_extras(m, mc, device)
    return m


def attach_extras(m, mc: dict, device="cuda"):
    """Add the project's extra modules to a freshly built model (all no-ops at init)."""
    if len(mc["d_model"]) == 1:
        return m
    D0 = mc["d_model"][0]
    if mc.get("dec_byte_emb") or mc.get("floor"):
        m.dec_emb = nn.Embedding(mc["vocab_size_padded"], D0, device=device)
        if mc.get("floor"):  # the decoder alone: initialised like the model's byte embedding
            nn.init.normal_(m.dec_emb.weight, std=1.0)
        else:  # zero init: R1 starts exactly at R0
            nn.init.zeros_(m.dec_emb.weight)
    router = mc.get("router", "learned")
    if router == "identity":  # option D: q, k frozen at their identity init
        rm = m.backbone.routing_module
        rm.q_proj_layer.weight.requires_grad_(False)
        rm.k_proj_layer.weight.requires_grad_(False)
    if router == "surprise":
        from latents.routing import SurpriseRouter
        m.surprise = SurpriseRouter(D0, mc.get("N", 6), **mc.get("surprise", {})).to(device)
    if mc.get("nest", {}).get("levels", 1) > 1:
        from latents.nested import Nest
        m.nest = Nest(m, mc, device)
    return m


def lr_multipliers(mc: dict, n_gpt: float = 4.6) -> list:
    """Per-stage LR multipliers, H-Net paper Appendix C (eq. 11): lambda^s = sqrt(N_GPT * prod_{i>=s} N^i / prod_{i>=0} N^i * D^S / D^s)."""
    D = mc["d_model"]
    if len(D) == 1:
        return [1.0]
    N = [mc.get("N", 6), 1.0]  # 1-stage: N^0, N^S=1
    S = len(D) - 1
    out = []
    for s in range(S + 1):
        num = 1.0
        for i in range(s, S + 1):
            num *= N[i]
        den = 1.0
        for i in range(S + 1):
            den *= N[i]
        out.append((n_gpt * num / den * D[S] / D[s]) ** 0.5)
    return out


def ratio_loss(bp: RoutingModuleOutput, N: float) -> torch.Tensor:
    """Paper eq. 10 (identical to hnet.utils.train.load_balancing_loss)."""
    F_ = bp.boundary_mask.float().mean()
    G_ = bp.boundary_prob[..., -1].float().mean()
    return N / (N - 1) * ((N - 1) * F_ * G_ + (1 - F_) * (1 - G_))


def hard_routing(mask: torch.Tensor) -> RoutingModuleOutput:
    """RoutingModuleOutput for a fixed (non-learned) boundary mask: no smoothing in dechunk, STE factor 1."""
    p = mask.float()
    prob = torch.stack((1 - p, p), dim=-1)
    sel = torch.ones(mask.shape[0], 1, device=mask.device)
    return RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)


def fixed_routing(B: int, L: int, k: int, device) -> RoutingModuleOutput:
    """H-Net (pool) baseline: a boundary every k bytes (and at the start of each row)."""
    pos = torch.arange(L, device=device).repeat(B)
    return hard_routing(pos % k == 0)


def segment_pool(x: torch.Tensor, p: torch.Tensor, mask: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    """p-weighted mean over each segment (router brief, 1A): z_j = sum_{t in S_j} p_t x_t / sum_{t in S_j} p_t, with
    S_j = (t_{j-1}, t_j], the bytes after the previous boundary up to and including boundary j (causal: z_j only
    depends on bytes <= t_j). x (T, D) packed rows, p (T,) boundary probabilities, mask (T,) boundaries (every row
    starts with one), cu (B+1,) row offsets. Returns (M, D), M = mask.sum(), in boundary order (= chunk_layer's
    order). Bytes after a row's last boundary are dropped. Segment id = cumsum(b) - b; two scatter-adds."""
    b = mask.long()
    M = int(b.sum())
    seg = torch.cumsum(b, 0) - b
    lens = (cu[1:] - cu[:-1]).long()
    row = torch.repeat_interleave(torch.arange(lens.numel(), device=x.device), lens)
    segc = seg.clamp(max=M - 1)
    keep = (seg < M) & (row[mask][segc] == row)  # the segment's boundary is in the byte's own row
    w = p.float() * keep
    num = torch.zeros(M, x.shape[-1], device=x.device, dtype=torch.float32).index_add_(0, segc, w[:, None] * x.float())
    den = torch.zeros(M, device=x.device, dtype=torch.float32).index_add_(0, segc, w)
    return (num / den.clamp(min=1e-6)[:, None]).to(x.dtype)


def pool_packed(cu: torch.Tensor, k: int) -> RoutingModuleOutput:
    """Fixed boundaries every k tokens of each packed row (rows of different lengths)."""
    T = int(cu[-1])
    lens = (cu[1:] - cu[:-1]).long()
    row = torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)
    pos = torch.arange(T, device=cu.device) - cu[:-1].long()[row]
    return hard_routing(pos % k == 0)


@dataclass
class LMOut:
    logits: torch.Tensor  # (B, L, V)
    bpred: list  # list[RoutingModuleOutput] (empty for isotropic)
    chunks: torch.Tensor = None  # (T1, D0) encoder outputs at boundaries (main-network inputs)
    z: torch.Tensor = None  # (T1, D0) main-network outputs (truncated to D0, as in the official code)
    cu1: torch.Tensor = None  # cu_seqlens over chunks
    enc: torch.Tensor = None  # (T0, D0) encoder outputs
    z_full: torch.Tensor = None  # (T1, D1) main-network outputs before truncation (predictor features)
    max1: int = None
    cu0: torch.Tensor = None  # cu_seqlens over bytes
    extras: dict = field(default_factory=dict)  # router aux (logp, surprise loss), nested levels, ema states


def forward_lm(m: HNetForCausalLM, input_ids: torch.Tensor, pool_k: int = 0, detach_main: bool = False,
               zero_main: bool = False, main_out_fn=None, opts: dict = None) -> LMOut:
    """detach_main: stop the byte-level AR gradient from reaching the main network (S3: main network is
    trained only by latent objectives; the decoder still reads its (detached) outputs).
    zero_main: control - the decoder gets zeros instead of the main-network output (no chunk-level context).
    main_out_fn: maps the full-width main output (T1, D1) -> (T1, D0) for the decoder (e.g. the latent head, so the
    decoder reads the *predicted next-chunk latent*); default = official truncation z_full[..., :D0].
    opts (all optional):
      detach_trunk  no byte-loss gradient reaches the trunk: residual input, main output and the boundary
                    probabilities used by dechunk/STE are detached (the decoder is a pure readout, ladder R6)
      floor         the decoder sees only its own byte embedding (control: what the decoder can do alone)
      pred_ste      multiply main-net inputs by STE(selected_probs): router gradient from the latent loss (R8)
      chunk_pool    hard (default: the boundary byte's encoder state) | mean (p-weighted segment mean, router brief 1A)
      router_fn     callable(h, cu) -> RoutingModuleOutput replacing the learned router (surprise / sampled)
      topdown       callable(hc, cu1, max1) -> (T1, D1) added to the main-net input (levels >= 2)
    """
    o = opts or {}
    B, L = input_ids.shape
    h = m.embeddings(input_ids).flatten(0, 1)
    cu = torch.arange(B + 1, device=h.device, dtype=torch.int32) * L
    bb = m.backbone
    if bb.is_innermost:  # isotropic
        h, _ = bb(h, cu_seqlens=cu, max_seqlen=L)
        return LMOut(logits=m.lm_head(h).view(B, L, -1), bpred=[], cu0=cu)

    h = bb.encoder(h, cu_seqlens=cu, max_seqlen=L)
    enc = h
    detach_trunk = bool(o.get("detach_trunk"))
    hres = h.detach() if detach_trunk else h
    residual = bb.residual_proj(hres.to(bb.residual_proj.weight.dtype))
    if pool_k:
        bp = fixed_routing(B, L, pool_k, h.device)
    elif o.get("router_fn") is not None:
        bp = o["router_fn"](h, cu)
    else:
        bp = bb.routing_module(h, cu_seqlens=cu)
    hc, cu1, max1, _ = bb.chunk_layer(h, bp.boundary_mask, cu)
    if o.get("chunk_pool") == "mean":  # router brief 1A: every byte's p gets gradient through its segment's weight
        hc = segment_pool(h, bp.boundary_prob[..., -1], bp.boundary_mask, cu).to(hc.dtype)
    if o.get("pred_ste"):
        hc = hc * ste_func(bp.selected_probs[bp.boundary_mask]).to(hc.dtype)
    # == bb.main_network(hc, ...) (innermost HNet: pad -> Isotropic -> truncate), keeping the full-width output
    inner = bb.main_network
    hp = hc if inner.pad_dimension is None else torch.cat((hc, inner.pad_dimension.expand(hc.shape[0], -1)), dim=-1)
    if o.get("topdown") is not None:
        hp = hp + o["topdown"](hc, cu1, max1).to(hp.dtype)
    z_full = inner.main_network(hp, cu_seqlens=cu1, max_seqlen=max1)
    z = z_full[..., : hc.shape[-1]]
    zsrc = z if main_out_fn is None else main_out_fn(z_full).to(z.dtype)
    if zero_main or o.get("floor"):
        zd = torch.zeros_like(zsrc)
    else:
        zd = zsrc.detach() if (detach_main or detach_trunk) else zsrc
    prob, sel = bp.boundary_prob, bp.selected_probs
    if detach_trunk:
        prob, sel = prob.detach(), sel.detach()
    out = bb.dechunk_layer(zd, bp.boundary_mask, prob, cu1)
    hd = bb.residual_func(out.to(residual.dtype), residual, sel).to(out.dtype)
    if o.get("floor"):
        hd = torch.zeros_like(hd)
    if getattr(m, "dec_emb", None) is not None:
        hd = hd + m.dec_emb(input_ids).flatten(0, 1).to(hd.dtype)
    hd = bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
    return LMOut(logits=m.lm_head(hd).view(B, L, -1), bpred=[bp], chunks=hc, z=z, cu1=cu1, enc=enc,
                 z_full=z_full, max1=max1, cu0=cu)


def forward_opts(mc: dict, lat=None) -> dict:
    """Static forward options from the run config (the dynamic ones are added by model_forward)."""
    lc = lat.cfg if lat is not None else {}
    return dict(detach_trunk=bool(lc.get("detach_trunk", False)), floor=bool(mc.get("floor", False)),
                pred_ste=lc.get("router_grad", "none") == "ste", chunk_pool=mc.get("chunk_pool", "hard"))


def model_forward(model, x, mc: dict, lat=None, step_frac: float = 1.0) -> LMOut:
    """The one forward used by training, eval and scripts, so every path sees the same model."""
    opts = forward_opts(mc, lat)
    extras = {}
    is_hnet = len(mc["d_model"]) > 1
    if is_hnet and lat is not None and lat.needs_ema_states(mc):
        extras["ema_h"] = lat.ema_states(model, x)
    if is_hnet and getattr(model, "surprise", None) is not None:
        tgt_h = extras.get("ema_h")
        opts["router_fn"] = lambda h, cu: model.surprise(h, cu, tgt_h, step_frac=step_frac, extras=extras)
    elif is_hnet and lat is not None and lat.cfg.get("router_grad", "none") == "reinforce":
        from latents.routing import sampled_routing
        opts["router_fn"] = lambda h, cu: sampled_routing(model.backbone.routing_module, h, cu, extras)
    nest = getattr(model, "nest", None)
    if is_hnet and nest is not None:
        ctx = nest.new_context()
        extras["levels"] = ctx
        opts["topdown"] = lambda hc, cu1, max1: nest.topdown(hc, cu1, max1, ctx)
    out = forward_lm(model, x, pool_k=mc.get("pool_k", 0),
                     detach_main=bool(lat is not None and lat.detach_main),
                     zero_main=bool(mc.get("zero_main", False)),
                     main_out_fn=(lat.head if (lat is not None and lat.decode_from_pred) else None), opts=opts)
    out.extras.update(extras)
    return out


def _lr_mult(p, default):
    return getattr(p, "_optim", {}).get("lr_multiplier", default)


def param_groups(m: HNetForCausalLM, mc: dict, wd: float):
    """AdamW groups: per-stage LR multiplier (official apply_lr_multiplier) x {decay, no-decay}.
    Project modules: dec_emb / surprise get the stage-0 multiplier, nested towers the stage-1 one."""
    mults = lr_multipliers(mc)
    m.apply_lr_multiplier(mults)
    for name, mod in (("dec_emb", 0), ("surprise", 0), ("nest", -1)):
        if getattr(m, name, None) is not None:
            for p in getattr(m, name).parameters():
                p._optim = {"lr_multiplier": mults[mod]}
    groups = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        lr_mult = _lr_mult(p, 1.0)
        no_decay = p.ndim < 2 or "embeddings" in name or "dec_emb" in name or "level_emb" in name
        key = (lr_mult, no_decay)
        groups.setdefault(key, {"params": [], "lr_mult": lr_mult, "weight_decay": 0.0 if no_decay else wd, "names": []})
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)
    return list(groups.values()), mults
