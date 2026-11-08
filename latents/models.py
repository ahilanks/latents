"""Model wrappers around the official H-Net code (goombalab/hnet, pinned in setup.sh).

All models are `HNetForCausalLM` instances (an isotropic model is a 0-stage H-Net with layout ["T8"]).
`forward_lm` re-implements HNet.forward for 0/1-stage models so intermediate quantities (chunk
representations, main-network outputs) are exposed for the latent/JEPA objectives; it is checked
against the official forward in tests/test_equiv.py.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
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


def fixed_routing(B: int, L: int, k: int, device) -> RoutingModuleOutput:
    """H-Net (pool) baseline: a boundary every k bytes (and at the start of each row)."""
    pos = torch.arange(L, device=device).repeat(B)
    mask = pos % k == 0
    p = mask.float()
    prob = torch.stack((1 - p, p), dim=-1)
    sel = torch.ones(B * L, 1, device=device)
    return RoutingModuleOutput(boundary_prob=prob, boundary_mask=mask, selected_probs=sel)


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


def forward_lm(m: HNetForCausalLM, input_ids: torch.Tensor, pool_k: int = 0, detach_main: bool = False,
               zero_main: bool = False, main_out_fn=None) -> LMOut:
    """detach_main: stop the byte-level AR gradient from reaching the main network (S3: main network is
    trained only by latent objectives; the decoder still reads its (detached) outputs).
    zero_main: control - the decoder gets zeros instead of the main-network output (no chunk-level context).
    main_out_fn: maps the full-width main output (T1, D1) -> (T1, D0) for the decoder (e.g. the latent head, so the
    decoder reads the *predicted next-chunk latent*); default = official truncation z_full[..., :D0]."""
    B, L = input_ids.shape
    h = m.embeddings(input_ids).flatten(0, 1)
    cu = torch.arange(B + 1, device=h.device, dtype=torch.int32) * L
    bb = m.backbone
    if bb.is_innermost:  # isotropic
        h, _ = bb(h, cu_seqlens=cu, max_seqlen=L)
        return LMOut(logits=m.lm_head(h).view(B, L, -1), bpred=[])

    h = bb.encoder(h, cu_seqlens=cu, max_seqlen=L)
    enc = h
    residual = bb.residual_proj(h.to(bb.residual_proj.weight.dtype))
    if pool_k:
        bp = fixed_routing(B, L, pool_k, h.device)
    else:
        bp = bb.routing_module(h, cu_seqlens=cu)
    hc, cu1, max1, _ = bb.chunk_layer(h, bp.boundary_mask, cu)
    # == bb.main_network(hc, ...) (innermost HNet: pad -> Isotropic -> truncate), keeping the full-width output
    inner = bb.main_network
    hp = hc if inner.pad_dimension is None else torch.cat((hc, inner.pad_dimension.expand(hc.shape[0], -1)), dim=-1)
    z_full = inner.main_network(hp, cu_seqlens=cu1, max_seqlen=max1)
    z = z_full[..., : hc.shape[-1]]
    zsrc = z if main_out_fn is None else main_out_fn(z_full).to(z.dtype)
    zd = torch.zeros_like(zsrc) if zero_main else (zsrc.detach() if detach_main else zsrc)
    out = bb.dechunk_layer(zd, bp.boundary_mask, bp.boundary_prob, cu1)
    h = bb.residual_func(out.to(residual.dtype), residual, bp.selected_probs).to(out.dtype)
    h = bb.decoder(h, cu_seqlens=cu, max_seqlen=L)
    return LMOut(logits=m.lm_head(h).view(B, L, -1), bpred=[bp], chunks=hc, z=z, cu1=cu1, enc=enc,
                 z_full=z_full, max1=max1)


def param_groups(m: HNetForCausalLM, mc: dict, wd: float):
    """AdamW groups: per-stage LR multiplier (official apply_lr_multiplier) x {decay, no-decay}."""
    mults = lr_multipliers(mc)
    m.apply_lr_multiplier(mults)
    groups = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        lr_mult = p._optim.get("lr_multiplier", 1.0)
        no_decay = p.ndim < 2 or "embeddings" in name
        key = (lr_mult, no_decay)
        groups.setdefault(key, {"params": [], "lr_mult": lr_mult, "weight_decay": 0.0 if no_decay else wd, "names": []})
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)
    return list(groups.values()), mults
