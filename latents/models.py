"""Model wrappers around the official H-Net code (goombalab/hnet, pinned in setup.sh).

All models are `HNetForCausalLM` instances (an isotropic model is a 0-stage H-Net with layout ["T8"]).
`forward_lm` re-implements HNet.forward for 0/1-stage models so intermediate quantities (chunk
representations, main-network outputs) are exposed for the latent/JEPA objectives; it is checked
against the official forward in tests/test_equiv.py.

Project modules that live on the model (so they are saved with it and seen by the optimizer):
  m.dec_emb   decoder's own byte embedding (model.dec_byte_emb, ladder R1)
  m.surprise  prediction-error router (model.router=surprise, ladder R9)
  m.nest      towers for levels >= 2 (latents/nested.py, ladder phases 4-5)
  m.nextlat   latent transition model for the next-latent loss (model.nextlat, latents/nextlat.py)
  m.feedback  hidden-state feedback (model.feedback): the main network's last-layer output at one chunk is added to
              its input at the next chunk (see main_pass)

Holes (model.holes = {mark: 3}, latents/holes.py): every byte equal to `mark` is a hole. Boundaries are forced at the
marker and at the byte after it, so a hole is exactly one chunk, and that chunk's encoder vector is replaced by zeros:
the main network's input at a hole is nothing but the fed-back output of the chunk before (plus the constant padding
every chunk input has). Forced boundaries are left out of the ratio loss (out.extras["forced"]).
"""
import contextlib
from contextlib import contextmanager
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
    if mc.get("nextlat"):
        from latents.nextlat import NextLat
        m.nextlat = NextLat(mc["d_model"][-1], D0, **mc["nextlat"]).to(device)
    if mc.get("feedback"):
        attach_feedback(m, mc, device)
    return m


def attach_feedback(m, mc, device="cuda"):
    """Add the feedback module described by mc["feedback"] to a model (also to a loaded plain model: it starts at zero)."""
    m.feedback = Feedback(mc["d_model"][-1], **{k: v for k, v in mc["feedback"].items()
                                                if k not in ("passes", "mode", "plain_frac", "center", "lr_mult", "detach")}).to(device)
    m._feedback_cfg = dict(mc["feedback"])
    return m


class Feedback(nn.Module):
    """Maps the main network's last-layer output at chunk t to an addition to its input at chunk t+1. The output
    layer starts at zero, so a model with feedback starts exactly as the same model without it.
    norm="layer", bias=True: LayerNorm then Linear (the original module; both have a learned term that is the same
    for every chunk). norm="rms", bias=False: W rms(z), the form of Latent Feedback (arXiv 2611.20404, eq. 1, and its
    code ifm-ai/LoRM lorm/model/feedback.py: F.rms_norm without a gain = affine False, W ~ N(0, 0.02) = init_std
    0.02), with nothing that can add a constant. init_std 0: the output layer starts at zero."""

    def __init__(self, D1, hidden=0, norm="layer", bias=True, affine=True, init_std=0.0):
        super().__init__()
        affine = bool(affine)
        self.norm = (nn.LayerNorm(D1, elementwise_affine=affine) if norm == "layer"
                     else nn.RMSNorm(D1, eps=1e-5, elementwise_affine=affine))
        bias = bool(bias)
        if hidden:
            self.net = nn.Sequential(nn.Linear(D1, int(hidden), bias=bias), nn.GELU(), nn.Linear(int(hidden), D1, bias=bias))
            last = self.net[-1]
        else:
            self.net = nn.Linear(D1, D1, bias=bias)
            last = self.net
        if float(init_std) > 0:
            nn.init.normal_(last.weight, std=float(init_std))
        else:
            nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, z):
        return self.net(self.norm(z.float()))


LOOP_MAX = 160  # longest row (in chunks) the exact loop accepts: its cost grows with the square of the length


@contextmanager
def feedback_mode(m, rounds=None, mode=None):
    """Temporarily run the feedback another way: rounds=k parallel rounds (1 = the loop switched off: one plain pass,
    nothing fed back), mode="loop" the exact loop. No-op for a model without feedback."""
    if getattr(m, "feedback", None) is None or (rounds is None and mode is None):
        yield
        return
    old = getattr(m, "_feedback_cfg", None)
    new = dict(old or {})
    if mode == "loop":
        new["mode"] = "loop"
    else:
        new["mode"], new["passes"] = "passes", int(rounds)
    m._feedback_cfg = new
    try:
        yield
    finally:
        m._feedback_cfg = old


def main_pass(m, hp, cu1, max1, valid=None):
    """The main network on packed chunk inputs hp (T1, D1). Hidden-state feedback (model.feedback):
      {mode: loop}   the last-layer output at chunk t is added (through m.feedback) to the bottom-layer input at chunk
                     t+1 of the same row: a real loop in one network, computed chunk by chunk (exact). The cost grows
                     with the square of the row length, so it is for short rows (one problem per row).
                     valid: per row, how many leading chunks matter (the rest, padding, get zeros).
      {passes: k}    an approximation with k parallel passes: pass i+1 gets pass i's outputs, shifted by one chunk.
                     Exact for a row's first k chunks; with k = the row length it equals the loop.
      center: true   the fed-back vector minus its average over the row's chunks so far (causal), so the loop can never
                     act as a constant offset (2026-11-24: without it the loop became a large near-constant addition
                     and the main network's outputs went flat, in HL-A-s0 and in the old two-pass run FB-h-s0)
      lr_mult: f     the feedback layer learns f times as fast as the rest of the main network
      detach: true   as in Latent Feedback's code: every pass but the last runs without gradient and the fed-back state
                     is detached, so the gradient reaches the feedback layer only through the last pass. A blocked
                     path: NOT end to end (the labelled comparison).
    Every variant sees only earlier chunks (causal), lets the gradient through the whole chain, and computes the same
    thing at inference."""
    inner = m.backbone.main_network.main_network
    fb = getattr(m, "feedback", None)
    if fb is None:
        return inner(hp, cu_seqlens=cu1, max_seqlen=max1)
    cfg = getattr(m, "_feedback_cfg", None) or {}
    if cfg.get("mode", "passes") == "loop":
        return loop_pass(m, hp, cu1, valid)
    first = torch.zeros(hp.shape[0], dtype=torch.bool, device=hp.device)
    first[cu1[:-1].long()] = True
    center = bool(cfg.get("center"))
    detach = bool(cfg.get("detach"))
    n = int(cfg.get("passes", 2))
    with torch.no_grad() if (detach and n > 1) else contextlib.nullcontext():
        z = inner(hp, cu_seqlens=cu1, max_seqlen=max1)

    def fed(zz):
        g = fb(zz)
        if center:
            g = g - causal_row_mean(g, cu1)
        add = torch.roll(g, 1, 0).to(hp.dtype)
        return torch.where(first[:, None], torch.zeros_like(add), add)
    if detach and n > 1:  # Latent Feedback's training: earlier passes without gradient, the fed state detached
        with torch.no_grad():
            for _ in range(n - 2):
                z = inner(hp + fed(z), cu_seqlens=cu1, max_seqlen=max1)
        return inner(hp + fed(z.detach()), cu_seqlens=cu1, max_seqlen=max1)
    for _ in range(n - 1):
        z = inner(hp + fed(z), cu_seqlens=cu1, max_seqlen=max1)
    return z


def causal_row_mean(g, cu1):
    """Per packed row, the mean of g over the row's chunks up to and including each chunk."""
    T = g.shape[0]
    lens = (cu1[1:] - cu1[:-1]).long()
    row = torch.repeat_interleave(torch.arange(lens.numel(), device=g.device), lens)
    start = cu1[:-1].long()[row]
    cs = torch.cat((g.new_zeros(1, g.shape[-1]).double(), g.double().cumsum(0)), 0)
    idx = torch.arange(T, device=g.device)
    return ((cs[idx + 1] - cs[start]) / (idx - start + 1).unsqueeze(-1).double()).to(g.dtype)


def loop_pass(m, hp, cu1, valid=None):
    """Exact hidden-state loop: x_1 = c_1, x_{t+1} = c_{t+1} + feedback(z_t), z_t = main network's last-layer output
    at chunk t given x_1..x_t. Step t runs the main network on every row's first t inputs and keeps the last output
    (earlier outputs cannot change: the network is causal)."""
    inner = m.backbone.main_network.main_network
    fb = m.feedback
    center = bool((getattr(m, "_feedback_cfg", None) or {}).get("center"))
    cu = cu1.long().tolist()
    rows = [hp[cu[b]:cu[b + 1]] for b in range(len(cu) - 1)]
    n = [r.shape[0] if valid is None else min(int(valid[b]), r.shape[0]) for b, r in enumerate(rows)]
    T = max(n, default=0)
    assert T <= LOOP_MAX, f"the exact loop is for short rows (<= {LOOP_MAX} chunks), got {T}; use feedback passes"
    xs, zs = [[] for _ in rows], [[] for _ in rows]
    gsum = [None for _ in rows]  # centred loop: running sum of the fed-back vectors of the row
    for t in range(T):
        act = [b for b in range(len(rows)) if n[b] > t]
        for b in act:
            x = rows[b][t:t + 1]
            if t == 0:
                xs[b].append(x)
                continue
            g = fb(zs[b][-1])
            if center:
                gsum[b] = g.float() if gsum[b] is None else gsum[b] + g.float()
                g = g - (gsum[b] / t).to(g.dtype)
            xs[b].append(x + g.to(x.dtype))
        cu_t = torch.arange(len(act) + 1, device=hp.device, dtype=torch.int32) * (t + 1)
        z = inner(torch.cat([torch.cat(xs[b], 0) for b in act], 0), cu_seqlens=cu_t, max_seqlen=t + 1)
        last = z[cu_t[1:].long() - 1]
        for i, b in enumerate(act):
            zs[b].append(last[i:i + 1])
    out = []
    for b, r in enumerate(rows):
        zb = torch.cat(zs[b], 0) if zs[b] else r.new_zeros(0, r.shape[-1])
        if zb.shape[0] < r.shape[0]:  # padding chunks: never read by a loss
            zb = torch.cat((zb, zb.new_zeros(r.shape[0] - zb.shape[0], zb.shape[-1])), 0)
        out.append(zb.to(hp.dtype))
    return torch.cat(out, 0)


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


def ratio_loss(bp: RoutingModuleOutput, N: float, keep: torch.Tensor = None) -> torch.Tensor:
    """Paper eq. 10 (identical to hnet.utils.train.load_balancing_loss). keep: only these positions count (holes: the
    forced boundaries are left out, so they do not push the learned cuts elsewhere)."""
    if keep is None:
        F_ = bp.boundary_mask.float().mean()
        G_ = bp.boundary_prob[..., -1].float().mean()
    else:
        F_ = bp.boundary_mask[keep].float().mean()
        G_ = bp.boundary_prob[..., -1][keep].float().mean()
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
      holes         marker byte value: markers are holes (forced one-byte chunks with a zero encoder vector; see top)
      z_perm        seed: every chunk's decoder-facing main-network state is replaced by a random other chunk's (swap
                    test: how much the decoder uses the main network)
      row_lens      per row, how many leading bytes are real (the rest is padding): the exact loop skips padding chunks
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
    holes = None
    if o.get("holes") is not None:
        holes = (input_ids == int(o["holes"])).flatten()
        if bool(holes.any()):
            bp, forced = force_cuts(bp, holes, B, L)
        else:
            forced = torch.zeros_like(holes)
    hc, cu1, max1, _ = bb.chunk_layer(h, bp.boundary_mask, cu)
    if holes is not None and bool(holes.any()):  # a hole's main-network input: only what is fed back
        hc = torch.where(holes[bp.boundary_mask][:, None], torch.zeros_like(hc), hc)
    if o.get("chunk_pool") == "mean":  # router brief 1A: every byte's p gets gradient through its segment's weight
        hc = segment_pool(h, bp.boundary_prob[..., -1], bp.boundary_mask, cu).to(hc.dtype)
    if o.get("pred_ste"):
        hc = hc * ste_func(bp.selected_probs[bp.boundary_mask]).to(hc.dtype)
    # == bb.main_network(hc, ...) (innermost HNet: pad -> Isotropic -> truncate), keeping the full-width output
    inner = bb.main_network
    hp = hc if inner.pad_dimension is None else torch.cat((hc, inner.pad_dimension.expand(hc.shape[0], -1)), dim=-1)
    if o.get("topdown") is not None:
        hp = hp + o["topdown"](hc, cu1, max1).to(hp.dtype)
    valid = None
    if o.get("row_lens") is not None:  # chunks that start inside each row's real bytes
        mv = bp.boundary_mask.view(B, L)
        valid = [int(mv[b, : int(n)].sum()) for b, n in enumerate(o["row_lens"])]
    z_full = main_pass(m, hp, cu1, max1, valid=valid)
    z = z_full[..., : hc.shape[-1]]
    zsrc = z if main_out_fn is None else main_out_fn(z_full).to(z.dtype)
    if zero_main or o.get("floor"):
        zd = torch.zeros_like(zsrc)
    else:
        zd = zsrc.detach() if (detach_main or detach_trunk) else zsrc
    if o.get("z_perm") is not None:
        g = torch.Generator(device="cpu").manual_seed(int(o["z_perm"]))
        zd = zd[torch.randperm(zd.shape[0], generator=g).to(zd.device)]
    prob, sel = bp.boundary_prob, bp.selected_probs
    if detach_trunk:
        prob, sel = prob.detach(), sel.detach()
    if not o.get("conf_grad", True):  # ablation: no gradient to p through the confidence scaling (forward unchanged)
        sel = sel.detach()
    if not o.get("smoothing", True):  # ablation: no smoothing between pieces (dechunk with p = 1: plain upsampling)
        prob = torch.ones_like(prob)
    out = bb.dechunk_layer(zd, bp.boundary_mask, prob, cu1)
    hd = bb.residual_func(out.to(residual.dtype), residual, sel).to(out.dtype)
    if o.get("floor"):
        hd = torch.zeros_like(hd)
    if getattr(m, "dec_emb", None) is not None:
        hd = hd + m.dec_emb(input_ids).flatten(0, 1).to(hd.dtype)
    hd = bb.decoder(hd, cu_seqlens=cu, max_seqlen=L)
    res = LMOut(logits=m.lm_head(hd).view(B, L, -1), bpred=[bp], chunks=hc, z=z, cu1=cu1, enc=enc,
                z_full=z_full, max1=max1, cu0=cu)
    if holes is not None:
        res.extras["holes"], res.extras["forced"] = holes, forced
    return res


def force_cuts(bp: RoutingModuleOutput, holes: torch.Tensor, B: int, L: int):
    """Boundaries at every marker byte and at the byte after it (same row), probability 1 there. Returns the new
    routing output and the forced positions (flat bool)."""
    hv = holes.view(B, L)
    forced = (hv | F.pad(hv[:, :-1], (1, 0))).flatten()
    prob = torch.where(forced[:, None], torch.tensor([0.0, 1.0], device=holes.device, dtype=bp.boundary_prob.dtype),
                       bp.boundary_prob)
    sel = torch.where(forced[:, None], torch.ones_like(bp.selected_probs), bp.selected_probs)
    return RoutingModuleOutput(boundary_prob=prob, boundary_mask=bp.boundary_mask | forced, selected_probs=sel), forced


def forward_opts(mc: dict, lat=None) -> dict:
    """Static forward options from the run config (the dynamic ones are added by model_forward)."""
    lc = lat.cfg if lat is not None else {}
    return dict(detach_trunk=bool(lc.get("detach_trunk", False)), floor=bool(mc.get("floor", False)),
                pred_ste=lc.get("router_grad", "none") == "ste", chunk_pool=mc.get("chunk_pool", "hard"),
                conf_grad=bool(mc.get("conf_grad", True)), smoothing=bool(mc.get("smoothing", True)),
                holes=hole_mark(mc))


def hole_mark(mc: dict):
    """The marker byte of a model trained with holes (model.holes.mark), else None."""
    h = mc.get("holes")
    return int(h.get("mark", 3)) if h else None


def model_forward(model, x, mc: dict, lat=None, step_frac: float = 1.0, extra_opts: dict = None) -> LMOut:
    """The one forward used by training, eval and scripts, so every path sees the same model.
    extra_opts: added forward options (z_perm, row_lens, ...; see forward_lm)."""
    opts = forward_opts(mc, lat)
    opts.update(extra_opts or {})
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
    Project modules: dec_emb / surprise get the stage-0 multiplier, nested towers, nextlat and feedback the stage-1 one."""
    mults = lr_multipliers(mc)
    m.apply_lr_multiplier(mults)
    for name, mod in (("dec_emb", 0), ("surprise", 0), ("nest", -1), ("nextlat", -1), ("feedback", -1)):
        if getattr(m, name, None) is not None:
            f = float((mc.get("feedback") or {}).get("lr_mult", 1.0)) if name == "feedback" else 1.0
            for p in getattr(m, name).parameters():
                p._optim = {"lr_multiplier": mults[mod] * f}
    groups = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        lr_mult = _lr_mult(p, 1.0)
        no_decay = p.ndim < 2 or "embeddings" in name or "dec_emb" in name or "level_emb" in name
        router = "routing_module" in name  # own group: model.router_wd / train.router_lr_warm act on it
        w = 0.0 if no_decay else (float(mc.get("router_wd", wd)) if router else wd)
        key = (lr_mult, no_decay, router)
        groups.setdefault(key, {"params": [], "lr_mult": lr_mult, "weight_decay": w, "names": [], "router": router})
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)
    return list(groups.values()), mults
