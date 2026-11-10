"""Levels >= 2 on top of the 1-stage H-Net (ladder phases 4-5).

Level 1 is the existing model: byte encoder -> router -> chunks c^1 -> main network (the level-1 predictor).
Level s >= 2 is a tower over the level-(s-1) chunk sequence:

    x^s = pad(c^{s-1})  ->  enc_s  ->  h^s  ->  router_s  ->  chunks c^s = h^s[mask_s]
        ->  pred_s(c^s + topdown from level s+1 [+ level embedding])  ->  z^s  ->  head_s  ->  ĉ^s_{k+1}

Latent loss at level s: head_s(z^s_k) vs sg(c̄^s_{k+1}), with c̄^s from EMA copies of the encoders (latent.py).
Top-down (optional): z^s is dechunked back to level-(s-1) resolution exactly like H-Net's decoder path
(EMA smoothing with the boundary probability, times STE(selected prob)), projected by a zero-initialised
linear layer and added to the level-(s-1) predictor input. Causal: the level-(s-1) position j only receives
z^s from the latest level-s boundary <= j, which saw level-(s-1) inputs <= j. Because the STE and the
smoothing use the router's probabilities, router_s gets gradient from the level-(s-1) latent loss.

Config (model.nest):
  levels        number of levels (>= 2), or the cap when depth_rule = min_len
  depth_rule    fixed | min_len (go up while the expected top-level length is above min_top_len chunks; the
                expected length = seq_len / (level-1 chunk size x ratios above), so the decision never depends
                on the content of the window and stays causal)
  enc / pred    layouts of the tower encoder / predictor (pred must be attention-only for rollout)
  router        per-level list or single value: pool | learned | identity
  ratio         per-level list or single value: target compression N_s (pool size for 'pool')
  compute_penalty, penalty_from_level   above that level, a penalty on the mean boundary probability
                replaces the ratio loss, so a level can send (almost) nothing up (R19)
  topdown       feed level-s outputs into the level-(s-1) predictor (False = passenger level)
  input_grad    let the level-s losses send gradient into the level-(s-1) chunks
  tie           none | towers (levels >= 2 share one block) | all (+ share the predictor with level 1)
  head          head kind for levels >= 2 (default: latent.head)
"""
import copy
import math

import torch
import torch.nn as nn

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.hnet import ste_func
from hnet.modules.dc import ChunkLayer, DeChunkLayer, RoutingModule
from hnet.modules.isotropic import Isotropic

from latents.models import pool_packed


def make_iso(layout, d, d_int, heads, rot, ssm_cfg, device):
    cfg = HNetConfig(arch_layout=[layout], d_model=[d], d_intermediate=[d_int], vocab_size=256,
                     ssm_cfg=SSMConfig(**ssm_cfg),
                     attn_cfg=AttnConfig(num_heads=[heads], rotary_emb_dim=[rot], window_size=[-1]))
    with torch.device(device):
        iso = Isotropic(cfg, pos_idx=0, stage_idx=0, device=device, dtype=torch.float32)
    n_res = iso.height
    for name, mod in iso.named_modules():
        if isinstance(mod, nn.Linear) and not getattr(mod.weight, "_no_reinit", False):
            std = 0.02 / math.sqrt(n_res) if ("out_proj" in name or "fc2" in name) else 0.02
            nn.init.normal_(mod.weight, std=std)
    return iso


def make_head(kind, d_in, d_out, dev):
    if kind == "linear":
        h = nn.Linear(d_in, d_out, bias=False, device=dev)
        nn.init.normal_(h.weight, std=0.02)
        return h
    if kind == "mlp":
        h = nn.Sequential(nn.Linear(d_in, 2 * d_in, device=dev), nn.GELU(), nn.Linear(2 * d_in, d_out, device=dev))
        for m in h:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        return h
    raise ValueError(kind)


class Tower(nn.Module):
    def __init__(self, D, nc, mc, device, pred=None):
        super().__init__()
        ssm = mc.get("ssm_cfg", {})
        heads, rot, d_int = mc["num_heads"][-1], mc["rotary_emb_dim"][-1], mc["d_intermediate"][-1]
        self.enc = make_iso(nc.get("enc", "m2"), D, d_int, heads, rot, ssm, device)
        self.router = RoutingModule(D, device=device)
        self.chunk = ChunkLayer()
        if pred is None:
            self.pred = make_iso(nc.get("pred", "T3"), D, d_int, heads, rot, ssm, device)
        else:  # tie=all: the level-1 predictor, held without registering it twice
            self.__dict__["pred"] = pred
        self.dechunk = DeChunkLayer(D)
        self.td_proj = nn.Linear(D, D, bias=False, device=device)
        nn.init.zeros_(self.td_proj.weight)  # top-down starts as a no-op
        self.head = make_head(nc.get("head", mc.get("_latent_head", "mlp")), D, D, device)


def _per_level(v, s, default):
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        return v[min(s - 2, len(v) - 1)]
    return v


class Nest(nn.Module):
    def __init__(self, model, mc, device="cuda"):
        super().__init__()
        nc = dict(mc.get("nest", {}))
        self.nc = nc
        self.D0, self.D = mc["d_model"][0], mc["d_model"][-1]
        self.levels = int(nc.get("levels", 2))
        self.max_levels = int(nc.get("max_levels", self.levels))
        self.depth_rule = nc.get("depth_rule", "fixed")
        self.min_top_len = int(nc.get("min_top_len", 128))
        self.topdown_on = bool(nc.get("topdown", True))
        self.input_grad = bool(nc.get("input_grad", True))
        self.tie = nc.get("tie", "none")
        self.alpha = float(mc.get("alpha", 0.03))
        n1 = mc.get("pool_k") or mc.get("N", 6)
        self.expected = {1: mc.get("seq_len", 8192) / n1}
        for s in range(2, self.max_levels + 1):
            self.expected[s] = self.expected[s - 1] / float(_per_level(nc.get("ratio"), s, 3))
        if self.depth_rule == "min_len":
            top = 1
            while top < self.max_levels and self.expected[top] > self.min_top_len:
                top += 1
        else:
            top = self.levels
        self.top = top
        main_iso = model.backbone.main_network.main_network
        n_blocks = 1 if self.tie in ("towers", "all") else top - 1
        self.blocks = nn.ModuleList([Tower(self.D, nc, mc, device, pred=main_iso if self.tie == "all" else None)
                                     for _ in range(n_blocks)])
        # per-level input pads: level 2 reads D0-dim byte-encoder chunks, levels >= 3 read D-dim tower chunks
        self.pads = nn.ParameterList([nn.Parameter(torch.zeros(self.D - (self.D0 if s == 2 else self.D), device=device))
                                      for s in range(2, top + 1)])
        self.level_emb = nn.Embedding(top + 1, self.D, device=device) if self.tie in ("towers", "all") else None
        if self.level_emb is not None:
            nn.init.zeros_(self.level_emb.weight)
        self.__dict__["main_iso"] = main_iso
        self.routers = {}
        for s in range(2, top + 1):
            kind = _per_level(nc.get("router"), s, "pool")
            if kind == "identity":
                blk = self.block(s)
                blk.router.q_proj_layer.weight.requires_grad_(False)
                blk.router.k_proj_layer.weight.requires_grad_(False)
            self.routers[s] = kind

    # ------------------------------------------------------------------ structure
    def block(self, s):
        return self.blocks[0] if len(self.blocks) == 1 else self.blocks[s - 2]

    def ratio(self, s):
        return float(_per_level(self.nc.get("ratio"), s, 3))

    def penalised(self, s):
        return self.nc.get("compute_penalty") is not None and s >= int(self.nc.get("penalty_from_level", 3))

    def new_context(self):
        return {"levels": []}

    def input_of(self, s, x_in):
        pad = self.pads[s - 2]
        return x_in if pad.numel() == 0 else torch.cat((x_in, pad.expand(x_in.shape[0], -1).to(x_in.dtype)), -1)

    def deeper(self, s, max_len):
        return s < self.top

    # ------------------------------------------------------------------ forward
    def topdown(self, hc, cu1, max1, ctx):
        """Called by forward_lm with the level-1 chunks; returns the additive input of the level-1 predictor."""
        td = self._level(2, hc, cu1, max1, ctx)
        out = torch.zeros(hc.shape[0], self.D, device=hc.device, dtype=hc.dtype) if td is None else td
        if self.tie == "all":
            out = out + self.level_emb.weight[1].to(out.dtype)
        cap = getattr(self.main_iso, "_qkv_capture", None)
        if cap is not None:
            cap.tag = "L1"
        return out

    def _level(self, s, x_in, cu, max_len, ctx):
        blk = self.block(s)
        x = self.input_of(s, x_in if self.input_grad else x_in.detach())
        h = blk.enc(x, cu_seqlens=cu, max_seqlen=max_len)
        kind = self.routers[s]
        bp = pool_packed(cu, int(self.ratio(s))) if kind == "pool" else blk.router(h, cu_seqlens=cu)
        hc, cu2, max2, _ = blk.chunk(h, bp.boundary_mask, cu)
        td_up = self._level(s + 1, hc, cu2, max2, ctx) if self.deeper(s, max2) else None
        hp = hc if td_up is None else hc + td_up.to(hc.dtype)
        if self.level_emb is not None:
            hp = hp + self.level_emb.weight[s].to(hp.dtype)
        cap = getattr(blk.pred, "_qkv_capture", None)
        if cap is not None:
            cap.tag = f"L{s}"
        z = blk.pred(hp, cu_seqlens=cu2, max_seqlen=max2)
        ctx["levels"].append(dict(level=s, x=x, h=h, bp=bp, chunks=hc, hp=hp, z=z, cu_in=cu, cu=cu2, max=max2,
                                  T_in=x.shape[0], T=hc.shape[0]))
        if not self.topdown_on:
            return None
        out = blk.dechunk(z, bp.boundary_mask, bp.boundary_prob, cu2)
        return blk.td_proj(out * ste_func(bp.selected_probs).to(out.dtype))

    # ------------------------------------------------------------------ losses
    def router_losses(self, ctx):
        loss, st = 0.0, {}
        for rec in ctx["levels"]:
            s = rec["level"]
            if self.routers[s] == "pool":
                continue
            bp = rec["bp"]
            if self.penalised(s):
                l = float(self.nc["compute_penalty"]) * bp.boundary_prob[..., -1].float().mean()
            else:
                from latents.models import ratio_loss
                l = self.alpha * ratio_loss(bp, self.ratio(s))
            loss = loss + l
            st[f"L{s}/router_loss"] = l.detach()
        return loss, st


class EmaNest(nn.Module):
    """EMA copies of what the level-s targets need: pads, level embedding, tower encoders."""

    def __init__(self, nest: Nest):
        super().__init__()
        self.pads = copy.deepcopy(nest.pads).requires_grad_(False)
        self.encs = nn.ModuleList([copy.deepcopy(b.enc) for b in nest.blocks]).requires_grad_(False)
        self.n_blocks = len(nest.blocks)

    def pairs(self, nest: Nest):
        out = list(zip(nest.pads, self.pads))
        for b, e in zip(nest.blocks, self.encs):
            out += list(zip(b.enc.parameters(), e.parameters()))
        return out

    @torch.no_grad()
    def targets(self, nest: Nest, c1_tgt, ctx):
        """c1_tgt: EMA level-1 chunk targets (T1, D0). Returns {s: c̄^s (T_s, D)} using the online boundary masks."""
        out = {}
        prev = c1_tgt
        recs = sorted(ctx["levels"], key=lambda r: r["level"])
        for rec in recs:
            s = rec["level"]
            pad = self.pads[s - 2]
            x = prev if pad.numel() == 0 else torch.cat((prev, pad.expand(prev.shape[0], -1).to(prev.dtype)), -1)
            enc = self.encs[0] if self.n_blocks == 1 else self.encs[s - 2]
            h = enc(x.to(rec["x"].dtype), cu_seqlens=rec["cu_in"], max_seqlen=int((rec["cu_in"][1:] - rec["cu_in"][:-1]).max()))
            prev = h[rec["bp"].boundary_mask]
            out[s] = prev
        return out
