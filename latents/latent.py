"""JEPA-style latent objectives on H-Net chunks (V-JEPA 2-AC recipe applied to dynamic chunks).

Notation (1-stage H-Net, packed rows):
  c_j     chunk representation = encoder output at the j-th boundary byte (main-network input)
  c̄_j     target representation of the same chunk: stop-grad online encoder ("online") or EMA encoder ("ema")
  P(c≤j)  predictor = main network output at chunk j (full width D1) -> linear head -> D0
Alignment: the main net's output at chunk j already drives the byte predictions inside chunk j; c̄_{j+1}
is the encoder state at the next boundary, which summarises exactly those bytes. So P(c≤j) -> c̄_{j+1} is
causal and consistent with the AR objective.

  L_TF = mean_j ‖P(c≤j) − sg(c̄_{j+1})‖₁                  (teacher forcing)
  L_R  = mean_j ‖P(c≤j, ĉ_{j+1}) − sg(c̄_{j+2})‖₁        (2-step rollout, ĉ_{j+1} = P(c≤j); see rollout.py)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build(lcfg, model, mc):
    return LatentObjective(lcfg, model, mc)


def next_pairs(cu1: torch.Tensor, T1: int, k: int = 1):
    """Indices j such that j and j+k are in the same packed row."""
    row = torch.zeros(T1, dtype=torch.long, device=cu1.device)
    row[cu1[1:-1].long()] = 1
    row = row.cumsum(0)
    j = torch.arange(T1 - k, device=cu1.device)
    return j[row[j] == row[j + k]]


class LatentObjective(nn.Module):
    def __init__(self, lcfg, model, mc):
        super().__init__()
        self.cfg = lcfg
        D0, D1 = mc["d_model"][0], mc["d_model"][-1]
        self.target = lcfg.get("target", "ema")  # online | ema
        self.beta_tf = float(lcfg.get("beta_tf", 1.0))
        self.beta_r = float(lcfg.get("beta_r", 0.0))
        self.detach_main = bool(lcfg.get("detach_main", False))
        # decoder reads head(P(c<=j)) = predicted next-chunk latent instead of the raw main-net output
        self.decode_from_pred = bool(lcfg.get("decode_from_pred", False))
        self.lr_mult = float(lcfg.get("lr_mult", 1.0))
        self.tau0, self.tau1 = float(lcfg.get("ema_tau", 0.996)), float(lcfg.get("ema_tau_end", 1.0))
        dev = next(model.parameters()).device
        # hillclimb knobs
        self.target_norm = lcfg.get("target_norm", "none")  # none | ln (layer-norm targets, as in V-JEPA)
        self.horizons = [int(k) for k in lcfg.get("extra_horizons", [])]  # extra TF horizons k>=2: P(c<=j) -> c̄_{j+k}
        self.beta_h = float(lcfg.get("beta_h", 0.5))
        self.head = make_head(lcfg.get("head", "linear"), D1, D0, dev)
        self.hheads = nn.ModuleList([make_head(lcfg.get("head", "linear"), D1, D0, dev) for _ in self.horizons])
        if self.target == "ema":
            # target encoder = EMA copy of (byte embedding, encoder); excluded from optimizer/state_dict grads
            self.tgt_emb = copy.deepcopy(model.embeddings).requires_grad_(False)
            self.tgt_enc = copy.deepcopy(model.backbone.encoder).requires_grad_(False)
        self.rollout = None
        if self.beta_r > 0:
            from latents.rollout import Rollout
            # predictor_only: rollout gradient trains only the rollout-token path (cached pass-1 context and the
            # fed-back prediction are detached), like V-JEPA 2-AC where encoder + TF context get no rollout grads
            self.rollout = Rollout(model, predictor_only=bool(lcfg.get("rollout_predictor_only", False)))

    # ------------------------------------------------------------------ targets
    @torch.no_grad()
    def targets(self, model, out, x):
        if self.target == "online":
            return out.chunks.detach()
        h = self.tgt_emb(x).flatten(0, 1)
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        h = self.tgt_enc(h, cu_seqlens=cu, max_seqlen=L)
        return h[out.bpred[0].boundary_mask]

    @torch.no_grad()
    def after_step(self, model, step, total_steps):
        if self.target != "ema":
            return
        tau = self.tau1 - (self.tau1 - self.tau0) * (math.cos(math.pi * step / total_steps) + 1) / 2
        for src, dst in ((model.embeddings, self.tgt_emb), (model.backbone.encoder, self.tgt_enc)):
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.lerp_(ps.detach(), 1 - tau)

    # ------------------------------------------------------------------ loss
    def loss(self, model, out, x, step=0, total_steps=1):
        tgt = self.targets(model, out, x).float()
        if self.target_norm == "ln":
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        T1 = tgt.shape[0]
        j = next_pairs(out.cu1, T1, 1)
        pred_all = self.head(out.z_full)  # (T1, D0)
        pred = pred_all[j].float()
        l_tf = (pred - tgt[j + 1]).abs().mean()
        loss = self.beta_tf * l_tf
        st = {"lat/tf_l1": l_tf}
        for k, hk in zip(self.horizons, self.hheads):
            jk = next_pairs(out.cu1, T1, k)
            l_k = (hk(out.z_full[jk]).float() - tgt[jk + k]).abs().mean()
            loss = loss + self.beta_h * l_k / len(self.horizons)
            st[f"lat/tf{k}_l1"] = l_k
        if self.rollout is not None:
            l_r = self.rollout.loss(model, self, out, pred_all, tgt)
            loss = loss + self.beta_r * l_r
            st["lat/r_l1"] = l_r
        with torch.no_grad():
            st["lat/copy_l1"] = (tgt[j] - tgt[j + 1]).abs().mean()  # persistence baseline
            st["lat/tgt_std"] = tgt.std(0).mean()
            st["lat/pred_std"] = pred.std(0).mean()
            st["lat/cos"] = F.cosine_similarity(pred, tgt[j + 1], dim=-1).mean()
        return loss, st

    @torch.no_grad()
    def eval_metrics(self, model, out, x):
        _, st = self.loss(model, out, x)
        tgt = self.targets(model, out, x).float()
        st["lat/tgt_eff_rank"] = eff_rank(tgt)
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in st.items()}


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


def eff_rank(x: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of the normalised singular-value spectrum (RankMe) — collapse detector."""
    x = x - x.mean(0)
    s = torch.linalg.svdvals(x[:4096].float())
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum())
