"""JEPA-style latent objectives on H-Net chunks (V-JEPA 2-AC recipe applied to dynamic chunks).

Notation (1-stage H-Net, packed rows):
  c_j     chunk representation = encoder output at the j-th boundary byte (main-network input)
  c̄_j     target representation of the same chunk: stop-grad online encoder ("online") or EMA encoder ("ema")
  P(c≤j)  predictor = main network output at chunk j (full width D1) -> head -> D0
Alignment: the main net's output at chunk j already drives the byte predictions inside chunk j; c̄_{j+1}
is the encoder state at the next boundary, which summarises exactly those bytes. So P(c≤j) -> c̄_{j+1} is
causal and consistent with the AR objective.

  L_TF = mean_j ‖P(c≤j) − sg(c̄_{j+1})‖₁                       (teacher forcing)
  L_R  = mean_{s=2..k} mean_j ‖P(c≤j, ĉ_{j+1..j+s-1}) − sg(c̄_{j+s})‖₁   (k-step rollout, see rollout.py)

Ladder additions (all off by default):
  reg          var | vicreg | sigreg anti-collapse term on the online chunk latents; reg_scope=within centres
               each sequence first, so a per-document constant code cannot satisfy it (R5)
  contrast=cpc InfoNCE against the other chunks of the same sequence (R6b)
  detach_trunk no byte-loss gradient into the trunk (handled in models.forward_lm, R6)
  router_grad  ste (R8, handled in forward_lm) | reinforce (option E)
  levels       latent losses of nested levels >= 2 (latents/nested.py)
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from latents.nested import EmaNest, make_head  # noqa: F401  (make_head re-exported for old imports)


def build(lcfg, model, mc):
    return LatentObjective(lcfg, model, mc)


def next_pairs(cu1: torch.Tensor, T1: int, k: int = 1):
    """Indices j such that j and j+k are in the same packed row."""
    row = torch.zeros(T1, dtype=torch.long, device=cu1.device)
    row[cu1[1:-1].long()] = 1
    row = row.cumsum(0)
    j = torch.arange(max(T1 - k, 0), device=cu1.device)
    return j[row[j] == row[j + k]]


def row_ids(cu: torch.Tensor, T: int):
    lens = (cu[1:] - cu[:-1]).long()
    return torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens), lens


def center_rows(x, cu):
    """Subtract each packed row's mean."""
    row, lens = row_ids(cu, x.shape[0])
    sums = torch.zeros(lens.numel(), x.shape[-1], device=x.device, dtype=x.dtype).index_add_(0, row, x)
    return x - (sums / lens.clamp(min=1)[:, None].to(x.dtype))[row]


@torch.no_grad()
def within_frac(x, cu):
    """Share of the variance that is within sequences (1 = all variation inside rows, 0 = per-row constant code)."""
    x = x.float()
    tot = x.var(0).sum()
    return (center_rows(x, cu).pow(2).mean(0).sum() / tot.clamp(min=1e-12)).item() if x.shape[0] > 1 else float("nan")


def sigreg(x, n_proj=64, seed=0):
    """Epps-Pulley test against N(0, 1) along random unit directions (LeJEPA's SIGReg), averaged over directions."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(x.shape[-1], n_proj, generator=g).to(x.device, x.dtype)
    A = A / A.norm(dim=0, keepdim=True)
    y = x @ A  # (N, P)
    t = torch.linspace(0.2, 3.0, 15, device=x.device, dtype=x.dtype)
    ty = y[:, :, None] * t  # (N, P, T)
    phi = torch.exp(-t ** 2 / 2)
    err = (ty.cos().mean(0) - phi) ** 2 + ty.sin().mean(0) ** 2  # (P, T)
    w = phi * (t[1] - t[0])
    return (err * w).sum(-1).mean()


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
        # ladder knobs
        self.dist = lcfg.get("dist", "l1")  # l1 | l2 (squared error): the prediction distance of the TF losses
        self.reg = lcfg.get("reg", "none")  # none | var | vicreg | vicreg_corr | sigreg
        self.beta_cov = float(lcfg.get("beta_cov", 1.0))  # weight of the off-diagonal term (vicreg, vicreg_corr)
        self.reg_scope = lcfg.get("reg_scope", "within")  # within | batch
        self.beta_reg = float(lcfg.get("beta_reg", 1.0))
        self.reg_gamma = lcfg.get("reg_gamma", "auto")  # target std for var/vicreg, or auto-calibrated
        self.reg_gamma_frac = float(lcfg.get("reg_gamma_frac", 0.5))
        self.reg_calib_frac = float(lcfg.get("reg_calib_frac", 0.1))
        self.register_buffer("gamma_est", torch.tensor(-1.0, device=dev))
        self.contrast = lcfg.get("contrast", "none")  # none | cpc
        self.beta_cpc = float(lcfg.get("beta_cpc", 1.0))
        self.cpc_temp = float(lcfg.get("cpc_temp", 0.1))
        self.router_grad = lcfg.get("router_grad", "none")  # none | ste | reinforce
        self.beta_rf = float(lcfg.get("beta_rf", 1.0))
        self.register_buffer("rf_baseline", torch.tensor(0.0, device=dev))
        self.beta_level = float(lcfg.get("beta_level", 1.0))
        self.__dict__["nest"] = getattr(model, "nest", None)  # a reference, not a submodule (no duplicate params)
        if self.target == "ema":
            # target encoder = EMA copy of (byte embedding, encoder); excluded from optimizer/state_dict grads
            self.tgt_emb = copy.deepcopy(model.embeddings).requires_grad_(False)
            self.tgt_enc = copy.deepcopy(model.backbone.encoder).requires_grad_(False)
            if self.nest is not None:
                self.tgt_nest = EmaNest(self.nest)
        self.rollout = None
        tie_all = self.nest is not None and self.nest.tie == "all"
        if self.beta_r > 0:
            from latents.rollout import Rollout
            # predictor_only: rollout gradient trains only the rollout-token path (cached pass-1 context and the
            # fed-back prediction are detached), like V-JEPA 2-AC where encoder + TF context get no rollout grads
            # With nesting, Nest.topdown tags the level-1 predictor's qkv capture "L1" (for any tie), so read that.
            self.rollout = Rollout.for_main(
                model, predictor_only=bool(lcfg.get("rollout_predictor_only", False)),
                k=int(lcfg.get("rollout_k", 2)), tag="L1" if self.nest is not None else None,
                in_add=(lambda: self.nest.level_emb.weight[1]) if tie_all else None)
        self.level_rollouts = {}
        nk = int(self.nest.nc.get("rollout_k", 0)) if self.nest is not None else 0
        if nk >= 2:
            from latents.rollout import Rollout
            for s in range(2, self.nest.top + 1):
                blk = self.nest.block(s)
                emb = self.nest.level_emb
                self.level_rollouts[s] = Rollout(
                    blk.pred, pad=None, predictor_only=bool(lcfg.get("rollout_predictor_only", True)), k=nk,
                    tag=f"L{s}", in_add=(lambda s=s: emb.weight[s]) if emb is not None else None)

    # ------------------------------------------------------------------ targets
    def needs_ema_states(self, mc):
        return self.target == "ema"

    @torch.no_grad()
    def ema_states(self, model, x):
        """EMA byte-encoder states for every byte (T0, D0)."""
        h = self.tgt_emb(x).flatten(0, 1)
        B, L = x.shape
        cu = torch.arange(B + 1, device=x.device, dtype=torch.int32) * L
        return self.tgt_enc(h, cu_seqlens=cu, max_seqlen=L)

    @torch.no_grad()
    def targets(self, model, out, x):
        if self.target == "online":
            return out.chunks.detach()
        h = out.extras.get("ema_h")
        if h is None:
            h = self.ema_states(model, x)
        return h[out.bpred[0].boundary_mask]

    @torch.no_grad()
    def after_step(self, model, step, total_steps):
        if self.target != "ema":
            return
        tau = self.tau1 - (self.tau1 - self.tau0) * (math.cos(math.pi * step / total_steps) + 1) / 2
        pairs = list(zip(model.embeddings.parameters(), self.tgt_emb.parameters()))
        pairs += list(zip(model.backbone.encoder.parameters(), self.tgt_enc.parameters()))
        if self.nest is not None:
            pairs += self.tgt_nest.pairs(self.nest)
        for ps, pd in pairs:
            pd.lerp_(ps.detach().to(pd.dtype), 1 - tau)

    # ------------------------------------------------------------------ pieces
    def gamma(self, x_std, step, total_steps):
        """Target std of the variance hinge; 'auto' = reg_gamma_frac x the std measured early in training."""
        if self.reg_gamma != "auto":
            return float(self.reg_gamma)
        if step < self.reg_calib_frac * total_steps:
            v = x_std.detach()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(v)
                v = v / torch.distributed.get_world_size()
            self.gamma_est.copy_(v if self.gamma_est < 0 else 0.9 * self.gamma_est + 0.1 * v)
            return None  # calibrating: no penalty yet
        return float(self.reg_gamma_frac * self.gamma_est)

    def reg_loss(self, c, cu, step, total_steps):
        x = c.float()
        if self.reg_scope == "within":
            x = center_rows(x, cu)
        else:
            x = x - x.mean(0)
        std = (x.pow(2).mean(0) + 1e-6).sqrt()
        st = {"lat/reg_std": std.mean().detach()}
        if self.reg == "sigreg":
            return sigreg(x, seed=step), st
        g = self.gamma(std.mean(), step, total_steps)
        if g is None:
            return x.new_zeros(()), st
        st["lat/reg_gamma"] = torch.tensor(g)
        loss = F.relu(g - std).mean()
        if self.reg in ("vicreg", "vicreg_corr"):
            # vicreg: covariance; vicreg_corr: correlation (scale-free, Barlow-style decorrelation)
            xs = x / std if self.reg == "vicreg_corr" else x
            cov = (xs.T @ xs) / max(x.shape[0] - 1, 1)
            off = cov - torch.diag(torch.diag(cov))
            l_cov = off.pow(2).sum() / x.shape[-1]
            loss = loss + self.beta_cov * l_cov
            st["lat/reg_cov"] = l_cov.detach()
        with torch.no_grad():  # participation ratio of the (centred) chunk latents: effective dimensionality
            ev = torch.linalg.eigvalsh((x.T @ x) / max(x.shape[0] - 1, 1)).clamp(min=0)
            st["lat/participation"] = ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)
        return loss, st

    def cpc_loss(self, pred_all, tgt, cu):
        """InfoNCE: P(c<=j) must pick c̄_{j+1} among all chunks of the same sequence (incl. c̄_j, the copy)."""
        row, lens = row_ids(cu, tgt.shape[0])
        B, M = lens.numel(), int(lens.max())
        pos = torch.arange(tgt.shape[0], device=tgt.device) - cu[:-1].long()[row]
        P = F.normalize(pred_all.float(), dim=-1)
        Tn = F.normalize(tgt.float(), dim=-1)
        Pp = P.new_zeros(B, M, P.shape[-1])
        Tp = Tn.new_zeros(B, M, Tn.shape[-1])
        Pp[row, pos] = P
        Tp[row, pos] = Tn
        logits = Pp @ Tp.transpose(1, 2) / self.cpc_temp  # (B, M, M)
        col_ok = torch.arange(M, device=tgt.device)[None, :] < lens[:, None]
        logits = logits.masked_fill(~col_ok[:, None, :], float("-inf"))
        q_ok = torch.arange(M, device=tgt.device)[None, :] < (lens[:, None] - 1)
        label = torch.arange(M, device=tgt.device)[None, :].expand(B, -1) + 1
        ce = F.cross_entropy(logits[q_ok], label[q_ok])
        acc = (logits[q_ok].argmax(-1) == label[q_ok]).float().mean()
        return ce, acc.detach()

    def reinforce_loss(self, out, pred_all, tgt, j):
        """Option E: per-sequence reward = -latent L1 of that sequence; logp from routing.sampled_routing."""
        logp = out.extras.get("logp")
        if logp is None:
            return pred_all.new_zeros(()), {}
        row1, lens1 = row_ids(out.cu1, tgt.shape[0])
        err = (pred_all[j].float() - tgt[j + 1]).abs().mean(-1).detach()
        B = lens1.numel()
        r_sum = torch.zeros(B, device=err.device).index_add_(0, row1[j], err)
        r_cnt = torch.zeros(B, device=err.device).index_add_(0, row1[j], torch.ones_like(err))
        reward = -(r_sum / r_cnt.clamp(min=1))
        row0, lens0 = row_ids(out.cu0, logp.shape[0])
        lp = torch.zeros(B, device=err.device).index_add_(0, row0, logp.float()) / lens0.float()
        adv = reward - self.rf_baseline
        self.rf_baseline.mul_(0.95).add_(0.05 * reward.mean())
        return -(adv * lp).mean(), {"lat/rf_reward": reward.mean()}

    def level_losses(self, out, tgt1_raw):
        ctx = out.extras.get("levels")
        if ctx is None or not ctx["levels"]:
            return 0.0, {}
        nest = self.nest
        if self.target == "ema":
            tg = self.tgt_nest.targets(nest, tgt1_raw, ctx)
        else:
            tg = {r["level"]: r["chunks"].detach() for r in ctx["levels"]}
        loss, st = 0.0, {}
        for rec in ctx["levels"]:
            s = rec["level"]
            tgt = tg[s].float()
            blk = nest.block(s)
            pred_all = blk.head(rec["z"])
            j = next_pairs(rec["cu"], tgt.shape[0], 1)
            if j.numel() == 0:
                continue
            pred = pred_all[j].float()
            l = (pred - tgt[j + 1]).abs().mean()
            loss = loss + self.beta_level * l
            st[f"L{s}/tf_l1"] = l
            if s in self.level_rollouts:
                lr_, cp = self.level_rollouts[s].losses(blk.head, rec["chunks"], rec["cu"], pred_all, tgt)
                if lr_:
                    l_r = sum(lr_.values()) / len(lr_)
                    loss = loss + self.beta_r * l_r if self.beta_r > 0 else loss + l_r
                    st[f"L{s}/r_l1"] = l_r
                    for k_, v in lr_.items():
                        st[f"L{s}/r{k_}_l1"] = v
                        st[f"L{s}/copy{k_}_l1"] = cp[k_]
            with torch.no_grad():
                st[f"L{s}/copy_l1"] = (tgt[j] - tgt[j + 1]).abs().mean()
                st[f"L{s}/cos"] = F.cosine_similarity(pred, tgt[j + 1], dim=-1).mean()
                st[f"L{s}/tgt_std"] = tgt.std(0).mean()
                st[f"L{s}/compression"] = torch.tensor(rec["T"] / max(rec["T_in"], 1))
                st[f"L{s}/len"] = torch.tensor(float(rec["max"]))
        st["levels/depth"] = torch.tensor(float(1 + len(ctx["levels"])))
        return loss, st

    # ------------------------------------------------------------------ loss
    def dist_fn(self, pred, tgt):
        """Element-wise prediction distance (mean it yourself): |p - t| (l1, default) or (p - t)^2 (l2)."""
        d = pred - tgt
        return d.pow(2) if self.dist == "l2" else d.abs()

    def loss(self, model, out, x, step=0, total_steps=1, scale=1.0):
        tgt_raw = self.targets(model, out, x).float()
        tgt = F.layer_norm(tgt_raw, tgt_raw.shape[-1:]) if self.target_norm == "ln" else tgt_raw
        T1 = tgt.shape[0]
        j = next_pairs(out.cu1, T1, 1)
        pred_all = self.head(out.z_full)  # (T1, D0)
        pred = pred_all[j].float()
        l_tf = self.dist_fn(pred, tgt[j + 1]).mean()
        loss = self.beta_tf * l_tf
        st = {"lat/tf_l1": l_tf if self.dist == "l1" else (pred - tgt[j + 1]).abs().mean().detach()}
        for k, hk in zip(self.horizons, self.hheads):
            jk = next_pairs(out.cu1, T1, k)
            l_k = self.dist_fn(hk(out.z_full[jk]).float(), tgt[jk + k]).mean()
            loss = loss + self.beta_h * l_k / len(self.horizons)
            st[f"lat/tf{k}_l1"] = l_k
        if self.rollout is not None:
            lr_, cp = self.rollout.losses(self.head, out.chunks, out.cu1, pred_all, tgt)
            if lr_:
                l_r = sum(lr_.values()) / len(lr_)
                loss = loss + self.beta_r * l_r
                st["lat/r_l1"] = l_r
                for k_, v in lr_.items():
                    st[f"lat/r{k_}_l1"] = v
                    st[f"lat/copy{k_}_l1"] = cp[k_]
        if self.contrast == "cpc":
            l_c, acc = self.cpc_loss(pred_all, tgt, out.cu1)
            loss = loss + self.beta_cpc * l_c
            st["lat/cpc"] = l_c
            st["lat/cpc_acc"] = acc
        if self.reg != "none":
            l_reg, st_r = self.reg_loss(out.chunks, out.cu1, step, total_steps)
            loss = loss + self.beta_reg * l_reg
            st["lat/reg"] = l_reg
            st.update(st_r)
        if self.router_grad == "reinforce":
            l_rf, st_rf = self.reinforce_loss(out, pred_all, tgt, j)
            loss = loss + self.beta_rf * l_rf
            st.update(st_rf)
        l_lev, st_lev = self.level_losses(out, tgt_raw)
        loss = loss + l_lev
        st.update(st_lev)
        loss = loss * scale
        if "surprise_loss" in out.extras:  # trains the surprise router's own predictor (unscaled)
            loss = loss + out.extras["surprise_loss"]
            st["lat/surprise_l1"] = out.extras["surprise_loss"].detach()
            st["lat/surprise_thr"] = out.extras["surprise_thr"]
        with torch.no_grad():
            st["lat/copy_l1"] = (tgt[j] - tgt[j + 1]).abs().mean()  # persistence baseline
            st["lat/tgt_std"] = tgt.std(0).mean()
            st["lat/pred_std"] = pred.std(0).mean()
            st["lat/cos"] = F.cosine_similarity(pred, tgt[j + 1], dim=-1).mean()
        return loss, st

    @torch.no_grad()
    def eval_metrics(self, model, out, x):
        _, st = self.loss(model, out, x, step=10 ** 9, total_steps=10 ** 9)
        tgt = self.targets(model, out, x).float()
        st["lat/tgt_eff_rank"] = eff_rank(tgt)
        st["lat/tgt_within_frac"] = within_frac(tgt, out.cu1)
        st["lat/chunk_within_frac"] = within_frac(out.chunks, out.cu1)
        st["lat/ratio_tf_copy"] = (st["lat/tf_l1"] / st["lat/copy_l1"].clamp(min=1e-9))
        for s in range(2, 10):
            if f"L{s}/tf_l1" in st:
                st[f"L{s}/ratio_tf_copy"] = st[f"L{s}/tf_l1"] / st[f"L{s}/copy_l1"].clamp(min=1e-9)
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in st.items()}


def eff_rank(x: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of the normalised singular-value spectrum (RankMe) — collapse detector."""
    x = x - x.mean(0)
    s = torch.linalg.svdvals(x[:4096].float())
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum())
