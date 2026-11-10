"""Alternative boundary rules for the byte-level router (ladder phase 2).

SurpriseRouter (option C): cut where a small byte-level latent predictor is surprised. The predictor g maps the
  (detached) online encoder state h_{t-1} to the target encoder state at t (EMA encoder if available); the
  surprise is s_t = mean |g(h_{t-1}) - sg(h̄_t)|. A boundary is placed at t when s_t exceeds a running threshold
  that is steered to hit the target ratio 1/N. Causal: s_t depends on bytes <= t only (like H-Net's router).
  The threshold is a buffer updated only in training, so eval never looks at batch statistics.
  During warmup (first `warmup` fraction of training) boundaries are fixed every N bytes while g learns.
sampled_routing (option E): boundaries sampled from the learned router's Bernoulli(p) in training (greedy in
  eval); the log-probabilities feed a REINFORCE term in latents/latent.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.modules.dc import RoutingModuleOutput

from latents.models import hard_routing


class SurpriseRouter(nn.Module):
    def __init__(self, d, N, hidden_mult=1, warmup=0.1, thr_lr=0.05, beta=1.0):
        super().__init__()
        self.N = float(N)
        self.warmup = float(warmup)
        self.thr_lr = float(thr_lr)
        self.beta = float(beta)
        hdim = int(hidden_mult * d)
        self.g = nn.Sequential(nn.Linear(d, hdim), nn.GELU(), nn.Linear(hdim, d))
        for mod in self.g:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
                nn.init.zeros_(mod.bias)
        self.register_buffer("thr", torch.tensor(-1.0))  # < 0: not initialised yet

    def forward(self, h, cu, tgt_h=None, step_frac=1.0, extras=None):
        T = h.shape[0]
        tgt = (h if tgt_h is None else tgt_h).detach().float()
        pred = self.g(h.detach().float())
        starts = torch.zeros(T, dtype=torch.bool, device=h.device)
        starts[cu[:-1].long()] = True
        err = (pred[:-1] - tgt[1:]).abs().mean(-1)  # surprise of byte t+1 given state at t
        s = F.pad(err.detach(), (1, 0))
        valid = ~starts[1:]  # pairs inside one row
        if extras is not None and self.training:
            extras["surprise_loss"] = self.beta * err[valid].mean()
        if self.training and float(self.thr) < 0:
            self.thr.fill_(float(torch.quantile(s[~starts][:100_000].float(), 1 - 1 / self.N)))
        warm = self.training and step_frac < self.warmup
        if warm:
            lens = (cu[1:] - cu[:-1]).long()
            row = torch.repeat_interleave(torch.arange(lens.numel(), device=h.device), lens)
            pos = torch.arange(T, device=h.device) - cu[:-1].long()[row]
            mask = pos % int(round(self.N)) == 0
        else:
            mask = s > self.thr
        mask = mask | starts
        if self.training:
            rate = (s[~starts] > self.thr).float().mean()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(rate)
                rate = rate / torch.distributed.get_world_size()
            # multiplicative update: rate above target -> raise the threshold
            self.thr.mul_(torch.exp(self.thr_lr * (rate - 1 / self.N) * self.N))
        if extras is not None:
            extras["surprise_thr"] = self.thr.detach().clone()
            extras["surprise_mean"] = s[~starts].mean().detach()
        return hard_routing(mask)


def sampled_routing(router, h, cu, extras):
    """Option E: sample boundaries from the learned router's probabilities (training only)."""
    bp = router(h, cu_seqlens=cu)
    if not router.training:
        return bp
    p = bp.boundary_prob[..., 1].clamp(1e-4, 1 - 1e-4)
    mask = torch.bernoulli(p.detach()).bool()
    mask[cu[:-1].long()] = True
    sel = torch.where(mask, p, 1 - p)
    extras["logp"] = torch.log(sel)  # (T0,)
    return RoutingModuleOutput(boundary_prob=bp.boundary_prob, boundary_mask=mask, selected_probs=sel[:, None])
