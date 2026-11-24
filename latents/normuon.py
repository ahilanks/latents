"""NorMuon, as Latent Feedback trains its matrices (the feedback matrix W included).

Ported from ifm-ai/LoRM @ 2460ca8, lorm/train/optim/normuon.py (Apache License 2.0), which adapts the Q Labs
scaling-exponents code (Chen et al., 2026, arXiv 2609.19107). Same algorithm, per update of each matrix W (rows x cols):

1. Nesterov momentum: m <- lerp(m, g, 1 - mu); u = lerp(g, m, mu).
2. Divide each row of u by its norm.
3. Polar Express orthogonalization (Amsel et al., 2025), five bfloat16 steps.
4. NorMuon variance reduction: an EMA (beta2) of the mean square along the longer dimension rescales each row (tall)
   or column (wide), then the Frobenius norm is restored.
5. Step lr * max(1, rows / cols)^0.5, with cautious weight decay: decay only entries where the update and the weight
   share a sign, at the same scaled rate.

Momentum ramps linearly from momentum_start to momentum over momentum_warmup_steps updates. Parameters without a
gradient in an update (e.g. W on Latent Feedback's single-pass updates) are skipped, as in theirs.
"""
import math

import torch
from torch.optim import Optimizer

POLAR_EXPRESS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(update, steps=5):
    """Approximate the orthogonal polar factor of each matrix in a (..., rows, cols) batch."""
    x = update.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    tall = x.size(-2) > x.size(-1)
    for a, b, c in POLAR_EXPRESS[:steps]:
        if tall:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


class NorMuon(Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, momentum_start=None,
                 momentum_warmup_steps=0, beta2=0.95, ns_steps=5):
        momentum_start = momentum if momentum_start is None else momentum_start
        for name, value in (("lr", lr), ("weight_decay", weight_decay)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        for name, value in (("momentum", momentum), ("momentum_start", momentum_start), ("beta2", beta2)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      momentum_start=momentum_start, momentum_warmup_steps=int(momentum_warmup_steps),
                                      beta2=beta2, ns_steps=int(ns_steps)))
        for group in self.param_groups:
            if any(p.ndim != 2 for p in group["params"]):
                raise ValueError("NorMuon updates only two-dimensional matrices.")

    @staticmethod
    def momentum_at(group, step):
        warmup = group["momentum_warmup_steps"]
        fraction = 1.0 if warmup == 0 else min(step / warmup, 1.0)
        return (1 - fraction) * group["momentum_start"] + fraction * group["momentum"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            by_shape = {}
            for p in group["params"]:
                if p.grad is not None:
                    by_shape.setdefault((p.shape, p.dtype, p.device), []).append(p)
            for (shape, _, _), params in by_shape.items():
                self._update(group, shape, params)
        return loss

    def _update(self, group, shape, params):
        rows, cols = shape
        states = [self.state[p] for p in params]
        for p, state in zip(params, states):
            if not state:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                state["second_momentum_buffer"] = torch.zeros(
                    (rows, 1) if rows >= cols else (1, cols), dtype=torch.float32, device=p.device)
        mu = self.momentum_at(group, states[0]["step"])
        grads = torch.stack([p.grad.float() for p in params])
        buffers = torch.stack([state["momentum_buffer"] for state in states])
        buffers.lerp_(grads, 1 - mu)
        update = grads.lerp_(buffers, mu)
        update = update / update.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        update = polar_express(update, group["ns_steps"]).float()
        reduce = -1 if rows >= cols else -2
        size = update.size(reduce)
        mean_square = update.square().mean(dim=reduce, keepdim=True)
        norm = (mean_square.sum(dim=(-2, -1), keepdim=True) * size).sqrt()
        second = torch.stack([state["second_momentum_buffer"] for state in states])
        second.lerp_(mean_square, 1 - group["beta2"])
        scale = second.clamp_min(1e-10).rsqrt()
        rescaled = ((mean_square * size) * scale.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
        update = update * (scale * (norm / rescaled.clamp_min(1e-10)))
        lr = group["lr"] * max(1.0, rows / cols) ** 0.5
        for index, (p, state) in enumerate(zip(params, states)):
            state["step"] += 1
            state["momentum_buffer"].copy_(buffers[index])
            state["second_momentum_buffer"].copy_(second[index])
            direction = update[index].to(p.dtype)
            decay = (direction * p >= 0).to(p.dtype) * p
            p.sub_(lr * direction + lr * group["weight_decay"] * decay)
