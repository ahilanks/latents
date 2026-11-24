"""H-Net health numbers (NEXT_STEPS.md "H-Net health during training"), measured at every eval next to the ones
`latents/train.py evaluate` already logs (bits per byte, cut rate, edge precision, space recall, cut-probability
histogram, router rank, cuts after a space, cuts moved).

batch_health (first eval batch):
  main_variation   variation of the main network's outputs between chunks, as a share of their mean square
  main_rank        effective rank (participation ratio) of the main network's outputs
  main_state_norm  size of the main network's internal state right before its final normalisation (normal H-Net ~15)
  main_state_const share of that state that is the same for every chunk
  swap_nats        loss increase (nats per byte) when every chunk's main-network state is swapped for a random other
                   chunk's: how much the decoder uses the main network
  enc_cos_next     cosine similarity of consecutive encoder byte states
  loop only:
  fb_weight        size (Frobenius norm) of the feedback layer's output weights
  fb_share         size of the fed-back vector relative to the chunk input it is added to
  fb_variation     how much the fed-back vector differs between chunks (share of its mean square; ~0 = a constant)
  round_diff       main-network outputs at the trained number of rounds vs twice as many (relative difference)
  round_bpb_diff   bits per byte at twice the rounds minus at the trained rounds (this batch)
  loop_off_bpb     bits per byte with the loop switched off minus with it on (intact text, this batch)
question_cut_stats (reasoning text): cut rate on the question's bytes and on the rest, forced cuts left out.
evaluate_holes: the same validation windows with holes (fixed draw): bits per byte overall and on the text right after
holes, with the loop as trained and switched off; cut rate next to holes vs elsewhere.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from latents.holes import after_hole_targets, holed_window, target_mask
from latents.models import feedback_mode, hole_mark, model_forward

LN2 = math.log(2)


def participation_ratio(x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float() - x.float().mean(0)
        ev = torch.linalg.eigvalsh(x.T @ x / max(x.shape[0] - 1, 1)).clamp(min=0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12))


def _ce(out, y, keep):
    ce = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="none").view(y.shape)
    return ce, float((ce * keep).sum() / keep.sum().clamp(min=1))


def trained_rounds(mc):
    fb = mc.get("feedback") or {}
    return None if fb.get("mode") == "loop" else int(fb.get("passes", 2))


@torch.no_grad()
def batch_health(model, x, y, mc, lat=None, out=None):
    """out is ignored (the forward is rerun to see the main network's state before its final normalisation)."""
    mark = hole_mark(mc)
    keep = target_mask(y, mark)
    cap = {}
    iso = model.backbone.main_network.main_network
    hk = iso.rmsnorm.register_forward_pre_hook(
        lambda mod, args, kw: cap.__setitem__("pre", args[0].float() + (kw.get("residual") if kw.get("residual") is not None
                                                                         else 0)), with_kwargs=True)
    try:
        out = model_forward(model, x, mc, lat)
    finally:
        hk.remove()
    res = {}
    r = cap["pre"].reshape(-1, cap["pre"].shape[-1]).float()
    # the main network's internal state right before its final normalisation: size, and the share that is the same
    # for every chunk (a huge constant there makes the normalised output the same for every chunk)
    res["main_state_norm"] = float(r.norm(dim=-1).mean())
    res["main_state_const"] = float(r.mean(0).pow(2).sum() / r.pow(2).sum(-1).mean().clamp(min=1e-12))
    z = out.z_full.float()
    res["main_variation"] = float((z - z.mean(0)).pow(2).sum(-1).mean() / z.pow(2).sum(-1).mean().clamp(min=1e-12))
    res["main_rank"] = participation_ratio(z[:20000])
    B, L = x.shape
    h = out.enc.float().view(B, L, -1)
    res["enc_cos_next"] = float(F.cosine_similarity(h[:, :-1], h[:, 1:], dim=-1).mean())
    _, ce0 = _ce(out, y, keep)
    _, ce_sw = _ce(model_forward(model, x, mc, lat, extra_opts={"z_perm": 1234}), y, keep)
    res["swap_nats"] = ce_sw - ce0
    fb = getattr(model, "feedback", None)
    if fb is not None:
        last = fb.net if isinstance(fb.net, torch.nn.Linear) else fb.net[-1]
        res["fb_weight"] = float(last.weight.float().norm())
        cu1 = out.cu1.long()
        first = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
        first[cu1[:-1]] = True
        add = fb(torch.roll(out.z_full, 1, 0)).float()[~first]
        pad = model.backbone.main_network.pad_dimension
        inp = out.chunks.float() if pad is None else torch.cat((out.chunks.float(), pad.float().expand(out.chunks.shape[0], -1)), -1)
        res["fb_share"] = float(add.norm(dim=-1).mean() / inp[~first].norm(dim=-1).mean().clamp(min=1e-12))
        # how much the fed-back vector differs between chunks (share of its mean square): near 0 = a constant offset
        res["fb_variation"] = float((add - add.mean(0)).pow(2).sum(-1).mean() / add.pow(2).sum(-1).mean().clamp(min=1e-12))
        k = trained_rounds(mc)
        if k is not None:
            with feedback_mode(model, rounds=2 * k):
                o2 = model_forward(model, x, mc, lat)
            res["round_diff"] = float((o2.z_full.float() - z).norm() / z.norm().clamp(min=1e-12))
            res["round_bpb_diff"] = (_ce(o2, y, keep)[1] - ce0) / LN2
        with feedback_mode(model, rounds=1):
            res["loop_off_bpb"] = (_ce(model_forward(model, x, mc, lat), y, keep)[1] - ce0) / LN2
    return res


def question_mask(x):
    """(B, L) bool: bytes of a question ("Q: ..." up to and including its first newline)."""
    xn = x.detach().cpu().numpy()
    out = np.zeros(xn.shape, dtype=bool)
    for b in range(xn.shape[0]):
        r = xn[b]
        for s in np.flatnonzero((r[:-2] == ord("Q")) & (r[1:-1] == ord(":")) & (r[2:] == ord(" "))):
            if s > 0 and r[s - 1] != 254:
                continue
            nl = np.flatnonzero(r[s:] == 10)
            out[b, s: s + (nl[0] + 1 if len(nl) else len(r) - s)] = True
    return torch.from_numpy(out).to(x.device)


@torch.no_grad()
def question_cut_stats(x, out):
    """Cut rate on the question's bytes and on the rest (forced cuts left out); None if the text has no questions."""
    q = question_mask(x)
    if not bool(q.any()):
        return {}
    bm = out.bpred[0].boundary_mask.view(x.shape)
    free = ~out.extras["forced"].view(x.shape) if "forced" in out.extras else torch.ones_like(bm)
    rest = ~q & (x != 254) & (x != 0)
    return {"cut_rate_question": float((bm & q & free).sum() / (q & free).sum().clamp(min=1)),
            "cut_rate_rest": float((bm & rest & free).sum() / (rest & free).sum().clamp(min=1))}


def holed_val(val, L, n, seed=0, rate=0.1, max_run=3, mark=3):
    """n validation windows of L+1 bytes with holes (the same draw at every eval)."""
    rng = np.random.default_rng(seed)
    step = max(1, (len(val) - 2 * L) // max(n, 1))
    return np.stack([holed_window(val, i * step, L + 1, rng, rate, max_run, mark) for i in range(n)])


@torch.no_grad()
def evaluate_holes(model, mc, wins, mb, autocast, lat=None, prefix="val_holes"):
    """Bits per byte on windows with holes: all targets but markers, and the text right after holes; with the loop as
    trained and switched off. Cut rate next to holes (within 8 bytes) vs elsewhere, forced cuts left out."""
    mark = hole_mark(mc)
    dev = next(model.parameters()).device
    acc = {k: 0.0 for k in ("ce", "n", "ce_after", "n_after", "ce_off", "ce_after_off", "near_c", "near_n", "far_c", "far_n")}
    for i in range(0, len(wins), mb):
        w = torch.from_numpy(wins[i:i + mb].astype(np.int64)).to(dev)
        x, y = w[:, :-1], w[:, 1:]
        keep = target_mask(y, mark)
        aft = after_hole_targets(x, mark) & keep
        for tag, rounds in (("", None), ("_off", 1)):
            with autocast(), feedback_mode(model, rounds=rounds):
                out = model_forward(model, x, mc, lat)
            ce, _ = _ce(out, y, keep)
            acc["ce" + tag] += float((ce * keep).sum())
            acc["ce_after" + tag] += float((ce * aft).sum())
            if tag == "":
                acc["n"] += float(keep.sum())
                acc["n_after"] += float(aft.sum())
                h = x == mark
                near = F.max_pool1d(h.float()[:, None], 17, stride=1, padding=8)[:, 0].bool()
                free = ~out.extras["forced"].view(x.shape) if "forced" in out.extras else torch.ones_like(h)
                bm = out.bpred[0].boundary_mask.view(x.shape)
                acc["near_c"] += float((bm & near & free).sum())
                acc["near_n"] += float((near & free).sum())
                acc["far_c"] += float((bm & ~near & free).sum())
                acc["far_n"] += float((~near & free).sum())
    res = {"bpb": acc["ce"] / LN2 / max(acc["n"], 1), "bpb_after": acc["ce_after"] / LN2 / max(acc["n_after"], 1),
           "bpb_off": acc["ce_off"] / LN2 / max(acc["n"], 1),
           "bpb_after_off": acc["ce_after_off"] / LN2 / max(acc["n_after"], 1),
           "cut_rate_near": acc["near_c"] / max(acc["near_n"], 1), "cut_rate_far": acc["far_c"] / max(acc["far_n"], 1)}
    res["loop_gain_after"] = res["bpb_after_off"] - res["bpb_after"]
    return {f"{prefix}/{k}": v for k, v in res.items()}
