"""Analytic forward FLOPs (Chinchilla-style: 2*params per token + attention/SSD terms).
Train FLOPs = 3 x forward. Used for compute-matching configs and for logging cumulative FLOPs."""


def mamba2_fwd(d, expand=2, d_state=128, headdim=64, ngroups=1, chunk=256, d_conv=4, d_int=0):
    di = expand * d
    nh = di // headdim
    lin = 2 * (d * (2 * di + 2 * ngroups * d_state + nh) + di * d)
    conv = 2 * d_conv * (di + 2 * ngroups * d_state)
    ssd = chunk * (d_state + di) + 4 * di * d_state  # intra-chunk (causal half) + state passing
    mlp = 2 * 3 * d * d_int if d_int else 0
    return lin + conv + ssd + mlp


def attn_fwd(d, mean_ctx, d_int):
    """mean_ctx: average number of keys attended per query (L/2 for causal, ~window for SWA)."""
    lin = 2 * (4 * d * d + 3 * d * d_int)
    attn = 4 * mean_ctx * d
    return lin + attn


def causal_mean_ctx(L, window=-1):
    if window is None or window < 0 or window >= L:
        return L / 2
    # tokens beyond the window attend `window+1` keys
    w = window + 1
    return (w * (w + 1) / 2 + (L - w) * w) / L


def layout_fwd(layout, d, mean_ctx, d_int, ssm=None):
    """layout like 'm4' / 'T9' / 'm2T1'. Lowercase = no MLP."""
    import re
    ssm = ssm or {}
    tot = 0
    for kind, n in re.findall(r"([mMtT])(\d+)", layout):
        n = int(n)
        if kind in "mM":
            tot += n * mamba2_fwd(d, d_int=(d_int if kind == "M" else 0), **ssm)
        else:
            tot += n * attn_fwd(d, mean_ctx, d_int if kind == "T" else 0)
    return tot


def model_fwd_per_byte(cfg, compression=None):
    """Forward FLOPs per *byte* of text.
    cfg: dict with keys arch_layout, d_model, d_intermediate, vocab_size, seq_len, window_size, bytes_per_token.
    compression: L1/L0 (chunks per byte) for 1-stage H-Nets."""
    layout = cfg["arch_layout"]
    V = cfg["vocab_size"]
    L = cfg["seq_len"]
    win = cfg.get("window_size", [-1, -1])
    bpt = cfg.get("bytes_per_token", 1.0)  # >1 only for BPE models
    if len(layout) == 1:  # isotropic
        d, di = cfg["d_model"][0], cfg["d_intermediate"][0]
        per_tok = layout_fwd(layout[0], d, causal_mean_ctx(L, win[0]), di) + 2 * d * V
        return per_tok / bpt
    enc, (main,), dec = layout
    d0, d1 = cfg["d_model"]
    di0, di1 = cfg["d_intermediate"]
    r = compression if compression is not None else 1.0 / cfg.get("N", 6)
    outer = layout_fwd(enc, d0, causal_mean_ctx(L, win[0]), di0) + layout_fwd(dec, d0, causal_mean_ctx(L, win[0]), di0)
    outer += 2 * 2 * d0 * d0  # routing q,k
    outer += 2 * d0 * d0  # residual proj
    outer += 2 * d0 * V  # lm head
    inner = layout_fwd(main, d1, causal_mean_ctx(L * r, win[1]), di1)
    return outer + r * inner


def _per_token(layout, d, mean_ctx, d_int, ssm=None):
    return layout_fwd(layout, d, mean_ctx, d_int, ssm) if layout else 0.0


def _head_fwd(kind, d_in, d_out):
    return 2 * (d_in * 2 * d_in + 2 * d_in * d_out) if kind == "mlp" else 2 * d_in * d_out


def step_train_flops(mc, out, lat=None, nest=None):
    """Train FLOPs of one micro-batch from the token counts this batch actually produced.
    = 3 x forward of everything trained (encoder, decoder, main net, towers, heads, rollout passes)
    + 1 x forward of the EMA target encoders (no backward)."""
    B, L = out.logits.shape[:2]
    layout = mc["arch_layout"]
    ssm = mc.get("ssm_cfg", {})
    sk = {k: ssm[k] for k in ("expand", "d_state", "d_conv") if k in ssm}
    if "chunk_size" in ssm:
        sk["chunk"] = ssm["chunk_size"]
    win = mc.get("window_size", [-1, -1])
    V = mc["vocab_size"]
    if len(layout) == 1:
        d, di = mc["d_model"][0], mc["d_intermediate"][0]
        return 3 * B * L * (layout_fwd(layout[0], d, causal_mean_ctx(L, win[0]), di, sk) + 2 * d * V)
    enc, (main,), dec = layout
    d0, d1 = mc["d_model"]
    di0, di1 = mc["d_intermediate"]
    ctx0 = causal_mean_ctx(L, win[0])
    enc_f = layout_fwd(enc, d0, ctx0, di0, sk)
    outer = enc_f + layout_fwd(dec, d0, ctx0, di0, sk) + 2 * 2 * d0 * d0 + 2 * d0 * d0 + 2 * d0 * V
    T1 = out.chunks.shape[0]
    main_tok = layout_fwd(main, d1, causal_mean_ctx(T1 / B, win[1]), di1, sk)
    fwd = B * L * outer + T1 * main_tok
    ema = 0.0
    if lat is not None:
        hk = lat.cfg.get("head", "linear")
        fwd += T1 * _head_fwd(hk, d1, d0)
        if lat.target == "ema":
            ema += B * L * enc_f
        if lat.rollout is not None:
            fwd += (lat.rollout.k - 1) * T1 * (main_tok + _head_fwd(hk, d1, d0))
    ctx = out.extras.get("levels")
    if nest is not None and ctx is not None:
        nc = nest.nc
        for rec in ctx["levels"]:
            rows = rec["cu"].numel() - 1
            t_in, t = rec["T_in"], rec["T"]
            e = _per_token(nc.get("enc", "m2"), d1, causal_mean_ctx(t_in / rows, -1), di1, sk)
            p_layout = main if nest.tie == "all" else nc.get("pred", "T3")
            p = _per_token(p_layout, d1, causal_mean_ctx(t / rows, -1), di1, sk)
            fwd += t_in * (e + 2 * 2 * d1 * d1 + 2 * d1 * d1) + t * (p + _head_fwd(nc.get("head", "mlp"), d1, d1))
            if lat is not None and lat.target == "ema":
                ema += t_in * e
            if lat is not None and rec["level"] in lat.level_rollouts:
                fwd += (lat.level_rollouts[rec["level"]].k - 1) * t * p
    return 3 * fwd + ema


if __name__ == "__main__":
    import json, sys
    for p in sys.argv[1:]:
        import yaml
        c = yaml.safe_load(open(p))["model"]
        f = model_fwd_per_byte(c)
        print(f"{p}: fwd {f/1e6:.2f} MFLOP/byte, train {3*f/1e6:.2f} MFLOP/byte")
