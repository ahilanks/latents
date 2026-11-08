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


if __name__ == "__main__":
    import json, sys
    for p in sys.argv[1:]:
        import yaml
        c = yaml.safe_load(open(p))["model"]
        f = model_fwd_per_byte(c)
        print(f"{p}: fwd {f/1e6:.2f} MFLOP/byte, train {3*f/1e6:.2f} MFLOP/byte")
