"""Checks after pretraining the hole recipe (NEXT_STEPS.md, plan of 2026-11-23, "Checks after pretraining"), no
training. In the plan's order:
  1 rounds vs the exact loop: on short validation rows (--short bytes), intact and with holes: bits per byte and the
    main network's outputs at 1 (loop off), 4, 8, 16 rounds and with the exact loop. The numerical floor is the same
    comparison with the feedback zeroed (the exact loop and one plain pass then compute the same thing).
  2 does the loop fill holes: bits per byte on the text right after holes (full-length windows with holes, a fixed
    draw), as trained vs the loop switched off; also at 16 rounds.
  3 compression on intact text: full validation set, as trained, 16 rounds, loop off; the plain H-Net's number from
    its summary.
  4 cuts: cut rate, share after a space, edge precision on intact text, recipe vs plain on the same windows; cut rate
    next to holes vs elsewhere; pictures.
  5 is the feedback used: size of the feedback layer's weights and of the fed-back vector; what zeroing it changes.
usage: python scripts/holes_checks.py runs/<recipe> --plain runs/side-p06-plain-s0 [--out logs/holes/checks.json]
"""
import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents import health as HL  # noqa: E402
from latents.holes import holed_window, render_cuts, target_mask  # noqa: E402
from latents.models import feedback_mode, hole_mark, model_forward  # noqa: E402
from show_cuts import load_any  # noqa: E402

LN2 = math.log(2)
WS = (32, 9, 10, 13)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@contextlib.contextmanager
def feedback_zeroed(model):
    last = model.feedback.net if isinstance(model.feedback.net, torch.nn.Linear) else model.feedback.net[-1]
    keep = (last.weight.data.clone(), last.bias.data.clone())
    last.weight.data.zero_()
    last.bias.data.zero_()
    try:
        yield
    finally:
        last.weight.data.copy_(keep[0])
        last.bias.data.copy_(keep[1])


def ac():
    return torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=DEV == "cuda")


@torch.no_grad()
def run(model, mc, wins, mode, mb=16):
    """Bits per byte (marker targets left out) and the main network's outputs for each window batch."""
    ce, n, zs = 0.0, 0.0, []
    for i in range(0, len(wins), mb):
        w = torch.from_numpy(wins[i:i + mb].astype(np.int64)).to(DEV)
        x, y = w[:, :-1], w[:, 1:]
        with ac(), feedback_mode(model, **mode):
            out = model_forward(model, x, mc)
        keep = target_mask(y, hole_mark(mc))
        c = F.cross_entropy(out.logits.float().flatten(0, 1), y.flatten(), reduction="none").view(y.shape)
        ce += float((c * keep).sum())
        n += float(keep.sum())
        zs.append(out.z_full.float())
    return ce / LN2 / n, torch.cat(zs)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def windows(val, L, n, holes, mark, seed=0):
    if holes:
        return HL.holed_val(val, L, n, seed=seed, mark=mark)
    step = (len(val) - L - 1) // n
    return np.stack([val[i * step: i * step + L + 1] for i in range(n)])


@torch.no_grad()
def cut_stats(model, mc, wins, mb=16):
    c = dict(cuts=0.0, n=0.0, after_ws=0.0, edge=0.0)
    masks = []
    for i in range(0, len(wins), mb):
        x = torch.from_numpy(wins[i:i + mb, :-1].astype(np.int64)).to(DEV)
        with ac():
            out = model_forward(model, x, mc)
        b = out.bpred[0].boundary_mask.view(x.shape)
        isws = torch.isin(x, torch.tensor(WS, device=x.device))
        after = F.pad(isws[:, :-1], (1, 0))
        c["cuts"] += float(b.sum())
        c["n"] += b.numel()
        c["after_ws"] += float((b & after).sum())
        c["edge"] += float((b & (isws | after)).sum())
        masks.append(b.cpu())
    return {"cut_rate": c["cuts"] / c["n"], "cuts_after_space": c["after_ws"] / c["cuts"],
            "edge_prec": c["edge"] / c["cuts"]}, torch.cat(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--plain", default="runs/side-p06-plain-s0")
    ap.add_argument("--short", type=int, default=512)
    ap.add_argument("--n_short", type=int, default=128)
    ap.add_argument("--n_long", type=int, default=128)
    ap.add_argument("--full_val", type=int, default=1, help="check 3 on the whole validation set (else n_long windows)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pic_bytes", type=int, default=1200)
    a = ap.parse_args()
    cfg, model, _ = load_any(a.run, DEV)
    mc = cfg["model"]
    mark = hole_mark(mc)
    k = int(mc["feedback"].get("passes", 4))
    val = np.fromfile(ROOT / cfg.get("data_dir", "data/fwe") / "val.u8", dtype=np.uint8)
    L = cfg["train"]["seq_len"]
    res = {"run": a.run, "plain": a.plain, "trained_rounds": k}

    # 1. rounds vs the exact loop, short rows
    c1 = {}
    for tag, holes in (("intact", False), ("holes", True)):
        wins = windows(val, a.short, a.n_short, holes, mark)
        modes = {"off": dict(rounds=1), f"r{k}": dict(rounds=k), "r8": dict(rounds=8), "r16": dict(rounds=16),
                 "exact": dict(mode="loop")}
        out = {m: run(model, mc, wins, md) for m, md in modes.items()}
        ex_bpb, ex_z = out["exact"]
        c1[tag] = {m: {"bpb": b, "bpb_minus_exact": b - ex_bpb, "z_rel_diff_vs_exact": rel(z, ex_z)}
                   for m, (b, z) in out.items()}
        with feedback_zeroed(model):  # numerical floor: these two compute the same function
            fb0, z0 = run(model, mc, wins, dict(mode="loop"))
            fb1, z1 = run(model, mc, wins, dict(rounds=1))
        c1[tag]["floor"] = {"bpb_diff": fb1 - fb0, "z_rel_diff": rel(z1, z0)}
        print(f"[1] {tag}: " + "  ".join(f"{m} bpb {v['bpb']:.4f} (z diff vs exact {v['z_rel_diff_vs_exact']:.4f})"
                                         for m, v in c1[tag].items() if m != "floor")
              + f"  floor z diff {c1[tag]['floor']['z_rel_diff']:.4f}", flush=True)
    res["check1_rounds_vs_exact"] = c1

    # 2. does the loop fill holes (full-length windows with holes)
    hw = HL.holed_val(val, L, a.n_long, seed=0, mark=mark)
    model.eval()
    c2 = {f"r{k}": HL.evaluate_holes(model, mc, hw, 16, ac, prefix="h")}
    with feedback_mode(model, rounds=16):
        c2["r16"] = HL.evaluate_holes(model, mc, hw, 16, ac, prefix="h")
    res["check2_holes"] = c2
    r = c2[f"r{k}"]
    print(f"[2] after holes: loop on {r['h/bpb_after']:.4f}  off {r['h/bpb_after_off']:.4f}  "
          f"(gain {r['h/loop_gain_after']:.4f}); 16 rounds on {c2['r16']['h/bpb_after']:.4f}", flush=True)

    # 3. compression on intact text
    nv = (len(val) - 1) // L if a.full_val else a.n_long
    iw = np.stack([val[i * L: i * L + L + 1] for i in range(nv)])
    c3 = {m: run(model, mc, iw, md)[0] for m, md in ((f"r{k}", dict(rounds=k)), ("r16", dict(rounds=16)),
                                                       ("off", dict(rounds=1)))}
    psum = json.load(open(ROOT / a.plain / "summary.json"))
    c3["plain_summary"] = psum["val/bpb"]
    res["check3_compression"] = c3
    print(f"[3] intact val bpb: as trained {c3[f'r{k}']:.4f}, 16 rounds {c3['r16']:.4f}, loop off {c3['off']:.4f}; "
          f"plain {c3['plain_summary']:.4f}", flush=True)

    # 4. cuts vs plain on the same intact windows; pictures
    pcfg, plain, _ = load_any(a.plain, DEV)
    sw = iw[: a.n_long]
    s_r, m_r = cut_stats(model, mc, sw)
    s_p, m_p = cut_stats(plain, pcfg["model"], sw)
    agree = float((m_r & m_p).sum() / m_p.sum())
    near = {k2: v for k2, v in r.items() if "cut_rate" in k2}
    res["check4_cuts"] = {"recipe": s_r, "plain": s_p, "share_of_plain_cuts_also_cut": agree, "holes": near}
    pic = val[: a.pic_bytes]
    with torch.no_grad(), ac():
        xr = torch.from_numpy(pic.astype(np.int64))[None].to(DEV)
        mr = model_forward(model, xr, mc).bpred[0].boundary_mask.cpu().numpy()
        mp = model_forward(plain, xr, pcfg["model"]).bpred[0].boundary_mask.cpu().numpy()
        ph = holed_window(val, 0, a.pic_bytes, np.random.default_rng(0), 0.1, 3, mark)
        mh = model_forward(model, torch.from_numpy(ph.astype(np.int64))[None].to(DEV), mc).bpred[0].boundary_mask.cpu().numpy()
    res["check4_pictures"] = {"recipe": render_cuts(pic.tobytes(), mr), "plain": render_cuts(pic.tobytes(), mp),
                              "recipe_holes": render_cuts(ph.tobytes(), mh, mark)}
    print(f"[4] cuts: recipe {s_r}  plain {s_p}  share of plain's cuts also cut {agree:.3f}  holes {near}", flush=True)

    # 5. is the feedback used
    hb = HL.batch_health(model, torch.from_numpy(iw[:16, :-1].astype(np.int64)).to(DEV),
                         torch.from_numpy(iw[:16, 1:].astype(np.int64)).to(DEV), mc)
    res["check5_feedback"] = {k2: hb[k2] for k2 in ("fb_weight", "fb_share", "loop_off_bpb", "round_diff", "round_bpb_diff")
                              if k2 in hb}
    res["check5_feedback"]["loop_off_bpb_full_val"] = c3["off"] - c3[f"r{k}"]
    res["check5_feedback"]["loop_off_bpb_after_holes"] = r["h/loop_gain_after"]
    print(f"[5] feedback: {res['check5_feedback']}", flush=True)

    out = ROOT / (a.out or f"logs/holes/checks_{Path(a.run).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
