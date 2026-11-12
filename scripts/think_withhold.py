"""Leak-free read-then-compute test: withhold the last m teacher thoughts and let the model fill them in itself.

The model is trained exactly as before (teacher-forced thoughts, scripts/think_train.py --teacher_only). At test time a
problem whose written reasoning has k chunks gets
  thoughts 1 .. k-m    the true reasoning-chunk latents (read)
  thoughts k-m+1 .. k  the model's own rollouts: head(main-net output at the last position) fed back as the next
                       input (compute; exactly scripts/latent_cot.py's thoughts, started after the teacher prefix)
and the answer is decoded greedily from latents only, as in scripts/latent_cot.py (boundary forced at "####").
m = 0 is the teacher mode (the 97% number); m = k is the full own-rollout ("produce") mode. The total number of
thoughts is always k (the oracle count), as in training.

What a teacher thought can leak: a chunk's latent is the causal encoder state at the chunk's FIRST byte, so the given
thoughts have seen the written reasoning up to the first byte of the last given chunk, and nothing after it.
Fixed 4-byte chunks do not align with the reasoning steps, so for small m the prefix can still hold digits of the
final value. Three ways to withhold, all in one run:
  by_m      fixed m (chunks; m > k uses m = k), with the share of problems whose prefix still holds a digit of the
            final value ("leak")
  by_steps  n = 1, 2, ...: withhold every chunk whose first byte is at or after the start of the n-th last written
            step ("u=57-56=1."), so the model has the written values up to that step and must compute n steps.
            Only problems with at least n steps; leak-free by construction.
  leakfree  per problem, the smallest m whose prefix holds no digit of the final value
Each setting is also run with fill=drop (the withheld thoughts are removed: k-m thoughts, no fill-in), which shows
what the answer pass computes from the prefix on its own. Also reported: the share of answers equal to the last
value fully written in the given prefix (prev: reading, not computing), tens/units-digit accuracy, and the own
rollouts' latent error by rollout index vs the teacher-forced prediction at the same chunk and the copy baseline.

usage: python scripts/think_withhold.py runs/<think run> [--n 2000] [--batch 250]
Writes runs/<run>/think_withhold.json (summary) and think_withhold_rows.jsonl (per problem and setting).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from latents.think import BOS  # noqa: E402
from latents.think_lc import make_thinker  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from think_train import load  # noqa: E402

M_LIST = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, "all"]


def step_spans(cot):
    """Per written step (the restated assignment first): (start, value start, value end, value), offsets in cot."""
    out, i = [], 0
    for tok in cot.rstrip("\n").split(" "):
        eq = tok.rfind("=")
        out.append((i, i + eq + 1, i + len(tok) - 1, int(tok[eq + 1:-1])))
        i += len(tok) + 1
    return out


@torch.no_grad()
def prepare(th, probs):
    """Per problem: question chunk inputs, teacher thoughts (online encoder chunk states = the training inputs),
    EMA target latents of the reasoning chunks (= the training targets), and boundary offsets within the reasoning."""
    if hasattr(th, "prepare_eval"):  # learned-cut models: pooled chunks at the model's own boundaries
        return th.prepare_eval(probs)
    full = [bytes([BOS]) + (p["q"] + p["cot"]).encode() for p in probs]
    x = th.batch(full)
    B, L = x.shape
    h, cu, bp = th.encode(x)
    mask, hv = bp.boundary_mask.view(B, L), h.view(B, L, -1)
    ema = th.lat.ema_states(th.m, x).float()
    if getattr(th.lat, "target_norm", "none") == "ln":
        ema = F.layer_norm(ema, ema.shape[-1:])
    ema = ema.view(B, L, -1)
    ex = []
    for b, p in enumerate(probs):
        q = 1 + len(p["q"].encode())
        r = q + len(p["cot"].encode())
        pos = torch.nonzero(mask[b, :r]).squeeze(1)
        qpos, cpos = pos[pos < q], pos[pos >= q]
        ex.append(dict(qin=hv[b, qpos], teach=hv[b, cpos], tgt=ema[b, cpos], tgt_qlast=ema[b, qpos[-1]],
                       bpos=(cpos - q).tolist(), k=len(cpos), qb=bytes([BOS]) + p["q"].encode(),
                       spans=step_spans(p["cot"]), p=p))
    return ex


@torch.no_grad()
def teacher_preds(th, ex):
    """Teacher-forced prediction of every reasoning chunk (all inputs true): the no-compounding reference."""
    seqs = [torch.cat((th.pad(e["qin"]), th.pad(e["teach"])), 0) for e in ex]
    z, zcu = th.run_main(seqs)
    out = []
    for i, e in enumerate(ex):
        nq = e["qin"].shape[0]
        out.append(getattr(th, "th_head", th.lat.head)(z[zcu[i] + nq - 1: zcu[i] + nq - 1 + e["k"]]).float())
    return out


@torch.no_grad()
def rollout(th, ex, ms):
    """Given the first k-m teacher thoughts, the model rolls out m more. Returns the own thoughts per problem."""
    seqs = [torch.cat((th.pad(e["qin"]), th.pad(e["teach"][:e["k"] - m])), 0) for e, m in zip(ex, ms)]
    own = [[] for _ in ex]
    for t in range(max(ms, default=0)):
        act = [i for i, m in enumerate(ms) if t < m]
        z, zcu = th.run_main([seqs[i] for i in act])
        c = getattr(th, "th_head", th.lat.head)(z[zcu[1:].long() - 1]).to(seqs[0].dtype)
        for j, i in enumerate(act):
            own[i].append(c[j])
            seqs[i] = torch.cat((seqs[i], th.pad(c[j:j + 1])), 0)
    D = ex[0]["teach"].shape[-1]
    return [torch.stack(o) if o else ex[0]["teach"].new_zeros(0, D) for o in own]


@torch.no_grad()
def answers(th, qbs, thoughts, max_len=4):
    """Greedy answer bytes after "#### ", decoded from latents only (scripts/latent_cot.py answer_latent, batched)."""
    ctx = [bytearray(q + b"#### ") for q in qbs]
    done = [False] * len(qbs)
    for _ in range(max_len):
        act = [i for i in range(len(qbs)) if not done[i]]
        if not act:
            break
        lg = th.answer_logits([bytes(ctx[i]) for i in act], [len(qbs[i]) for i in act], [thoughts[i] for i in act])
        for j, i in enumerate(act):
            nxt = int(lg[j, len(ctx[i]) - 1].argmax())
            if nxt == 10:
                done[i] = True
            else:
                ctx[i].append(nxt)
    return [bytes(c[len(q) + 5:]).decode("utf-8", "replace").strip() for c, q in zip(ctx, qbs)]


def visible(e, m):
    """What the given prefix (k-m teacher thoughts) has seen: (holds a final-value digit?, last fully seen value)."""
    given = e["k"] - m
    s_last = e["bpos"][given - 1] if given > 0 else -1  # last byte (cot offset) any given thought has seen
    sp = e["spans"]
    leak = s_last >= sp[-1][1]
    prev = sp[0][3]  # the first value is in the question
    for st, vs, ve, v in sp:
        if ve - 1 <= s_last:
            prev = v
    return leak, prev


def run_setting(th, ex, ms, fill, tf):
    """One withholding setting over a batch of problems. Returns per-problem rows."""
    own = rollout(th, ex, ms) if fill == "own" else [None] * len(ex)
    th_in = []
    for e, m, o in zip(ex, ms, own):
        given = e["teach"][:e["k"] - m]
        th_in.append(torch.cat((given, o.to(given.dtype)), 0) if fill == "own" else given)
    ans = answers(th, [e["qb"] for e in ex], th_in)
    rows = []
    for e, m, o, a, ptf in zip(ex, ms, own, ans, tf):
        v = e["p"]["value"]
        leak, prev = visible(e, m)
        ai = int(a) if a.isdigit() else None
        row = dict(id=e["id"], steps=e["p"]["steps"], k=e["k"], m=m, ans=a, ok=a == str(v), leak=leak, prev=prev,
                   is_prev=(ai == prev and prev != v), tens=ai is not None and ai // 10 == v // 10,
                   units=ai is not None and ai % 10 == v % 10)
        if fill == "own" and m > 0:
            g = e["k"] - m
            tgt = e["tgt"][g:]
            prv = torch.cat((e["tgt_qlast"][None], e["tgt"]), 0)[g:g + m]  # copy baseline: the previous chunk's latent
            of = o.float()
            row["l1_own"] = (of - tgt).abs().mean(-1).tolist()
            row["l1_tf"] = (ptf[g:] - tgt).abs().mean(-1).tolist()
            row["l1_copy"] = (prv - tgt).abs().mean(-1).tolist()
            row["cos_own"] = F.cosine_similarity(of, tgt, dim=-1).tolist()
        rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    if n == 0:
        return dict(n=0)
    mean = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(r["steps"], []).append(r["ok"])
    s = dict(n=n, acc=mean("ok"), acc_by_steps={k: sum(v) / len(v) for k, v in sorted(by.items())},
             mean_m=mean("m"), leak_share=mean("leak"), prev_share=mean("is_prev"), tens_acc=mean("tens"),
             units_acc=mean("units"))
    if any("l1_own" in r for r in rows):
        agg = {}
        for r in rows:
            for t in range(len(r.get("l1_own", []))):
                a = agg.setdefault(t + 1, dict(own=0.0, tf=0.0, copy=0.0, cos=0.0, n=0))
                a["own"] += r["l1_own"][t]
                a["tf"] += r["l1_tf"][t]
                a["copy"] += r["l1_copy"][t]
                a["cos"] += r["cos_own"][t]
                a["n"] += 1
        s["latent_by_rollout_index"] = {t: dict(l1_own=a["own"] / a["n"], l1_tf=a["tf"] / a["n"],
                                                l1_copy=a["copy"] / a["n"], cos_own=a["cos"] / a["n"], n=a["n"])
                                        for t, a in sorted(agg.items())}
    return s


def settings(e):
    """All (name, m) withholding settings for one problem; None = the setting doesn't apply to it."""
    k, sp = e["k"], e["spans"]
    out = {}
    for m in M_LIST:
        out[f"m={m}"] = k if m == "all" else min(m, k)
    for nst in range(1, 13):
        out[f"steps={nst}"] = sum(1 for b in e["bpos"] if b >= sp[-nst][0]) if nst <= e["p"]["steps"] else None
    out["leakfree"] = sum(1 for b in e["bpos"] if b >= sp[-1][1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--only", default=None, help="comma list of setting names (debug), e.g. m=0,m=all")
    ap.add_argument("--data_dir", default=None, help="test set dir (default: the run's data_dir)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    cfg, model, lat = load(run, device)
    model.eval()
    th = make_thinker(model, lat, cfg["model"])
    probs = [json.loads(l) for l in open(ROOT / (a.data_dir or cfg.get("data_dir", "data/task_arith")) / "test.jsonl")][:a.n]
    ac = dict(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda")
    t0 = time.time()
    rows = {}  # (setting, fill) -> rows
    for i in range(0, len(probs), a.batch):
        with torch.autocast(**ac):
            ex = prepare(th, probs[i:i + a.batch])
            for j, e in enumerate(ex):
                e["id"] = i + j
            tf = teacher_preds(th, ex)
            per = [settings(e) for e in ex]
            names = list(per[0])
            if a.only:
                names = [nm for nm in names if nm in a.only.split(",")]
            for nm in names:
                idx = [j for j, s in enumerate(per) if s[nm] is not None]
                if not idx:
                    continue
                for fill in ("own", "drop"):
                    if fill == "drop" and nm == "m=0":
                        continue
                    rs = run_setting(th, [ex[j] for j in idx], [per[j][nm] for j in idx], fill, [tf[j] for j in idx])
                    rows.setdefault((nm, fill), []).extend(rs)
        print(f"[{time.time() - t0:.0f}s] {min(i + a.batch, len(probs))}/{len(probs)} problems", flush=True)
    res = dict(model=run.name, n=len(probs), settings={})
    for (nm, fill), rs in rows.items():
        res["settings"].setdefault(nm, {})[fill] = summarize(rs)
    for nm, d in res["settings"].items():
        print(f"{nm:10s} own {d['own']['acc']:.3f} (n={d['own']['n']}, mean m {d['own']['mean_m']:.2f}, "
              f"leak {d['own']['leak_share']:.2f}, prev {d['own']['prev_share']:.3f})"
              + (f"   drop {d['drop']['acc']:.3f}" if "drop" in d else ""), flush=True)
    suffix = (f"_{a.tag}" if a.tag else "") + ("_debug" if a.only else "")
    json.dump(res, open(run / f"think_withhold{suffix}.json", "w"), indent=1)
    with open(run / f"think_withhold_rows{suffix}.jsonl", "w") as f:
        for (nm, fill), rs in rows.items():
            for r in rs:
                f.write(json.dumps(dict(setting=nm, fill=fill) | r) + "\n")
    print(f"wrote {run / f'think_withhold{suffix}.json'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
