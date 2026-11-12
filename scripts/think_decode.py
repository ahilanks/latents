"""What do the model's own thoughts say, and can it decide by itself when to stop thinking? (evaluation only)

decode   A thought is a predicted chunk latent. The byte decoder reads it exactly as in pretraining: the bytes of chunk c
         read the predicted latent of chunk c+1 (fixed 4-byte chunks: no dechunk smoothing). Greedy byte generation
         with the decoder's sources fixed to the rollout's predictions spells out what the latents encode. The main
         network never sees the decoded bytes, so the rollout stays latent; this is a readout only.
         Each decoded chain is parsed step by step ("c=42*3=26.") and its first deviation is classified:
           stop_early  "####" before all steps     name      wrong variable      copy   wrong carried value
           operand     wrong operator/number       arith     wrong result of its own expression
           format      unparseable                 no_stop   no "####" within the rollout
halting  Roll out until the decoded text reaches "####" (cap --max_k thoughts). The thought count k' = the 4-byte
         chunks before "####" is then the model's own choice, and the answer is decoded from latents as usual with
         those k' thoughts. Compared with the oracle count k (scripts/think_withhold.py).
Settings: from the question alone (all thoughts own) and with the teacher prefix up to the n-th last step (steps=n).
Only fixed-chunk models (pool_k): learned boundaries would need the router run on the decoded bytes.

usage: python scripts/think_decode.py runs/<think run> [--n 2000] [--batch 250] [--max_k 48] [--data_dir DIR]
Writes runs/<run>/think_decode[_<tag>].json and think_decode_rows[_<tag>].jsonl (decoded texts per problem).
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents.think import Thinker  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from think_train import load  # noqa: E402
from think_withhold import answers, prepare  # noqa: E402

STEP_RE = re.compile(r"^([a-z])=(\d+)([+\-*])(\d+)=(\d+)\.$")
INIT_RE = re.compile(r"^([a-z])=(\d+)\.$")


@torch.no_grad()
def chain(th, ex, gs, K):
    """Per problem: decoder sources for every chunk from the first one, i.e. head(main-net output) at each position:
    question chunks (teacher-forced), the given teacher prefix of g reasoning chunks (teacher-forced), then K own
    rollout predictions. Row c = the predicted latent of chunk c+1. Also returns the own predictions (thought inputs)."""
    seqs = [torch.cat((th.pad(e["qin"]), th.pad(e["teach"][:g])), 0) for e, g in zip(ex, gs)]
    z, zcu = th.run_main(seqs)
    src = [th.lat.head(z[zcu[i]: zcu[i + 1]]).to(seqs[0].dtype) for i in range(len(ex))]  # nq + g rows
    own = [[] for _ in ex]
    for t in range(K):
        if t > 0:
            z, zcu = th.run_main(seqs)
        c = th.lat.head(z[zcu[1:].long() - 1]).to(seqs[0].dtype)
        for i in range(len(ex)):
            if t > 0:
                src[i] = torch.cat((src[i], c[i:i + 1]), 0)
            own[i].append(c[i])
            seqs[i] = torch.cat((seqs[i], th.pad(c[i:i + 1])), 0)
    return src, [torch.stack(o) for o in own]


@torch.no_grad()
def decode(th, ex, src, stop_after=8):
    """Greedy bytes from the end of the question on, the decoder reading src[c] for chunk c. Stops stop_after bytes
    after "####" (or when the sources run out). Returns the decoded text after the question."""
    k_ = th.mc["pool_k"]
    ctx = [bytearray(e["qb"]) for e in ex]
    q0 = [len(e["qb"]) for e in ex]
    limit = [k_ * (s.shape[0] - 1) + k_ + 1 for s in src]  # last chunk with a source predicts through its next boundary
    done = [len(c) >= lim for c, lim in zip(ctx, limit)]
    D = src[0].shape[-1]
    while not all(done):
        act = [i for i in range(len(ex)) if not done[i]]
        x = th.batch([bytes(ctx[i]) for i in act])
        B, L = x.shape
        h, cu, bp = th.encode(x)
        n_ch = L // k_
        S = h.new_zeros(B * n_ch, D)
        for j, i in enumerate(act):
            r = min(n_ch, src[i].shape[0])
            S[j * n_ch: j * n_ch + r] = src[i][:r].to(S.dtype)
        cu1 = torch.arange(B + 1, device=x.device, dtype=torch.int32) * n_ch
        lg = th._decode(x, h, bp.boundary_mask, bp.boundary_prob, bp.selected_probs, cu, cu1, S)
        for j, i in enumerate(act):
            ctx[i].append(int(lg[j, len(ctx[i]) - 1].argmax()))
            txt = bytes(ctx[i][q0[i]:])
            p = txt.find(b"####")
            if len(ctx[i]) >= limit[i] or (p >= 0 and len(txt) >= p + 4 + stop_after):
                done[i] = True
    return [bytes(c[q:]).decode("utf-8", "replace") for c, q in zip(ctx, q0)]


def parse_question(q):
    """Truth from the question: [(name, op, n)] per update, initial (name, value)."""
    parts = q.strip()[3:].split("; ")
    n0, v0 = parts[0].split("=")
    ups = []
    for p in parts[1:-1]:
        name, rhs = p.split("=")
        ups.append((name, rhs[1], int(rhs[2:])))
    return (n0, int(v0)), ups


def apply(op, a, n):
    return (a + n) % 100 if op == "+" else (a - n) % 100 if op == "-" else (a * n) % 100


def grade(p, text):
    """Parse the decoded chain against the question. Returns dict(first_error, steps_ok, n_steps_written, final)."""
    (n0, v0), ups = parse_question(p["q"])
    body = text.split("####")[0]
    toks = body.replace("\n", " ").split()
    stopped = "####" in text
    res = dict(steps_ok=0, n_written=0, first_error=None, err_step=None, final_written=None)
    if not toks or not INIT_RE.match(toks[0]) or INIT_RE.match(toks[0]).groups() != (n0, str(v0)):
        res["first_error"], res["err_step"] = "format", 0
        return res
    prev = v0
    for s, (name, op, n) in enumerate(ups, 1):
        if s >= len(toks):
            res["first_error"], res["err_step"] = ("stop_early" if stopped else "no_stop"), s
            return res
        m = STEP_RE.match(toks[s])
        if not m:
            res["first_error"], res["err_step"] = "format", s
            return res
        nm, a, o, b, c = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        res["n_written"] = s
        res["final_written"] = c
        err = ("name" if nm != name else "copy" if a != prev else "operand" if (o, b) != (op, n)
               else "arith" if c != apply(o, a, b) else None)
        if err:
            res["first_error"], res["err_step"] = err, s
            return res
        res["steps_ok"] = s
        prev = c
    if len(toks) > len(ups) + 1:
        res["first_error"], res["err_step"] = "stop_late", len(ups) + 1
    elif not stopped:
        res["first_error"], res["err_step"] = "no_stop", len(ups) + 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--max_k", type=int, default=48, help="rollout cap for self-halting (thoughts)")
    ap.add_argument("--data_dir", default=None, help="test set dir (default: the run's data_dir)")
    ap.add_argument("--settings", default="all,steps=1,steps=2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = ROOT / a.run if not Path(a.run).is_absolute() else Path(a.run)
    cfg, model, lat = load(run, device)
    model.eval()
    th = Thinker(model, lat, cfg["model"])
    assert th.mc.get("pool_k"), "fixed-chunk models only"
    kq = th.mc["pool_k"]
    data_dir = a.data_dir or cfg.get("data_dir", "data/task_arith")
    probs = [json.loads(l) for l in open(ROOT / data_dir / "test.jsonl")][:a.n]
    ac = dict(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16, enabled=device == "cuda")
    t0 = time.time()
    rows = []
    for i0 in range(0, len(probs), a.batch):
        with torch.autocast(**ac):
            ex = prepare(th, probs[i0:i0 + a.batch])
            for j, e in enumerate(ex):
                e["id"] = i0 + j
            for st in a.settings.split(","):
                if st == "all":
                    sel = list(range(len(ex)))
                    gs = [0] * len(ex)
                else:
                    n = int(st.split("=")[1])
                    sel = [j for j, e in enumerate(ex) if n <= e["p"]["steps"]]
                    gs = [sum(1 for b in ex[j]["bpos"] if b < ex[j]["spans"][-n][0]) for j in sel]
                if not sel:
                    continue
                E = [ex[j] for j in sel]
                K = a.max_k  # own thoughts after the teacher prefix
                src, own = chain(th, E, gs, K)
                texts = decode(th, E, src)
                # self-halting: k' = chunk boundaries in [q, position of "####")
                kself, th_in, stopped = [], [], []
                for e, g, txt, o in zip(E, gs, texts, own):
                    q = len(e["qb"])
                    p = txt.find("####")
                    stopped.append(p >= 0)
                    r = q + (p if p >= 0 else 10 ** 9)
                    k2 = sum(1 for pos in range(q, min(r, q + kq * (g + K))) if pos % kq == 0)
                    k2 = max(g, min(k2, g + K))
                    kself.append(k2)
                    th_in.append(torch.cat((e["teach"][:g], o[:k2 - g].to(e["teach"].dtype)), 0))
                ans = answers(th, [e["qb"] for e in E], th_in)
                for e, g, txt, k2, an, sp in zip(E, gs, texts, kself, ans, stopped):
                    gr = grade(e["p"], txt)
                    rows.append(dict(setting=st, id=e["id"] if "id" in e else None, steps=e["p"]["steps"], k=e["k"],
                                     given=g, k_self=k2, stopped=sp, ans_self=an, ok_self=an == str(e["p"]["value"]),
                                     decoded=txt, decoded_final_ok=gr["final_written"] == e["p"]["value"]
                                     and gr["steps_ok"] == e["p"]["steps"], **gr))
        print(f"[{time.time() - t0:.0f}s] {min(i0 + a.batch, len(probs))}/{len(probs)}", flush=True)
    # summary
    res = dict(model=run.name, data_dir=data_dir, n=len(probs), settings={})
    for st in a.settings.split(","):
        rs = [r for r in rows if r["setting"] == st]
        if not rs:
            continue
        by = {}
        for r in rs:
            by.setdefault(r["steps"], []).append(r)
        summ = lambda R: dict(  # noqa: E731
            n=len(R), acc_self_halt=sum(r["ok_self"] for r in R) / len(R),
            chain_all_correct=sum(r["first_error"] is None for r in R) / len(R),
            decoded_final_correct=sum(r["decoded_final_ok"] for r in R) / len(R),
            stopped=sum(r["stopped"] for r in R) / len(R),
            k_self_minus_k=sum(r["k_self"] - r["k"] for r in R) / len(R),
            k_exact=sum(r["k_self"] == r["k"] for r in R) / len(R),
            first_error=dict(Counter(r["first_error"] or "none" for r in R).most_common()))
        res["settings"][st] = summ(rs) | {"by_steps": {s: summ(R) for s, R in sorted(by.items())}}
        s = res["settings"][st]
        print(f"{st:8s} self-halt acc {s['acc_self_halt']:.3f}  chain all-correct {s['chain_all_correct']:.3f}  "
              f"stopped {s['stopped']:.2f}  k'=k {s['k_exact']:.2f}  first errors {s['first_error']}", flush=True)
        for stp, d in s["by_steps"].items():
            print(f"   {stp} steps: acc {d['acc_self_halt']:.3f} chain {d['chain_all_correct']:.3f} k'=k {d['k_exact']:.2f} "
                  f"errors {d['first_error']}", flush=True)
    tag = f"_{a.tag}" if a.tag else ""
    json.dump(res, open(run / f"think_decode{tag}.json", "w"), indent=1)
    with open(run / f"think_decode_rows{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {run / f'think_decode{tag}.json'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
