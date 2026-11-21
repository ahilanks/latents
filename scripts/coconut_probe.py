"""What do a Coconut model's thoughts hold? Linear probes on each thought for the digits of each hidden value.

For a run at stage s (problems with more than s written steps, so every problem has exactly c*s thoughts): value k is
the result of written step k (value 0 = the restated first value, value s-1 = the number the model needs when it starts
writing). For every thought and every value: a 10-way linear probe for the ones digit and one for the tens digit,
fitted on --n_train problems and scored on --n_test others (chance 10%).
usage: python scripts/coconut_probe.py runs/<name> --stage 3 [--n_train 6000] [--n_test 1500] [--mlp]
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents import tasks  # noqa: E402
from latents.coconut import Coconut, example  # noqa: E402
from latents.think_lc import step_spans  # noqa: E402
from show_cuts import load_any  # noqa: E402


def fit_probe(xtr, ytr, xte, yte, steps=300, mlp=False):
    """10-way probe: linear, or (mlp) one hidden layer of 512 units trained longer (is it there, but not linearly?)."""
    mu, sd = xtr.mean(0), xtr.std(0).clamp(min=1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    torch.manual_seed(0)
    if mlp:
        net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 10)).to(xtr.device)
        opt, steps = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4), 1500
    else:
        net = torch.nn.Linear(xtr.shape[1], 10).to(xtr.device)
        torch.nn.init.zeros_(net.weight)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(net(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((net(xte).argmax(-1) == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mlp", action="store_true", help="one-hidden-layer probes instead of linear ones")
    a = ap.parse_args()
    dev = "cuda"
    cfg, model, _ = load_any(a.run, dev)
    cc = cfg.get("coconut", {})
    co = Coconut(model, cfg["model"], c=cc.get("c", 1), feed=cc.get("feed", "full"), read=cc.get("read", "stop"),
                 fresh=cc.get("fresh", 0), ground=cc.get("ground", 0.0))
    rng = random.Random(77)
    probs = []
    while len(probs) < a.n_train + a.n_test:
        p = tasks.sample(rng, steps=(2, 6))
        if len(step_spans(p["cot"])) > a.stage:
            probs.append(p)
    T, V = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, len(probs), 250):
            ps = probs[i:i + 250]
            exs = [example(p, a.stage, co.c) for p in ps]
            _, _, _, th = co.forward([e[0] for e in exs], [e[1] for e in exs], [e[3] for e in exs])
            T.append(torch.stack(th).float())  # (B, n_thoughts, D1)
            V.append(torch.tensor([[s[3] for s in step_spans(p["cot"])[: a.stage]] for p in ps]))
    T, V = torch.cat(T), torch.cat(V).to(dev)
    ntr = a.n_train
    res = {}
    print(f"{a.run}, stage {a.stage}: {T.shape[1]} thoughts, {a.stage} hidden values; {'MLP' if a.mlp else 'linear'} probe accuracy ones / tens digit (chance 0.10)")
    print("thought".ljust(10) + "".join(f"value {k}".ljust(16) for k in range(a.stage)))
    for t in range(T.shape[1]):
        row = []
        for k in range(a.stage):
            u = fit_probe(T[:ntr, t], V[:ntr, k] % 10, T[ntr:, t], V[ntr:, k] % 10, mlp=a.mlp)
            te = fit_probe(T[:ntr, t], V[:ntr, k] // 10, T[ntr:, t], V[ntr:, k] // 10, mlp=a.mlp)
            res[f"thought{t + 1}/value{k}"] = dict(ones=u, tens=te)
            row.append(f"{u:.2f} / {te:.2f}")
        print(f"T{t + 1}".ljust(10) + "".join(r.ljust(16) for r in row), flush=True)
    if a.out:
        Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
