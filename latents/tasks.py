"""Synthetic multi-step arithmetic for the latent chain-of-thought test (ladder phase 6).

A problem is a chain of variable updates mod 100; the chain of thought writes out every intermediate value:

    Q: a=17; b=a+25; c=b*3; d=c-40; d=?
    a=17. b=17+25=42. c=42*3=26. d=26-40=86.
    #### 86

Training documents = Q + CoT + answer (byte stream, documents separated by byte 254 like FineWeb-Edu), so the
latent model learns to predict the chunk latents of the chain of thought. The test file keeps the parts apart.

usage: python -m latents.tasks --out data/task_arith [--train_bytes 2e8]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

SEP = 254
OPS = ("+", "-", "*")
NAMES = "abcdefghijklmnopqrstuvwxyz"


def problem(rng, steps):
    names = rng.sample(NAMES, steps + 1)
    v = rng.randrange(100)
    q = [f"{names[0]}={v}"]
    cot = [f"{names[0]}={v}."]
    for i in range(1, steps + 1):
        op = rng.choice(OPS)
        n = rng.randrange(2, 10) if op == "*" else rng.randrange(1, 100)
        prev = v
        v = (prev + n) % 100 if op == "+" else (prev - n) % 100 if op == "-" else (prev * n) % 100
        q.append(f"{names[i]}={names[i - 1]}{op}{n}")
        cot.append(f"{names[i]}={prev}{op}{n}={v}.")
    return dict(q="Q: " + "; ".join(q) + f"; {names[-1]}=?\n", cot=" ".join(cot) + "\n", answer=f"#### {v}\n",
                value=v, steps=steps)


def document(p):
    return (p["q"] + p["cot"] + p["answer"]).encode()


def write(out, train_bytes=2e8, val_bytes=2e6, n_test=2000, train_steps=(2, 6), test_steps=(2, 8), seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for split, nbytes in (("train", train_bytes), ("val", val_bytes)):
        buf, n = [], 0
        while n < nbytes:
            d = bytes([SEP]) + document(problem(rng, rng.randint(*train_steps)))
            buf.append(d)
            n += len(d)
        np.frombuffer(b"".join(buf), dtype=np.uint8).tofile(out / f"{split}.u8")
    trng = random.Random(seed + 1)
    with open(out / "test.jsonl", "w") as f:
        for i in range(n_test):
            f.write(json.dumps(problem(trng, test_steps[0] + i % (test_steps[1] - test_steps[0] + 1))) + "\n")
    print(f"wrote {out}: train {train_bytes:.2g} bytes, val {val_bytes:.2g}, {n_test} test problems")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/task_arith")
    ap.add_argument("--train_bytes", type=float, default=2e8)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    write(a.out, a.train_bytes, n_test=a.n_test)


if __name__ == "__main__":
    main()
