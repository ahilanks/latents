"""Multi-hop questions in plain English (in the style of ProntoQA / ProsQA, the language tasks of the Coconut paper).

Two chains of made-up categories of the same length are given as shuffled rules; a name belongs to the first category
of one chain; the question asks about the last category of one of the two chains (yes / no, half each). The written
reasoning follows the name's chain one hop per sentence:

    Q: Every wumpus is a gorpus. Every stimpus is a trompus. Every gorpus is a yumpus. Every brompus is a stimpus.
       Max is a wumpus. Is Max a yumpus?
    Max is a wumpus. Max is a gorpus. Max is a yumpus.
    #### yes

A problem with h hops has 2h rules and h + 1 written sentences (the restated fact first, like "a=17." in the
arithmetic task), so the number of written steps is read off the question: count("Every ") / 2 + 1.
Training documents = Q + reasoning + answer, separated by byte 254, as in latents/tasks.py.

usage: python -m latents.tasks_logic --out data/task_logic [--train_bytes 2e8] [--test_steps 2,8]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

SEP = 254
ONSETS = ["w", "y", "z", "d", "r", "n", "t", "v", "j", "g", "l", "st", "br", "sh", "f", "k", "m", "p", "sn", "tr",
          "bl", "gr", "kr", "pl"]
NUCLEI = ["um", "om", "im"]
CONCEPTS = [o + n + "pus" for o in ONSETS for n in NUCLEI]  # 72 made-up categories ("wumpus", "gorpus", ...)
NAMES = ["Max", "Alex", "Sam", "Polly", "Rex", "Fae", "Wren", "Stella", "Sally", "Tom", "Nina", "Leo"]


def problem(rng, steps):
    cs = rng.sample(CONCEPTS, 2 * (steps + 1))
    a, b = cs[: steps + 1], cs[steps + 1:]
    rules = [f"Every {c[i]} is a {c[i + 1]}." for c in (a, b) for i in range(steps)]
    rng.shuffle(rules)
    name = rng.choice(NAMES)
    yes = rng.random() < 0.5
    target = a[-1] if yes else b[-1]
    q = "Q: " + " ".join(rules) + f" {name} is a {a[0]}. Is {name} a {target}?\n"
    cot = " ".join(f"{name} is a {c}." for c in a) + "\n"
    v = "yes" if yes else "no"
    return dict(q=q, cot=cot, answer=f"#### {v}\n", value=v, steps=steps)


def document(p, cot=True):
    return (p["q"] + (p["cot"] if cot else "") + p["answer"]).encode()


def sample(rng, steps=(2, 6)):
    return problem(rng, rng.randint(*steps))


def write(out, train_bytes=2e8, val_bytes=2e6, n_test=2000, train_steps=(2, 6), test_steps=(2, 8), seed=0, cot=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for split, nbytes in (("train", train_bytes), ("val", val_bytes)):
        buf, n = [], 0
        while n < nbytes:
            d = bytes([SEP]) + document(problem(rng, rng.randint(*train_steps)), cot)
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
    ap.add_argument("--out", default="data/task_logic")
    ap.add_argument("--train_bytes", type=float, default=2e8)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--no_cot", action="store_true")
    ap.add_argument("--test_steps", default="2,8")
    a = ap.parse_args()
    write(a.out, a.train_bytes, n_test=a.n_test, cot=not a.no_cot,
          test_steps=tuple(int(v) for v in a.test_steps.split(",")))


if __name__ == "__main__":
    main()
