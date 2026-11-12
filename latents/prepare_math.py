"""Pretraining data with math in it: FineWeb-Edu + FineMath + synthetic arithmetic chains, mixed by bytes.

Sources
  web    data/fwe/train.u8, already prepared by latents.prepare_data (documents start with byte 254)
  math   FineMath (HuggingFaceTB/finemath, finemath-4plus), streamed from the Hub; the first --val_bytes go to
         val_math.u8 and never to training
  arith  latents/tasks.py problems with their written reasoning (seed 3; the reasoning test set uses seed 1,
         the reasoning-stage training problems seeds 1000+)
Mixing: every source is cut into blocks of whole documents (~1 MiB). Blocks are drawn with probability proportional
to the bytes each source still owes, so the mix holds all through the file and random 8 KB training windows see it
everywhere.
Outputs (in --out): train.u8, val.u8 (a copy of the web val set, so web BPB stays comparable with every earlier run),
val_math.u8, val_arith.u8, meta.json.

usage: python -m latents.prepare_math --out data/fwe_math [--total 3.4e9] [--frac 0.75,0.20,0.05]
       (tests: --math_jsonl <file with {"text": ...} lines> instead of the Hub)
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from latents import tasks

SEP = 254
BLOCK = 1 << 20


def web_blocks(path, block=BLOCK):
    """~block-sized byte blocks of the prepared web stream, each ending right before a document start."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    i, n = 0, len(data)
    while i < n:
        j = min(n, i + block)
        if j < n:
            nxt = np.flatnonzero(data[j: j + 4 * block] == SEP)
            j = j + int(nxt[0]) if len(nxt) else n
        yield bytes(data[i:j])
        i = j


def doc_blocks(docs, block=BLOCK):
    """Group documents (bytes, each starting with SEP) into ~block-sized blocks."""
    buf, n = [], 0
    for d in docs:
        buf.append(d)
        n += len(d)
        if n >= block:
            yield b"".join(buf)
            buf, n = [], 0
    if buf:
        yield b"".join(buf)


def math_texts(math_jsonl=None, config="finemath-4plus"):
    if math_jsonl:
        for line in open(math_jsonl):
            yield json.loads(line)["text"]
        return
    from datasets import load_dataset
    for ex in load_dataset("HuggingFaceTB/finemath", config, split="train", streaming=True):
        if ex.get("text"):
            yield ex["text"]


def arith_docs(seed):
    rng = random.Random(seed)
    while True:
        yield bytes([SEP]) + tasks.document(tasks.problem(rng, rng.randint(2, 6)))


def write(out, total=3.4e9, frac=(0.75, 0.20, 0.05), web="data/fwe", math_jsonl=None, val_bytes=2_000_000,
          arith_val_bytes=1_000_000, seed=0, block=BLOCK):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    web = Path(web)
    shutil.copy(web / "val.u8", out / "val.u8")
    # math: the first val_bytes of the stream are held out
    mdocs = (bytes([SEP]) + t.encode("utf-8") for t in math_texts(math_jsonl))
    held, n = [], 0
    for d in mdocs:
        held.append(d)
        n += len(d)
        if n >= val_bytes:
            break
    np.frombuffer(b"".join(held), dtype=np.uint8).tofile(out / "val_math.u8")
    av, n = [], 0
    for d in arith_docs(seed + 4):
        av.append(d)
        n += len(d)
        if n >= arith_val_bytes:
            break
    np.frombuffer(b"".join(av), dtype=np.uint8).tofile(out / "val_arith.u8")
    gens = {"web": web_blocks(web / "train.u8", block), "math": doc_blocks(mdocs, block),
            "arith": doc_blocks(arith_docs(seed + 3), block)}
    owed = {k: f * total for k, f in zip(("web", "math", "arith"), frac) if f > 0}
    got = {k: 0 for k in owed}
    rng = np.random.default_rng(seed)
    with open(out / "train.u8", "wb") as fo:
        while owed:
            keys = list(owed)
            p = np.array([owed[k] for k in keys])
            k = keys[rng.choice(len(keys), p=p / p.sum())]
            try:
                b = next(gens[k])
            except StopIteration:
                raise SystemExit(f"source {k} ran out after {got[k] / 1e6:.1f}MB (wanted {(owed[k] + got[k]) / 1e6:.1f}MB): "
                                 "lower --total or give it more data")
            fo.write(b)
            got[k] += len(b)
            owed[k] -= len(b)
            if owed[k] <= 0:
                del owed[k]
    meta = dict(total=sum(got.values()), bytes=got, frac={k: v / max(sum(got.values()), 1) for k, v in got.items()},
                target_frac=dict(zip(("web", "math", "arith"), frac)), val_math=(out / "val_math.u8").stat().st_size,
                val_arith=(out / "val_arith.u8").stat().st_size, seed=seed)
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fwe_math")
    ap.add_argument("--total", type=float, default=3.4e9)
    ap.add_argument("--frac", default="0.75,0.20,0.05", help="web,math,arith shares of the training bytes")
    ap.add_argument("--web", default="data/fwe")
    ap.add_argument("--math_jsonl", default=None)
    ap.add_argument("--val_bytes", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write(a.out, a.total, tuple(float(v) for v in a.frac.split(",")), a.web, a.math_jsonl, a.val_bytes, seed=a.seed)


if __name__ == "__main__":
    main()
