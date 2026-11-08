"""Tokenize FineWeb-Edu sample-10BT shards into flat byte (uint8) and GPT-2 BPE (uint16) streams.

Each document is prefixed with a separator: byte 254 (H-Net ByteTokenizer BOS) for the byte stream,
<|endoftext|> (50256) for the BPE stream. Both streams are built from the *same* documents so BPB is
directly comparable.

Outputs (in --out):
  train.u8 / train.gpt2.u16      train streams (shards 000-002)
  val.u8   / val.gpt2.u16        fixed val set: first docs of shard 013 up to --val_bytes
  gpt2_token_bytes.u16           byte length of every GPT-2 token (for exact BPB of BPE models)
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

BYTE_SEP = 254
_enc = None


def _get_enc():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def encode_rowgroup(args):
    path, rg = args
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"]).column("text").to_pylist()
    return encode_texts(texts)


def encode_texts(texts):
    enc = _get_enc()
    eot = enc.eot_token
    b_parts, t_parts = [], []
    for t in texts:
        b = t.encode("utf-8")
        b_parts.append(bytes([BYTE_SEP]) + b)
        t_parts.append([eot] + enc.encode_ordinary(t))
    bytes_arr = np.frombuffer(b"".join(b_parts), dtype=np.uint8)
    tok_arr = np.fromiter((x for ts in t_parts for x in ts), dtype=np.uint16)
    return bytes_arr, tok_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/sample/10BT")
    ap.add_argument("--out", default="data/fwe")
    ap.add_argument("--train_shards", default="000,001,002")
    ap.add_argument("--val_shard", default="013")
    ap.add_argument("--val_bytes", type=int, default=8_000_000)
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # token byte lengths (for exact BPB on BPE models); eot counts as 1 byte (the separator)
    enc = _get_enc()
    lens = np.array([len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab - 1)] + [1], dtype=np.uint16)
    lens.tofile(out / "gpt2_token_bytes.u16")

    # val: first docs of the val shard until val_bytes
    pf = pq.ParquetFile(f"{a.raw}/{a.val_shard}_00000.parquet")
    texts, nb = [], 0
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            texts.append(t)
            nb += len(t.encode("utf-8")) + 1
            if nb >= a.val_bytes:
                break
        if nb >= a.val_bytes:
            break
    vb, vt = encode_texts(texts)
    vb.tofile(out / "val.u8")
    vt.tofile(out / "val.gpt2.u16")
    print(f"val: {len(texts)} docs, {len(vb)/1e6:.2f}MB bytes, {len(vt)/1e6:.2f}M tokens, {len(vb)/len(vt):.3f} bytes/token")

    # train
    jobs = []
    for s in a.train_shards.split(","):
        p = f"{a.raw}/{s}_00000.parquet"
        jobs += [(p, rg) for rg in range(pq.ParquetFile(p).num_row_groups)]
    with open(out / "train.u8", "wb") as fb, open(out / "train.gpt2.u16", "wb") as ft, Pool(a.workers) as pool:
        nb = nt = 0
        for i, (b, t) in enumerate(pool.imap(encode_rowgroup, jobs, chunksize=1)):
            fb.write(b.tobytes())
            ft.write(t.tobytes())
            nb += len(b)
            nt += len(t)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} rowgroups, {nb/1e9:.2f}GB bytes, {nt/1e9:.3f}B tokens", flush=True)
    print(f"train: {nb/1e9:.3f}B bytes, {nt/1e9:.3f}B tokens, {nb/nt:.3f} bytes/token")


if __name__ == "__main__":
    main()
