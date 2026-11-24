"""Health numbers of finished models on the same 16 web validation windows (no training; latents/health.py
batch_health): main-network variation and rank, its internal state's size and constant share, the decoder's use of
the main network, the loop's size and variation.
usage: python scripts/holes_diag.py runs/<a> runs/<b> ... [--out logs/holes/flat_diag.json]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from latents import health as HL  # noqa: E402
from show_cuts import load_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="logs/holes/flat_diag.json")
    a = ap.parse_args()
    val = np.fromfile(ROOT / "data/fwe/val.u8", dtype=np.uint8)
    L = 8192
    w = np.stack([val[i * L: i * L + L + 1] for i in range(16)])
    x = torch.from_numpy(w[:, :-1].astype(np.int64)).cuda()
    y = torch.from_numpy(w[:, 1:].astype(np.int64)).cuda()
    out = ROOT / a.out
    res = json.load(open(out)) if out.exists() else {}
    for r in a.runs:
        cfg, m, _ = load_any(r, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res[Path(r).name] = HL.batch_health(m, x, y, cfg["model"])
        print(r, {k: round(v, 4) for k, v in res[Path(r).name].items()}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
