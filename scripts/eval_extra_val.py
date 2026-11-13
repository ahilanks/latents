"""Validation BPB and cut statistics of finished runs on the extra (math) validation sets of a data dir, so models
trained before those sets existed can be compared with the math-pretrained ones.

usage: python scripts/eval_extra_val.py runs/<run> [runs/<run> ...] [--data data/fwe_math] [--windows 64]
Writes runs/<run>/extra_val.json and prints a table.
"""
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
from latents.models import build_model  # noqa: E402

from latents import train as T  # noqa: E402
from think_train import load  # noqa: E402

SETS = {"val": "val.u8", "val_math": "val_math.u8", "val_arith": "val_arith.u8"}
KEYS = ("bpb", "compression_L1/L0", "edge_prec", "space_recall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/fwe_math")
    ap.add_argument("--windows", type=int, default=64)
    a = ap.parse_args()
    dd = ROOT / a.data
    rows = {}
    for run in a.runs:
        rd = ROOT / run if not Path(run).is_absolute() else Path(run)
        ck = torch.load(rd / "final.pt", map_location="cpu", weights_only=False)
        if "latent" in ck["cfg"]:
            cfg, model, lat = load(rd, T.DEV)
        else:  # normal H-Net (no latent objective)
            cfg, lat = ck["cfg"], None
            model = build_model(cfg["model"], device=T.DEV)
            model.load_state_dict(ck["model"])
        del ck
        mc, tc = cfg["model"], dict(cfg["train"])
        data = T.Data.__new__(T.Data)
        data.bpe, data.tok_bytes, data.bytes_per_token = False, None, 1.0
        data.val = np.fromfile(dd / "val.u8", dtype=np.uint8)
        res = {}
        for prefix, f in SETS.items():
            if (dd / f).exists():
                res.update(T.evaluate(model, data, mc, tc, lat=lat, max_windows=a.windows,
                                      val=np.fromfile(dd / f, dtype=np.uint8), prefix=prefix))
        json.dump(res, open(Path(run) / "extra_val.json", "w"), indent=1)
        rows[Path(run).name] = res
        del model, lat
    cols = [f"{p}/{k}" for p in SETS for k in KEYS]
    print("| run | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for nm, r in rows.items():
        print(f"| {nm} | " + " | ".join("–" if c not in r else f"{r[c]:.4f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
