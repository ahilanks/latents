"""Print progress / results of local runs.  usage: python scripts/status.py [substring]"""
import json, sys
from pathlib import Path

pat = sys.argv[1] if len(sys.argv) > 1 else ""
for d in sorted(Path(__file__).resolve().parents[1].glob("runs/*/")):
    if pat not in d.name or not (d / "metrics.jsonl").exists():
        continue
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    tr = [r for r in rows if "train/ce" in r]
    ev = [r for r in rows if "val/bpb" in r]
    s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else None
    t = tr[-1] if tr else {}
    msg = f"{d.name:42s} step {t.get('step', 0):>6} s/step {t.get('perf/sec_per_step', 0):.3f} train_bpb {t.get('train/bpb', 0):.4f}"
    if "train/compression_L1/L0" in t:
        msg += f" L1/L0 {t['train/compression_L1/L0']:.3f}"
    if ev:
        msg += f" | val_bpb {ev[-1]['val/bpb']:.4f}"
    if s:
        msg += f" | FINAL val_bpb {s['val/bpb']:.4f} ({s['train_flops']:.2e} FLOPs, {s['wall_min']:.0f} min)"
    print(msg)
