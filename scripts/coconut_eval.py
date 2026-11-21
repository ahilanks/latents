"""Evaluate a Coconut-trained H-Net (scripts/coconut_train.py) on the arithmetic test set.

Modes (greedy decoding, one problem's answer = the number after its last '####'):
  all_thoughts   question, <start>, the model's own thoughts (c per written step, the count read off the question),
                 <stop>, then the model writes only the answer — the Coconut result
  no_thoughts    the same with zero thoughts (control: do the thoughts carry the reasoning?)
  written        no markers, the model writes its reasoning (did written reasoning survive?)
By number of computation steps (2-6 = training lengths, 7-8 = longer).

usage: python scripts/coconut_eval.py runs/<name> [--n 2000] [--modes all_thoughts,no_thoughts,written]
Writes runs/<name>/coconut_eval.json and prints a table.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents.coconut import Coconut, score  # noqa: E402
from show_cuts import load_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--modes", default="all_thoughts,no_thoughts,written")
    ap.add_argument("--data_dir", default="data/task_arith")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", default="", help="write coconut_eval_<tag>.json (e.g. another test set) instead of coconut_eval.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = a.run if Path(a.run).is_absolute() else str(ROOT / a.run)
    cfg, model, _ = load_any(Path(run).relative_to(ROOT) if run.startswith(str(ROOT)) else run, dev)
    cc = cfg.get("coconut", {})
    co = Coconut(model, cfg["model"], c=cc.get("c", 1), feed=cc.get("feed", "full"), read=cc.get("read", "stop"),
                 fresh=cc.get("fresh", 0), ground=cc.get("ground", 0.0))
    probs = [json.loads(l) for l in open(ROOT / a.data_dir / "test.jsonl")][: a.n]
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    res = {"run": a.run, "coconut": cc, "n": len(probs)}
    for mode in a.modes.split(","):
        outs = []
        with torch.autocast(**ac):
            for i in range(0, len(probs), a.batch):
                chunk = probs[i:i + a.batch]
                if mode == "all_thoughts":
                    outs += co.answer(chunk, stage=None)
                elif mode == "no_thoughts":
                    outs += co.answer(chunk, stage=None, n_thoughts=[0] * len(chunk))
                elif mode == "written":
                    outs += co.answer(chunk, stage=0, max_new=400)
                else:
                    raise SystemExit(f"unknown mode {mode}")
        res[mode] = score(probs, outs)
        res[mode]["examples"] = [dict(q=p["q"].strip(), out=o[0], value=p["value"]) for p, o in zip(probs[:5], outs[:5])]
        r = res[mode]
        print(f"{mode:13s} acc {r['acc']:.3f}  by steps {r['by_steps']}  thoughts {r['mean_thoughts']:.1f}  "
              f"bytes written {r['mean_bytes_written']:.1f}", flush=True)
    json.dump(res, open(Path(run) / (f"coconut_eval_{a.tag}.json" if a.tag else "coconut_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
