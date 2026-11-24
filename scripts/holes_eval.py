"""Final answers right on a reasoning test set (latents/holes.py), greedy, one problem per row; models with the loop run
the exact loop (one chunk at a time) unless --rounds is given.

Modes (comma list):
  written        the model writes its reasoning and the answer
  hidden         every step hidden: question, --per_step markers per step (the count read off the question), answer
  stage<s>       the first s steps hidden, the rest written
  nothoughts     every step deleted, nothing in its place (answering directly)
usage: python scripts/holes_eval.py runs/<name> [--task arith|logic] [--modes written] [--n 2000] [--data_dir ...]
           [--tag long] [--rounds 0]
Writes runs/<name>/holes_eval[_<tag>].json (merged with what is already there) and prints a table.
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

from latents import holes as H  # noqa: E402
from show_cuts import load_any  # noqa: E402

DATA = {"arith": "data/task_arith", "logic": "data/task_logic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--task", choices=list(DATA), default="arith")
    ap.add_argument("--modes", default="written")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--per_step", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=0, help="parallel rounds instead of the exact loop (0: exact loop)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--name", default="", help="label for the job runner's log (unused)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model, _ = load_any(a.run, dev)
    mc = cfg["model"]
    if not mc.get("holes"):
        mc["holes"] = {"mark": H.MARK}
    probs = [json.loads(l) for l in open(ROOT / (a.data_dir or DATA[a.task]) / "test.jsonl")][: a.n]
    ac = dict(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev == "cuda")
    path = ROOT / a.run / (f"holes_eval_{a.tag}.json" if a.tag else "holes_eval.json")
    res = json.load(open(path)) if path.exists() else {}
    res.update({"run": a.run, "n": len(probs), "loop": "exact" if not a.rounds else f"{a.rounds} rounds",
                "has_loop": bool(mc.get("feedback"))})
    for mode in a.modes.split(","):
        k, marks = {"written": (0, True), "hidden": (None, True), "nothoughts": (None, False)}.get(mode, (None, True))
        if mode.startswith("stage"):
            k = int(mode[5:])
        with torch.autocast(**ac):
            r = H.evaluate_reasoning(model, mc, probs, k, a.per_step, marks, exact=not a.rounds, bs=a.batch,
                                     rounds=a.rounds or None)
        res[mode] = r
        print(f"{mode:11s} acc {r['acc']:.3f}  by steps {r['by_steps']}  bytes written {r['mean_bytes_written']:.1f}",
              flush=True)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
