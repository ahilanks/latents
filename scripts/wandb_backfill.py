"""Upload a finished run's local metrics to W&B (for runs trained with wandb=False, or by a script that only writes
runs/<name>/metrics.jsonl, e.g. scripts/coconut_train.py before it logged to W&B).

usage: python scripts/wandb_backfill.py <name> [<name> ...] [--group G]
Each row of metrics.jsonl is logged at its "step" (nested dicts flattened with "/"); the config comes from
config.yaml or summary.json. A run that already exists on W&B under the same name is skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latents.train import WANDB_ENTITY, WANDB_PROJECT  # noqa: E402  (also loads .env)


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}/"))
        elif isinstance(v, (int, float)):
            out[f"{pre}{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--group", default=None)
    a = ap.parse_args()
    import wandb
    have = {r.name for r in wandb.Api(timeout=60).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": {"$in": a.names}})}
    for n in a.names:
        d = ROOT / "runs" / n
        if n in have:
            print("already on W&B:", n)
            continue
        if not (d / "metrics.jsonl").exists():
            print("no metrics.jsonl:", n)
            continue
        cfg = yaml.safe_load(open(d / "config.yaml")) if (d / "config.yaml").exists() else {}
        summ = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=n, group=a.group or cfg.get("group") or summ.get("group"),
                         config=cfg or {k: v for k, v in summ.items() if not isinstance(v, dict)}, dir=str(ROOT / "runs"),
                         notes="backfilled from metrics.jsonl")
        last = -1
        for line in open(d / "metrics.jsonl"):
            r = json.loads(line)
            step = int(r.pop("step"))
            wandb.log(flat(r), step=max(step, last))
            last = max(step, last)
        for k, v in flat(summ).items():
            run.summary[f"final/{k}"] = v
        wandb.finish()
        print("backfilled", n)


if __name__ == "__main__":
    main()
