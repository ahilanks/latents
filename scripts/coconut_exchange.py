"""Does the state the decoder reads after thinking carry the hidden value by itself? Exchange test for a Coconut model.

For problems at stage s (s steps replaced by thoughts): generate the first number written after thinking (the hidden
value) with the model's own read-out state, then again with only that state (the main-network output the decoder
reads at the <stop> chunk) taken from another problem. If the state carries the value, the model now writes the other
problem's (the donor's) value; if the value is made jointly with something else, it writes neither.
usage: python scripts/coconut_exchange.py runs/<name> --stage 2 [--n 512] [--out logs/coconut/exchange_<name>.json]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import latents  # noqa: E402,F401
import torch  # noqa: E402

from latents import tasks  # noqa: E402
from latents.coconut import BOS, START, STOP, Coconut  # noqa: E402
from latents.think_lc import step_spans  # noqa: E402
from show_cuts import load_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg, model, _ = load_any(a.run, "cuda")
    cc = cfg.get("coconut", {})
    co = Coconut(model, cfg["model"], c=cc.get("c", 1), feed=cc.get("feed", "full"), read=cc.get("read", "stop"),
                 fresh=cc.get("fresh", 0), ground=cc.get("ground", 0.0))
    ac = dict(device_type="cuda", dtype=torch.bfloat16)
    rng = random.Random(5)
    probs = []
    while len(probs) < a.n:
        p = tasks.sample(rng, steps=(2, 6))
        if len(step_spans(p["cot"])) > a.stage:
            probs.append(p)
    real = co.bb.dechunk_layer
    st = {"mode": "none", "saved": None, "perm": None, "stops": None}

    class Patch(torch.nn.Module):  # exchanges the <stop> chunk's state just before it is spread over the bytes
        def forward(self, zc, mask, prob, cu1):
            B = len(st["stops"])
            mv = mask.view(B, -1)
            idx = torch.stack([cu1[b].long() + mv[b, : st["stops"][b]].sum() for b in range(B)])
            if st["mode"] == "record":
                st["saved"] = zc[idx].clone()
            elif st["mode"] == "patch":
                zc = zc.clone()
                zc[idx] = st["saved"][st["perm"]]
            return real(zc, mask, prob, cu1)
    co.bb.dechunk_layer = Patch()

    def gen(ps, mode, perm=None, max_new=6):
        ctx = [bytearray(bytes([BOS]) + p["q"].encode() + bytes([START, STOP])) for p in ps]
        st["stops"], st["perm"] = [len(c) - 1 for c in ctx], perm
        starts = [len(c) for c in ctx]
        th = None
        for i in range(max_new):
            st["mode"] = mode if (mode != "record" or i == 0) else "none"
            with torch.no_grad(), torch.autocast(**ac):
                logits, _, _, th = co.forward([bytes(c) for c in ctx], st["stops"], [co.c * a.stage] * len(ps), thoughts=th)
            for b in range(len(ctx)):
                ctx[b].append(int(logits[b, len(ctx[b]) - 1].argmax()))
        return [bytes(c[s:]).decode("utf-8", "replace") for c, s in zip(ctx, starts)]

    def val(txt):
        m = re.match(r"[a-z]=(\d+)", txt)
        return int(m.group(1)) if m else -1
    true = [step_spans(p["cot"])[a.stage - 1][3] for p in probs]
    keys = ("own_state_own_value", "other_state_own_value", "other_state_other_value")
    res = {k: [0, 0, 0] for k in keys}
    n = 0
    for i in range(0, len(probs), 128):
        ps = probs[i:i + 128]
        outs = gen(ps, "record")
        perm = torch.tensor(list(range(1, len(ps))) + [0], device="cuda")
        outs_p = gen(ps, "patch", perm)
        for b in range(len(ps)):
            t, d = true[i + b], true[i + int(perm[b])]
            if t == d:
                continue  # the two problems share the value: nothing to tell apart
            n += 1
            for key, got, tr in ((keys[0], val(outs[b]), t), (keys[1], val(outs_p[b]), t), (keys[2], val(outs_p[b]), d)):
                res[key][0] += got == tr
                res[key][1] += got >= 0 and got % 10 == tr % 10
                res[key][2] += got >= 0 and got // 10 == tr // 10
    out = {"run": a.run, "stage": a.stage, "n": n}
    for k, v in res.items():
        out[k] = v[0] / n
        out[k + "_ones"] = v[1] / n
        out[k + "_tens"] = v[2] / n
        print(f"{k:26s} exact {v[0] / n:.3f}  ones {v[1] / n:.3f}  tens {v[2] / n:.3f}", flush=True)
    path = ROOT / (a.out or f"logs/coconut/exchange_{Path(a.run).name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
