"""Evaluations beyond FineWeb-Edu val BPB, for any finished run (BPE, byte Transformer, H-Net, latent, nested).

usage:
  python scripts/eval_suite.py fetch                                download eval data into data/eval/
  python scripts/eval_suite.py run runs/<name> [runs/<name> ...] [--tasks bpb,perturb,mc,chunks] [--limit N]
  python scripts/eval_suite.py table runs/<name> [...]              markdown table of runs/<name>/evals.json

Tasks
  bpb      BPB on out-of-distribution text: code, Chinese / English Wikipedia, FineWeb-2 languages
  perturb  BPB on FineWeb-Edu val text after BLT/H-Net-style noise (antspeak, drop, randomcase, repeat, uppercase)
  mc       zero-shot multiple choice scored by the log-likelihood of each continuation: hellaswag, piqa, arc_easy,
           sciq (acc, acc_norm = per byte, bpb of the correct answer); lambada (target-word bpb + greedy exact
           match, from one teacher-forced pass); blimp (good vs bad sentence)
  chunks   boundary stats on val (edge precision, space recall), morpheme-boundary F1 against SIGMORPHON 2022
           (if fetched), linear probes on chunk latents (word identity of the current chunk / of the next chunk)
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "data" / "eval"
BOS = 254
LN2 = math.log(2)


# ----------------------------------------------------------------------------- data fetch
def _take_text(ds, field, n_bytes):
    out, n = [], 0
    for ex in ds:
        t = ex[field]
        if not t:
            continue
        out.append(t)
        n += len(t.encode("utf-8")) + 1
        if n >= n_bytes:
            break
    return out


def fetch(n_text=4_000_000):
    from datasets import load_dataset
    (EVAL / "text").mkdir(parents=True, exist_ok=True)
    (EVAL / "mc").mkdir(parents=True, exist_ok=True)
    texts = {
        "code": ("codeparrot/codeparrot-clean-valid", None, "valid", "content"),
        "en_wiki": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
        "zh_wiki": ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
        "fw2_rus": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "test", "text"),
        "fw2_cmn": ("HuggingFaceFW/fineweb-2", "cmn_Hani", "test", "text"),
        "fw2_hin": ("HuggingFaceFW/fineweb-2", "hin_Deva", "test", "text"),
        "fw2_arb": ("HuggingFaceFW/fineweb-2", "arb_Arab", "test", "text"),
        "fw2_deu": ("HuggingFaceFW/fineweb-2", "deu_Latn", "test", "text"),
    }
    for name, (path, conf, split, field) in texts.items():
        try:
            ds = load_dataset(path, conf, split=split, streaming=True)
            docs = _take_text(ds, field, n_text)
            (EVAL / "text" / f"{name}.txt").write_text("\x00".join(docs))
            print("text", name, len(docs), "docs")
        except Exception as e:
            print("SKIP text", name, type(e).__name__, str(e)[:200])

    def dump(name, rows):
        with open(EVAL / "mc" / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("mc", name, len(rows))

    def pre_hs(t):
        t = t.strip().replace(" [title]", ". ")
        import re
        t = re.sub(r"\[.*?\]", "", t)
        return t.replace("  ", " ")

    jobs = {
        "hellaswag": lambda: [dict(ctx=pre_hs(e["activity_label"] + ": " + e["ctx_a"] + " " + e["ctx_b"].capitalize()),
                                   choices=[" " + pre_hs(x) for x in e["endings"]], label=int(e["label"]))
                              for e in load_dataset("Rowan/hellaswag", split="validation")],
        "arc_easy": lambda: [dict(ctx=f"Question: {e['question']}\nAnswer:", choices=[" " + c for c in e["choices"]["text"]],
                                  label=e["choices"]["label"].index(e["answerKey"]))
                             for e in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")],
        "sciq": lambda: [dict(ctx=f"{e['support'].strip()}\nQuestion: {e['question']}\nAnswer:".lstrip(),
                              choices=[" " + e[k] for k in ("distractor1", "distractor2", "distractor3", "correct_answer")], label=3)
                         for e in load_dataset("allenai/sciq", split="test")],
        "lambada": lambda: [dict(ctx=e["text"].rsplit(" ", 1)[0], target=" " + e["text"].rsplit(" ", 1)[1])
                            for e in load_dataset("EleutherAI/lambada_openai", "default", split="test")],
    }
    for name, fn in jobs.items():
        try:
            dump(name, fn())
        except Exception as e:
            print("SKIP mc", name, type(e).__name__, str(e)[:200])
    for path in ("ybisk/piqa", "baber/piqa"):
        try:
            ds = load_dataset(path, split="validation")
            dump("piqa", [dict(ctx=f"Question: {e['goal']}\nAnswer:", choices=[" " + e["sol1"], " " + e["sol2"]], label=int(e["label"]))
                          for e in ds])
            break
        except Exception as e:
            print("SKIP piqa from", path, type(e).__name__, str(e)[:120])
    try:
        from datasets import get_dataset_config_names
        rows = []
        for conf in get_dataset_config_names("nyu-mll/blimp"):
            for e in load_dataset("nyu-mll/blimp", conf, split="train"):
                rows.append(dict(good=e["sentence_good"], bad=e["sentence_bad"], subset=conf))
        dump("blimp", rows)
    except Exception as e:
        print("SKIP blimp", type(e).__name__, str(e)[:200])
    (EVAL / "seg").mkdir(parents=True, exist_ok=True)
    import urllib.request
    for fname in ("eng.word.test.gold.tsv", "eng.word.dev.tsv"):
        url = f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{fname}"
        try:
            urllib.request.urlretrieve(url, EVAL / "seg" / fname)
            print("seg", fname)
        except Exception as e:
            print("SKIP seg", fname, type(e).__name__)


# ----------------------------------------------------------------------------- model wrapper
class Scorer:
    """Byte-level log-likelihoods for any of our checkpoints (BPE models via token log-probs)."""

    def __init__(self, run_dir, device=None):
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import latents  # noqa: F401
        from latents.models import build_model
        from latents import latent as latent_lib
        self.torch = torch
        ck = torch.load(Path(run_dir) / "final.pt", map_location=device, weights_only=False)
        self.cfg = ck["cfg"]
        self.mc = self.cfg["model"]
        self.device = device
        self.model = build_model(self.mc, device=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.lat = None
        if self.cfg.get("latent"):
            self.lat = latent_lib.build(self.cfg["latent"], self.model, self.mc)
            self.lat.load_state_dict(ck["latent"])
        self.bpe = self.mc.get("tokenizer") == "gpt2"
        self.L = self.mc["seq_len"]
        if self.bpe:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")

    def forward(self, x):
        from latents.models import model_forward
        with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16, enabled=self.device.startswith("cuda")):
            return model_forward(self.model, x, self.mc, self.lat)

    def _batch_logp(self, seqs, bs=16):
        """seqs: list of token lists (each starts with BOS / <|endoftext|>). Returns per-position log p(next)."""
        torch = self.torch
        out = []
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            n = max(len(s) for s in chunk)
            n = (n + 63) // 64 * 64
            x = torch.zeros(len(chunk), n, dtype=torch.long, device=self.device)
            for r, s in enumerate(chunk):
                x[r, :len(s)] = torch.tensor(s, device=self.device)
            lp = torch.log_softmax(self.forward(x).logits.float(), -1)
            for r, s in enumerate(chunk):
                tgt = torch.tensor(s[1:], device=self.device)
                out.append(lp[r, : len(s) - 1].gather(-1, tgt[:, None]).squeeze(-1).cpu())
        return out

    def tokens(self, text_bytes):
        if self.bpe:
            return self.enc.encode_ordinary(text_bytes.decode("utf-8", "replace"))
        return list(text_bytes)

    def bos(self):
        return [self.enc.eot_token] if self.bpe else [BOS]

    def continuation(self, pairs, bs=16):
        """pairs: [(ctx_bytes, cont_bytes)] -> [(sum log p of cont in nats, n_bytes(cont), greedy_exact)]."""
        torch = self.torch
        seqs, spans = [], []
        for ctx, cont in pairs:
            c, t = self.tokens(ctx), self.tokens(cont)
            s = (self.bos() + c + t)[-self.L:]
            seqs.append(s)
            spans.append(len(t))
        res = []
        for start in range(0, len(seqs), bs):
            chunk = seqs[start:start + bs]
            n = (max(len(s) for s in chunk) + 63) // 64 * 64
            x = torch.zeros(len(chunk), n, dtype=torch.long, device=self.device)
            for r, s in enumerate(chunk):
                x[r, :len(s)] = torch.tensor(s, device=self.device)
            lp = torch.log_softmax(self.forward(x).logits.float(), -1)
            for r, s in enumerate(chunk):
                k = spans[start + r]
                pos = torch.arange(len(s) - 1 - k, len(s) - 1, device=self.device)
                tgt = torch.tensor(s[len(s) - k:], device=self.device)
                lps = lp[r, pos].gather(-1, tgt[:, None]).squeeze(-1)
                greedy = bool((lp[r, pos].argmax(-1) == tgt).all())
                res.append((float(lps.sum()), len(pairs[start + r][1]), greedy))
        return res

    def text_bpb(self, text_bytes, max_windows=256):
        """BPB of a long text in non-overlapping windows of the model's context (like latents.train.evaluate)."""
        torch = self.torch
        toks = self.tokens(text_bytes)
        L = self.L
        n = min((len(toks) - 1) // L, max_windows)
        if n == 0:
            return None
        ce = nbytes = 0.0
        if self.bpe:
            tb = [len(self.enc.decode_single_token_bytes(t)) for t in toks[: n * L + 1]]
        for i in range(0, n, 8):
            rows = [toks[j * L: j * L + L + 1] for j in range(i, min(n, i + 8))]
            x = torch.tensor([r[:-1] for r in rows], device=self.device)
            y = torch.tensor([r[1:] for r in rows], device=self.device)
            logits = self.forward(x).logits.float()
            ce += float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
            if self.bpe:
                nbytes += sum(sum(tb[j * L + 1: j * L + L + 1]) for j in range(i, min(n, i + 8)))
            else:
                nbytes += y.numel()
        return ce / LN2 / nbytes


# ----------------------------------------------------------------------------- tasks
PERTURB = {
    "antspeak": lambda s, r: " ".join(s),
    "drop": lambda s, r: "".join(c for c in s if r.random() > 0.1),
    "randomcase": lambda s, r: "".join(c.swapcase() if r.random() < 0.5 else c for c in s),
    "repeat": lambda s, r: "".join(c * (2 if r.random() < 0.2 else 1) for c in s),
    "uppercase": lambda s, r: s.upper(),
}


def task_bpb(sc, limit):
    out = {}
    for p in sorted((EVAL / "text").glob("*.txt")):
        t = p.read_text(errors="replace").replace("\x00", "\n").encode("utf-8")
        out[p.stem] = sc.text_bpb(t, max_windows=limit or 256)
    return out


def task_perturb(sc, limit):
    import numpy as np
    val = np.fromfile(ROOT / "data" / "fwe" / "val.u8", dtype=np.uint8)[:1_000_000].tobytes().decode("utf-8", "replace")
    out = {"clean": sc.text_bpb(val.encode("utf-8"), max_windows=limit or 64)}
    for name, fn in PERTURB.items():
        out[name] = sc.text_bpb(fn(val, random.Random(0)).encode("utf-8"), max_windows=limit or 64)
        out[name + "_vs_clean"] = out[name] - out["clean"] if out[name] and out["clean"] else None
    return out


def load_jsonl(name, limit):
    p = EVAL / "mc" / f"{name}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if limit:
        random.Random(0).shuffle(rows)
        rows = rows[:limit]
    return rows


def task_mc(sc, limit):
    out = {}
    for name in ("hellaswag", "piqa", "arc_easy", "sciq"):
        rows = load_jsonl(name, limit)
        if not rows:
            continue
        pairs = [(r["ctx"].encode(), c.encode()) for r in rows for c in r["choices"]]
        sc_ = sc.continuation(pairs)
        i = acc = accn = bpb_c = 0
        for r in rows:
            k = len(r["choices"])
            s = sc_[i:i + k]
            i += k
            acc += max(range(k), key=lambda j: s[j][0]) == r["label"]
            accn += max(range(k), key=lambda j: s[j][0] / max(s[j][1], 1)) == r["label"]
            bpb_c += -s[r["label"]][0] / LN2 / max(s[r["label"]][1], 1)
        out[name] = dict(acc=acc / len(rows), acc_norm=accn / len(rows), bpb_correct=bpb_c / len(rows), n=len(rows))
    rows = load_jsonl("lambada", limit)
    if rows:
        s = sc.continuation([(r["ctx"].encode(), r["target"].encode()) for r in rows])
        out["lambada"] = dict(acc=sum(g for _, _, g in s) / len(s),
                              bpb_target=sum(-lp / LN2 / max(n, 1) for lp, n, _ in s) / len(s), n=len(s))
    rows = load_jsonl("blimp", limit * 10 if limit else None)
    if rows:
        g = sc.continuation([(b"", r["good"].encode()) for r in rows])
        b = sc.continuation([(b"", r["bad"].encode()) for r in rows])
        per = {}
        for r, x, y in zip(rows, g, b):
            per.setdefault(r["subset"], []).append(x[0] > y[0])
        out["blimp"] = dict(acc=sum(sum(v) for v in per.values()) / len(rows),
                            acc_macro=sum(sum(v) / len(v) for v in per.values()) / len(per), n=len(rows))
    return out


def task_chunks(sc, limit):
    torch = sc.torch
    if sc.bpe or len(sc.mc["d_model"]) == 1:
        return {}
    import numpy as np
    out = {}
    # morpheme boundary F1 (SIGMORPHON 2022 word-level: "word\tmor @@phe @@me\t...")
    for p in sorted((EVAL / "seg").glob("*.tsv")):
        tp = fp = fn = 0
        rows = [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8") if "\t" in l]
        if limit:
            rows = rows[: limit * 10]
        for i in range(0, len(rows), 64):
            batch = rows[i:i + 64]
            seqs = [[BOS, 32] + list(r[0].encode()) for r in batch]
            n = (max(len(s) for s in seqs) + 63) // 64 * 64
            x = torch.zeros(len(seqs), n, dtype=torch.long, device=sc.device)
            for j, s in enumerate(seqs):
                x[j, :len(s)] = torch.tensor(s, device=sc.device)
            mask = sc.forward(x).bpred[0].boundary_mask.view(x.shape).cpu()
            for j, r in enumerate(batch):
                morphs = r[1].replace(" @@", "@@").split("@@") if "@@" in r[1] else r[1].split()
                gold, off = set(), 0
                for m in morphs[:-1]:
                    off += len(m.strip().encode())
                    gold.add(off)
                wl = len(r[0].encode())
                pred = {k for k in range(1, wl) if bool(mask[j, 2 + k])}
                tp += len(gold & pred)
                fp += len(pred - gold)
                fn += len(gold - pred)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"morph_f1_{p.stem}"] = 2 * prec * rec / max(prec + rec, 1e-9)
        out[f"morph_prec_{p.stem}"], out[f"morph_rec_{p.stem}"] = prec, rec
    # linear probes on chunk latents: word id of the word containing the chunk (current) and of the next chunk
    val = np.fromfile(ROOT / "data" / "fwe" / "val.u8", dtype=np.uint8)
    L = sc.L
    feats, nfeats, cur, nxt = [], [], [], []
    words = {}
    nwin = min(limit or 48, (len(val) - 1) // L)
    for w0 in range(0, nwin, 4):
        xs = np.stack([val[j * L:(j + 1) * L] for j in range(w0, min(nwin, w0 + 4))]).astype(np.int64)
        x = torch.from_numpy(xs).to(sc.device)
        o = sc.forward(x)
        mask = o.bpred[0].boundary_mask.view(x.shape).cpu().numpy()
        pred = sc.lat.head(o.z_full).float().cpu() if sc.lat is not None else o.z.float().cpu()
        ch = o.chunks.float().cpu()
        k = 0
        for r in range(xs.shape[0]):
            text = xs[r]
            starts = np.nonzero(mask[r])[0]
            ids = []
            for s_ in starts:
                a = s_
                while a > 0 and text[a - 1] not in (32, 10, 9):
                    a -= 1
                b = s_
                while b < L and text[b] not in (32, 10, 9):
                    b += 1
                wd = bytes(text[a:b]).lower()
                ids.append(words.setdefault(wd, len(words)) if wd else -1)
            for t in range(len(starts)):
                feats.append(ch[k + t])
                cur.append(ids[t])
                if t + 1 < len(starts):
                    nfeats.append(pred[k + t])
                    nxt.append(ids[t + 1])
            k += len(starts)
    for tag, X, Y in (("probe_current_word", feats, cur), ("probe_next_word", nfeats, nxt)):
        Y = torch.tensor(Y)
        top = torch.bincount(Y[Y >= 0]).argsort(descending=True)[:1000]
        keep = torch.isin(Y, top)
        Xk, Yk = torch.stack(X)[keep], Y[keep]
        remap = {int(w): i for i, w in enumerate(top)}
        Yk = torch.tensor([remap[int(y)] for y in Yk])
        n = len(Yk)
        if n < 1000:
            continue
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
        clf = torch.nn.Linear(Xk.shape[1], len(top))
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        mu, sd = Xk[tr].mean(0), Xk[tr].std(0) + 1e-6
        Z = (Xk - mu) / sd
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(clf(Z[tr]), Yk[tr]).backward()
            opt.step()
        out[tag] = float((clf(Z[te]).argmax(-1) == Yk[te]).float().mean())
        out[tag + "_majority"] = float((Yk[te] == torch.bincount(Yk[tr]).argmax()).float().mean())
    return out


TASKS = {"bpb": task_bpb, "perturb": task_perturb, "mc": task_mc, "chunks": task_chunks}


def run(dirs, tasks, limit):
    for d in dirs:
        sc = Scorer(d)
        p = Path(d) / "evals.json"
        res = json.load(open(p)) if p.exists() else {}
        for t in tasks:
            print(f"[{Path(d).name}] {t}", flush=True)
            res[t] = TASKS[t](sc, limit)
            json.dump(res, open(p, "w"), indent=1)
        print(json.dumps(res, indent=1))
        del sc


def table(dirs):
    rows = {Path(d).name: json.load(open(Path(d) / "evals.json")) for d in dirs if (Path(d) / "evals.json").exists()}
    keys = []
    for r in rows.values():
        for t, v in r.items():
            for k, x in v.items():
                if isinstance(x, dict):
                    keys += [f"{t}/{k}/{m}" for m in x if m != "n"]
                else:
                    keys.append(f"{t}/{k}")
    keys = list(dict.fromkeys(keys))
    print("| metric | " + " | ".join(rows) + " |")
    print("|---" * (len(rows) + 1) + "|")
    for k in keys:
        vals = []
        for r in rows.values():
            v = r
            for part in k.split("/"):
                v = v.get(part) if isinstance(v, dict) else None
            vals.append("–" if v is None else f"{v:.4f}")
        print(f"| {k} | " + " | ".join(vals) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "table"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--tasks", default="bpb,perturb,mc,chunks")
    ap.add_argument("--limit", type=int, default=0, help="examples per task (0 = all)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "run":
        run(a.runs, a.tasks.split(","), a.limit)
    else:
        table(a.runs)


if __name__ == "__main__":
    main()
