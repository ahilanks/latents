"""Holes: pretraining text with some words deleted and a marker byte left in their place (NEXT_STEPS.md, plan of
2026-11-23; diagram docs/holes_recipe.png).

Data (pretraining): about `rate` of the words of a window are deleted, in runs of 1..max_run consecutive words
(truncated geometric, p = 1/2: 4/7, 2/7, 1/7). Each deleted word becomes one marker byte (MARK = 3, never in the web
text); the spaces inside a run go with it, so a run of k words becomes k markers in a row ("the \\x03\\x03 fox").
Runs never span a document separator, and the first and last word of a window (possibly cut by the window edge) are
never deleted. In the model (latents/models.py forward_lm, opts["holes"]) a marker is its own chunk and its slot gets
nothing but the fed-back output; there is no loss on predicting a marker byte; the bytes after it are normal targets.

Reasoning: thoughts are the same holes. Stage s of the curriculum replaces the first s written steps with `per_step`
markers each (the count read off the question); the model then writes the rest and the answer, with the exact loop.

Both tasks (arithmetic, latents/tasks.py; multi-hop English, latents/tasks_logic.py) write their reasoning as
sentences ending in ". " ("a=17. b=17+25=42." / "Max is a wumpus. Max is a yumpus."), split by `steps_of`.
"""
import re

import numpy as np
import torch
import torch.nn.functional as F

MARK = 3
BOS = 254
SEPS = (32, 9, 10, 13, BOS)  # word separators: whitespace and the document separator
RUN_P = np.array([4, 2, 1]) / 7.0  # run length 1, 2, 3 (truncated geometric, p = 1/2)


# ----------------------------------------------------------------------------- pretraining data
def start_prob(rate, max_run=3):
    """Chance to start a run at a free word so that about `rate` of the words are deleted. Per decision either one word
    is kept, or a run of mean length m is deleted and the next word kept: deleted share = q m / (1 + q m)."""
    p = RUN_P[:max_run] / RUN_P[:max_run].sum()
    m = float((np.arange(1, max_run + 1) * p).sum())
    return rate / ((1 - rate) * m)


def word_spans(w, seps=SEPS):
    """(starts, ends) of the maximal runs of non-separator bytes (ends exclusive)."""
    nz = ~np.isin(w, seps)
    d = np.diff(np.concatenate(([False], nz, [False])).astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def punch(w, rng, rate=0.1, max_run=3, mark=MARK):
    """Delete about `rate` of the words of byte window w in runs; each deleted word -> one marker byte.
    Returns (new window, number of words, number deleted)."""
    w = np.asarray(w, dtype=np.uint8)
    starts, ends = word_spans(w)
    n = len(starts)
    if n < 3 or rate <= 0:
        return w.copy(), n, 0
    p = RUN_P[:max_run] / RUN_P[:max_run].sum()
    cand = np.flatnonzero(rng.random(n) < start_prob(rate, max_run))
    lens = rng.choice(np.arange(1, max_run + 1), size=len(cand), p=p)
    pieces, prev, free, gone = [], 0, 1, 0
    for c, k in zip(cand.tolist(), lens.tolist()):
        if c < free or c >= n - 1:
            continue
        j = c
        while j < min(c + k - 1, n - 2) and BOS not in w[ends[j]:starts[j + 1]]:
            j += 1
        pieces += [w[prev:starts[c]], np.full(j - c + 1, mark, dtype=np.uint8)]
        prev, free, gone = ends[j], j + 2, gone + j - c + 1
    pieces.append(w[prev:])
    return np.concatenate(pieces), n, gone


def holed_window(src, i, L, rng, rate=0.1, max_run=3, mark=MARK):
    """L bytes of text with holes, read from src at offset i (more is read: deleting words shortens the text)."""
    need = L + L // 4 + 64
    while True:
        w, _, _ = punch(np.asarray(src[i:i + need]), rng, rate, max_run, mark)
        if len(w) >= L or i + need >= len(src):
            break
        need *= 2
    if len(w) < L:  # end of the stream: pad with separators (never a target: such windows are not drawn in practice)
        w = np.concatenate((w, np.full(L - len(w), BOS, dtype=np.uint8)))
    return w[:L]


# ----------------------------------------------------------------------------- masks on model inputs
def after_hole_targets(x, mark=MARK):
    """(B, L) bool over target positions t (target = byte t+1): the text right after a run of holes, from the byte after
    the last marker through the end of the first word that follows (the bytes that only a filled hole can help with)."""
    xn = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    out = np.zeros(xn.shape, dtype=bool)
    for b in range(xn.shape[0]):
        r = xn[b]
        h = r == mark
        ends_ = np.flatnonzero(h & ~np.concatenate((h[1:], [False])))  # last marker of each run
        if not len(ends_):
            continue
        ws, we = word_spans(r, SEPS + (mark,))
        for e in ends_:
            k = np.searchsorted(ws, e + 1)
            stop = we[k] if k < len(ws) else len(r)
            out[b, e: stop - 1] = True  # targets at positions e .. stop-2 = bytes e+1 .. stop-1
    return torch.from_numpy(out).to(x.device) if torch.is_tensor(x) else out


def target_mask(y, mark):
    """No loss on predicting a marker byte."""
    return torch.ones_like(y, dtype=torch.bool) if mark is None else y != mark


# ----------------------------------------------------------------------------- reasoning with holes
def steps_of(cot):
    """(start, end) of each written step (a sentence ending in '.') in a reasoning text."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S[^.]*\.", cot)]


def count_steps(q):
    """Written steps of a problem, read off its question: arithmetic, one per assignment (the restated first value
    counts); multi-hop English, one per hop plus the restated fact (two chains of equal length are given)."""
    if "Every " in q:
        return q.count("Every ") // 2 + 1
    return q.count(";")


def example(p, k, per_step=3, marks=True, mark=MARK, bos=BOS):
    """Bytes of one problem with its first k written steps hidden: question, per_step * k markers (marks=False: the
    steps are just deleted, the no-thoughts control), the rest of the reasoning, the answer. k = 0: written reasoning."""
    q = bytes([bos]) + p["q"].encode()
    sp = steps_of(p["cot"])
    k = min(k, len(sp))
    rest = p["cot"] if k == 0 else (p["cot"][sp[k][0]:] if k < len(sp) else "")
    holes = bytes([mark]) * (per_step * k if marks else 0)
    return q + holes + (rest + p["answer"]).encode(), len(q) + len(holes)


def pad_batch(seqs, device, mult=64):
    L = (max(len(s) for s in seqs) + mult - 1) // mult * mult
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(list(s))
    return x.to(device)


@torch.no_grad()
def generate(model, mc, prompts, max_new=160, exact=True, rounds=None, stop=b"\n####"):
    """Greedy continuation of each prompt (bytes). exact: the exact loop (one chunk at a time); else `rounds` parallel
    rounds (None: as trained). Stops a row at a newline after '####' (or after max_new bytes)."""
    from latents.models import feedback_mode, model_forward
    ctx = [bytearray(p) for p in prompts]
    n0 = [len(c) for c in ctx]
    done = [False] * len(ctx)
    dev = next(model.parameters()).device
    for _ in range(max_new):
        live = [b for b in range(len(ctx)) if not done[b]]
        if not live:
            break
        x = pad_batch([bytes(ctx[b]) for b in live], dev)
        with feedback_mode(model, mode="loop" if exact else None, rounds=None if exact else rounds):
            out = model_forward(model, x, mc, extra_opts={"row_lens": [len(ctx[b]) for b in live]})
        nxt = out.logits[torch.arange(len(live)), torch.tensor([len(ctx[b]) - 1 for b in live])].argmax(-1).tolist()
        for b, t in zip(live, nxt):
            ctx[b].append(t)
            gen = bytes(ctx[b][n0[b]:])
            if (b"####" in gen and gen.endswith(b"\n")) or len(gen) >= max_new:
                done[b] = True
    return [bytes(c[n:]).decode("utf-8", "replace") for c, n in zip(ctx, n0)]


def answer_of(txt):
    return txt.split("####")[-1].strip() if "####" in txt else ""


def score(probs, outs):
    by, right = {}, 0
    for p, txt in zip(probs, outs):
        ok = answer_of(txt) == str(p["value"])
        right += ok
        by.setdefault(p["steps"], []).append(ok)
    return {"acc": right / max(len(probs), 1), "by_steps": {k: sum(v) / len(v) for k, v in sorted(by.items())},
            "mean_bytes_written": sum(len(t.encode()) for t in outs) / max(len(outs), 1)}


def prompts_for(probs, k, per_step=3, marks=True, mark=MARK):
    """Prompts with the first k steps hidden (k None: every step), up to where the model writes."""
    out = []
    for p in probs:
        kk = count_steps(p["q"]) if k is None else min(k, count_steps(p["q"]))
        b, n = example(p, kk, per_step, marks, mark)
        out.append(b[:n])
    return out


def max_new_for(probs):
    """Room for the whole written reasoning and the answer (with slack): a row stops as soon as its answer is written."""
    return int(max(len(p["cot"]) + len(p["answer"]) for p in probs) * 1.5) + 16


def evaluate_reasoning(model, mc, probs, k, per_step=3, marks=True, exact=True, bs=128, rounds=None):
    """Answers right with the first k steps hidden (k = 0 written reasoning, None every step hidden)."""
    outs = []
    mn = max_new_for(probs)
    for i in range(0, len(probs), bs):
        chunk = probs[i:i + bs]
        outs += generate(model, mc, prompts_for(chunk, k, per_step, marks, hole_mark_of(mc)), max_new=mn, exact=exact,
                         rounds=rounds)
    res = score(probs, outs)
    res["examples"] = [dict(q=p["q"].strip(), out=o, value=p["value"]) for p, o in zip(probs[:4], outs[:4])]
    return res


def hole_mark_of(mc):
    h = mc.get("holes")
    return int(h.get("mark", MARK)) if h else MARK


# ----------------------------------------------------------------------------- pictures
def render_cuts(bytes_, mask, mark=MARK):
    """Text with '|' before every chunk; markers shown as '□'."""
    pieces, cur = [], bytearray()
    for b, m in zip(bytes_, mask):
        if m and cur:
            pieces.append(bytes(cur).decode("utf-8", "replace"))
            cur = bytearray()
        cur.append(b)
    pieces.append(bytes(cur).decode("utf-8", "replace"))
    return "|".join(p.replace("�", "?") for p in pieces).replace(chr(BOS), "<s>").replace(chr(mark), "□")
